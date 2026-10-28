"""Run the original TIMM benchmark, optionally collecting FP32 eager/Inductor trajectories."""
from pathlib import Path
import csv
import importlib.metadata
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TORCHINDUCTOR_EMULATE_PRECISION_CASTS', '0')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def configure_hf_download():
    """Opt in to a process-local certificate bypass for benchmark weight downloads."""
    if os.environ.get('HF_DOWNLOAD_INSECURE') != '1':
        return
    # Set before importing the Hub: requests/httpx settings do not affect Rust hf-xet.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    import huggingface_hub as hub

    if hasattr(hub, 'set_client_factory'):
        # Preserve the HTTP client type used by this installed Hub version.
        client_type = type(hub.get_session())
        hub.set_client_factory(lambda: client_type(verify=False, follow_redirects=True, timeout=60.0))
    else:
        import requests

        def backend_factory():
            session = requests.Session()
            session.verify = False
            return session

        hub.configure_http_backend(backend_factory=backend_factory)
    print('HF_DOWNLOAD_INSECURE=1: Hugging Face TLS verification disabled for this process; Xet disabled.',
          file=sys.stderr, flush=True)


print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)
configure_hf_download()

import torch
import common
from trajectory_metrics import compare_model
from timm_models import TimmRunner


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def diagnose(self, name, model, example_inputs, optimize_ctx, experiment, tag):
    assert self.args.training and self.args.float32 and self.args.devices == ['cuda']
    assert name == 'vit_base_patch14_dinov2.lvd142m'
    assert self.args.backend == 'inductor'
    lr = self.get_learning_rate(True, 'cuda', name)
    tolerance, cosine = self.get_tolerance_and_cosine_flag(True, 'cuda', name)
    assert lr == 0.01 and tolerance == 0.02 and not cosine
    out = Path(self.args.output).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    base, inputs = self.maybe_cast(model, example_inputs)
    assert tuple(inputs[0].shape) == (self.args.batch_size, 3, 518, 518), inputs[0].shape
    raw = {key: self.deepcopy_and_maybe_ddp(base) for key in ['eager', 'eager_check', 'compiled']}
    optimizers = {}
    for key, mod in raw.items():
        self.init_optimizer(name, 'cuda', mod.parameters(), lr)
        optimizers[key] = self.optimizer
        assert isinstance(self.optimizer, torch.optim.SGD)
        assert all(not group['foreach'] and group['momentum'] == 0 and group['weight_decay'] == 0
                   for group in self.optimizer.param_groups)
    torch._dynamo.reset()
    models = dict(raw)
    models['compiled'] = optimize_ctx(raw['compiled'])
    common.reset_rng_state()
    rng = {key: rng_state() for key in models}
    report = dict(model=name, baseline='848e9feaa86f1cefe82ea1307c62c2d8b89bfc6f',
                  iterations=self.args.iterations, batch_size=inputs[0].shape[0], input_shape=list(inputs[0].shape),
                  seed=1337, optimizer='SGD', lr=lr, foreach=False, tolerance=tolerance,
                  reference='same-input FP32 eager; no FP64 golden in trajectory mode',
                  first_failure={}, steps=[])
    versions = {package: importlib.metadata.version(package)
                for package in ['torch', 'torchvision', 'timm', 'triton']}
    write_json(out / 'environment.json', dict(
        versions=versions, torch_git=torch.version.git_version, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        emulate_precision_casts=torch._inductor.config.emulate_precision_casts,
        triton_cudagraphs=torch._inductor.config.triton.cudagraphs,
        input_note='Generated on CUDA using the original benchmark procedure; not bitwise-matched to NPU input.',
        command=sys.argv))
    with (out / 'trajectory.jsonl').open('w', buffering=1) as journal:
        for step in range(1, self.args.iterations + 1):
            outputs = {}
            for key in models:
                restore_rng(rng[key])
                self.optimizer = optimizers[key]
                outputs[key] = self.model_iter_fn(models[key], inputs, collect_outputs=True)
                rng[key] = rng_state()
            eager_same = bool(common.same(outputs['eager'], outputs['eager_check'], fp64_ref=None,
                                          tol=0, equal_nan=True, log_error=lambda *args, **kwargs: None))
            if not eager_same:
                report['eager_failure_step'] = step
                write_json(out / 'trajectory_summary.json', report)
                raise RuntimeError(f'eager_two_runs_differ at step {step}; Inductor accuracy is not determined')
            comparison = compare_model(outputs['eager'], outputs['compiled'])
            record = dict(step=step, eager_repeat_equal=eager_same, paths={'compiled_free': comparison})
            for field in ['prediction', 'loss', 'grads', 'params_after_step']:
                if comparison[field]['mismatches_runner_002']:
                    report['first_failure'].setdefault(field, step)
            if not comparison['runner_same_002']:
                report['first_failure'].setdefault('compiled_free', step)
            report['steps'].append(record)
            journal.write(json.dumps(record) + '\n')
            write_json(out / 'trajectory_summary.json', report)
            print('TRAJECTORY_STEP', step, json.dumps(dict(
                pass_002=comparison['runner_same_002'], prediction_max_abs=comparison['prediction']['max_abs'],
                grad_max_abs=comparison['grads']['max_abs'])), flush=True)
            del outputs
    status = 'pass_accuracy' if all(row['paths']['compiled_free']['runner_same_002']
                                    for row in report['steps']) else 'fail_accuracy'
    with Path(self.args.output).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dev', 'name', 'batch_size', 'accuracy'])
        writer.writerow(['cuda', name, inputs[0].shape[0], status])
    print('TRAJECTORY_RESULT', status, 'first_failure=', report['first_failure'], flush=True)
    return status


if __name__ == '__main__':
    trajectory = '--trajectory' in sys.argv
    if trajectory:
        sys.argv.remove('--trajectory')
        common.BenchmarkRunner.check_accuracy = diagnose
    common.main(TimmRunner())
