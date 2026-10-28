import torch
import common


@torch.no_grad()
def compare_tensor(ref, res):
    assert ref.shape == res.shape and ref.dtype == res.dtype, (ref.shape, res.shape, ref.dtype, res.dtype)
    diff = res - ref
    absolute = diff.abs()
    small = ~torch.isclose(ref, res, atol=1e-5, rtol=1e-5, equal_nan=True)
    runner = ~torch.isclose(ref, res, atol=0.02, rtol=0.02, equal_nan=True)
    counts = torch.stack([small.sum(), runner.sum(), (ref != res).sum(), (~torch.isfinite(res)).sum()]).cpu().tolist()
    floats = torch.stack([absolute.max(), diff.square().mean().sqrt(), ref.square().mean().sqrt(),
        (absolute / (1e-5 + 1e-5 * ref.abs())).max(), ref.abs().max()]).cpu().tolist()
    return dict(max_abs=floats[0], rms_diff=floats[1], ref_rms=floats[2],
        relative_l2=floats[1] / max(floats[2], 1e-30), max_tolerance_ratio=floats[3], ref_max=floats[4],
        mismatches_1e5=counts[0], mismatches_runner_002=counts[1], unequal=counts[2],
        nonfinite=counts[3], numel=ref.numel())

def compare_model(ref, res):
    row = dict(prediction=compare_tensor(ref[0], res[0]), loss=compare_tensor(ref[1], res[1]))
    for index, key in ((2, 'grads'), (3, 'params_after_step')):
        names = sorted(ref[index])
        a = torch.cat([ref[index][name].detach().reshape(-1) for name in names])
        b = torch.cat([res[index][name].detach().reshape(-1) for name in names])
        row[key] = compare_tensor(a, b)
        row[key]['failed_tensors_002'] = [name for name in names
            if not torch.allclose(ref[index][name], res[index][name], atol=0.02, rtol=0.02, equal_nan=True)]
    row['runner_same_002'] = bool(common.same(ref, res, fp64_ref=None, tol=0.02,
                                            equal_nan=True, log_error=lambda *a, **k: None))
    return row
