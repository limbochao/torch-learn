import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from torch_learn.profiler import NpuProfilerConfig, ProfileResultParser, TorchNpuProfiler


def write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ProfileResultParserTest(unittest.TestCase):
    def test_kernel_time_by_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_csv(
                root / "run_a" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv",
                "Name,Duration(us)\ntriton_add,10\ntriton_add,14\ntriton_mul,3\n",
            )

            summaries = ProfileResultParser(root).kernel_time_by_name(name_prefix="triton")

            self.assertEqual(summaries[0].key, "triton_add")
            self.assertEqual(summaries[0].count, 2)
            self.assertEqual(summaries[0].mean_us, 12)
            self.assertEqual(summaries[1].key, "triton_mul")
            self.assertEqual(summaries[1].mean_us, 3)

    def test_step_average_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_csv(
                root / "prof" / "ASCEND_PROFILER_OUTPUT" / "step_trace_time.csv",
                "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,Stage\n"
                "0,0,10,0,0,0,0,100\n"
                "0,1,10,0,0,0,0,120\n",
            )

            parser = ProfileResultParser(root)

            self.assertEqual(parser.average_step_time_us(), 110)
            self.assertEqual(parser.step_time_summary()[0].count, 2)

    def test_kernel_time_by_shape_uses_profile_label_when_shape_column_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_csv(
                root / "shape_1x32" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv",
                "Name,Duration(us)\ntriton_add,10\n",
            )
            write_csv(
                root / "shape_2x32" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv",
                "Name,Duration(us)\ntriton_add,20\n",
            )

            summaries = ProfileResultParser(root).kernel_time_by_shape(name_prefix="triton")
            result = {summary.key: summary.mean_us for summary in summaries}

            self.assertEqual(result["triton_add | shape=shape_1x32"], 10)
            self.assertEqual(result["triton_add | shape=shape_2x32"], 20)

    def test_default_total_steps_matches_profiler_schedule(self):
        profiler = TorchNpuProfiler(wait=2, warmup=1, active=3, repeat=2, skip_first=1)

        self.assertEqual(profiler.default_total_steps(), 13)

    def test_profiler_defaults_enable_shapes_and_stack_without_aic_metrics(self):
        self.assertTrue(NpuProfilerConfig().record_shapes)
        self.assertTrue(NpuProfilerConfig().with_stack)
        self.assertFalse(hasattr(NpuProfilerConfig(), "aic_metrics"))

    def test_profile_does_not_configure_aic_metrics(self):
        captured_experimental_config_kwargs = {}

        class FakeExperimentalConfig:
            def __init__(self, **kwargs):
                captured_experimental_config_kwargs.update(kwargs)

        fake_profiler = SimpleNamespace(
            ProfilerLevel=SimpleNamespace(Level1="level1"),
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            _ExperimentalConfig=FakeExperimentalConfig,
            schedule=lambda **kwargs: ("schedule", kwargs),
            tensorboard_trace_handler=lambda output_dir: ("handler", output_dir),
            profile=lambda **kwargs: ("profile", kwargs),
        )
        fake_torch_npu = SimpleNamespace(profiler=fake_profiler)

        with mock.patch.dict("sys.modules", {"torch_npu": fake_torch_npu}):
            _, profile_kwargs = TorchNpuProfiler("/tmp/prof").profile()

        self.assertEqual(captured_experimental_config_kwargs, {"profiler_level": "level1"})
        self.assertTrue(profile_kwargs["record_shapes"])
        self.assertTrue(profile_kwargs["with_stack"])


if __name__ == "__main__":
    unittest.main()
