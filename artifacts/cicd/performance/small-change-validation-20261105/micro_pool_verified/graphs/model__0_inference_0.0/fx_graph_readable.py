class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 1536, 14, 14]"):
        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:102 in <lambda>, code: eager_pool = lambda x: torch.nn.functional.avg_pool2d(x, 2, 2)
        avg_pool2d: "f32[128, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(arg0_1, [2, 2], [2, 2]);  arg0_1 = None
        return (avg_pool2d,)
