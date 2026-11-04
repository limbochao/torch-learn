class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10000, 64]"):
        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:155 in eager, code: return (a.sum(0, keepdim=True),)
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0], True);  arg0_1 = None
        return (sum_1,)
