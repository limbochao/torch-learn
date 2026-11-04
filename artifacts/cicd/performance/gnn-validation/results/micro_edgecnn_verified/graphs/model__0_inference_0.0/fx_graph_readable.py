class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[200000, 64]", arg1_1: "f32[200000, 64]"):
        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:153 in eager, code: v = torch.where(a <= 0, 0., g)
        le: "b8[200000, 64]" = torch.ops.aten.le.Scalar(arg0_1, 0);  arg0_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[200000, 64]" = torch.ops.aten.where.self(le, full_default, arg1_1);  le = full_default = arg1_1 = None

        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:154 in eager, code: return v, v.sum(0, keepdim=True)
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where, [0], True)
        return (where, sum_1)
