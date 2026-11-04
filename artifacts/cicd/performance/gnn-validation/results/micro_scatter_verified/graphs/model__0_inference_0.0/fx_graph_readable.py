class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[209978]", arg1_1: "i64[2, 209978]", arg2_1: "f32[10000, 64]"):
        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:89 in eager, code: message = weight[:, None] * x[edge[0]]
        unsqueeze: "f32[209978, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
        select: "i64[209978]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        index: "f32[209978, 64]" = torch.ops.aten.index.Tensor(arg2_1, [select]);  arg2_1 = select = None
        mul: "f32[209978, 64]" = torch.ops.aten.mul.Tensor(unsqueeze, index);  unsqueeze = index = None

        # File: <job>/bundle/scripts/repro/cicd/performance/gnn_validation/run.py:90 in eager, code: return torch.zeros_like(x).scatter_add(0, edge[1, :, None].expand(-1, 64), message)
        full_default: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        select_1: "i64[209978]" = torch.ops.aten.select.int(arg1_1, 0, 1);  arg1_1 = None
        unsqueeze_1: "i64[209978, 1]" = torch.ops.aten.unsqueeze.default(select_1, 1);  select_1 = None
        expand: "i64[209978, 64]" = torch.ops.aten.expand.default(unsqueeze_1, [-1, 64]);  unsqueeze_1 = None
        scatter_add: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  full_default = expand = mul = None
        return (scatter_add,)
