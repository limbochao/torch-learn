class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64]", arg1_1: "f32[10000, 64]", arg2_1: "i64[2, 209978]", arg3_1: "f32[209978]", arg4_1: "f32[64]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[64, 64]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "f32[10000, 64]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None

        # File: /tmp/root_pyg/tmpqeocx1_f.py:88 in _lift, code: index = edge_index[dim]
        select: "i64[209978]" = torch.ops.aten.select.int(arg2_1, 0, 0)

        # File: /tmp/root_pyg/tmpqeocx1_f.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f32[209978, 64]" = torch.ops.aten.index.Tensor(mm, [select]);  mm = select = None

        # File: /tmp/root_pyg/tmpqeocx1_f.py:139 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[209978]" = torch.ops.aten.select.int(arg2_1, 0, 1);  arg2_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:241 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[209978, 1]" = torch.ops.aten.view.default(arg3_1, [-1, 1]);  arg3_1 = None
        mul: "f32[209978, 64]" = torch.ops.aten.mul.Tensor(view, index);  view = index = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[209978, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[209978, 64]" = torch.ops.aten.expand.default(view_1, [209978, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  full_default = expand = mul = None

        # File: /tmp/root_pyg/tmpqeocx1_f.py:247 in torch_dynamo_resume_in_forward_at_221, code: out = out + self.bias
        add: "f32[10000, 64]" = torch.ops.aten.add.Tensor(scatter_add, arg4_1);  scatter_add = arg4_1 = None
        return (add,)
