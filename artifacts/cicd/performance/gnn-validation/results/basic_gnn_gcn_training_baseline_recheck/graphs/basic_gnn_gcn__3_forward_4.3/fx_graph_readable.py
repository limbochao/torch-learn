class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 64]", primals_2: "f32[10000, 64]", primals_3: "i64[2, 209978]", primals_4: "f32[209978]", primals_5: "f32[64]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[64, 64]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[10000, 64]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None

        # File: /tmp/root_pyg/tmpcz_mm2eu.py:88 in _lift, code: index = edge_index[dim]
        select: "i64[209978]" = torch.ops.aten.select.int(primals_3, 0, 0)

        # File: /tmp/root_pyg/tmpcz_mm2eu.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f32[209978, 64]" = torch.ops.aten.index.Tensor(mm, [select]);  mm = select = None

        # File: /tmp/root_pyg/tmpcz_mm2eu.py:139 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[209978]" = torch.ops.aten.select.int(primals_3, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:241 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[209978, 1]" = torch.ops.aten.view.default(primals_4, [-1, 1])
        mul: "f32[209978, 64]" = torch.ops.aten.mul.Tensor(view, index);  view = index = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[209978, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[209978, 64]" = torch.ops.aten.expand.default(view_1, [209978, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  expand = mul = None

        # File: /tmp/root_pyg/tmpcz_mm2eu.py:247 in torch_dynamo_resume_in_forward_at_221, code: out = out + self.bias
        add: "f32[10000, 64]" = torch.ops.aten.add.Tensor(scatter_add, primals_5);  scatter_add = primals_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_1: "f32[64, 10000]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (add, primals_3, primals_4, full_default, permute_1)
