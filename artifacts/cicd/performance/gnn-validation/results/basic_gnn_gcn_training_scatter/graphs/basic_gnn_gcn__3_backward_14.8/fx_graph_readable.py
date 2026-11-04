class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[2, 209978]", primals_4: "f32[209978]", full_default: "f32[10000, 64]", permute_1: "f32[64, 10000]", tangents_1: "f32[10000, 64]"):
        # File: /tmp/root_pyg/tmpsa0jmb8n.py:247 in torch_dynamo_resume_in_forward_at_221, code: out = out + self.bias
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view_2: "f32[64]" = torch.ops.aten.view.default(sum_1, [64]);  sum_1 = None

        # File: /tmp/root_pyg/tmpsa0jmb8n.py:139 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[209978]" = torch.ops.aten.select.int(primals_3, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[209978, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[209978, 64]" = torch.ops.aten.expand.default(view_1, [209978, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        gather: "f32[209978, 64]" = torch.ops.aten.gather.default(tangents_1, 0, expand);  tangents_1 = expand = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:241 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[209978, 1]" = torch.ops.aten.view.default(primals_4, [-1, 1]);  primals_4 = None
        mul_1: "f32[209978, 64]" = torch.ops.aten.mul.Tensor(gather, view);  gather = view = None

        # File: /tmp/root_pyg/tmpsa0jmb8n.py:88 in _lift, code: index = edge_index[dim]
        select: "i64[209978]" = torch.ops.aten.select.int(primals_3, 0, 0);  primals_3 = None

        # File: /tmp/root_pyg/tmpsa0jmb8n.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        index_put: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select], mul_1, True);  full_default = select = mul_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        mm_1: "f32[64, 64]" = torch.ops.aten.mm.default(permute_1, index_put);  permute_1 = index_put = None
        permute_2: "f32[64, 64]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        return (permute_2, None, None, None, view_2)
