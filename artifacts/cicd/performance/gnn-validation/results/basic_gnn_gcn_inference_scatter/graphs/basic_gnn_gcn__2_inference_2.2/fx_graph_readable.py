class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 209978]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:95 in torch_dynamo_resume_in_gcn_norm_at_91, code: edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
        full_default: "f32[209978]" = torch.ops.aten.full.default([209978], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:98 in torch_dynamo_resume_in_gcn_norm_at_91, code: row, col = edge_index[0], edge_index[1]
        select: "i64[209978]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1: "i64[209978]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        expand: "i64[209978]" = torch.ops.aten.expand.default(select_1, [209978])

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default_1: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default_1, 0, expand, full_default);  full_default_1 = expand = full_default = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:101 in torch_dynamo_resume_in_gcn_norm_at_91, code: deg_inv_sqrt = deg.pow_(-0.5)
        pow_1: "f32[10000]" = torch.ops.aten.pow.Tensor_Scalar(scatter_add, -0.5);  scatter_add = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:102 in torch_dynamo_resume_in_gcn_norm_at_91, code: deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        eq: "b8[10000]" = torch.ops.aten.eq.Scalar(pow_1, inf)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[10000]" = torch.ops.aten.where.self(eq, full_default_2, pow_1);  eq = full_default_2 = pow_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/gcn_conv.py:103 in torch_dynamo_resume_in_gcn_norm_at_91, code: edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        index: "f32[209978]" = torch.ops.aten.index.Tensor(where, [select]);  select = None
        index_1: "f32[209978]" = torch.ops.aten.index.Tensor(where, [select_1]);  where = select_1 = None
        mul_1: "f32[209978]" = torch.ops.aten.mul.Tensor(index, index_1);  index = index_1 = None
        return (mul_1,)
