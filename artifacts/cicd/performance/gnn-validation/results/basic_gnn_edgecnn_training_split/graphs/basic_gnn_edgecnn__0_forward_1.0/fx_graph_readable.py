class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10000, 64]", primals_2: "i64[2, 200000]", primals_3: "f32[64, 128]", primals_4: "f32[64]", primals_5: "f32[64, 64]", primals_6: "f32[64]", primals_7: "f32[64, 128]", primals_8: "f32[64]", primals_9: "f32[64, 64]", primals_10: "f32[64]", primals_11: "f32[64, 128]", primals_12: "f32[64]", primals_13: "f32[64, 64]", primals_14: "f32[64]"):
        # File: /tmp/root_pyg/tmpqr2ah85s.py:87 in _lift, code: index = edge_index[dim]
        select: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 0)

        # File: /tmp/root_pyg/tmpqr2ah85s.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f32[200000, 64]" = torch.ops.aten.index.Tensor(primals_1, [select])

        # File: /tmp/root_pyg/tmpqr2ah85s.py:87 in _lift, code: index = edge_index[dim]
        select_1: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 1);  primals_2 = None

        # File: /tmp/root_pyg/tmpqr2ah85s.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index_1: "f32[200000, 64]" = torch.ops.aten.index.Tensor(primals_1, [select_1]);  primals_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/edge_conv.py:63 in message, code: return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        sub: "f32[200000, 64]" = torch.ops.aten.sub.Tensor(index, index_1);  index = None
        cat: "f32[200000, 128]" = torch.ops.aten.cat.default([index_1, sub], -1);  index_1 = sub = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[128, 64]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_4, cat, permute);  primals_4 = permute = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        relu: "f32[200000, 64]" = torch.ops.aten.relu.default(addmm);  addmm = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm_1: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_6, relu, permute_1);  primals_6 = permute_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1])
        expand: "i64[200000, 64]" = torch.ops.aten.expand.default(view, [200000, 64]);  view = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        full_default: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_reduce: "f32[10000, 64]" = torch.ops.aten.scatter_reduce.two(full_default, 0, expand, addmm_1, 'amax', include_self = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu_1: "f32[10000, 64]" = torch.ops.aten.relu.default(scatter_reduce)

        # File: /tmp/root_pyg/tmpeb8dfx4a.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index_2: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_1, [select])
        index_3: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_1, [select_1])

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/edge_conv.py:63 in message, code: return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        sub_1: "f32[200000, 64]" = torch.ops.aten.sub.Tensor(index_2, index_3);  index_2 = None
        cat_1: "f32[200000, 128]" = torch.ops.aten.cat.default([index_3, sub_1], -1);  index_3 = sub_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_2: "f32[128, 64]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_2: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_8, cat_1, permute_2);  primals_8 = permute_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        relu_2: "f32[200000, 64]" = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_3: "f32[64, 64]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_3: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_10, relu_2, permute_3);  primals_10 = permute_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        scatter_reduce_1: "f32[10000, 64]" = torch.ops.aten.scatter_reduce.two(full_default, 0, expand, addmm_3, 'amax', include_self = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu_3: "f32[10000, 64]" = torch.ops.aten.relu.default(scatter_reduce_1)

        # File: /tmp/root_pyg/tmphufwnlsw.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index_4: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_3, [select])
        index_5: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_3, [select_1])

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/edge_conv.py:63 in message, code: return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        sub_2: "f32[200000, 64]" = torch.ops.aten.sub.Tensor(index_4, index_5);  index_4 = None
        cat_2: "f32[200000, 128]" = torch.ops.aten.cat.default([index_5, sub_2], -1);  index_5 = sub_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_4: "f32[128, 64]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_4: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_12, cat_2, permute_4);  primals_12 = permute_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        relu_4: "f32[200000, 64]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_5: "f32[64, 64]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_5: "f32[200000, 64]" = torch.ops.aten.addmm.default(primals_14, relu_4, permute_5);  primals_14 = permute_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        scatter_reduce_2: "f32[10000, 64]" = torch.ops.aten.scatter_reduce.two(full_default, 0, expand, addmm_5, 'amax', include_self = False)
        gather: "f32[200000, 64]" = torch.ops.aten.gather.default(scatter_reduce_2, 0, expand)
        eq: "b8[10000, 64]" = torch.ops.aten.eq.Tensor(full_default, scatter_reduce_2)
        eq_1: "b8[200000, 64]" = torch.ops.aten.eq.Tensor(addmm_5, gather);  addmm_5 = gather = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        le_1: "b8[10000, 64]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        gather_2: "f32[200000, 64]" = torch.ops.aten.gather.default(scatter_reduce_1, 0, expand)
        eq_4: "b8[10000, 64]" = torch.ops.aten.eq.Tensor(full_default, scatter_reduce_1);  scatter_reduce_1 = None
        eq_5: "b8[200000, 64]" = torch.ops.aten.eq.Tensor(addmm_3, gather_2);  addmm_3 = gather_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        le_3: "b8[10000, 64]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        gather_4: "f32[200000, 64]" = torch.ops.aten.gather.default(scatter_reduce, 0, expand);  expand = None
        eq_8: "b8[10000, 64]" = torch.ops.aten.eq.Tensor(full_default, scatter_reduce);  scatter_reduce = None
        eq_9: "b8[200000, 64]" = torch.ops.aten.eq.Tensor(addmm_1, gather_4);  addmm_1 = gather_4 = None
        return (scatter_reduce_2, primals_5, primals_7, primals_9, primals_11, primals_13, select, select_1, cat, relu, full_default, cat_1, relu_2, cat_2, relu_4, eq, eq_1, le_1, eq_4, eq_5, le_3, eq_8, eq_9)
