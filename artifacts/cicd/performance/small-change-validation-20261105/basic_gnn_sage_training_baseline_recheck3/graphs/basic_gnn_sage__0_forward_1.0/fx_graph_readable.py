class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10000, 64]", primals_2: "i64[2, 200000]", primals_3: "f32[64, 64]", primals_4: "f32[64]", primals_5: "f32[64, 64]", primals_6: "f32[64, 64]", primals_7: "f32[64]", primals_8: "f32[64, 64]", primals_9: "f32[64, 64]", primals_10: "f32[64]", primals_11: "f32[64, 64]"):
        # File: /tmp/root_pyg/tmpxhf5t7p1.py:86 in _lift, code: index = edge_index[dim]
        select: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 0)

        # File: /tmp/root_pyg/tmpxhf5t7p1.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f32[200000, 64]" = torch.ops.aten.index.Tensor(primals_1, [select])

        # File: /tmp/root_pyg/tmpxhf5t7p1.py:136 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:77 in scatter, code: count = src.new_zeros(dim_size)
        full_default: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:78 in scatter, code: count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        full_default_1: "f32[200000]" = torch.ops.aten.full.default([200000], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default, 0, select_1, full_default_1);  full_default = full_default_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:79 in scatter, code: count = count.clamp(min=1)
        clamp_min: "f32[10000]" = torch.ops.aten.clamp_min.default(scatter_add, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[200000, 64]" = torch.ops.aten.expand.default(view, [200000, 64]);  view = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        full_default_2: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_1: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_2, 0, expand, index);  index = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "f32[10000, 1]" = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        expand_1: "f32[10000, 64]" = torch.ops.aten.expand.default(view_1, [10000, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_1, expand_1);  scatter_add_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[64, 64]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm: "f32[10000, 64]" = torch.ops.aten.addmm.default(primals_4, div, permute);  primals_4 = permute = None
        permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        mm: "f32[10000, 64]" = torch.ops.aten.mm.default(primals_1, permute_1);  permute_1 = None

        # File: /tmp/root_pyg/tmpxhf5t7p1.py:233 in forward, code: out = out + self.lin_r(x_r)
        add: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm, mm);  addmm = mm = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu: "f32[10000, 64]" = torch.ops.aten.relu.default(add);  add = None

        # File: /tmp/root_pyg/tmpojsc0ehz.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_1: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu, [select])

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        scatter_add_3: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_2, 0, expand, index_1);  index_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_1: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_3, expand_1);  scatter_add_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_2: "f32[64, 64]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_1: "f32[10000, 64]" = torch.ops.aten.addmm.default(primals_7, div_1, permute_2);  primals_7 = permute_2 = None
        permute_3: "f32[64, 64]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        mm_1: "f32[10000, 64]" = torch.ops.aten.mm.default(relu, permute_3);  permute_3 = None

        # File: /tmp/root_pyg/tmpojsc0ehz.py:233 in forward, code: out = out + self.lin_r(x_r)
        add_1: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm_1, mm_1);  addmm_1 = mm_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu_1: "f32[10000, 64]" = torch.ops.aten.relu.default(add_1);  add_1 = None

        # File: /tmp/root_pyg/tmp1a8_ynjh.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_2: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_1, [select]);  select = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        scatter_add_5: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_2, 0, expand, index_2);  expand = index_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_2: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_5, expand_1);  scatter_add_5 = expand_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_4: "f32[64, 64]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_2: "f32[10000, 64]" = torch.ops.aten.addmm.default(primals_10, div_2, permute_4);  primals_10 = permute_4 = None
        permute_5: "f32[64, 64]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        mm_2: "f32[10000, 64]" = torch.ops.aten.mm.default(relu_1, permute_5);  permute_5 = None

        # File: /tmp/root_pyg/tmp1a8_ynjh.py:233 in forward, code: out = out + self.lin_r(x_r)
        add_2: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm_2, mm_2);  addmm_2 = mm_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_20: "f32[64, 10000]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        return (add_2, primals_2, primals_6, primals_8, primals_9, primals_11, scatter_add, full_default_2, div, relu, div_1, relu_1, div_2, permute_20)
