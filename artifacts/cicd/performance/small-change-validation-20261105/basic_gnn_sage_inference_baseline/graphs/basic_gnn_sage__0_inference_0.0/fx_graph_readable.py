class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10000, 64]", arg1_1: "i64[2, 200000]", arg2_1: "f32[64, 64]", arg3_1: "f32[64]", arg4_1: "f32[64, 64]", arg5_1: "f32[64, 64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64]", arg8_1: "f32[64, 64]", arg9_1: "f32[64]", arg10_1: "f32[64, 64]"):
        # File: /tmp/root_pyg/tmpmymq84s9.py:86 in _lift, code: index = edge_index[dim]
        select: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 0)

        # File: /tmp/root_pyg/tmpmymq84s9.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f32[200000, 64]" = torch.ops.aten.index.Tensor(arg0_1, [select]);  select = None

        # File: /tmp/root_pyg/tmpmymq84s9.py:136 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:77 in scatter, code: count = src.new_zeros(dim_size)
        full_default: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:78 in scatter, code: count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        full_default_1: "f32[200000]" = torch.ops.aten.full.default([200000], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default, 0, select_1, full_default_1);  full_default = full_default_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:79 in scatter, code: count = count.clamp(min=1)
        clamp_min: "f32[10000]" = torch.ops.aten.clamp_min.default(scatter_add, 1);  scatter_add = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[200000, 64]" = torch.ops.aten.expand.default(view, [200000, 64]);  view = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        full_default_2: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_1: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_2, 0, expand, index);  full_default_2 = expand = index = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "f32[10000, 1]" = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        expand_1: "f32[10000, 64]" = torch.ops.aten.expand.default(view_1, [10000, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_1, expand_1);  scatter_add_1 = expand_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[64, 64]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "f32[10000, 64]" = torch.ops.aten.addmm.default(arg3_1, div, permute);  arg3_1 = div = permute = None
        permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "f32[10000, 64]" = torch.ops.aten.mm.default(arg0_1, permute_1);  arg0_1 = permute_1 = None

        # File: /tmp/root_pyg/tmpmymq84s9.py:233 in forward, code: out = out + self.lin_r(x_r)
        add: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm, mm);  addmm = mm = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu: "f32[10000, 64]" = torch.ops.aten.relu.default(add);  add = None

        # File: /tmp/root_pyg/tmpzw4anfji.py:86 in _lift, code: index = edge_index[dim]
        select_2: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 0)

        # File: /tmp/root_pyg/tmpzw4anfji.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_1: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu, [select_2]);  select_2 = None

        # File: /tmp/root_pyg/tmpzw4anfji.py:136 in _collect, code: edge_index_i = edge_def[i]
        select_3: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:77 in scatter, code: count = src.new_zeros(dim_size)
        full_default_3: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:78 in scatter, code: count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        full_default_4: "f32[200000]" = torch.ops.aten.full.default([200000], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_2: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default_3, 0, select_3, full_default_4);  full_default_3 = full_default_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:79 in scatter, code: count = count.clamp(min=1)
        clamp_min_1: "f32[10000]" = torch.ops.aten.clamp_min.default(scatter_add_2, 1);  scatter_add_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_2: "i64[200000, 1]" = torch.ops.aten.view.default(select_3, [-1, 1]);  select_3 = None
        expand_2: "i64[200000, 64]" = torch.ops.aten.expand.default(view_2, [200000, 64]);  view_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        full_default_5: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_3: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_5, 0, expand_2, index_1);  full_default_5 = expand_2 = index_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_3: "f32[10000, 1]" = torch.ops.aten.view.default(clamp_min_1, [-1, 1]);  clamp_min_1 = None
        expand_3: "f32[10000, 64]" = torch.ops.aten.expand.default(view_3, [10000, 64]);  view_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_1: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_3, expand_3);  scatter_add_3 = expand_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_2: "f32[64, 64]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[10000, 64]" = torch.ops.aten.addmm.default(arg6_1, div_1, permute_2);  arg6_1 = div_1 = permute_2 = None
        permute_3: "f32[64, 64]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_1: "f32[10000, 64]" = torch.ops.aten.mm.default(relu, permute_3);  relu = permute_3 = None

        # File: /tmp/root_pyg/tmpzw4anfji.py:233 in forward, code: out = out + self.lin_r(x_r)
        add_1: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm_1, mm_1);  addmm_1 = mm_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        relu_1: "f32[10000, 64]" = torch.ops.aten.relu.default(add_1);  add_1 = None

        # File: /tmp/root_pyg/tmpw1bc2f7f.py:86 in _lift, code: index = edge_index[dim]
        select_4: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 0)

        # File: /tmp/root_pyg/tmpw1bc2f7f.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_2: "f32[200000, 64]" = torch.ops.aten.index.Tensor(relu_1, [select_4]);  select_4 = None

        # File: /tmp/root_pyg/tmpw1bc2f7f.py:136 in _collect, code: edge_index_i = edge_def[i]
        select_5: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 1);  arg1_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:77 in scatter, code: count = src.new_zeros(dim_size)
        full_default_6: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:78 in scatter, code: count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        full_default_7: "f32[200000]" = torch.ops.aten.full.default([200000], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_4: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default_6, 0, select_5, full_default_7);  full_default_6 = full_default_7 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:79 in scatter, code: count = count.clamp(min=1)
        clamp_min_2: "f32[10000]" = torch.ops.aten.clamp_min.default(scatter_add_4, 1);  scatter_add_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_4: "i64[200000, 1]" = torch.ops.aten.view.default(select_5, [-1, 1]);  select_5 = None
        expand_4: "i64[200000, 64]" = torch.ops.aten.expand.default(view_4, [200000, 64]);  view_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        full_default_8: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        scatter_add_5: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default_8, 0, expand_4, index_2);  full_default_8 = expand_4 = index_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_5: "f32[10000, 1]" = torch.ops.aten.view.default(clamp_min_2, [-1, 1]);  clamp_min_2 = None
        expand_5: "f32[10000, 64]" = torch.ops.aten.expand.default(view_5, [10000, 64]);  view_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_2: "f32[10000, 64]" = torch.ops.aten.div.Tensor(scatter_add_5, expand_5);  scatter_add_5 = expand_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_4: "f32[64, 64]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_2: "f32[10000, 64]" = torch.ops.aten.addmm.default(arg9_1, div_2, permute_4);  arg9_1 = div_2 = permute_4 = None
        permute_5: "f32[64, 64]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_2: "f32[10000, 64]" = torch.ops.aten.mm.default(relu_1, permute_5);  relu_1 = permute_5 = None

        # File: /tmp/root_pyg/tmpw1bc2f7f.py:233 in forward, code: out = out + self.lin_r(x_r)
        add_2: "f32[10000, 64]" = torch.ops.aten.add.Tensor(addmm_2, mm_2);  addmm_2 = mm_2 = None
        return (add_2,)
