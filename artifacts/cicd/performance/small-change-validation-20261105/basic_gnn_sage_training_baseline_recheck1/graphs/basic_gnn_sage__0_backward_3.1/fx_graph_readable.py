class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[2, 200000]", primals_6: "f32[64, 64]", primals_8: "f32[64, 64]", primals_9: "f32[64, 64]", primals_11: "f32[64, 64]", scatter_add: "f32[10000]", full_default_2: "f32[10000, 64]", div: "f32[10000, 64]", relu: "f32[10000, 64]", div_1: "f32[10000, 64]", relu_1: "f32[10000, 64]", div_2: "f32[10000, 64]", permute_20: "f32[64, 10000]", tangents_1: "f32[10000, 64]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_5: "f32[64, 64]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_6: "f32[64, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_3: "f32[10000, 64]" = torch.ops.aten.mm.default(tangents_1, permute_6);  permute_6 = None
        permute_7: "f32[64, 10000]" = torch.ops.aten.permute.default(relu_1, [1, 0])
        mm_4: "f32[64, 64]" = torch.ops.aten.mm.default(permute_7, tangents_1);  permute_7 = None
        permute_8: "f32[64, 64]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
        permute_4: "f32[64, 64]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_9: "f32[64, 64]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_5: "f32[10000, 64]" = torch.ops.aten.mm.default(tangents_1, permute_9);  permute_9 = None
        permute_10: "f32[64, 10000]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_6: "f32[64, 64]" = torch.ops.aten.mm.default(permute_10, div_2);  permute_10 = div_2 = None
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_6: "f32[64]" = torch.ops.aten.view.default(sum_1, [64]);  sum_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:79 in scatter, code: count = count.clamp(min=1)
        clamp_min: "f32[10000]" = torch.ops.aten.clamp_min.default(scatter_add, 1);  scatter_add = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "f32[10000, 1]" = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        expand_1: "f32[10000, 64]" = torch.ops.aten.expand.default(view_1, [10000, 64]);  view_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_3: "f32[10000, 64]" = torch.ops.aten.div.Tensor(mm_5, expand_1);  mm_5 = None

        # File: /tmp/root_pyg/tmp1eut46ek.py:136 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 1)

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[200000, 64]" = torch.ops.aten.expand.default(view, [200000, 64]);  view = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        gather: "f32[200000, 64]" = torch.ops.aten.gather.default(div_3, 0, expand);  div_3 = None

        # File: /tmp/root_pyg/tmp1eut46ek.py:86 in _lift, code: index = edge_index[dim]
        select: "i64[200000]" = torch.ops.aten.select.int(primals_2, 0, 0);  primals_2 = None

        # File: /tmp/root_pyg/tmppbka81bm.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_put: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default_2, [select], gather, True);  gather = None
        add_3: "f32[10000, 64]" = torch.ops.aten.add.Tensor(mm_3, index_put);  mm_3 = index_put = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        le: "b8[10000, 64]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[10000, 64]" = torch.ops.aten.where.self(le, full_default_10, add_3);  le = add_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_3: "f32[64, 64]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_13: "f32[64, 64]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_7: "f32[10000, 64]" = torch.ops.aten.mm.default(where, permute_13);  permute_13 = None
        permute_14: "f32[64, 10000]" = torch.ops.aten.permute.default(relu, [1, 0])
        mm_8: "f32[64, 64]" = torch.ops.aten.mm.default(permute_14, where);  permute_14 = None
        permute_15: "f32[64, 64]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
        permute_2: "f32[64, 64]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_16: "f32[64, 64]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_9: "f32[10000, 64]" = torch.ops.aten.mm.default(where, permute_16);  permute_16 = None
        permute_17: "f32[64, 10000]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_10: "f32[64, 64]" = torch.ops.aten.mm.default(permute_17, div_1);  permute_17 = div_1 = None
        sum_2: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_7: "f32[64]" = torch.ops.aten.view.default(sum_2, [64]);  sum_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:84 in scatter, code: return out / broadcast(count, out, dim)
        div_4: "f32[10000, 64]" = torch.ops.aten.div.Tensor(mm_9, expand_1);  mm_9 = expand_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:82 in scatter, code: out = src.new_zeros(size).scatter_add_(dim, index, src)
        gather_1: "f32[200000, 64]" = torch.ops.aten.gather.default(div_4, 0, expand);  div_4 = expand = None

        # File: /tmp/root_pyg/tmpbaqe0i8_.py:87 in _lift, code: return src.index_select(self.node_dim, index)
        index_put_1: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default_2, [select], gather_1, True);  full_default_2 = select = gather_1 = None
        add_4: "f32[10000, 64]" = torch.ops.aten.add.Tensor(mm_7, index_put_1);  mm_7 = index_put_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        le_1: "b8[10000, 64]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_1: "f32[10000, 64]" = torch.ops.aten.where.self(le_1, full_default_10, add_4);  le_1 = full_default_10 = add_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        mm_11: "f32[64, 64]" = torch.ops.aten.mm.default(permute_20, where_1);  permute_20 = None
        permute_21: "f32[64, 64]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        permute_22: "f32[64, 10000]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_12: "f32[64, 64]" = torch.ops.aten.mm.default(permute_22, div);  permute_22 = div = None
        sum_3: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_8: "f32[64]" = torch.ops.aten.view.default(sum_3, [64]);  sum_3 = None
        return (None, None, mm_12, view_8, permute_21, mm_10, view_7, permute_15, mm_6, view_6, permute_8)
