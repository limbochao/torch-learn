class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "f32[64, 64]", primals_7: "f32[64, 128]", primals_9: "f32[64, 64]", primals_11: "f32[64, 128]", primals_13: "f32[64, 64]", select: "i64[200000]", select_1: "i64[200000]", cat: "f32[200000, 128]", relu: "f32[200000, 64]", full_default: "f32[10000, 64]", cat_1: "f32[200000, 128]", relu_2: "f32[200000, 64]", cat_2: "f32[200000, 128]", relu_4: "f32[200000, 64]", eq: "b8[10000, 64]", eq_1: "b8[200000, 64]", le_1: "b8[10000, 64]", eq_4: "b8[10000, 64]", eq_5: "b8[200000, 64]", le_3: "b8[10000, 64]", eq_8: "b8[10000, 64]", eq_9: "b8[200000, 64]", tangents_1: "f32[10000, 64]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view: "i64[200000, 1]" = torch.ops.aten.view.default(select_1, [-1, 1])
        expand: "i64[200000, 64]" = torch.ops.aten.expand.default(view, [200000, 64]);  view = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        convert_element_type: "f32[10000, 64]" = torch.ops.prims.convert_element_type.default(eq, torch.float32);  eq = None
        convert_element_type_1: "f32[200000, 64]" = torch.ops.prims.convert_element_type.default(eq_1, torch.float32)
        scatter_add: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(convert_element_type, 0, expand, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        div: "f32[10000, 64]" = torch.ops.aten.div.Tensor(tangents_1, scatter_add);  tangents_1 = scatter_add = None
        gather_1: "f32[200000, 64]" = torch.ops.aten.gather.default(div, 0, expand);  div = None
        mul_1: "f32[200000, 64]" = torch.ops.aten.mul.Tensor(eq_1, gather_1);  eq_1 = gather_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_5: "f32[64, 64]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_6: "f32[64, 64]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm: "f32[200000, 64]" = torch.ops.aten.mm.default(mul_1, permute_6);  permute_6 = None
        permute_7: "f32[64, 200000]" = torch.ops.aten.permute.default(mul_1, [1, 0])
        mm_1: "f32[64, 64]" = torch.ops.aten.mm.default(permute_7, relu_4);  permute_7 = None
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True);  mul_1 = None
        view_3: "f32[64]" = torch.ops.aten.view.default(sum_1, [64]);  sum_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        le: "b8[200000, 64]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[200000, 64]" = torch.ops.aten.where.self(le, full_default_3, mm);  le = mm = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_4: "f32[128, 64]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_10: "f32[64, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_2: "f32[200000, 128]" = torch.ops.aten.mm.default(where, permute_10);  permute_10 = None
        permute_11: "f32[64, 200000]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_3: "f32[64, 128]" = torch.ops.aten.mm.default(permute_11, cat_2);  permute_11 = cat_2 = None
        sum_2: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_4: "f32[64]" = torch.ops.aten.view.default(sum_2, [64]);  sum_2 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/edge_conv.py:63 in message, code: return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        slice_1: "f32[200000, 64]" = torch.ops.aten.slice.Tensor(mm_2, 1, 0, 64)
        slice_2: "f32[200000, 64]" = torch.ops.aten.slice.Tensor(mm_2, 1, 64, 128);  mm_2 = None
        neg: "f32[200000, 64]" = torch.ops.aten.neg.default(slice_2)
        add: "f32[200000, 64]" = torch.ops.aten.add.Tensor(slice_1, neg);  slice_1 = neg = None

        # File: /tmp/root_pyg/tmp1fuv3bdv.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index_put: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select_1], add, True);  add = None
        index_put_1: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select], slice_2, True);  slice_2 = None
        add_1: "f32[10000, 64]" = torch.ops.aten.add.Tensor(index_put, index_put_1);  index_put = index_put_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        where_1: "f32[10000, 64]" = torch.ops.aten.where.self(le_1, full_default_3, add_1);  le_1 = add_1 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        convert_element_type_2: "f32[10000, 64]" = torch.ops.prims.convert_element_type.default(eq_4, torch.float32);  eq_4 = None
        convert_element_type_3: "f32[200000, 64]" = torch.ops.prims.convert_element_type.default(eq_5, torch.float32)
        scatter_add_1: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(convert_element_type_2, 0, expand, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None
        div_1: "f32[10000, 64]" = torch.ops.aten.div.Tensor(where_1, scatter_add_1);  where_1 = scatter_add_1 = None
        gather_3: "f32[200000, 64]" = torch.ops.aten.gather.default(div_1, 0, expand);  div_1 = None
        mul_3: "f32[200000, 64]" = torch.ops.aten.mul.Tensor(eq_5, gather_3);  eq_5 = gather_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_3: "f32[64, 64]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_14: "f32[64, 64]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_4: "f32[200000, 64]" = torch.ops.aten.mm.default(mul_3, permute_14);  permute_14 = None
        permute_15: "f32[64, 200000]" = torch.ops.aten.permute.default(mul_3, [1, 0])
        mm_5: "f32[64, 64]" = torch.ops.aten.mm.default(permute_15, relu_2);  permute_15 = None
        sum_3: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(mul_3, [0], True);  mul_3 = None
        view_5: "f32[64]" = torch.ops.aten.view.default(sum_3, [64]);  sum_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        le_2: "b8[200000, 64]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_2: "f32[200000, 64]" = torch.ops.aten.where.self(le_2, full_default_3, mm_4);  le_2 = mm_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_2: "f32[128, 64]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_18: "f32[64, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_6: "f32[200000, 128]" = torch.ops.aten.mm.default(where_2, permute_18);  permute_18 = None
        permute_19: "f32[64, 200000]" = torch.ops.aten.permute.default(where_2, [1, 0])
        mm_7: "f32[64, 128]" = torch.ops.aten.mm.default(permute_19, cat_1);  permute_19 = cat_1 = None
        sum_4: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where_2, [0], True);  where_2 = None
        view_6: "f32[64]" = torch.ops.aten.view.default(sum_4, [64]);  sum_4 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/conv/edge_conv.py:63 in message, code: return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        slice_3: "f32[200000, 64]" = torch.ops.aten.slice.Tensor(mm_6, 1, 0, 64)
        slice_4: "f32[200000, 64]" = torch.ops.aten.slice.Tensor(mm_6, 1, 64, 128);  mm_6 = None
        neg_1: "f32[200000, 64]" = torch.ops.aten.neg.default(slice_4)
        add_2: "f32[200000, 64]" = torch.ops.aten.add.Tensor(slice_3, neg_1);  slice_3 = neg_1 = None

        # File: /tmp/root_pyg/tmp7wrdye85.py:88 in _lift, code: return src.index_select(self.node_dim, index)
        index_put_2: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select_1], add_2, True);  select_1 = add_2 = None
        index_put_3: "f32[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select], slice_4, True);  full_default = select = slice_4 = None
        add_3: "f32[10000, 64]" = torch.ops.aten.add.Tensor(index_put_2, index_put_3);  index_put_2 = index_put_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/basic_gnn.py:234 in forward, code: x = self.act(x)
        where_3: "f32[10000, 64]" = torch.ops.aten.where.self(le_3, full_default_3, add_3);  le_3 = add_3 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/scatter.py:98 in scatter, code: return src.new_zeros(size).scatter_reduce_(
        convert_element_type_4: "f32[10000, 64]" = torch.ops.prims.convert_element_type.default(eq_8, torch.float32);  eq_8 = None
        convert_element_type_5: "f32[200000, 64]" = torch.ops.prims.convert_element_type.default(eq_9, torch.float32)
        scatter_add_2: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(convert_element_type_4, 0, expand, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        div_2: "f32[10000, 64]" = torch.ops.aten.div.Tensor(where_3, scatter_add_2);  where_3 = scatter_add_2 = None
        gather_5: "f32[200000, 64]" = torch.ops.aten.gather.default(div_2, 0, expand);  div_2 = expand = None
        mul_5: "f32[200000, 64]" = torch.ops.aten.mul.Tensor(eq_9, gather_5);  eq_9 = gather_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_1: "f32[64, 64]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_22: "f32[64, 64]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_8: "f32[200000, 64]" = torch.ops.aten.mm.default(mul_5, permute_22);  permute_22 = None
        permute_23: "f32[64, 200000]" = torch.ops.aten.permute.default(mul_5, [1, 0])
        mm_9: "f32[64, 64]" = torch.ops.aten.mm.default(permute_23, relu);  permute_23 = None
        sum_5: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(mul_5, [0], True);  mul_5 = None
        view_7: "f32[64]" = torch.ops.aten.view.default(sum_5, [64]);  sum_5 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/models/mlp.py:211 in forward, code: x = self.act(x)
        le_4: "b8[200000, 64]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_4: "f32[200000, 64]" = torch.ops.aten.where.self(le_4, full_default_3, mm_8);  le_4 = full_default_3 = mm_8 = None

        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_26: "f32[64, 200000]" = torch.ops.aten.permute.default(where_4, [1, 0])
        mm_10: "f32[64, 128]" = torch.ops.aten.mm.default(permute_26, cat);  permute_26 = cat = None
        sum_6: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where_4, [0], True);  where_4 = None
        view_8: "f32[64]" = torch.ops.aten.view.default(sum_6, [64]);  sum_6 = None
        return (None, None, mm_10, view_8, mm_9, view_7, mm_7, view_6, mm_5, view_5, mm_3, view_4, mm_1, view_3)
