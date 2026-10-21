class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s4)", arg1_1: "i64[1, s4]", arg2_1: "Sym(s70)", arg3_1: "i64[s70]"):
        # File: <workspace>/three_model_boundary_20261021/fastnlp_length_graph.py:12 in word_piece_lengths, code: mask = words.ne(0)
        ne: "b8[1, s4]" = torch.ops.aten.ne.Scalar(arg1_1, 0)

        # File: <workspace>/three_model_boundary_20261021/fastnlp_length_graph.py:13 in word_piece_lengths, code: seq_len = mask.sum(dim=-1)
        sum_1: "i64[1]" = torch.ops.aten.sum.dim_IntList(ne, [-1])

        # File: <workspace>/three_model_boundary_20261021/fastnlp_length_graph.py:14 in word_piece_lengths, code: lengths = table[words].masked_fill(mask.eq(False), 0)
        index: "i64[1, s4]" = torch.ops.aten.index.Tensor(arg3_1, [arg1_1]);  arg3_1 = arg1_1 = None
        eq_2: "b8[1, s4]" = torch.ops.aten.eq.Scalar(ne, False)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "i64[1, s4]" = torch.ops.aten.where.self(eq_2, full_default, index);  eq_2 = full_default = index = None

        # File: <workspace>/three_model_boundary_20261021/fastnlp_length_graph.py:15 in word_piece_lengths, code: total = lengths.sum(dim=-1)
        sum_2: "i64[1]" = torch.ops.aten.sum.dim_IntList(where, [-1])

        # File: <workspace>/three_model_boundary_20261021/fastnlp_length_graph.py:16 in word_piece_lengths, code: maximum = lengths.sum(dim=-1).max()
        sum_3: "i64[1]" = torch.ops.aten.sum.dim_IntList(where, [-1])
        max_1: "i64[]" = torch.ops.aten.max.default(sum_3);  sum_3 = None
        return (max_1, ne, sum_1, where, sum_2)
