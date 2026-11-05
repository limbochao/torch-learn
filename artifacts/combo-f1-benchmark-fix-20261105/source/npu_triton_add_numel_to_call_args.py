# Source: torch_npu/_inductor/codegen/triton.py
# Version: e32cdebf77e797fc85a1a8c3fe1e0eef050f9c0e
    def add_numel_to_call_args(self, name, call_args, arg_types):
        for node in self.sorted_axis:
            if isinstance(node.expr, ModularIndexing):
                numel_expr = node.length
            else:
                numel_expr = node.expr.subs(
                    {sympy_index_symbol(r.name): r.numel for r in self.range_trees}
                )

            if isinstance(numel_expr, (sympy.Integer, sympy.Symbol)):
                expr = numel_expr
            else:
                expr = V.graph.wrapper_code.generate_node_numel_expr(
                    name, node, numel_expr
                )
            call_args.append(expr)
            arg_types.append(type(expr))
