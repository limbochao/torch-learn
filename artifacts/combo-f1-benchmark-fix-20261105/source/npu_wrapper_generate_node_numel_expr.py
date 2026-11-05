# Source: torch_npu/_inductor/codegen/wrapper.py
# Version: e32cdebf77e797fc85a1a8c3fe1e0eef050f9c0e
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        simplified = V.graph.sizevars.simplify(numel_expr)
        # Ensure all PRECOMPUTED_SIZE symbols in the *simplified* expression
        # are defined before we emit the line that uses them.
        for sym in simplified.free_symbols:
            self.ensure_size_computed(sym)
        self.writeline(f"{expr} = {pexpr(simplified)}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, numel_expr)
