# Source: torch/_inductor/codegen/wrapper.py
# Version: v2.10.0
    def generate_numel_expr(self, kernel_name: str, tree, suffix: Optional[str] = None):
        sym_name = f"{kernel_name}_{tree.prefix}numel"
        if suffix is not None:
            sym_name += f"_{suffix}"
        sym = sympy.Symbol(sym_name, is_integer=True, is_positive=True)

        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        arg = SymbolicCallArg(sym, tree.numel)

        is_benchmark_kernel = kernel_name == ""
        if not is_benchmark_kernel:
            self.writeline(SymbolicCallArgLine(self, arg, V.graph))

        return arg
