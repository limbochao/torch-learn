# PyTorch v2.10.0 上游源码对照摘录

标签 commit：`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`。由本地社区仓对应标签提取，不代表最新 master。

### `torch/_inductor/codegen/common.py`：is_buffer_removed

原始行 1417–1426；内容原样摘录。

```python
def is_buffer_removed(name: str) -> bool:
    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )
```

### `torch/_inductor/codegen/common.py`：DeferredLine

原始行 1429–1443；内容原样摘录。

```python
class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name: str, line: str):
        super().__init__(line)
        self.name = name
        assert not isinstance(line, DeferredLineBase)

    def __call__(self) -> Optional[str]:
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> DeferredLine:
        return DeferredLine(self.name, line)
```

### `torch/_inductor/codegen/simd.py`：generate_combo_kernel_code

原始行 2313–2342；内容原样摘录。

```python
            "ComboKernels: %d nodes partitioned into %s groups",
            len(subkernel_nodes),
            [len(p) for p in partitions],
        )
        kernel_code_list = []
        for node_group in partitions:
            if len(node_group) == 0:
                continue
            kernel = ComboKernel(
                enable_autotune=enable_autotune,
                mixed_sizes=mixed_sizes,
            )

            for pn in node_group:
                self.codegen_node_schedule_with_kernel(
                    node_schedule_map[pn][0],
                    kernel.create_sub_kernel(subkernel_map[pn]),
                )
                subkernel = subkernel_map[pn]
                node_schedule = node_schedule_map[pn][0]
                if not only_gen_src_code:
                    with V.set_kernel_handler(subkernel):  # type: ignore[call-arg]
                        for node in NodeScheduleMarker.only_nodes(node_schedule):
                            node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            src_code = kernel.codegen_kernel()
            kernel_code_list.append((src_code, kernel, node_group))
        return kernel_code_list
```

### `torch/_inductor/codegen/wrapper.py`：generate_numel_expr

原始行 2621–2640；内容原样摘录。

```python
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
```

### `torch/_inductor/codegen/triton.py`：kernel_benchmark_extra_args

原始行 4921–4945；内容原样摘录。

```python
    def kernel_benchmark_extra_args(self) -> list[str]:
        args = []
        if self.need_numel_args():
            numel_args: list[sympy.Expr] = []
            self.add_numel_to_call_args("", numel_args, [])
            for arg in numel_args:
                if isinstance(arg, int):
                    args.append(str(arg))
                elif isinstance(arg, SymbolicCallArg):
                    hint = V.graph.sizevars.size_hint(
                        arg.inner_expr,
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    args.append(str(hint))
                elif isinstance(arg, sympy.Expr):
                    hint = V.graph.sizevars.size_hint(
                        arg,
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    args.append(str(hint))
                else:
                    raise ValueError(f"Unsupported numel argument type: {type(arg)}")
        return args
```

### `torch/_inductor/codegen/triton.py`：for arg_num in equal_1_arg_indices(signature):

原始行 5385–5392；内容原样摘录。

```python
        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/triton-lang/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]
        triton_meta["enable_fp_fusion"] = not config.emulate_precision_casts
```

### `torch/_inductor/codegen/triton_combo_kernel.py`：for arg_num in equal_1_arg_indices(signature):

原始行 612–618；内容原样摘录。

```python
            "constants": {},
        }

        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        # pyrefly: ignore [unsupported-operation]
```

### `torch/_inductor/codegen/triton_utils.py`：_arg_equals_1

原始行 193–198；内容原样摘录。

```python
def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )
```

### `torch/_inductor/codegen/triton_utils.py`：equal_1_arg_indices

原始行 201–211；内容原样摘录。

```python
def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: Optional[list[int]] = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1
```

### `torch/_inductor/codegen/triton.py`：filter_masks

原始行 5710–5716；内容原样摘录。

```python
    def filter_masks(self, mask_vars: OrderedSet[str]) -> None:
        for tree in self.range_trees:
            if self._has_constant_mask(tree):
                mask_vars.discard(f"{tree.prefix}mask")

        # can be added as an override_mask
        mask_vars.discard("None")
```

### `torch/_inductor/codegen/triton.py`：_print_FloorDiv

原始行 812–819；内容原样摘录。

```python
    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        assert expr.is_integer
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " // ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.div_floor_integer({quot_s},  {div_s})"
```

### `torch/utils/_sympy/functions.py`：class FloorDiv(sympy.Function):

原始行 188–245；内容原样摘录。

```python
class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)

    NB: This is Python-style floor division, round to -Inf
    """

    nargs: tuple[int, ...] = (2,)
    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @property
    def base(self) -> sympy.Basic:
        # pyrefly: ignore [missing-attribute]
        return self.args[0]

    @property
    def divisor(self) -> sympy.Basic:
        # pyrefly: ignore [missing-attribute]
        return self.args[1]

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        base = printer.parenthesize(self.base, PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.divisor, PRECEDENCE["Atom"] - 0.5)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base: sympy.Integer, divisor: sympy.Integer) -> sympy.Basic | None:
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # Assert triggered by inequality solver
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if base in (int_oo, -int_oo, sympy.oo, -sympy.oo) and divisor in (
            int_oo,
            -int_oo,
            sympy.oo,
            -sympy.oo,
        ):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and equal_valued(divisor, 1):
            return base
        if base.is_integer and equal_valued(divisor, -1):
            return sympy.Mul(base, -1)
        if (
```

### `torch/_inductor/codegen/common.py`：remove_buffer

原始行 2332–2338；内容原样摘录。

```python
    def remove_buffer(self, name: str) -> None:
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        self.args.output_buffers[name] = REMOVED
        self.removed_buffers.add(name)
```

### `torch/_inductor/codegen/common.py`：remove_inplace_buffer

原始行 2340–2343；内容原样摘录。

```python
    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        self.args.inplace_buffers[name] = REMOVED
        self.removed_buffers.add(name)
```
