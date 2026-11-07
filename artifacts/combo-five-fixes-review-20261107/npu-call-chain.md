# NPU 修复后的调用链摘录

以下固定取自已验证提交 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，不是随工作区 HEAD 变化的内容。

### `torch_npu/_inductor/codegen/scheduling.py`：codegen_combo_kernel

原始行 703–734；内容原样摘录。

```python
    def codegen_combo_kernel(self, combo_kernel_node):
        subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
        custom_part_algorithm = combo_kernel_node.use_custom_partition_algo
        enable_autotune = combo_kernel_node.enable_autotune
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_part_algorithm
        )

        kernel_code_list = self.generate_combo_kernel_code(
            subkernel_nodes, custom_part_algorithm, enable_autotune, mixed_sizes,
            only_gen_src_code=True,
        )

        # Validate every partition before allocating buffers or emitting any
        # wrapper code. A failed check can then fall back to ordinary kernels.
        for src_code, kernel, _ in kernel_code_list:
            with V.set_kernel_handler(kernel):
                self.make_ttir_for_check(src_code)

        for src_code, kernel, node_group in kernel_code_list:
            for subkernel_node, subkernel in zip(node_group, kernel.sub_kernels):
                with V.set_kernel_handler(subkernel):
                    for snode in subkernel_node.get_nodes():
                        snode.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove
            kernel_name, _ = self.define_kernel(src_code, [combo_kernel_node], kernel, None)
            self.codegen_comment([combo_kernel_node])
            log.debug("ComboKernels: generated kernel %s.", kernel_name)
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.free_buffers_in_scheduler()
```

### `torch_npu/_inductor/codegen/scheduling.py`：generate_combo_kernel_code

原始行 773–797；内容原样摘录。

```python
        kernel_code_list = []
        for node_group in partitions:
            fused_node_lists = [node.get_nodes() for node in node_group]
            kernel = NPUComboKernel(
                enable_autotune=enable_autotune,
                mixed_sizes=mixed_sizes,
            )

            for pn, nodes in zip(node_group, fused_node_lists):
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

### `torch_npu/_inductor/codegen/wrapper.py`：generate_node_numel_expr

原始行 73–91；内容原样摘录。

```python
    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"
        # Benchmark argument discovery must not emit runtime size assignments.
        if not kernel_name:
            return SymbolicCallArg(expr, numel_expr)
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
```

### `torch_npu/_inductor/codegen/triton.py`：add_numel_to_call_args

原始行 2585–2601；内容原样摘录。

```python
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
```

### `torch_npu/_inductor/codegen/triton.py`：filter_masks

原始行 4600–4685；内容原样摘录。

```python
    def filter_masks(self, mask_vars, index_vars=None):
        variable_mask_vars = {  # noqa: set_linter
            mask_var
            for mask_var in mask_vars
            if isinstance(mask_var, TritonCSEVariable)
        }
        normal_mask_vars = mask_vars.copy() - variable_mask_vars
        # This function is only allowed to remove *axis* masks that are known
        # to be redundant/invalid for current kernel shape lowering.
        #
        # It must NOT drop "semantic" masks (e.g. tmp masks guarding padded
        # loads/stores, indirect indexing validity masks, etc.), otherwise we
        # can generate incorrect tl.load/tl.store.
        masked_axis_name = []
        # Size symbols are not loop axes: "y0 < s0" only constrains y0, and
        # ops.masked skips them when it ands the axis masks in. Left in, they
        # fail the subset test below for every load whose index does not spell
        # the size symbol out, e.g. the first slice of a cat over a dynamic
        # dimension, whose guarding mask would then be dropped.
        subblock_axis = {  # noqa: set_linter
            axis
            for axis in V.kernel.current_subblock_axis
            if not str(axis).startswith(("s", "ps", "i"))
        }
        save_variable_mask = True
        if index_vars and subblock_axis:
            save_variable_mask = subblock_axis.issubset(
                {str(var) for var in index_vars}  # noqa: set_linter
            )

        # cumsum is codegen'ed as ops.scan, which relies on keeping the axis
        # masks (including the persistent reduction axis) for its partial
        # scan / accumulator update.  Other kernels must drop the redundant
        # persistent reduction axis masks, otherwise the generated
        # tl.load/tl.store carries useless masks (e.g. Layernorm perf
        # regression).
        kernel_features = getattr(self, "features", None)
        keep_persistent_reduction_mask = (
            kernel_features is not None and kernel_features.contains_op("scan")
        )

        for node in self.sorted_axis:
            allow_dynamic = get_allow_dynamic()
            is_persistent_reduction_axis = (
                self.persistent_reduction and node.is_reduction
            )
            if node.is_vectorized_split:
                masked_axis_name.append(node.name)
                continue
            if self.is_unified_simt_kernel():
                if not node.is_tiling_axis:
                    continue
            else:
                if (
                    not node.is_tiling_axis
                    or (
                        is_persistent_reduction_axis
                        and not keep_persistent_reduction_mask
                    )
                    or (node.is_no_loop_axis and not allow_dynamic)
                ):
                    continue

            if save_variable_mask and node.name in subblock_axis:
                continue

            masked_axis_name.append(node.name)

        # Semantic masks can depend on axes absent from the address (for
        # example, a pooling window boundary). They must still guard the load.

        # Be careful when remove mask from load store
        # 1. Current only leave axis mask from sorted axis
        # 2. If z0 is permute, maybe have z0_mask, z0_1_mask
        # 3. xmask, x0_mask: if axis is x, will not remove x0_mask, can't just use startswith
        for mask_var in normal_mask_vars:
            valid_mask_var = False
            mask_var_str = str(mask_var)
            for axis_name in masked_axis_name:
                if mask_var_str == f"{axis_name}mask" or mask_var_str.startswith(
                    f"{axis_name}_"
                ):
                    valid_mask_var = True

            if not valid_mask_var:
                mask_vars.discard(mask_var)
```
