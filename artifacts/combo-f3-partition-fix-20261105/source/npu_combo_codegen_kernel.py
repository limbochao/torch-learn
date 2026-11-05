    def codegen_kernel(self, name: Optional[str] = None) -> str:
        # TODO: is it correct to use the first sub kernel's heuristics?
        heuristics_list, size_hints_list = [], []
        for subkernel in self.sub_kernels:
            h, s = self.select_heuristics(subkernel)
            heuristics_list.append(h)
            size_hints_list.append(s)
        heuristics, size_hints, selected_kernel = self.select_combo_heuristics(
            heuristics_list, size_hints_list
        )

        pointwise_with_reduction, heuristics = (
            (True, "pointwise")
            if heuristics == "pointwise_with_reduction"
            else (False, heuristics)
        )
        code = IndentedBuffer()

        from torch._inductor.codegen.triton import gen_common_triton_imports
        code.splice(gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()
        argdefs = self.add_numel_to_args(argdefs, signature)
        # remove enable_autotune condition and add autotune args
        self.add_autotune_args(self.sub_kernels[0], argdefs, signature)

        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=False,
                signature=signature,
                argdefs=argdefs,
            )
        )
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(x.full_name() for x in argdefs)}):"
        )

        with code.indent():
            code.splice("pid = tl.program_id(0)")

            for num, sub_kernel in enumerate(self.sub_kernels):
                assert self.dispatch_class is not None
                self.dispatch_class.codegen_pid_range(self, num, code)
                with code.indent():
                    uniquify = self.codegen_static_numels_sub_kernel(
                        code, sub_kernel, num
                    )
                    if len(sub_kernel.range_tree_nodes) == 0:
                        sub_kernel.write_scalar()
                    else:
                        sub_kernel.codegen_body()
                    uniquified_body = self.uniquify_block_sizes(
                        sub_kernel.body, num, uniquify
                    )
                    code.splice(uniquified_body)

            code.splice("else:")
            with code.indent():
                code.splice("pass")

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=0))

        return code.getvalue()
