    def codegen_kernel(self, name=None):
        """
        codegen triton kernel with linear and loop
        """

        code = IndentedBuffer()
        size_hints = self.get_size_hints()
        heuristics = self._get_heuristic()
        if name is None:
            code.splice(self.gen_common_triton_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()

        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype, argdefs=argdefs
        )

        triton_meta = {
            "signature": triton_meta_signature,
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            # special config for NPU, specify compile target
            "mix_mode": "aiv",
        }

        inductor_meta = self.create_inductor_meta()
        self.triton_meta = triton_meta
        self.inductor_meta = inductor_meta
        self._disable_grouped_autotune_if_unsupported(inductor_meta)
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        self.gen_numel_args(signature, triton_meta, triton_meta_signature, argdefs)
        if self.is_unified_simt_kernel():
            triton_meta["configs"] = [config_of(signature)]

        if inductor_meta.get("group_enabled", False):
            try:
                inductor_meta.update(
                    self.build_grouped_benchmark_arg_specs(argdefs, signature)
                )
                inductor_meta.setdefault("extra_launcher_arg_specs", ())
                self._disable_grouped_autotune_if_benchmark_too_large(
                    inductor_meta,
                    V.graph.get_current_device_or_throw(),
                )
            except RuntimeError as exc:
                self._disable_grouped_autotune(inductor_meta, str(exc))

        self._record_legacy_auto_blockify_for_grouped_fallback(inductor_meta)

        # add in tiling args
        self.add_autotune_args(argdefs, signature, triton_meta_signature)
        # for scalar codegen
        if len(self.range_tree_nodes) == 0:
            self.write_scalar()
        else:
            self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        # Note: override original triton_heuristics
        if self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint()
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(x.full_name() for x in argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()
