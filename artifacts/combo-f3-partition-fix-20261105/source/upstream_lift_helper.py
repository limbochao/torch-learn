    def _lift_helper(
        self, fn, values: tuple[CSEVariable, ...], dtypes: tuple[torch.dtype, ...]
    ) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")
        cse = CSE()

        args = [
            tuple(
                cse.namedvar(f"arg{i}_{n}", dtype=dtype, shape=value.shape)
                for n, (value, dtype) in enumerate(zip(values, dtypes))
            )
            for i in range(2)
        ]
        signature = ", ".join(str(x) for x in itertools.chain.from_iterable(args))
        helper.writeline(f"def {{name}}({signature}):")

        overrides = TritonOverrides()

        # Build a name that changes depending on fn to workaround a triton bug
        # where the combine_fn to reduce and scan is not hashed, and so different
        # scan ops may collide in the triton cache.
        # This is fixed with the latest triton pin, but not the triton-rocm pin.
        helper_name = "_triton_helper_fn"

        from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
        from torch._inductor.shape_propagation import ShapePropagationOpsHandler

        shape_handler = ShapePropagationOpsHandler()
        dtype_handler = DtypePropagationOpsHandler()

        class CSEProxy(DefaultHandler):
            def _default(
                self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> Any:
                nonlocal helper_name
                helper_name += f"_{name}"

                output_dtype = getattr(
                    dtype_handler,
                    name,
                )(*args, **kwargs)

                output_shape = getattr(
                    shape_handler,
                    name,
                )(*args, **kwargs)

                return cse.generate(
                    helper,
                    getattr(overrides, name)(*args, **kwargs),
                    dtype=output_dtype,
                    shape=output_shape,
                )

        with helper.indent(), V.set_ops_handler(CSEProxy()):
            outputs = fn(*args)
            outputs = ", ".join(str(output) for output in outputs)
            helper.writeline(f"return {outputs}")

        return self.helper_functions.add(helper.getvalue(), base_name=helper_name)
