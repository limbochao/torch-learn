# Excerpt from the generated wrapper; not an executable module.
            buf182 = empty_strided((1568, 512), (512, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [linear_82, permute_355, view_807], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:67
            extern_kernels.mm(buf181, primals_304, out=buf182)
            del buf181
            del primals_304
            buf189 = reinterpret_tensor(buf159, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [view_782, view_809, permute_359, clone_292, constant_pad_nd_31, sub_51, mul_177, mul_178, mul_179, sum_86, mul_180, sum_87, mul_181, sub_52, sub_53, div_80, mul_182, add_146], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view, aten.native_layer_norm_backward, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_native_46.run(buf189, buf182, primals_302, view_547, getitem_190, getitem_191, 8, 2, 7, 2, 7, 512, stream=raw_stream0)
            del primals_302
            buf187 = empty_strided((512, ), (1, ), device='npu', dtype=torch.float32)
            buf188 = empty_strided((512, ), (1, ), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [view_809, permute_359, clone_292, constant_pad_nd_31, sub_51, mul_177, mul_183, sum_88, sum_89], Original ATen: [aten.view, aten.permute, aten.clone, aten._unsafe_view, aten.native_layer_norm_backward]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_clone_native_lay_47.run(buf182, view_547, getitem_190, getitem_191, buf187, buf188, 512, 8, 7, 2, 14, stream=raw_stream0)
            del buf182
            del getitem_190
