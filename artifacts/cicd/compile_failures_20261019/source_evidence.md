# Source evidence

Extracted from installed torch_npu on the validation target.

## torch_npu/_inductor/codegen/triton.py

SHA256: `dc7ff8c142f979645111c13ad56f3a8bfff38d418d4428989ba4100f84b51674`

```text
1874:     def create_inductor_meta(self):
1875:         mutated_args = set()  # noqa: set_linter
1876:         for mutation in self.mutations:
1877:             if mutation in self.args.input_buffers:
1878:                 mutated_args.add(self.args.input_buffers[mutation])
1879:             if (
1880:                 mutation in self.args.inplace_buffers
1881:                 and mutation not in V.graph.removed_buffers
1882:                 and mutation not in self.removed_buffers
1883:             ):
1884:                 mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
1885:             if mutation in self.args.output_buffers:
1886:                 mutated_args.add(self.args.output_buffers[mutation])
1887:         mutated_args = sorted(mutated_args)
1888:         tiling_axis = [x.sorted_order for x in self.tiling_axis]
1889:         if (
1890:             self.vectorized_welford_axis is not None
1891:             and self.vectorized_welford_axis.sorted_order not in tiling_axis
1892:         ):
1893:             tiling_axis.append(self.vectorized_welford_axis.sorted_order)
1894:         no_loop_axis = [x.sorted_order for x in self.tiling_axis if x.is_no_loop_axis]
1895:         split_axis = [x.sorted_order for x in self.split_axis]
1896:         axis_names = [x.name for x in self.sorted_axis]
1897:         axis_static_values = []
1898:         for axis in self.sorted_axis:
1899:             length = axis.length
1900:             if isinstance(length, (int, sympy.Integer)):
1901:                 axis_static_values.append((axis.name, int(length)))
1902:         split_axis_dtype = (
1903:             self.get_axis_dtype(self.split_axis[0]) if self.split_axis else None
1904:         )
1905:         runtime_block_arg_names = tuple(
1906:             f"{axis.name.upper()}BLOCK" for axis in self.split_axis
1907:         )
1908: 
1909:         inductor_meta = {
1910:             "grid_type": self._get_grid_type().__name__,
1911:             "autotune_hints": set(self.autotune_hints),  # noqa: set_linter
1912:             "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
1913:             "mutated_arg_names": mutated_args,
1914:             # Due to breaking change of triton 3.0, the original invocation is broken
1915:             "backend_hash": torch.utils._triton.triton_hash_with_backend(),
1916:             "split_axis": split_axis,
1917:             "tiling_axis": tiling_axis,
1918:             "no_loop_axis": no_loop_axis,
1919:             "axis_names": axis_names,
1920:             "axis_static_values": tuple(axis_static_values),
1921:             "low_dims": self.low_dims,
1922:             "numof_reduction_axis": self.numof_reduction_axis(),
1923:             "split_axis_dtype": split_axis_dtype,
1924:             "dual_reduction": self.numof_reduction_axis() > 1,
1925:             "npu_kernel_type": str(self.npu_kernel_type),
1926:             "traced_graph_hash": "TRACED_GRAPH_HASH",
1927:             "traced_graph_dir": "TRACED_GRAPH_DIR",
1928:             "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
1929:             "runtime_block_arg_names": runtime_block_arg_names,
1930:             **TritonKernel.inductor_meta_common(),
```

```text
2026:     def add_autotune_args(self, argdefs, signature, triton_meta_signature):
2027:         for axis in self.split_axis:
2028:             arg_name = f"{axis.name.upper()}BLOCK"
2029:             sizearg = SizeArg(arg_name, axis.length)
2030:             signature.append(sizearg)
2031:             triton_meta_signature[arg_name] = signature_of(
2032:                 sizearg, size_dtype=self.index_dtype
2033:             )
2034:             argdefs.append(ArgName(arg_name))
2035: 
2036:         for axis in self.tiling_axis:
2037:             # Skip persistent_reduction reduction axis only if length is static (handled in codegen_static_numels)
2038:             if axis.name[0] == "r" and self.persistent_reduction:
2039:                 simplified = V.graph.sizevars.simplify(axis.length)
2040:                 if isinstance(simplified, (sympy.Integer, int)):
2041:                     continue  # Static length: handled by codegen_static_numels
2042:                 # Dynamic length: need to pass as parameter
2043:             elif axis.is_no_loop_axis:
2044:                 simplified = V.graph.sizevars.simplify(axis.length)
2045:                 if isinstance(simplified, (sympy.Integer, int)):
2046:                     continue  # Static length: handled by codegen_static_numels
2047:                 # Dynamic length: need to pass as parameter
2048:             argdefs.append(ArgName(f"{axis.name.upper()}BLOCK_SUB", is_constexpr=True))
2049: 
2050:         if (
2051:             self.vectorized_welford_axis is not None
2052:             and self.vectorized_welford_axis not in self.tiling_axis
2053:         ):
2054:             argdefs.append(
2055:                 ArgName(
```

```text
2634:     def codegen_body(self):
2635:         """
2636:         codegen triton kernel body
2637:         """
2638: 
2639:         if not (
2640:             self.loads
2641:             or self.stores
2642:             or self.compute
2643:             or self.post_loop_store
2644:             or self._original_loads_buf
2645:             or self._original_compute_buf
2646:             or self._original_stores_buf
2647:         ):
2648:             return
2649: 
2650:         def write_pointwise(allow_stores=None):
2651:             if allow_stores is None:
2652:                 allow_stores = self.numof_reduction_axis() <= 1
2653:             self._emit_coordinate_transforms()
2654:             self.body.splice(self.indexing_code)
2655:             loads = (
2656:                 self._original_loads_buf
2657:                 if self._original_loads_buf is not None
2658:                 else self.loads
2659:             )
2660:             self.body.splice(loads)
2661:             compute = (
2662:                 self._original_compute_buf
2663:                 if self._original_compute_buf is not None
2664:                 else self.compute
2665:             )
2666:             self.body.splice(compute)
2667:             if allow_stores:
2668:                 stores = (
2669:                     self._original_stores_buf
2670:                     if self._original_stores_buf is not None
2671:                     else self.stores
2672:                 )
2673:                 self.body.splice(stores)
2674: 
2675:         def collect_store_unified_vars():
2676:             """
2677:             Collect unified axis names that are directly referenced by Store indices.
```

```text
3681:                 "dims_to_remove": dims_to_remove,
3682:                 "unified_axis": unified_axis,
3683:             }
3684:         return guarded_expansions
3685: 
3686:     def _remove_expansion_dims_from_axes(self, dims_to_remove):
3687:         """
3688:         Remove sub-axis dimensions from active tiling/sorted axes.
3689: 
3690:         Args:
3691:             dims_to_remove: Dimension names to remove from active structure axes.
3692:         """
3693:         self.tiling_axis = [
3694:             axis for axis in self.tiling_axis if axis.name not in dims_to_remove
3695:         ]
3696: 
3697:         if hasattr(self, "sorted_axis") and self.sorted_axis:
3698:             self.sorted_axis = [
3699:                 axis for axis in self.sorted_axis if axis.name not in dims_to_remove
3700:             ]
3701: 
3702:             for i, axis in enumerate(self.sorted_axis):
3703:                 axis.sorted_order = i
3704: 
3705:             self.low_dims = set()  # noqa: set_linter
3706: 
3707:     def _append_unified_axis_to_active_axes(self, unified_axis):
3708:         """
3709:         Add the unified axis back into active tiling and sorted axes.
3710: 
3711:         Args:
3712:             unified_axis: The Store-side unified axis.
3713:         """
3714:         if not unified_axis:
3715:             return
3716: 
3717:         unified_axis.is_tiling_axis = True
3718:         unified_axis.is_no_loop_axis = False
3719:         if unified_axis not in self.tiling_axis:
3720:             unified_axis.tiling_order = len(self.tiling_axis)
3721:             self.tiling_axis.append(unified_axis)
3722:         else:
3723:             unified_axis.tiling_order = self.tiling_axis.index(unified_axis)
3724: 
3725:         if (
3726:             hasattr(self, "sorted_axis")
3727:             and self.sorted_axis is not None
3728:             and unified_axis not in self.sorted_axis
3729:         ):
3730:             unified_axis.sorted_order = len(self.sorted_axis)
3731:             self.sorted_axis.append(unified_axis)
3732:         elif hasattr(self, "sorted_axis") and self.sorted_axis is not None:
3733:             unified_axis.sorted_order = self.sorted_axis.index(unified_axis)
3734: 
3735:     def _clear_sub_axis_states(self, dims_to_remove):
```

```text
3780:     def _apply_guarded_expansions(self, guarded_expansions):
3781:         """
3782:         Apply unified-axis special path using guarded expansion metadata.
3783: 
3784:         Args:
3785:             guarded_expansions: Guarded expansion groups keyed by prefix.
3786:         """
3787:         self.coordinate_transforms = {}
3788: 
3789:         for prefix, guarded_info in guarded_expansions.items():
3790:             expansion_list = guarded_info["expansion_list"]
3791:             dims_to_remove = guarded_info["dims_to_remove"]
3792:             unified_axis = guarded_info["unified_axis"]
3793:             self._remove_expansion_dims_from_axes(dims_to_remove)
3794:             self._append_unified_axis_to_active_axes(unified_axis)
3795:             self._clear_sub_axis_states(dims_to_remove)
3796:             self._update_coordinate_transforms(prefix, expansion_list, unified_axis)
3797:             log.info(
3798:                 "Apply unified-axis special path for prefix %s: unified_axis=%s, removed_dims=%s",
3799:                 prefix,
3800:                 unified_axis.name if unified_axis else None,
3801:                 sorted(dims_to_remove),
3802:             )
3803: 
3804:         self.golden_var_list = tuple([x.symbol() for x in self.tiling_axis])
3805:         log.info(
3806:             "Unified-axis final structure: golden=%s, tiling_axis=%s, sorted_axis=%s",
3807:             self.golden_var_list,
3808:             [x.name for x in self.tiling_axis],
3809:             [x.name for x in self.sorted_axis] if self.sorted_axis else [],
3810:         )
3811: 
3812:     def _select_golden_varlist_normal_case(self):
```

```text
4533: 
4534:         if has_tmpmask:
4535:             # Masked loads must come after the mask is computed
4536:             load_buffer = self.compute
4537:         elif (
4538:             self.inside_reduction
4539:             and self.range_trees[-1].is_loop
4540:             and not indirect_indexing
4541:             and not has_rindex
4542:         ):
4543:             # can lift a common load outside of reduction loop
4544:             # One exception is when this is an indirect_load.
4545:             load_buffer = self.prefix
4546: 
4547:         else:
4548:             load_buffer = self.loads
4549: 
4550:         result_var = self.cse.generate(load_buffer, line, dtype=dtype, shape=shape)
4551:         if not (isinstance(result_var, TritonCSEVariable)):
```

## torch_npu/_inductor/runtime/tile_generator.py

SHA256: `2d7a514a3a182b492965871c335ecbdefe2753c183cfb1f9245096e60520c9e6`

```text
22:     def __init__(self, numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims, persistent_reduction,
23:                  dtype, npu_kernel_type=NPUKernelType.SIMD, input_ptr_num=0, dual_reduction=False):
24:         self.numels = copy.deepcopy(numels)
25:         self.blocks = list(self.numels)
26:         self.candidate_blocks = []
27:         self.sub_blocks = self.blocks.copy()
28:         self.axis_name = axis_names
29:         self.tiling_axis = tiling_axis
30:         self.no_loop_axis = no_loop_axis
31:         self.split_axis = split_axis
32:         self.low_dims = low_dims
33:         self.configs = []
34:         self.dtype_bytes = get_byte_per_numel(dtype)
35:         self.rebuild_dynamic_axis()
36:         self.block_name = {}
37:         self.sub_block_name = {}
38:         self.persistent_reduction = persistent_reduction
39:         self.dual_reduction = dual_reduction
40:         self.npu_kernel_type = npu_kernel_type
41:         self.max_total_numel = functools.reduce(lambda x, y: x * y, self.blocks) if self.blocks else 1
42:         self.tiny_kernel = self.max_total_numel < 128 * 1024
43:         self.input_ptr_num = 3 if input_ptr_num == 0 else min(input_ptr_num, 3)
44: 
45:         local_mem_size = 128 * 1024 if self.npu_kernel_type == NPUKernelType.SIMT_ONLY else config.ub_size
46:         self.max_numel_threshold = local_mem_size // self.input_ptr_num // self.dtype_bytes
47:         self.stop_numel = min(self.max_numel_threshold, self.max_total_numel // config.num_vector_core) // 8
```

```text
121:     def valid_tile_numel(self, total_numel):
122:         max_numel = self.max_numel_threshold
123:         return total_numel <= max_numel
124: 
125:     def calculate_config_numel(self, cfg):
126:         total_numel = 1
127:         for i, _ in enumerate(self.sub_blocks):
128:             sub_block_name = self.sub_block_name.get(i, "")
129:             config_sub_block = cfg.get(sub_block_name, None)
130:             if config_sub_block is None:
131:                 total_numel = total_numel * 1
132:             else:
133:                 total_numel = total_numel * config_sub_block
134:         return total_numel
135: 
136:     def calculate_total_numel(self):
137:         smallest = sys.maxsize
138: 
139:         def calculate_total_numel_candi(blocks):
140:             total_numel = 1
141:             for axis in self.sub_blocks:
142:                 total_numel = total_numel * axis
143:             return total_numel
144: 
145:         for candi_blocks in self.candidate_blocks:
146:             numel = calculate_total_numel_candi(candi_blocks)
147:             if numel < smallest:
148:                 smallest = numel
149:         return smallest
150: 
```

```text
175:     def find_config(self, cfg):
176:         for config_var in self.configs:
177:             if config_var.kwargs == cfg:
178:                 return True
179:         return False
180: 
181:     def add_to_configs(self, candi_block):
182:         newcfg = {}
183:         self.fill_config(newcfg, candi_block)
184:         total_numel = self.calculate_config_numel(newcfg)
185:         stop_numel_threshold = 0 if len(self.configs) < 10 or self.tiny_kernel else self.stop_numel + 100
186:         if not self.valid_tile_numel(total_numel):
187:             return False
188:         if self.find_config(newcfg):
189:             return False
190:         if total_numel < stop_numel_threshold:
191:             return False
192: 
193:         # This is tmp check for simt overflow, and will be removed latter
194:         if self.npu_kernel_type == NPUKernelType.SIMT_ONLY and total_numel * self.dtype_bytes > 8 * 1024:
195:             return False
196:         self.configs.append(Config(newcfg, num_warps=1, num_stages=2))
197:         return True
198: 
199:     def desecnd_all_low_dims_with_all_blocks(self):
```

```text
359:     def descend_all_low_dims(self):
360:         low_dim_numels = [self.sub_blocks[x] for x in self.low_dims]
361:         if not low_dim_numels:
362:             return
363: 
364:         def descent_all_axis(min_numel):
365:             for axis in self.low_dims:
366:                 if self.axis_name[axis][0] == "r" and self.persistent_reduction:
367:                     continue
368:                 if axis in self.no_loop_axis:
369:                     continue
370: 
371:                 numel = self.sub_blocks[axis]
372:                 if numel == 1:
373:                     continue
374:                 if min_numel > 1 and abs(numel - min_numel) / min_numel < 0.2:
375:                     continue
376:                 if numel >= 128:
377:                     self.sub_blocks[axis] = next_power_of_2(numel // 2)
378:                 else:  # numel >4 and numel < 128 :
379:                     numel = self.sub_blocks[axis]
380:                     numel = numel // 2
381:                     self.sub_blocks[axis] = min(aligned_numel_32byte(numel, self.dtype_bytes), next_power_of_2(numel))
382: 
383:         count = 0
384:         total_numel = self.calculate_total_numel()
385:         while total_numel > self.stop_numel and count < 100:
386:             count += 1
387:             total_numel = self.calculate_total_numel()
388:             for candi_block in self.candidate_blocks:
389:                 self.add_to_configs(candi_block)
390:             if not self.candidate_blocks:
391:                 self.add_to_configs(self.sub_blocks)
392:             min_numel = min(low_dim_numels)
393:             descent_all_axis(min_numel)
394:             total_numel_2 = self.calculate_total_numel()
395:             if total_numel == total_numel_2:
396:                 descent_all_axis(0)
397: 
398:         return total_numel < self.stop_numel
```

```text
428:     def descend_split_tiling(self):
429:         self.reset_configs()
430:         tiling_not_low_dims = [x for x in self.tiling_axis if x not in self.low_dims]
431: 
432:         def descend_split_axis():
433: 
434:             for axis_index in range(len(self.split_axis)):
435:                 if self.descend_split_axis_inner(axis_index):
436:                     return True
437: 
438:             total = self.calculate_total_numel()
439:             return total <= self.stop_numel
440: 
441:         def desceond_tiling_not_low_dims():
442:             for axis in tiling_not_low_dims:
443:                 if self.axis_name[axis][0] == "r" and self.persistent_reduction:
444:                     continue
445:                 if axis in self.no_loop_axis:
446:                     continue
447: 
448:                 if self.descend_tiling_axis(axis):
449:                     return True
450:             total = self.calculate_total_numel()
451:             return total <= self.stop_numel
452: 
453:         while True:
454:             # descend split axis
455:             if descend_split_axis():
456:                 break
457:             if len(self.candidate_blocks) > 0:
458:                 self.sub_blocks = list(self.candidate_blocks[0])
459:             # descend tiling but not low dims
460:             if desceond_tiling_not_low_dims():
461:                 break
462:                 # descend low dims, need to descend all axis at the same time
463:             self.descend_all_low_dims()
464:             break
465:         self.add_extra_options()
```

## torch_npu/_inductor/runtime/triton_heuristics.py

SHA256: `ffc45bd8fae9c72f7a3ad2dc2be56275166eb9d41e925040bbe094f5bcda4763`

```text
1004:     def _precompile_config(self, cfg: Config) -> TritonCompileResultNpu:
1005:         """Ahead of time compile a given autotuner config."""
1006:         compile_meta = copy.deepcopy(self.triton_meta)
1007:         cfg_kwargs = cfg.kwargs
1008:         constexpr_arg_names = {self.fn.arg_names[i] for i in self.fn.constexprs}
1009:         for k, v in cfg_kwargs.items():
1010:             if k not in constexpr_arg_names:
1011:                 continue
1012:             compile_meta["constants"][k] = v
1013: 
1014:         for i in self.fn.constexprs:
1015:             arg_name = self.fn.arg_names[i]
1016:             if arg_name not in compile_meta["constants"] and (
1017:                 arg_name == "num_warps" or arg_name == "num_stages"
1018:             ):
1019:                 compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
1020:         compile_meta["num_warps"] = cfg.num_warps
```

```text
3586: def _triton_config_npu_index_legacy(
3587:     size_hints,
3588:     inductor_meta,
3589:     triton_meta=None,
3590:     is_reduction=False,
3591:     is_persistent_reduction=False,
3592: ) -> List[Config]:
3593:     num_warps = 1
3594:     num_stages = 2
3595:     configs = []
3596:     split_axis = inductor_meta["split_axis"]
3597:     tiling_axis = inductor_meta["tiling_axis"]
3598:     no_loop_axis = inductor_meta.get("no_loop_axis", [])
3599:     low_dims = inductor_meta["low_dims"]
3600:     split_axis_dtype = inductor_meta["split_axis_dtype"]
3601:     axis_names = inductor_meta["axis_names"]
3602:     dual_reduction = inductor_meta["dual_reduction"]
3603:     input_signature = triton_meta["signature"]
3604:     input_ptr_num = len(list(filter(lambda k: 'ptr' in k, input_signature))) if triton_meta is not None else 0
3605:     npu_kernel_type = NPUKernelType(inductor_meta.get("npu_kernel_type", "simd"))
3606:     size_hints = [size_hints.get(axis_name, 1) for axis_name in axis_names]
3607: 
3608:     if npu_config.fasta_autotune:
3609:         from .fasta_autotune import FastATileGenerator
3610:         tile_generator = FastATileGenerator(size_hints, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims,
3611:                                             persistent_reduction=is_persistent_reduction,
3612:                                             dtype=split_axis_dtype,
3613:                                             npu_kernel_type=npu_kernel_type,
3614:                                             input_ptr_num=input_ptr_num, dual_reduction=dual_reduction)
3615:         configs = tile_generator.descend_split_tiling()
3616:     else:
3617:         tile_generator = TileGenerator(size_hints, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims,
3618:                                        persistent_reduction=is_persistent_reduction,
3619:                                        dtype=split_axis_dtype,
3620:                                        npu_kernel_type=npu_kernel_type,
3621:                                        input_ptr_num=input_ptr_num, dual_reduction=dual_reduction)
3622:         if npu_kernel_type == NPUKernelType.SIMD_SIMT_MIX:
3623:             tile_generator.set_kernel_type(NPUKernelType.SIMT_ONLY)
3624:             configs.extend(tile_generator.descend_split_tiling())
3625:             tile_generator.set_kernel_type(NPUKernelType.SIMT_TEMPLATE)
3626:             configs.extend(tile_generator.descend_split_tiling())
3627:         else:
3628:             configs = tile_generator.descend_split_tiling()
3629: 
3630:     if not configs:
3631:         cfg = {}
3632:         for x in split_axis:
3633:             cfg[f"{axis_names[x].upper()}BLOCK"] = size_hints[x]
3634:         for x in tiling_axis:
3635:             cfg[f"{axis_names[x].upper()}BLOCK_SUB"] = size_hints[x]
3636:         if not cfg:
3637:             cfg["dummy"] = 1
3638:         tmp = Config(cfg, num_warps=num_warps, num_stages=num_stages)
3639:         configs.append(tmp)
3640: 
3641:     for cfg in configs:
3642:         split_blocks = [None for x in split_axis]
3643:         for i, axis in enumerate(split_axis):
3644:             name = axis_names[axis]
3645:             block_name = f"{name.upper()}BLOCK"
3646:             split_blocks[i] = cfg.kwargs[block_name]
3647:         cfg.kwargs["split_axis"] = tuple(split_axis)
3648:         cfg.kwargs["split_blocks"] = tuple(split_blocks)
3649: 
3650:     _remap_fallback_block_subs(configs, inductor_meta)
```

## benchmarks/torchbench/timm_models.py

SHA256: `c21c644e642c8ccd9183358a77aaeee53573c139fb4ac1f3fb17bf1c11318cd4`

```text
86: TIMM_MODELS = dict()
87: filename = os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
88: 
89: abspath = os.path.abspath(filename)
90: PathManager.check_directory_path_readable(abspath)
91: 
92: with open(filename) as fh:
93:     lines = fh.readlines()
94:     lines = [line.rstrip() for line in lines]
95:     for line in lines:
96:         model_name, batch_size = line.split(" ")
97:         TIMM_MODELS[model_name] = int(batch_size)
```

```text
155:     def load_model(
156:         self,
157:         device,
158:         model_name,
159:         batch_size=None,
160:     ):
161:         is_training = self.args.training
162:         model = self._download_model(model_name)
163: 
164:         if model is None:
165:             raise RuntimeError(f"Failed to load model '{model_name}'")
166:         model.to(device=device)
167: 
168:         self.num_classes = model.num_classes
169: 
170:         data_config = resolve_data_config(
171:             vars(self._args) if timmversion >= "0.8.0" else self._args,
172:             model=model,
173:             use_test_size=not is_training,
174:         )
175:         input_size = data_config["input_size"]
176:         recorded_batch_size = TIMM_MODELS[model_name]
177: 
178:         batch_size = batch_size or recorded_batch_size
```

Current `timm_models_list.txt`, line 13: `swin_base_patch4_window7_224 128`.

## Candidate failures versus kernel failure

`torch_npu/_inductor/runtime/triton_heuristics.py`:

```text
910:     def _precompile_configs_once(self, configs):
911:         compile_results = [None] * len(configs)
912:         compile_exceptions = [None] * len(configs)
913:         compile_exception_stacks = [""] * len(configs)
914:         for index, candidate_config in enumerate(configs):
915:             try:
916:                 compile_results[index] = self._precompile_config(candidate_config)
917:             except Exception as exc:
918:                 import traceback
919:                 compile_exception_stacks[index] = traceback.format_exc()
920:                 compile_exceptions[index] = exc
921:         return compile_results, compile_exceptions, compile_exception_stacks
922: 
923:     def _configs_with_vf_fusion(self, configs):
924:         retry_configs = []
925:         for candidate_config in configs:
926:             retry_config = copy.deepcopy(candidate_config)
927:             retry_config.kwargs["enable_vf_fusion"] = True
928:             retry_configs.append(retry_config)
929:         return retry_configs
930: 
931:     def _precompile_configs_with_vf_retry(self, configs, compile_once):
932:         if not configs:
933:             raise NoTritonConfigsError("No triton configs are available")
934: 
935:         compile_start_time = time.perf_counter()
936:         compile_results, exceptions, exception_stacks = compile_once(configs)
937:         successful_results = [
938:             result for result in compile_results if result is not None
939:         ]
940:         if not successful_results:
941:             retry_results, retry_exceptions, retry_exception_stacks = compile_once(
942:                 self._configs_with_vf_fusion(configs)
943:             )
944:             successful_results = [
945:                 result for result in retry_results if result is not None
946:             ]
947:             exceptions = retry_exceptions + exceptions
948:             exception_stacks = retry_exception_stacks + exception_stacks
949: 
950:         if not successful_results:
951:             exc, exc_stack = next(
952:                 (
953:                     (exc, stack)
954:                     for exc, stack in zip(exceptions, exception_stacks)
955:                     if exc is not None
956:                 ),
957:                 (None, ""),
958:             )
959:             raise NoTritonConfigsError(
960:                 f"No valid triton configs for kernel {self.get_fn_name()}. "
961:                 f"{type(exc).__name__}: {exc} \nStack trace:{exc_stack}"
962:             )
963:         log.debug(
```
