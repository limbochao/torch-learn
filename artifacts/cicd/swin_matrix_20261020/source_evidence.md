# Source evidence

安装版本与 runner 源码摘录；行号对应原文件。

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
48:         for axis, name in enumerate(self.axis_name):
49:             if axis not in tiling_axis and axis not in split_axis:
50:                 self.blocks[axis] = 1
51:                 self.sub_blocks[axis] = 1
52:                 continue
53:             if axis in self.split_axis:
54:                 self.block_name[axis] = f"{name.upper()}BLOCK"
55:             if axis in self.tiling_axis:
56:                 self.sub_block_name[axis] = f"{name.upper()}BLOCK_SUB"
57: 
58:         self.program_threshold = (
59:             config.num_vector_core // 8
60:             if self.tiny_kernel and self.npu_kernel_type != NPUKernelType.SIMT_ONLY
61:             else config.num_vector_core // 2
62:         )
63:         self.program_threshold = min(self.max_total_numel // 128 - 1, self.program_threshold)
64:         self.program_threshold = max(0, self.program_threshold)
65:         self.npu_kernel_type = npu_kernel_type
66:         self.real_tiling_axis = []
67:         for tiling_axis in self.tiling_axis:
68:             if tiling_axis in self.no_loop_axis:
69:                 continue
70:             if self.axis_name[tiling_axis][0] == "r" and self.persistent_reduction:
71:                 continue
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
151:     def fill_config(self, cfg, blocks):
```

```text
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
```

```text
207:     def descend_tiling_axis(self, axis):
208:         def calc_total_programs():
209:             grids = []
210:             for axis in self.split_axis:
211:                 numel = self.numels[axis]
212:                 block_size = self.blocks[axis]
213:                 programs = (numel + block_size - 1) // block_size
214:                 grids.append(programs)
215: 
216:             total_programs = functools.reduce(lambda x, y: x * y, grids) if grids else 1
217:             return total_programs
218: 
219:         reached_stop_numel = False
220: 
221:         while True:
222:             if axis in self.real_tiling_axis:
223:                 do_tiling_axis_index = self.real_tiling_axis.index(axis)
224:                 if do_tiling_axis_index + 1 < len(self.real_tiling_axis):
225:                     tmp_block = copy.deepcopy(self.blocks)
226:                     tmp_sub_block = copy.deepcopy(self.sub_blocks)
227:                     self.descend_tiling_axis(self.real_tiling_axis[do_tiling_axis_index + 1])
228:                     self.blocks = tmp_block
229:                     self.sub_blocks = tmp_sub_block
230: 
231:             for candi_block in self.candidate_blocks:
232:                 if self.add_to_configs(candi_block):
233:                     self.desecnd_all_low_dims_with_all_blocks()
234: 
235:             if not self.candidate_blocks:
236:                 if self.add_to_configs(self.blocks):
237:                     self.desecnd_all_low_dims_with_all_blocks()
238: 
239:             # tile numel reached threshold
240:             total_numel = self.calculate_total_numel()
241:             if total_numel <= self.stop_numel:
242:                 if self.add_to_configs(self.blocks):
243:                     self.desecnd_all_low_dims_with_all_blocks()
244:                 reached_stop_numel = True
245:                 break
246: 
247:             numel = self.sub_blocks[axis]
248:             if numel == 1:
249:                 if self.add_to_configs(self.blocks):
250:                     self.desecnd_all_low_dims_with_all_blocks()
251:                 break
252: 
253:             if numel >= 32:
254:                 self.sub_blocks[axis] = next_power_of_2(numel // 2)
255: 
256:             else:  # numel < 32 :
257:                 numel = self.sub_blocks[axis]
258:                 self.sub_blocks[axis] = numel - 1
259:         return reached_stop_numel
```

```text
291:                 self.descend_split_axis_inner(axis_index + 1)
292:                 self.blocks = tmp_block
293:                 self.sub_blocks = tmp_sub_block
294: 
295:             for candi_block in self.candidate_blocks:
296:                 if self.add_to_configs(candi_block):
297:                     self.desecnd_all_low_dims_with_all_blocks()
298: 
299:             # tile numel reached threshold
300:             total_numel = self.calculate_total_numel()
301:             if total_numel <= self.stop_numel:
302:                 if self.add_to_configs(self.blocks):
303:                     self.desecnd_all_low_dims_with_all_blocks()
304:                 reached_stop_numel = True
305:                 break
306: 
307:             numel = self.blocks[axis]
```

## benchmarks/torchbench/timm_models.py

SHA256: `c21c644e642c8ccd9183358a77aaeee53573c139fb4ac1f3fb17bf1c11318cd4`

```text
174:         )
175:         input_size = data_config["input_size"]
176:         recorded_batch_size = TIMM_MODELS[model_name]
177: 
178:         batch_size = batch_size or recorded_batch_size
179: 
180:         torch.manual_seed(1337)
181:         input_tensor = torch.randint(
```

```text
233:     def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
234:         cosine = self.args.cosine
235:         tolerance = 1e-3
236:         if is_training:
237:             if name in REQUIRE_HIGHER_TOLERANCE:
238:                 tolerance = 4 * 1e-2
239:             else:
240:                 tolerance = 2 * 1e-2
241:         return tolerance, cosine
```

```text
258:         with self.autocast():
259:             return mod(*inputs)
260: 
261:     def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
262:         cloned_inputs = clone_inputs(inputs)
263:         self.optimizer_zero_grad(mod)
264:         with self.autocast():
265:             pred = mod(*cloned_inputs)
266:             if isinstance(pred, tuple):
267:                 pred = pred[0]
268:             loss = self.compute_loss(pred)
269:         self.grad_scaler.scale(loss).backward()
270:         self.optimizer_step()
271:         if collect_outputs:
272:             return collect_results(mod, pred, loss, cloned_inputs)
273:         return loss.item()
274: 
275: 
```

## benchmarks/torchbench/common.py

SHA256: `2338ffaa9f59ba70264d039b5f56c766973a9ed88071cf733cb4f053b63e710e`

```text
1876:     if args.dynamic_batch_only:
1877:         args.dynamic_shapes = True
1878:         torch._dynamo.config.assume_static_by_default = True
1879:     if args.dynamic_shapes:
1880:         if not args.dynamic_batch_only:
1881:             torch._dynamo.config.assume_static_by_default = False
1882: 
1883:     if args.ddp:
1884:         # but just to measure impact on singlenode of performing graph-breaks.
1885:         # Left it as a follow up to keep this PR isolated.
1886:         if not args.accuracy:
1887:             raise AssertionError(
1888:                 "DDP benchmark is currently only hooked up to --accuracy bench"
1889:             )
1890:         if not args.training:
1891:             raise AssertionError("DDP benchmark requires --training mode")
1892:     if args.accuracy or args.precision_checker:
1893:         # Use small batch size. We use >1 batch size to ensure we test
1894:         # batch_norm type of operators that work on batch dims.
1895:         if args.batch_size is None:
1896:             if runner.suite_name == "huggingface":
1897:                 args.batch_size = 1
1898:             elif runner.suite_name == "torchbench":
1899:                 args.batch_size = 4
1900:             else:
1901:                 # Larger batch size of TIMM models to have stable batch_norm
1902:                 if runner.suite_name != "timm_models":
1903:                     raise AssertionError
1904:                 args.batch_size = 8
1905: 
1906:         # Remove sources of randomness
1907:         inductor_config.fallback_random = True
1908:         if args.only is not None and args.only not in {
```

```text
2249:     backend = None
2250:     if args.backend:
2251:         backend = args.backend
2252:     else:
2253:         backend = None
2254: 
2255:     # Dynamic shape Config
2256:     dynamic = None
2257:     if args.dynamic_shapes:
2258:         dynamic = True
2259: 
2260:     if backend:
2261:         compile_kwargs = {
2262:             "backend": backend,
2263:             "fullgraph": args.nopython,
2264:             "dynamic": dynamic,
2265:         }
2266:         compile_kwargs["mode"] = mode
2267:         if backend == "inductor" and hasattr(args, "npu_backend"):
2268:             if npu_backend == "default":
```

## Training kernel 47: additional TileGenerator excerpts

```text
76:     def rebuild_dynamic_axis(self):
77:         if len(self.tiling_axis) > 3 and not self.no_loop_axis:
78:             current_numels = 1
79:             for low_dim in self.low_dims:
80:                 if self.axis_name[low_dim] == 'r':
81:                     continue
82:                 current_numels *= self.numels[low_dim]
83:                 if current_numels * self.dtype_bytes > 4 * 1024:
84:                     return
85:                 self.no_loop_axis.append(low_dim)
86: 
87: 
```

```text
151:     def fill_config(self, cfg, blocks):
152:         for axis in self.split_axis:
153:             cfg[self.block_name[axis]] = aligned_numel_32byte(blocks[axis], self.dtype_bytes)
154:         for axis in self.tiling_axis:
155:             if self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
156:                 tiling_numel = next_power_of_2(self.sub_blocks[axis])
157:                 while tiling_numel > blocks[axis]:
158:                     tiling_numel = tiling_numel // 2
159:             else:
160:                 tiling_numel = min(self.sub_blocks[axis], blocks[axis])
161:             cfg[self.sub_block_name[axis]] = aligned_numel_32byte(tiling_numel, self.dtype_bytes)
162:         cfg["compile_mode"] = self.npu_kernel_type.compile_mode()
163: 
```

```text
317:             if not slow_decend_split or self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
318:                 self.blocks[axis] = numel // 2
319:                 self.sub_blocks[axis] = self.blocks[axis]
320: 
321:             else:
322:                 step = numel // 4 if numel // 4 > 1 else 1
323:                 self.blocks[axis] = numel - step
324:                 self.sub_blocks[axis] = self.blocks[axis]
325: 
326:             total_programs = calc_total_programs()
327:             if total_programs > config.num_vector_core:
328:                 last_blocks = self.calcu_last_split_blocks(axis)
329:                 if last_blocks != self.blocks[axis]:
330:                     self.blocks[axis] = last_blocks
331:                 if tuple(self.blocks) not in self.candidate_blocks:
332:                     self.candidate_blocks.append(tuple(self.blocks))
333:                     self.add_to_configs(self.candidate_blocks[-1])
334:                 break
```

```text
357: 
358: 
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
```
