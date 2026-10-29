[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_0(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c131328_i64 = arith.constant 131328 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c98560_i64 = arith.constant 98560 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c65792_i64 = arith.constant 65792 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant 28 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  scf.for %arg8 = %3 to %1 step %c28_i32  : i32 {
    %4 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %5 = arith.remui %4, %c2_i64 : i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = arith.index_cast %6 : index to i1
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %8 = arith.select %7, %c4_i64, %c5_i64 : i64
    %9 = arith.index_cast %6 : index to i1
    %c2_i64_0 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %10 = arith.select %9, %c2_i64_0, %c3_i64 : i64
    %11 = arith.index_cast %6 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_1, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %14 = hivm.hir.pointer_cast(%c65536_i64, %c131328_i64) : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %15 = hivm.hir.pointer_cast(%c32768_i64, %c98560_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c65792_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %17 = arith.remsi %arg8, %arg5 : i32
    %18 = arith.subi %c1_i32, %17 : i32
    %19 = arith.minsi %18, %c1_i32 : i32
    %20 = arith.remsi %17, %19 : i32
    %21 = arith.addi %17, %20 : i32
    %22 = arith.divsi %c0_i32, %19 : i32
    %23 = arith.cmpi sge, %21, %c0_i32 : i32
    "llvm.intr.assume"(%23) : (i1) -> ()
    %24 = arith.cmpi sge, %22, %c0_i32 : i32
    "llvm.intr.assume"(%24) : (i1) -> ()
    %25 = arith.muli %21, %c128_i32 : i32
    %26 = arith.muli %22, %c128_i32 : i32
    %27 = arith.index_cast %25 : i32 to index
    %28 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%27]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %12]
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%16 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    %29 = arith.index_cast %26 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %10]
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%15 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %30 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%29]
    %31 = arith.maxsi %29, %c128 : index
    %32 = arith.minsi %30, %31 : index
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %29]
    %34 = arith.cmpi slt, %33, %c128 : index
    %subview = memref.subview %reinterpret_cast_3[0] [%33] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %14[0] [%33] [1] : memref<128xf16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %8]
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f16 init_out_buffer = true init_condition = %34 : i1 eviction_policy = <EvictFirst>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %12]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %c-1_i64 = arith.constant -1 : i64
    hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C, normalized_init_or_bias} ins(%16, %15, %true, %c128, %c128, %c128, %14 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index, memref<128xf16, #hivm.address_space<cbuf>>) outs(%13 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64, %c-1_i64, %12, %10, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %8]
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%29, %27]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %36 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%27]
    %37 = arith.maxsi %27, %c128 : index
    %38 = arith.minsi %36, %37 : index
    %39 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %27]
    %40 = arith.minsi %39, %c128 : index
    %41 = arith.maxsi %40, %c0 : index
    %42 = arith.minsi %33, %c128 : index
    %43 = arith.maxsi %42, %c0 : index
    %44 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%41]
    %45 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%43]
    %subview_6 = memref.subview %13[0, 0, 0, 0] [%45, %44, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%41, %43] [1, 1] : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_FIX>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview_6 : memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>) outs(%subview_7 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %12]
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: argument unused during compilation: '--cce-disable-asc-reserved-ubuf' [-Wunused-command-line-argument]
