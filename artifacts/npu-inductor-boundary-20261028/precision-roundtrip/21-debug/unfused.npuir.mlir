[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c56_i32 = arith.constant 56 : i32
  %c128_i64 = arith.constant 128 : i64
  %c384_i64 = arith.constant 384 : i64
  %c66048_i64 = arith.constant 66048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant 28 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.index_cast %4 : i64 to index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%5]
  annotation.mark %1 {logical_block_num} : i32
  %7 = arith.addi %1, %c56_i32 : i32
  %8 = arith.cmpi slt, %3, %1 : i32
  %9 = arith.select %8, %7, %3 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg8 = %3 to %9 step %c28_i32  : i32 {
    %10 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %11 = arith.remui %10, %c2_i64 : i64
    %12 = arith.index_cast %11 : i64 to index
    %13 = arith.index_cast %12 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64_0 = arith.constant 2 : i64
    %14 = arith.select %13, %c1_i64, %c2_i64_0 : i64
    %15 = arith.subi %arg8, %c28_i32 : i32
    %16 = hivm.hir.pointer_cast(%c0_i64, %c384_i64) : memref<64xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c384_i64) : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c66048_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    %19 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    %subview = memref.subview %18[%6] [64] [1] {to_be_bubbled_slice} : memref<128xf32, #hivm.address_space<cbuf>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %20 = arith.cmpi sge, %arg8, %3 : i32
    %21 = arith.cmpi slt, %arg8, %1 : i32
    %22 = arith.andi %20, %21 : i1
    annotation.mark %22 {hivm.preload_num = 1 : i32} : i1
    scf.if %22 {
      %26 = arith.remsi %arg8, %arg5 : i32
      %27 = arith.subi %c1_i32, %26 : i32
      %28 = arith.minsi %27, %c1_i32 : i32
      %29 = arith.divsi %c0_i32, %28 : i32
      %30 = arith.cmpi sge, %29, %c0_i32 : i32
      "llvm.intr.assume"(%30) : (i1) -> ()
      %31 = arith.muli %29, %c128_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %33 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%32]
      %34 = arith.maxsi %32, %c128 : index
      %35 = arith.minsi %33, %34 : index
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %32]
      %37 = arith.cmpi slt, %36, %c128 : index
      %38 = arith.minsi %6, %36 : index
      %39 = affine.apply affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>()[%38, %35, %32]
      %40 = arith.minsi %39, %c64 : index
      %subview_1 = memref.subview %reinterpret_cast[%6] [%40] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %17[0] [%40] [1] : memref<64xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        func.call @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_0(%17) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      hivm.hir.load ins(%subview_1 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %41 = hivm.hir.pointer_cast(%c128_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_1(%17, %41) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      annotation.mark %18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index} : memref<128xf32, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.copy ins(%41 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    }
    %23 = arith.cmpi sge, %15, %3 : i32
    %24 = arith.cmpi slt, %15, %1 : i32
    %25 = arith.andi %23, %24 : i1
    annotation.mark %25 {hivm.preload_num = 0 : i32} : i1
    scf.if %25 {
      %26 = arith.remsi %15, %arg5 : i32
      %27 = arith.subi %c1_i32, %26 : i32
      %28 = arith.minsi %27, %c1_i32 : i32
      %29 = arith.remsi %26, %28 : i32
      %30 = arith.addi %26, %29 : i32
      %31 = arith.divsi %c0_i32, %28 : i32
      %32 = arith.cmpi sge, %30, %c0_i32 : i32
      "llvm.intr.assume"(%32) : (i1) -> ()
      %33 = arith.cmpi sge, %31, %c0_i32 : i32
      "llvm.intr.assume"(%33) : (i1) -> ()
    }
  } {hivm.max_preload_num = 2 : i32}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c56_i32 = arith.constant 56 : i32
  %c65536_i64 = arith.constant 65536 : i64
  %c99328_i64 = arith.constant 99328 : i64
  %c33280_i64 = arith.constant 33280 : i64
  %c66560_i64 = arith.constant 66560 : i64
  %c512_i64 = arith.constant 512 : i64
  %c66048_i64 = arith.constant 66048 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c28_i32 = arith.constant 28 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  %4 = arith.addi %1, %c56_i32 : i32
  %5 = arith.cmpi slt, %3, %1 : i32
  %6 = arith.select %5, %4, %3 : i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  scf.for %arg8 = %3 to %6 step %c28_i32  : i32 {
    %7 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %8 = arith.remui %7, %c2_i64 : i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = arith.index_cast %9 : index to i1
    %c2_i64_1 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %11 = arith.select %10, %c2_i64_1, %c3_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_2, %c1_i64_3 : i64
    %14 = arith.subi %arg8, %c28_i32 : i32
    %15 = hivm.hir.pointer_cast(%c0_i64_0, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %16 = hivm.hir.pointer_cast(%c0_i64_0, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %17 = hivm.hir.pointer_cast(%c33280_i64, %c99328_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %18 = hivm.hir.pointer_cast(%c33280_i64, %c99328_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %19 = hivm.hir.pointer_cast(%c512_i64, %c66560_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c512_i64, %c66560_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c0_i64_0, %c66048_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64_0) : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %23 = arith.cmpi sge, %arg8, %3 : i32
    %24 = arith.cmpi slt, %arg8, %1 : i32
    %25 = arith.andi %23, %24 : i1
    annotation.mark %25 {hivm.preload_num = 1 : i32} : i1
    scf.if %25 {
      %29 = arith.remsi %arg8, %arg5 : i32
      %30 = arith.subi %c1_i32, %29 : i32
      %31 = arith.minsi %30, %c1_i32 : i32
      %32 = arith.divsi %c0_i32, %31 : i32
      %33 = arith.cmpi sge, %32, %c0_i32 : i32
      "llvm.intr.assume"(%33) : (i1) -> ()
      annotation.mark %21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = -1 : index} : memref<128xf32, #hivm.address_space<cbuf>>
    }
    %26 = arith.cmpi sge, %14, %3 : i32
    %27 = arith.cmpi slt, %14, %1 : i32
    %28 = arith.andi %26, %27 : i1
    annotation.mark %28 {hivm.preload_num = 0 : i32} : i1
    scf.if %28 {
      %29 = arith.remsi %14, %arg5 : i32
      %30 = arith.subi %c1_i32, %29 : i32
      %31 = arith.minsi %30, %c1_i32 : i32
      %32 = arith.remsi %29, %31 : i32
      %33 = arith.addi %29, %32 : i32
      %34 = arith.divsi %c0_i32, %31 : i32
      %35 = arith.cmpi sge, %33, %c0_i32 : i32
      "llvm.intr.assume"(%35) : (i1) -> ()
      %36 = arith.cmpi sge, %34, %c0_i32 : i32
      "llvm.intr.assume"(%36) : (i1) -> ()
      %37 = arith.muli %33, %c128_i32 : i32
      %38 = arith.muli %34, %c128_i32 : i32
      %39 = arith.index_cast %37 : i32 to index
      %40 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%39]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %13]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%19 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %41 = arith.index_cast %38 : i32 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %11]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%17 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %42 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%41]
      %43 = arith.maxsi %41, %c128 : index
      %44 = arith.minsi %42, %43 : index
      %45 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%44, %41]
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %13]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C, normalized_init_or_bias} ins(%19, %17, %true, %c128, %c128, %c128, %22 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index, memref<128xf32, #hivm.address_space<cbuf>>) outs(%15 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %13, %11, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%41, %39]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%39]
      %48 = arith.maxsi %39, %c128 : index
      %49 = arith.minsi %47, %48 : index
      %50 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%49, %39]
      %51 = arith.minsi %50, %c128 : index
      %52 = arith.maxsi %51, %c0 : index
      %53 = arith.minsi %45, %c128 : index
      %54 = arith.maxsi %53, %c0 : index
      %55 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%52]
      %56 = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%54]
      %subview = memref.subview %15[0, 0, 0, 0] [%56, %55, 16, 16] [1, 1, 1, 1] : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>
      %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%52, %54] [1, 1] : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_FIX>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>} ins(%subview : memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>) outs(%subview_6 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %13]
    }
  } {hivm.max_preload_num = 2 : i32}
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: argument unused during compilation: '--cce-disable-asc-reserved-ubuf' [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
