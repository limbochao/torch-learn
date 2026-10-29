[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c56_i32 = arith.constant 56 : i32
  %c81920_i64 = arith.constant 81920 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant 28 : i32
  %c0 = arith.constant 0 : index
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.index_cast %4 : i64 to index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%5]
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
  %7 = arith.addi %1, %c56_i32 : i32
  %8 = arith.cmpi slt, %3, %1 : i32
  %9 = arith.select %8, %7, %3 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg8 = %3 to %9 step %c28_i32  : i32 {
    %10 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %11 = arith.remui %10, %c2_i64 : i64
    %12 = arith.index_cast %11 : i64 to index
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_0, %c1_i64 : i64
    %15 = arith.subi %arg8, %c28_i32 : i32
    %16 = hivm.hir.pointer_cast(%c32768_i64, %c81920_i64) : memref<64x128xf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c32768_i64, %c81920_i64) : memref<64x128xf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<64x128xf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<64x128xf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c49152_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c49152_i64, %c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %19 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
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
      %29 = arith.remsi %26, %28 : i32
      %30 = arith.addi %26, %29 : i32
      %31 = arith.divsi %c0_i32, %28 : i32
      %32 = arith.cmpi sge, %30, %c0_i32 : i32
      "llvm.intr.assume"(%32) : (i1) -> ()
      %33 = arith.cmpi sge, %31, %c0_i32 : i32
      "llvm.intr.assume"(%33) : (i1) -> ()
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
      %34 = arith.muli %30, %c128_i32 : i32
      %35 = arith.muli %31, %c128_i32 : i32
      %36 = arith.index_cast %34 : i32 to index
      %37 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%36]
      %38 = arith.index_cast %35 : i32 to index
      %39 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%38]
      %40 = arith.maxsi %38, %c128 : index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%38, %36]
      %42 = arith.minsi %39, %40 : index
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%42, %38]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      func.call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%19, %16) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf16, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %44 = arith.maxsi %36, %c128 : index
      %45 = arith.minsi %37, %44 : index
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%45, %36]
      %47 = arith.minsi %46, %c128 : index
      %48 = arith.maxsi %47, %c0 : index
      %49 = arith.minsi %43, %c128 : index
      %50 = arith.maxsi %49, %c0 : index
      %51 = arith.minsi %6, %48 : index
      %52 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%48, %51]
      %53 = arith.minsi %52, %c64 : index
      %subview = memref.subview %16[0, 0] [%53, %50] [1, 1] : memref<64x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %reinterpret_cast[%6, 0] [%53, %50] [1, 1] : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    }
  } {hivm.max_preload_num = 2 : i32}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_relu_sub_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c56_i32 = arith.constant 56 : i32
  %c131328_i64 = arith.constant 131328 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c98560_i64 = arith.constant 98560 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c65792_i64 = arith.constant 65792 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c0_i64 = arith.constant 0 : i64
  %c28_i32 = arith.constant 28 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %cst = arith.constant 0.000000e+00 : f16
  %c128 = arith.constant 128 : index
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.addi %1, %c56_i32 : i32
  %5 = arith.cmpi slt, %3, %1 : i32
  %6 = arith.select %5, %4, %3 : i32
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
  scf.for %arg8 = %3 to %6 step %c28_i32  : i32 {
    %7 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %8 = arith.remui %7, %c2_i64 : i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = arith.index_cast %9 : index to i1
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %11 = arith.select %10, %c4_i64, %c5_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c2_i64_0 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %13 = arith.select %12, %c2_i64_0, %c3_i64 : i64
    %14 = arith.index_cast %9 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_1, %c1_i64 : i64
    %16 = arith.subi %arg8, %c28_i32 : i32
    %17 = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %19 = hivm.hir.pointer_cast(%c65536_i64, %c131328_i64) : memref<128xf16, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c65536_i64, %c131328_i64) : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c32768_i64, %c98560_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c32768_i64, %c98560_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %23 = hivm.hir.pointer_cast(%c0_i64, %c65792_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %24 = hivm.hir.pointer_cast(%c0_i64, %c65792_i64) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %25 = hivm.hir.pointer_cast(%c0_i64, %c49152_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c49152_i64, %c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %27 = arith.cmpi sge, %arg8, %3 : i32
    %28 = arith.cmpi slt, %arg8, %1 : i32
    %29 = arith.andi %27, %28 : i1
    annotation.mark %29 {hivm.preload_num = 1 : i32} : i1
    scf.if %29 {
      %33 = arith.remsi %arg8, %arg5 : i32
      %34 = arith.subi %c1_i32, %33 : i32
      %35 = arith.minsi %34, %c1_i32 : i32
      %36 = arith.remsi %33, %35 : i32
      %37 = arith.addi %33, %36 : i32
      %38 = arith.divsi %c0_i32, %35 : i32
      %39 = arith.cmpi sge, %37, %c0_i32 : i32
      "llvm.intr.assume"(%39) : (i1) -> ()
      %40 = arith.cmpi sge, %38, %c0_i32 : i32
      "llvm.intr.assume"(%40) : (i1) -> ()
      %41 = arith.muli %37, %c128_i32 : i32
      %42 = arith.muli %38, %c128_i32 : i32
      %43 = arith.index_cast %41 : i32 to index
      %44 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%43]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %15]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%24 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
      %45 = arith.index_cast %42 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%45], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %13]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_2 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%22 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %46 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%45]
      %47 = arith.maxsi %45, %c128 : index
      %48 = arith.minsi %46, %47 : index
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%48, %45]
      %50 = arith.cmpi slt, %49, %c128 : index
      %subview = memref.subview %reinterpret_cast_3[0] [%49] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %20[0] [%49] [1] : memref<128xf16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f16 init_out_buffer = true init_condition = %50 : i1 eviction_policy = <EvictFirst>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %15]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C, normalized_init_or_bias} ins(%24, %22, %true, %c128, %c128, %c128, %20 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index, memref<128xf16, #hivm.address_space<cbuf>>) outs(%18 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64, %c-1_i64, %15, %13, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %11]
      annotation.mark %25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%18 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%25 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %15]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    }
    %30 = arith.cmpi sge, %16, %3 : i32
    %31 = arith.cmpi slt, %16, %1 : i32
    %32 = arith.andi %30, %31 : i1
    annotation.mark %32 {hivm.preload_num = 0 : i32} : i1
    scf.if %32 {
      %33 = arith.remsi %16, %arg5 : i32
      %34 = arith.subi %c1_i32, %33 : i32
      %35 = arith.minsi %34, %c1_i32 : i32
      %36 = arith.remsi %33, %35 : i32
      %37 = arith.addi %33, %36 : i32
      %38 = arith.divsi %c0_i32, %35 : i32
      %39 = arith.cmpi sge, %37, %c0_i32 : i32
      "llvm.intr.assume"(%39) : (i1) -> ()
      %40 = arith.cmpi sge, %38, %c0_i32 : i32
      "llvm.intr.assume"(%40) : (i1) -> ()
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
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: argument unused during compilation: '--cce-disable-asc-reserved-ubuf' [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
