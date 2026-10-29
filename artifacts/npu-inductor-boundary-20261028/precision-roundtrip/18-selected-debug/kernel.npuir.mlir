[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused__to_copy_bmm_2_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c56_i32 = arith.constant 56 : i32
  %c163840_i64 = arith.constant 163840 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c98304_i64 = arith.constant 98304 : i64
  %c0_i64 = arith.constant 0 : i64
  %c256_i32 = arith.constant 256 : i32
  %c128_i32 = arith.constant 128 : i32
  %c762880_i32 = arith.constant 762880 : i32
  %c512_i32 = arith.constant 512 : i32
  %c4_i32 = arith.constant 4 : i32
  %c6_i32 = arith.constant 6 : i32
  %c1234_i32 = arith.constant 1234 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c28_i32 = arith.constant 28 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.index_cast %4 : i64 to index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%5]
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
  %7 = arith.addi %1, %c56_i32 : i32
  %8 = arith.cmpi slt, %3, %1 : i32
  %9 = arith.select %8, %7, %3 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %9 step %c28_i32  : i32 {
    %10 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %11 = arith.remui %10, %c2_i64 : i64
    %12 = arith.index_cast %11 : i64 to index
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_0, %c1_i64 : i64
    %15 = arith.subi %arg9, %c28_i32 : i32
    %16 = hivm.hir.pointer_cast(%c65536_i64, %c163840_i64) : memref<64x256xbf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c65536_i64, %c163840_i64) : memref<64x256xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<64x256xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<64x256xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c98304_i64) : memref<64x256xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c98304_i64, %c0_i64) : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %19 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %20 = arith.cmpi sge, %arg9, %3 : i32
    %21 = arith.cmpi slt, %arg9, %1 : i32
    %22 = arith.andi %20, %21 : i1
    annotation.mark %22 {hivm.preload_num = 1 : i32} : i1
    scf.if %22 {
      %alloca = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %c0_i32, %alloca[] : memref<i32>
      scf.for %arg10 = %c0_i32 to %c512_i32 step %c256_i32  : i32 {
        %26 = memref.load %alloca[] : memref<i32>
        %27 = arith.addi %26, %c1_i32 : i32
        memref.store %27, %alloca[] : memref<i32>
      }
    }
    %23 = arith.cmpi sge, %15, %3 : i32
    %24 = arith.cmpi slt, %15, %1 : i32
    %25 = arith.andi %23, %24 : i1
    annotation.mark %25 {hivm.preload_num = 0 : i32} : i1
    scf.if %25 {
      %26 = arith.remsi %15, %arg6 : i32
      %27 = arith.divsi %15, %arg6 : i32
      %28 = arith.remsi %27, %arg7 : i32
      %29 = arith.divsi %26, %c6_i32 : i32
      %30 = arith.subi %c4_i32, %29 : i32
      %31 = arith.minsi %30, %c1_i32 : i32
      %32 = arith.remsi %26, %31 : i32
      %33 = arith.addi %29, %32 : i32
      %34 = arith.remsi %26, %c6_i32 : i32
      %35 = arith.divsi %34, %31 : i32
      %36 = arith.muli %33, %c128_i32 : i32
      %37 = arith.muli %35, %c256_i32 : i32
      %38 = arith.minsi %37, %c1234_i32 : i32
      %39 = arith.muli %28, %c762880_i32 : i32
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      func.call @triton_tem_fused__to_copy_bmm_2_mix_aiv_outlined_vf_0(%19, %16) {hivm.vector_function, no_inline} : (memref<64x256xf32, #hivm.address_space<ub>>, memref<64x256xbf16, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %40 = arith.index_cast %38 : i32 to index
      %41 = arith.index_cast %36 : i32 to index
      %42 = arith.index_cast %39 : i32 to index
      %43 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1490)>()[%40, %42, %41]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [128, 256], strides: [1490, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[%6, 0] [64, 256] [1, 1] {to_be_bubbled_slice} : memref<128x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%16 : memref<64x256xbf16, #hivm.address_space<ub>>) outs(%subview : memref<64x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%43], sizes: [128, 256], strides: [1490, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %reinterpret_cast_1[%6, 0] [64, 256] [1, 1] {to_be_bubbled_slice} : memref<128x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%19 : memref<64x256xf32, #hivm.address_space<ub>>) outs(%subview_2 : memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%19 : memref<64x256xf32, #hivm.address_space<ub>>) outs(%subview_2 : memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    }
  } {hivm.max_preload_num = 2 : i32}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused__to_copy_bmm_2_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c56_i32 = arith.constant 56 : i32
  %c262144_i64 = arith.constant 262144 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c196608_i64 = arith.constant 196608 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c98304_i64 = arith.constant 98304 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c28_i32 = arith.constant 28 : i32
  %c1_i32 = arith.constant 1 : i32
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c0_i32 = arith.constant 0 : i32
  %c1234_i32 = arith.constant 1234 : i32
  %c6_i32 = arith.constant 6 : i32
  %c4_i32 = arith.constant 4 : i32
  %c512_i32 = arith.constant 512 : i32
  %c762880_i32 = arith.constant 762880 : i32
  %c262144_i32 = arith.constant 262144 : i32
  %c128_i32 = arith.constant 128 : i32
  %c256_i32 = arith.constant 256 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.addi %1, %c56_i32 : i32
  %5 = arith.cmpi slt, %3, %1 : i32
  %6 = arith.select %5, %4, %3 : i32
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %6 step %c28_i32  : i32 {
    %7 = hivm.hir.multi_buffer_counter -> i64
    %c2_i64 = arith.constant 2 : i64
    %8 = arith.remui %7, %c2_i64 : i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64_2 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_1, %c1_i64_2 : i64
    %12 = hivm.hir.pointer_cast(%c0_i64_0, %c131072_i64) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    %13 = hivm.hir.pointer_cast(%c0_i64_0, %c131072_i64) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    %14 = hivm.hir.pointer_cast(%c0_i64_0, %c98304_i64) : memref<64x256xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c98304_i64, %c0_i64_0) : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %16 = arith.cmpi sge, %arg9, %3 : i32
    %17 = arith.cmpi slt, %arg9, %1 : i32
    %18 = arith.andi %16, %17 : i1
    annotation.mark %18 {hivm.preload_num = 1 : i32} : i1
    scf.if %18 {
      %19 = arith.remsi %arg9, %arg6 : i32
      %20 = arith.divsi %arg9, %arg6 : i32
      %21 = arith.remsi %20, %arg7 : i32
      %22 = arith.divsi %19, %c6_i32 : i32
      %23 = arith.subi %c4_i32, %22 : i32
      %24 = arith.minsi %23, %c1_i32 : i32
      %25 = arith.remsi %19, %24 : i32
      %26 = arith.addi %22, %25 : i32
      %27 = arith.remsi %19, %c6_i32 : i32
      %28 = arith.divsi %27, %24 : i32
      %29 = arith.muli %26, %c128_i32 : i32
      %30 = arith.muli %28, %c256_i32 : i32
      %31 = arith.minsi %30, %c1234_i32 : i32
      %32 = arith.muli %21, %c262144_i32 : i32
      %33 = arith.muli %21, %c762880_i32 : i32
      %alloca = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
      memref.store %c0_i32, %alloca[] : memref<i32>
      %34 = arith.index_cast %29 : i32 to index
      %35 = arith.index_cast %32 : i32 to index
      %36 = arith.index_cast %31 : i32 to index
      %37 = arith.index_cast %33 : i32 to index
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %11]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      scf.for %arg10 = %c0_i32 to %c512_i32 step %c256_i32  : i32 {
        %38 = hivm.hir.multi_buffer_counter -> i64
        %c2_i64_3 = arith.constant 2 : i64
        %39 = arith.remui %38, %c2_i64_3 : i64
        %40 = arith.index_cast %39 : i64 to index
        %41 = arith.index_cast %40 : index to i1
        %c2_i64_4 = arith.constant 2 : i64
        %c3_i64 = arith.constant 3 : i64
        %42 = arith.select %41, %c2_i64_4, %c3_i64 : i64
        %43 = arith.index_cast %40 : index to i1
        %c0_i64_5 = arith.constant 0 : i64
        %c1_i64_6 = arith.constant 1 : i64
        %44 = arith.select %43, %c0_i64_5, %c1_i64_6 : i64
        %45 = hivm.hir.pointer_cast(%c65536_i64, %c262144_i64) : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>
        %46 = hivm.hir.pointer_cast(%c0_i64_0, %c196608_i64) : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
        %47 = arith.index_cast %arg10 : i32 to index
        %48 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 512)>()[%47, %35, %34]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%48], sizes: [128, 256], strides: [512, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        %49 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 1490)>()[%36, %37, %47]
        %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [256, 256], strides: [1490, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %44]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%46 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %42]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_7 : memref<256x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>) outs(%45 : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        %50 = memref.load %alloca[] : memref<i32>
        %51 = arith.cmpi eq, %50, %c0_i32 : i32
        %c-1_i64 = arith.constant -1 : i64
        hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C} ins(%46, %45, %51, %c128, %c256, %c256 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>, memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%13 : memref<16x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %44, %42, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
        %52 = arith.addi %50, %c1_i32 : i32
        memref.store %52, %alloca[] : memref<i32>
      }
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      annotation.mark %14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%13 : memref<16x8x16x16xf32, #hivm.address_space<cc>>) outs(%14 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %11]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
    }
  } {hivm.max_preload_num = 2 : i32}
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: argument unused during compilation: '--cce-disable-asc-reserved-ubuf' [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
