[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c56_i32 = arith.constant 56 : i32
  %c84_i32 = arith.constant 84 : i32
  %c180736_i64 = arith.constant 180736 : i64
  %c65920_i64 = arith.constant 65920 : i64
  %c147840_i64 = arith.constant 147840 : i64
  %c115072_i64 = arith.constant 115072 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c65664_i64 = arith.constant 65664 : i64
  %c180608_i64 = arith.constant 180608 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c66048_i64 = arith.constant 66048 : i64
  %c82304_i64 = arith.constant 82304 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
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
  %7 = arith.addi %1, %c84_i32 : i32
  %8 = arith.cmpi slt, %3, %1 : i32
  %9 = arith.select %8, %7, %3 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
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
    %16 = arith.subi %arg8, %c56_i32 : i32
    %17 = hivm.hir.pointer_cast(%c65920_i64, %c180736_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c65920_i64, %c180736_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c65920_i64, %c180736_i64) : memref<64x128xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<64x128xbf16, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<64x128xbf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<64x128xbf16, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c32768_i64, %c115072_i64, %c147840_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c147840_i64, %c32768_i64, %c115072_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c115072_i64, %c147840_i64, %c32768_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c65536_i64, %c180608_i64) : memref<64xbf16, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c65536_i64, %c180608_i64) : memref<64xbf16, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c65536_i64, %c180608_i64) : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<64xbf16, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c0_i64, %c66048_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    %29 = hivm.hir.pointer_cast(%c82304_i64, %c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c0_i64, %c82304_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c82304_i64, %c0_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %31 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %30 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %29 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
    %subview = memref.subview %26[%6] [64] [1] {to_be_bubbled_slice} : memref<128xf32, #hivm.address_space<cbuf>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %32 = arith.cmpi sge, %arg8, %3 : i32
    %33 = arith.cmpi slt, %arg8, %1 : i32
    %34 = arith.andi %32, %33 : i1
    annotation.mark %34 {hivm.preload_num = 2 : i32} : i1
    scf.if %34 {
      %41 = arith.remsi %arg8, %arg5 : i32
      %42 = arith.subi %c1_i32, %41 : i32
      %43 = arith.minsi %42, %c1_i32 : i32
      %44 = arith.divsi %c0_i32, %43 : i32
      %45 = arith.cmpi sge, %44, %c0_i32 : i32
      "llvm.intr.assume"(%45) : (i1) -> ()
      %46 = arith.muli %44, %c128_i32 : i32
      %47 = arith.index_cast %46 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %48 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%47]
      %49 = arith.maxsi %47, %c128 : index
      %50 = arith.minsi %48, %49 : index
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %47]
      %52 = arith.cmpi slt, %51, %c128 : index
      %53 = arith.minsi %6, %51 : index
      %54 = affine.apply affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>()[%53, %50, %47]
      %55 = arith.minsi %54, %c64 : index
      %subview_1 = memref.subview %reinterpret_cast[%6] [%55] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %25[0] [%55] [1] : memref<64xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %52 {
        func.call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%25) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      hivm.hir.load ins(%subview_1 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      annotation.mark %26 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index} : memref<128xf32, #hivm.address_space<cbuf>>
      %56 = hivm.hir.pointer_cast(%c65664_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      func.call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_merged_vf_0(%25, %56, %20) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xbf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.copy ins(%56 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    }
    %35 = arith.cmpi sge, %15, %3 : i32
    %36 = arith.cmpi slt, %15, %1 : i32
    %37 = arith.andi %35, %36 : i1
    annotation.mark %37 {hivm.preload_num = 1 : i32} : i1
    scf.if %37 {
      %41 = arith.remsi %15, %arg5 : i32
      %42 = arith.subi %c1_i32, %41 : i32
      %43 = arith.minsi %42, %c1_i32 : i32
      %44 = arith.remsi %41, %43 : i32
      %45 = arith.addi %41, %44 : i32
      %46 = arith.divsi %c0_i32, %43 : i32
      %47 = arith.cmpi sge, %45, %c0_i32 : i32
      "llvm.intr.assume"(%47) : (i1) -> ()
      %48 = arith.cmpi sge, %46, %c0_i32 : i32
      "llvm.intr.assume"(%48) : (i1) -> ()
    }
    %38 = arith.cmpi sge, %16, %3 : i32
    %39 = arith.cmpi slt, %16, %1 : i32
    %40 = arith.andi %38, %39 : i1
    annotation.mark %40 {hivm.preload_num = 0 : i32} : i1
    scf.if %40 {
      %41 = arith.remsi %16, %arg5 : i32
      %42 = arith.subi %c1_i32, %41 : i32
      %43 = arith.minsi %42, %c1_i32 : i32
      %44 = arith.remsi %41, %43 : i32
      %45 = arith.addi %41, %44 : i32
      %46 = arith.divsi %c0_i32, %43 : i32
      %47 = arith.cmpi sge, %45, %c0_i32 : i32
      "llvm.intr.assume"(%47) : (i1) -> ()
      %48 = arith.cmpi sge, %46, %c0_i32 : i32
      "llvm.intr.assume"(%48) : (i1) -> ()
      %49 = arith.muli %45, %c128_i32 : i32
      %50 = arith.muli %46, %c128_i32 : i32
      %51 = arith.index_cast %49 : i32 to index
      %52 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%51]
      %53 = arith.index_cast %50 : i32 to index
      %54 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%53]
      %55 = arith.maxsi %53, %c128 : index
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%53, %51]
      %57 = arith.minsi %54, %55 : index
      %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %53]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%56], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      func.call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_3(%22, %31, %17) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xbf16, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %59 = arith.maxsi %51, %c128 : index
      %60 = arith.minsi %52, %59 : index
      %61 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%60, %51]
      %62 = arith.minsi %61, %c128 : index
      %63 = arith.maxsi %62, %c0 : index
      %64 = arith.minsi %58, %c128 : index
      %65 = arith.maxsi %64, %c0 : index
      %66 = arith.minsi %6, %63 : index
      %67 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%63, %66]
      %68 = arith.minsi %67, %c64 : index
      %subview_1 = memref.subview %17[0, 0] [%68, %65] [1, 1] : memref<64x128xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %reinterpret_cast[%6, 0] [%68, %65] [1, 1] : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_1 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_1 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    }
  } {hivm.max_preload_num = 3 : i32}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @triton_tem_fused_addmm_relu_sub_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c56_i32 = arith.constant 56 : i32
  %c84_i32 = arith.constant 84 : i32
  %c65536_i64 = arith.constant 65536 : i64
  %c99328_i64 = arith.constant 99328 : i64
  %c33280_i64 = arith.constant 33280 : i64
  %c66560_i64 = arith.constant 66560 : i64
  %c512_i64 = arith.constant 512 : i64
  %c66048_i64 = arith.constant 66048 : i64
  %c82304_i64 = arith.constant 82304 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c28_i32 = arith.constant 28 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
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
  %4 = arith.addi %1, %c84_i32 : i32
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
    %15 = arith.subi %arg8, %c56_i32 : i32
    %16 = hivm.hir.pointer_cast(%c0_i64_0, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %17 = hivm.hir.pointer_cast(%c0_i64_0, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %18 = hivm.hir.pointer_cast(%c0_i64_0, %c65536_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %19 = hivm.hir.pointer_cast(%c33280_i64, %c99328_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c33280_i64, %c99328_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c33280_i64, %c99328_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c512_i64, %c66560_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %23 = hivm.hir.pointer_cast(%c512_i64, %c66560_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %24 = hivm.hir.pointer_cast(%c512_i64, %c66560_i64) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %25 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64_0) : memref<128xf32, #hivm.address_space<cbuf>>
    %26 = hivm.hir.pointer_cast(%c0_i64_0, %c66048_i64) : memref<128xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c66048_i64, %c0_i64_0) : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c82304_i64, %c0_i64_0) : memref<64x128xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c0_i64_0, %c82304_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c82304_i64, %c0_i64_0) : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %31 = arith.cmpi sge, %arg8, %3 : i32
    %32 = arith.cmpi slt, %arg8, %1 : i32
    %33 = arith.andi %31, %32 : i1
    annotation.mark %33 {hivm.preload_num = 2 : i32} : i1
    scf.if %33 {
      %40 = arith.remsi %arg8, %arg5 : i32
      %41 = arith.subi %c1_i32, %40 : i32
      %42 = arith.minsi %41, %c1_i32 : i32
      %43 = arith.divsi %c0_i32, %42 : i32
      %44 = arith.cmpi sge, %43, %c0_i32 : i32
      "llvm.intr.assume"(%44) : (i1) -> ()
      annotation.mark %25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = -1 : index} : memref<128xf32, #hivm.address_space<cbuf>>
    }
    %34 = arith.cmpi sge, %14, %3 : i32
    %35 = arith.cmpi slt, %14, %1 : i32
    %36 = arith.andi %34, %35 : i1
    annotation.mark %36 {hivm.preload_num = 1 : i32} : i1
    scf.if %36 {
      %40 = arith.remsi %14, %arg5 : i32
      %41 = arith.subi %c1_i32, %40 : i32
      %42 = arith.minsi %41, %c1_i32 : i32
      %43 = arith.remsi %40, %42 : i32
      %44 = arith.addi %40, %43 : i32
      %45 = arith.divsi %c0_i32, %42 : i32
      %46 = arith.cmpi sge, %44, %c0_i32 : i32
      "llvm.intr.assume"(%46) : (i1) -> ()
      %47 = arith.cmpi sge, %45, %c0_i32 : i32
      "llvm.intr.assume"(%47) : (i1) -> ()
      %48 = arith.muli %44, %c128_i32 : i32
      %49 = arith.muli %45, %c128_i32 : i32
      %50 = arith.index_cast %48 : i32 to index
      %51 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%50]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %13]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%23 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %52 = arith.index_cast %49 : i32 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %11]
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %13]
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C, normalized_init_or_bias} ins(%23, %20, %true, %c128, %c128, %c128, %26 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index, memref<128xf32, #hivm.address_space<cbuf>>) outs(%17 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %13, %11, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      annotation.mark %29 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%17 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%29 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %13]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    }
    %37 = arith.cmpi sge, %15, %3 : i32
    %38 = arith.cmpi slt, %15, %1 : i32
    %39 = arith.andi %37, %38 : i1
    annotation.mark %39 {hivm.preload_num = 0 : i32} : i1
    scf.if %39 {
      %40 = arith.remsi %15, %arg5 : i32
      %41 = arith.subi %c1_i32, %40 : i32
      %42 = arith.minsi %41, %c1_i32 : i32
      %43 = arith.remsi %40, %42 : i32
      %44 = arith.addi %40, %43 : i32
      %45 = arith.divsi %c0_i32, %42 : i32
      %46 = arith.cmpi sge, %44, %c0_i32 : i32
      "llvm.intr.assume"(%46) : (i1) -> ()
      %47 = arith.cmpi sge, %45, %c0_i32 : i32
      "llvm.intr.assume"(%47) : (i1) -> ()
    }
  } {hivm.max_preload_num = 3 : i32}
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
bisheng: warning: argument unused during compilation: '--cce-disable-asc-reserved-ubuf' [-Wunused-command-line-argument]
ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
