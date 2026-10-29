#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_merged_vf_0(%arg0: memref<64xbf16, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(14 : i32) : i32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(64 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(128 : i32) : i32
    %7 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = builtin.unrealized_conversion_cast %arg0 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%2, %9, %5) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %11 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%11, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %14 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%12, %13, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %15 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %16 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%14, %16, %3, %0, %3, %15) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    cf.br ^bb1(%3 : i32)
  ^bb1(%17: i32):  // 2 preds: ^bb0, ^bb5
    %18 = arith.cmpi slt, %17, %4 : i32
    cf.cond_br %18, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %19 = llvm.sext %17 : i32 to i64
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    cf.br ^bb3(%3 : i32)
  ^bb3(%21: i32):  // 2 preds: ^bb2, ^bb4
    %22 = arith.cmpi slt, %21, %6 : i32
    cf.cond_br %22, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %23 = llvm.sext %21 : i32 to i64
    %24 = builtin.unrealized_conversion_cast %23 : i64 to index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg2 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %c128 = arith.constant 128 : index
    %25 = arith.muli %20, %c128 : index
    %26 = arith.addi %25, %24 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%26], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.extractvalue %27[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%10, %31, %3, %0, %3, %28) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %32 = arith.addi %21, %4 overflow<nsw> : i32
    cf.br ^bb3(%32 : i32)
  ^bb5:  // pred: ^bb3
    %33 = arith.addi %17, %5 overflow<nsw> : i32
    cf.br ^bb1(%33 : i32)
  ^bb6:  // pred: ^bb1
    return
  }
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(1 : i8) : i8
    %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.mlir.constant(65536 : i64) : i64
    %9 = llvm.mlir.constant(33280 : i64) : i64
    %10 = llvm.mlir.constant(99328 : i64) : i64
    %11 = llvm.mlir.constant(512 : i64) : i64
    %12 = llvm.mlir.constant(66560 : i64) : i64
    %13 = llvm.mlir.constant(66048 : i64) : i64
    %14 = llvm.mlir.constant(82304 : i64) : i64
    %15 = llvm.mlir.constant(-1 : i64) : i64
    %16 = llvm.mlir.constant(3 : i64) : i64
    %17 = llvm.mlir.constant(2 : i64) : i64
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.mlir.constant(56 : i32) : i32
    %20 = llvm.mlir.constant(84 : i32) : i32
    %21 = llvm.mlir.constant(28 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(true) : i1
    %24 = llvm.mlir.constant(128 : index) : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = builtin.unrealized_conversion_cast %1 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%26] : memref<1xi64>
    %27 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %28 = builtin.unrealized_conversion_cast %27 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %29 = hivm.hir.pointer_cast(%8) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %30 = builtin.unrealized_conversion_cast %29 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %31 = hivm.hir.pointer_cast(%9) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %32 = builtin.unrealized_conversion_cast %31 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %33 = hivm.hir.pointer_cast(%10) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %35 = hivm.hir.pointer_cast(%11) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %37 = hivm.hir.pointer_cast(%12) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %39 = hivm.hir.pointer_cast(%0) : memref<128xf32, #hivm.address_space<cbuf>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = hivm.hir.pointer_cast(%13) : memref<128xf32, #hivm.address_space<cbuf>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = hivm.hir.pointer_cast(%0) : memref<64x128xf32, #hivm.address_space<ub>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = hivm.hir.pointer_cast(%14) : memref<64x128xf32, #hivm.address_space<ub>>
    %46 = builtin.unrealized_conversion_cast %45 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.mul %arg5, %arg6 : i32
    %48 = llvm.mul %47, %arg7 : i32
    %49 = hivm.hir.get_block_idx -> i64
    %50 = llvm.trunc %49 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    %51 = llvm.add %48, %20 : i32
    %52 = llvm.icmp "slt" %50, %48 : i32
    %53 = llvm.select %52, %51, %50 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    cf.br ^bb1(%50 : i32)
  ^bb1(%54: i32):  // 2 preds: ^bb0, ^bb8
    %55 = arith.cmpi slt, %54, %53 : i32
    cf.cond_br %55, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %56 = memref.load %alloca[%26] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %57 = llvm.urem %56, %17  : i64
    %58 = llvm.trunc %57 : i64 to i1
    %59 = llvm.select %58, %17, %16 : i1, i64
    %60 = llvm.xor %58, %23  : i1
    %61 = llvm.zext %60 : i1 to i64
    %62 = llvm.sub %54, %21 : i32
    %63 = llvm.sub %54, %19 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %64 = llvm.icmp "sge" %54, %50 : i32
    %65 = llvm.icmp "slt" %54, %48 : i32
    %66 = llvm.and %64, %65  : i1
    cf.cond_br %66, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %67 = llvm.srem %54, %arg5  : i32
    %68 = llvm.sub %22, %67 : i32
    %69 = llvm.intr.smin(%68, %22)  : (i32, i32) -> i32
    %70 = llvm.sdiv %2, %69  : i32
    %71 = llvm.icmp "sge" %70, %2 : i32
    "llvm.intr.assume"(%71) : (i1) -> ()
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %72 = llvm.icmp "sge" %62, %50 : i32
    %73 = llvm.icmp "slt" %62, %48 : i32
    %74 = llvm.and %72, %73  : i1
    cf.cond_br %74, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %75 = llvm.icmp "eq" %57, %18 : i64
    %76 = llvm.select %75, %30, %28 : i1, !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %77 = builtin.unrealized_conversion_cast %76 : !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %78 = llvm.select %75, %34, %32 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %79 = builtin.unrealized_conversion_cast %78 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %80 = llvm.select %75, %38, %36 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %81 = builtin.unrealized_conversion_cast %80 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %82 = llvm.select %75, %42, %40 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %83 = builtin.unrealized_conversion_cast %82 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf32, #hivm.address_space<cbuf>>
    %84 = llvm.select %75, %46, %44 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %85 = builtin.unrealized_conversion_cast %84 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %86 = llvm.srem %62, %arg5  : i32
    %87 = llvm.sub %22, %86 : i32
    %88 = llvm.intr.smin(%87, %22)  : (i32, i32) -> i32
    %89 = llvm.srem %86, %88  : i32
    %90 = llvm.add %86, %89 : i32
    %91 = llvm.sdiv %2, %88  : i32
    %92 = llvm.icmp "sge" %90, %2 : i32
    "llvm.intr.assume"(%92) : (i1) -> ()
    %93 = llvm.icmp "sge" %91, %2 : i32
    "llvm.intr.assume"(%93) : (i1) -> ()
    %94 = llvm.mul %90, %3 : i32
    %95 = llvm.mul %91, %3 : i32
    %96 = llvm.sext %94 : i32 to i64
    %97 = builtin.unrealized_conversion_cast %96 : i64 to index
    %c128 = arith.constant 128 : index
    %98 = arith.muli %97, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %61]
    %cast = memref.cast %reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %81 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast, %cast_0, %2) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %99 = llvm.sext %95 : i32 to i64
    %100 = builtin.unrealized_conversion_cast %99 : i64 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%100], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %59]
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %79 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_2, %cast_3, %2) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %61]
    %cast_4 = memref.cast %81 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_5 = memref.cast %79 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_6 = memref.cast %83 : memref<128xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %77 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_with_float_bias_bfloat16_t_to_float(%cast_4, %cast_5, %23, %25, %25, %25, %cast_6, %cast_7, %0, %18, %61, %59, %15, %15, %15, %7, %0) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_8 = memref.cast %77 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_9 = memref.cast %85 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_8, %cast_9, %0, %5, %0, %6, %23, %7, %0, %4) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %61]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %101 = llvm.icmp "sge" %63, %50 : i32
    %102 = llvm.icmp "slt" %63, %48 : i32
    %103 = llvm.and %101, %102  : i1
    cf.cond_br %103, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %104 = llvm.srem %63, %arg5  : i32
    %105 = llvm.sub %22, %104 : i32
    %106 = llvm.intr.smin(%105, %22)  : (i32, i32) -> i32
    %107 = llvm.srem %104, %106  : i32
    %108 = llvm.add %104, %107 : i32
    %109 = llvm.sdiv %2, %106  : i32
    %110 = llvm.icmp "sge" %108, %2 : i32
    "llvm.intr.assume"(%110) : (i1) -> ()
    %111 = llvm.icmp "sge" %109, %2 : i32
    "llvm.intr.assume"(%111) : (i1) -> ()
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %112 = llvm.add %56, %18 : i64
    memref.store %112, %alloca[%26] : memref<1xi64>
    %113 = arith.addi %54, %21 overflow<nsw> : i32
    cf.br ^bb1(%113 : i32)
  ^bb9:  // pred: ^bb1
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
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%arg0: memref<64xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(12 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%2, %2) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%3, %5, %1) : (bf16, vector<256xi1>, i32) -> vector<128xbf16>
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%0, %2) {mask_bit_width = 16 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
    %8 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%6, %8, %2, %1, %2, %7) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_3(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(128 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(64 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%6, %7, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%5, %9, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    cf.br ^bb1(%4 : i32)
  ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb5
    %12 = arith.cmpi slt, %11, %3 : i32
    cf.cond_br %12, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %13 = llvm.sext %11 : i32 to i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    cf.br ^bb3(%4 : i32)
  ^bb3(%15: i32):  // 2 preds: ^bb2, ^bb4
    %16 = arith.cmpi slt, %15, %1 : i32
    cf.cond_br %16, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %17 = llvm.sext %15 : i32 to i64
    %18 = builtin.unrealized_conversion_cast %17 : i64 to index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %c128 = arith.constant 128 : index
    %19 = arith.muli %14, %c128 : index
    %20 = arith.addi %19, %18 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%20], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %21[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.extractvalue %21[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.getelementptr %22[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %25 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%24, %4, %4, %4) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %c128_4 = arith.constant 128 : index
    %26 = arith.muli %14, %c128_4 : index
    %27 = arith.addi %26, %18 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_0 to offset: [%27], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %28 = builtin.unrealized_conversion_cast %reinterpret_cast_5 : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.extractvalue %28[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %32 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%31, %4, %4, %4) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %33 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %34 = "hivm_regbaseintrins.intr.hivm.vcmp.gt.s.z"(%25, %32, %33) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<256xi1>
    %35 = "hivm_regbaseintrins.intr.hivm.vsel"(%10, %32, %34) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %36 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %37 = "hivm_regbaseintrins.intr.hivm.vsub.s.x"(%35, %8, %36) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %38 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %39 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%37, %38, %4, %4, %4) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %arg2 : memref<64x128xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
    %c128_10 = arith.constant 128 : index
    %40 = arith.muli %14, %c128_10 : index
    %41 = arith.addi %40, %18 : index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_6 to offset: [%41], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, #map, #hivm.address_space<ub>>
    %42 = builtin.unrealized_conversion_cast %reinterpret_cast_11 : memref<64xbf16, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %44 = llvm.bitcast %39 : vector<128xbf16> to vector<64xf32>
    %45 = llvm.extractvalue %42[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.extractvalue %42[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.getelementptr %45[%46] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%44, %47, %4, %0, %4, %43) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %48 = arith.addi %15, %3 overflow<nsw> : i32
    cf.br ^bb3(%48 : i32)
  ^bb5:  // pred: ^bb3
    %49 = arith.addi %11, %2 overflow<nsw> : i32
    cf.br ^bb1(%49 : i32)
  ^bb6:  // pred: ^bb1
    return
  }
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_with_float_bias_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_cbuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(180736 : i64) : i64
    %3 = llvm.mlir.constant(65920 : i64) : i64
    %4 = llvm.mlir.constant(32768 : i64) : i64
    %5 = llvm.mlir.constant(115072 : i64) : i64
    %6 = llvm.mlir.constant(147840 : i64) : i64
    %7 = llvm.mlir.constant(65536 : i64) : i64
    %8 = llvm.mlir.constant(180608 : i64) : i64
    %9 = llvm.mlir.constant(66048 : i64) : i64
    %10 = llvm.mlir.constant(82304 : i64) : i64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(56 : i32) : i32
    %14 = llvm.mlir.constant(84 : i32) : i32
    %15 = llvm.mlir.constant(65664 : i64) : i64
    %16 = llvm.mlir.constant(64 : index) : i64
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(128 : i32) : i32
    %19 = llvm.mlir.constant(128 : index) : i64
    %20 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(28 : i32) : i32
    %23 = llvm.mlir.constant(3 : i64) : i64
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%25] : memref<1xi64>
    %26 = hivm.hir.pointer_cast(%3) : memref<64x128xbf16, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<64x128xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = hivm.hir.pointer_cast(%2) : memref<64x128xbf16, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<64x128xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = hivm.hir.pointer_cast(%4) : memref<64x128xf32, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = hivm.hir.pointer_cast(%5) : memref<64x128xf32, #hivm.address_space<ub>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = hivm.hir.pointer_cast(%6) : memref<64x128xf32, #hivm.address_space<ub>>
    %35 = builtin.unrealized_conversion_cast %34 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = hivm.hir.pointer_cast(%5) : memref<64x128xf32, #hivm.address_space<ub>>
    %37 = builtin.unrealized_conversion_cast %36 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = hivm.hir.pointer_cast(%6) : memref<64x128xf32, #hivm.address_space<ub>>
    %39 = builtin.unrealized_conversion_cast %38 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = hivm.hir.pointer_cast(%4) : memref<64x128xf32, #hivm.address_space<ub>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = hivm.hir.pointer_cast(%7) : memref<64xbf16, #hivm.address_space<ub>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = hivm.hir.pointer_cast(%8) : memref<64xbf16, #hivm.address_space<ub>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = hivm.hir.pointer_cast(%9) : memref<128xf32, #hivm.address_space<cbuf>>
    %47 = builtin.unrealized_conversion_cast %46 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = hivm.hir.pointer_cast(%0) : memref<128xf32, #hivm.address_space<cbuf>>
    %49 = builtin.unrealized_conversion_cast %48 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = hivm.hir.pointer_cast(%10) : memref<64x128xf32, #hivm.address_space<ub>>
    %51 = builtin.unrealized_conversion_cast %50 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = hivm.hir.pointer_cast(%0) : memref<64x128xf32, #hivm.address_space<ub>>
    %53 = builtin.unrealized_conversion_cast %52 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.mul %arg5, %arg6 : i32
    %55 = llvm.mul %54, %arg7 : i32
    %56 = hivm.hir.get_block_idx -> i64
    %57 = llvm.trunc %56 : i64 to i32
    %58 = hivm.hir.get_sub_block_idx -> i64
    %59 = builtin.unrealized_conversion_cast %58 : i64 to index
    %c64 = arith.constant 64 : index
    %60 = arith.muli %59, %c64 : index
    %61 = builtin.unrealized_conversion_cast %60 : index to i64
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    %62 = llvm.add %55, %14 : i32
    %63 = llvm.icmp "slt" %57, %55 : i32
    %64 = llvm.select %63, %62, %57 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%57 : i32)
  ^bb1(%65: i32):  // 2 preds: ^bb0, ^bb10
    %66 = arith.cmpi slt, %65, %64 : i32
    cf.cond_br %66, ^bb2, ^bb11
  ^bb2:  // pred: ^bb1
    %67 = memref.load %alloca[%25] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %68 = llvm.urem %67, %12  : i64
    %69 = llvm.urem %67, %23  : i64
    %70 = llvm.icmp "eq" %68, %11 : i64
    %71 = llvm.select %70, %45, %43 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<64xbf16, #hivm.address_space<ub>>
    %73 = llvm.select %70, %49, %47 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %74 = builtin.unrealized_conversion_cast %73 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf32, #hivm.address_space<cbuf>>
    %75 = llvm.trunc %68 : i64 to i1
    %76 = llvm.select %75, %11, %12 : i1, i64
    %77 = llvm.sub %65, %22 : i32
    %78 = llvm.sub %65, %13 : i32
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %74 : memref<128xf32, #hivm.address_space<cbuf>> -> memref<f32, #hivm.address_space<cbuf>>, index, index, index
    %c64_0 = arith.constant 64 : index
    %79 = arith.muli %59, %c64_0 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%79], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<cbuf>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %80 = llvm.icmp "sge" %65, %57 : i32
    %81 = llvm.icmp "slt" %65, %55 : i32
    %82 = llvm.and %80, %81  : i1
    cf.cond_br %82, ^bb3, ^bb6
  ^bb3:  // pred: ^bb2
    %83 = llvm.icmp "eq" %69, %11 : i64
    %84 = llvm.select %83, %33, %31 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %85 = llvm.icmp "eq" %69, %12 : i64
    %86 = llvm.select %85, %35, %84 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %87 = builtin.unrealized_conversion_cast %86 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %88 = llvm.srem %65, %arg5  : i32
    %89 = llvm.sub %21, %88 : i32
    %90 = llvm.intr.smin(%89, %21)  : (i32, i32) -> i32
    %91 = llvm.sdiv %17, %90  : i32
    %92 = llvm.icmp "sge" %91, %17 : i32
    "llvm.intr.assume"(%92) : (i1) -> ()
    %93 = llvm.mul %91, %18 : i32
    %94 = llvm.sext %93 : i32 to i64
    %95 = builtin.unrealized_conversion_cast %94 : i64 to index
    %c128 = arith.constant 128 : index
    %96 = arith.addi %95, %c128 : index
    %97 = builtin.unrealized_conversion_cast %96 : index to i64
    %98 = llvm.intr.smax(%94, %19)  : (i64, i64) -> i64
    %99 = llvm.intr.smin(%97, %98)  : (i64, i64) -> i64
    %100 = builtin.unrealized_conversion_cast %99 : i64 to index
    %c-1 = arith.constant -1 : index
    %101 = arith.muli %95, %c-1 : index
    %102 = arith.addi %100, %101 : index
    %103 = builtin.unrealized_conversion_cast %102 : index to i64
    %104 = llvm.icmp "slt" %103, %19 : i64
    %105 = llvm.intr.smin(%61, %103)  : (i64, i64) -> i64
    %106 = builtin.unrealized_conversion_cast %105 : i64 to index
    %c-1_1 = arith.constant -1 : index
    %107 = arith.muli %106, %c-1_1 : index
    %108 = arith.addi %107, %100 : index
    %c-1_2 = arith.constant -1 : index
    %109 = arith.muli %95, %c-1_2 : index
    %110 = arith.addi %108, %109 : index
    %111 = builtin.unrealized_conversion_cast %110 : index to i64
    %112 = llvm.intr.smin(%111, %16)  : (i64, i64) -> i64
    %113 = builtin.unrealized_conversion_cast %112 : i64 to index
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %c64_7 = arith.constant 64 : index
    %114 = arith.muli %59, %c64_7 : index
    %115 = arith.addi %95, %114 : index
    %reinterpret_cast_8 = memref.reinterpret_cast %base_buffer_3 to offset: [%115], sizes: [%113], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %72 : memref<64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [%113], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    cf.cond_br %104, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%72) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %76]
    %cast = memref.cast %reinterpret_cast_8 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_14 = memref.cast %reinterpret_cast_13 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast, %cast_14, %1, %20, %25, %17) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %116 = hivm.hir.pointer_cast(%15) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_merged_vf_0(%72, %116, %87) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xbf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %76]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_15 = memref.cast %116 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_16 = memref.cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    call @copy_ubuf_to_cbuf_1d_float(%cast_15, %cast_16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb2, ^bb5
    %117 = llvm.icmp "sge" %77, %57 : i32
    %118 = llvm.icmp "slt" %77, %55 : i32
    %119 = llvm.and %117, %118  : i1
    cf.cond_br %119, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %120 = llvm.srem %77, %arg5  : i32
    %121 = llvm.sub %21, %120 : i32
    %122 = llvm.intr.smin(%121, %21)  : (i32, i32) -> i32
    %123 = llvm.srem %120, %122  : i32
    %124 = llvm.add %120, %123 : i32
    %125 = llvm.sdiv %17, %122  : i32
    %126 = llvm.icmp "sge" %124, %17 : i32
    "llvm.intr.assume"(%126) : (i1) -> ()
    %127 = llvm.icmp "sge" %125, %17 : i32
    "llvm.intr.assume"(%127) : (i1) -> ()
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %128 = llvm.icmp "sge" %78, %57 : i32
    %129 = llvm.icmp "slt" %78, %55 : i32
    %130 = llvm.and %128, %129  : i1
    cf.cond_br %130, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %131 = llvm.select %70, %29, %27 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %132 = builtin.unrealized_conversion_cast %131 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xbf16, #hivm.address_space<ub>>
    %133 = llvm.icmp "eq" %69, %11 : i64
    %134 = llvm.select %133, %39, %37 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %135 = llvm.icmp "eq" %69, %12 : i64
    %136 = llvm.select %135, %41, %134 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %137 = builtin.unrealized_conversion_cast %136 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %138 = llvm.select %70, %53, %51 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %139 = builtin.unrealized_conversion_cast %138 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %140 = llvm.srem %78, %arg5  : i32
    %141 = llvm.sub %21, %140 : i32
    %142 = llvm.intr.smin(%141, %21)  : (i32, i32) -> i32
    %143 = llvm.srem %140, %142  : i32
    %144 = llvm.add %140, %143 : i32
    %145 = llvm.sdiv %17, %142  : i32
    %146 = llvm.icmp "sge" %144, %17 : i32
    "llvm.intr.assume"(%146) : (i1) -> ()
    %147 = llvm.icmp "sge" %145, %17 : i32
    "llvm.intr.assume"(%147) : (i1) -> ()
    %148 = llvm.mul %144, %18 : i32
    %149 = llvm.mul %145, %18 : i32
    %150 = llvm.sext %148 : i32 to i64
    %151 = builtin.unrealized_conversion_cast %150 : i64 to index
    %c128_17 = arith.constant 128 : index
    %152 = arith.addi %151, %c128_17 : index
    %153 = builtin.unrealized_conversion_cast %152 : index to i64
    %154 = llvm.sext %149 : i32 to i64
    %155 = builtin.unrealized_conversion_cast %154 : i64 to index
    %c128_18 = arith.constant 128 : index
    %156 = arith.addi %155, %c128_18 : index
    %157 = builtin.unrealized_conversion_cast %156 : index to i64
    %158 = llvm.intr.smax(%154, %19)  : (i64, i64) -> i64
    %c128_19 = arith.constant 128 : index
    %159 = arith.muli %151, %c128_19 : index
    %160 = arith.addi %155, %159 : index
    %161 = llvm.intr.smin(%157, %158)  : (i64, i64) -> i64
    %162 = builtin.unrealized_conversion_cast %161 : i64 to index
    %c-1_20 = arith.constant -1 : index
    %163 = arith.muli %155, %c-1_20 : index
    %164 = arith.addi %162, %163 : index
    %165 = builtin.unrealized_conversion_cast %164 : index to i64
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %76]
    call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_3(%137, %139, %132) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    %166 = llvm.intr.smax(%150, %19)  : (i64, i64) -> i64
    %167 = llvm.intr.smin(%153, %166)  : (i64, i64) -> i64
    %168 = builtin.unrealized_conversion_cast %167 : i64 to index
    %c-1_21 = arith.constant -1 : index
    %169 = arith.muli %151, %c-1_21 : index
    %170 = arith.addi %168, %169 : index
    %171 = builtin.unrealized_conversion_cast %170 : index to i64
    %172 = llvm.intr.smin(%171, %19)  : (i64, i64) -> i64
    %173 = llvm.intr.smax(%172, %24)  : (i64, i64) -> i64
    %174 = builtin.unrealized_conversion_cast %173 : i64 to index
    %175 = llvm.intr.smin(%165, %19)  : (i64, i64) -> i64
    %176 = llvm.intr.smax(%175, %24)  : (i64, i64) -> i64
    %177 = builtin.unrealized_conversion_cast %176 : i64 to index
    %178 = llvm.intr.smin(%61, %173)  : (i64, i64) -> i64
    %179 = builtin.unrealized_conversion_cast %178 : i64 to index
    %c-1_22 = arith.constant -1 : index
    %180 = arith.muli %179, %c-1_22 : index
    %181 = arith.addi %174, %180 : index
    %182 = builtin.unrealized_conversion_cast %181 : index to i64
    %183 = llvm.intr.smin(%182, %16)  : (i64, i64) -> i64
    %184 = builtin.unrealized_conversion_cast %183 : i64 to index
    %base_buffer_23, %offset_24, %sizes_25:2, %strides_26:2 = memref.extract_strided_metadata %132 : memref<64x128xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [%184, %177], strides: [128, 1] : memref<bf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %base_buffer_28, %offset_29, %sizes_30, %strides_31 = memref.extract_strided_metadata %arg4 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %c8192 = arith.constant 8192 : index
    %185 = arith.muli %59, %c8192 : index
    %186 = arith.addi %160, %185 : index
    %reinterpret_cast_32 = memref.reinterpret_cast %base_buffer_28 to offset: [%186], sizes: [%184, %177], strides: [128, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_33 = memref.cast %reinterpret_cast_27 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_34 = memref.cast %reinterpret_cast_32 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_bfloat16_t(%cast_33, %cast_34, %17) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_35 = memref.cast %reinterpret_cast_27 : memref<?x?xbf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_36 = memref.cast %reinterpret_cast_32 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_bfloat16_t(%cast_35, %cast_36, %17) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %76]
    cf.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %187 = llvm.add %67, %11 : i64
    memref.store %187, %alloca[%25] : memref<1xi64>
    %188 = arith.addi %65, %22 overflow<nsw> : i32
    cf.br ^bb1(%188 : i32)
  ^bb11:  // pred: ^bb1
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
}
