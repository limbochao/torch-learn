#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(128 : i32) : i32
    %5 = llvm.mlir.constant(1 : i8) : i8
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(false) : i1
    %8 = llvm.mlir.constant(0 : i8) : i8
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.constant(65536 : i64) : i64
    %11 = llvm.mlir.constant(131328 : i64) : i64
    %12 = llvm.mlir.constant(32768 : i64) : i64
    %13 = llvm.mlir.constant(98560 : i64) : i64
    %14 = llvm.mlir.constant(65792 : i64) : i64
    %15 = llvm.mlir.constant(49152 : i64) : i64
    %16 = llvm.mlir.constant(-1 : i64) : i64
    %17 = llvm.mlir.constant(3 : i64) : i64
    %18 = llvm.mlir.constant(5 : i64) : i64
    %19 = llvm.mlir.constant(4 : i64) : i64
    %20 = llvm.mlir.constant(2 : i64) : i64
    %21 = llvm.mlir.constant(56 : i32) : i32
    %22 = llvm.mlir.constant(28 : i32) : i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.mlir.constant(true) : i1
    %25 = llvm.mlir.constant(128 : index) : i64
    %26 = builtin.unrealized_conversion_cast %25 : i64 to index
    %27 = builtin.unrealized_conversion_cast %1 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%27] : memref<1xi64>
    %28 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %30 = hivm.hir.pointer_cast(%10) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %32 = hivm.hir.pointer_cast(%10) : memref<128xf16, #hivm.address_space<cbuf>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<128xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = hivm.hir.pointer_cast(%11) : memref<128xf16, #hivm.address_space<cbuf>>
    %35 = builtin.unrealized_conversion_cast %34 : memref<128xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = hivm.hir.pointer_cast(%12) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %37 = builtin.unrealized_conversion_cast %36 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %38 = hivm.hir.pointer_cast(%13) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %39 = builtin.unrealized_conversion_cast %38 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %40 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %42 = hivm.hir.pointer_cast(%14) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %44 = hivm.hir.pointer_cast(%0) : memref<64x128xf32, #hivm.address_space<ub>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = hivm.hir.pointer_cast(%15) : memref<64x128xf32, #hivm.address_space<ub>>
    %47 = builtin.unrealized_conversion_cast %46 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mul %arg5, %arg6 : i32
    %49 = llvm.mul %48, %arg7 : i32
    %50 = hivm.hir.get_block_idx -> i64
    %51 = llvm.trunc %50 : i64 to i32
    %52 = llvm.add %49, %21 : i32
    %53 = llvm.icmp "slt" %51, %49 : i32
    %54 = llvm.select %53, %52, %51 : i1, i32
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
    cf.br ^bb1(%51 : i32)
  ^bb1(%55: i32):  // 2 preds: ^bb0, ^bb6
    %56 = arith.cmpi slt, %55, %54 : i32
    cf.cond_br %56, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %57 = memref.load %alloca[%27] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %58 = llvm.urem %57, %20  : i64
    %59 = llvm.trunc %58 : i64 to i1
    %60 = llvm.select %59, %19, %18 : i1, i64
    %61 = llvm.select %59, %20, %17 : i1, i64
    %62 = llvm.xor %59, %24  : i1
    %63 = llvm.zext %62 : i1 to i64
    %64 = llvm.sub %55, %22 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %65 = llvm.icmp "sge" %55, %51 : i32
    %66 = llvm.icmp "slt" %55, %49 : i32
    %67 = llvm.and %65, %66  : i1
    cf.cond_br %67, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %68 = llvm.icmp "eq" %58, %2 : i64
    %69 = llvm.select %68, %31, %29 : i1, !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %71 = llvm.select %68, %35, %33 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf16, #hivm.address_space<cbuf>>
    %73 = llvm.select %68, %39, %37 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %74 = builtin.unrealized_conversion_cast %73 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %75 = llvm.select %68, %43, %41 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %77 = llvm.select %68, %47, %45 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %78 = builtin.unrealized_conversion_cast %77 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %79 = llvm.srem %55, %arg5  : i32
    %80 = llvm.sub %23, %79 : i32
    %81 = llvm.intr.smin(%80, %23)  : (i32, i32) -> i32
    %82 = llvm.srem %79, %81  : i32
    %83 = llvm.add %79, %82 : i32
    %84 = llvm.sdiv %3, %81  : i32
    %85 = llvm.icmp "sge" %83, %3 : i32
    "llvm.intr.assume"(%85) : (i1) -> ()
    %86 = llvm.icmp "sge" %84, %3 : i32
    "llvm.intr.assume"(%86) : (i1) -> ()
    %87 = llvm.mul %83, %4 : i32
    %88 = llvm.mul %84, %4 : i32
    %89 = llvm.sext %87 : i32 to i64
    %90 = builtin.unrealized_conversion_cast %89 : i64 to index
    %c128 = arith.constant 128 : index
    %91 = arith.muli %90, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%91], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %63]
    %cast = memref.cast %reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %76 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_half(%cast, %cast_0, %3) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %92 = llvm.sext %88 : i32 to i64
    %93 = builtin.unrealized_conversion_cast %92 : i64 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%93], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %61]
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %74 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_half(%cast_2, %cast_3, %3) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %c128_4 = arith.constant 128 : index
    %94 = arith.addi %93, %c128_4 : index
    %95 = builtin.unrealized_conversion_cast %94 : index to i64
    %96 = llvm.intr.smax(%92, %25)  : (i64, i64) -> i64
    %97 = llvm.intr.smin(%95, %96)  : (i64, i64) -> i64
    %98 = builtin.unrealized_conversion_cast %97 : i64 to index
    %c-1 = arith.constant -1 : index
    %99 = arith.muli %93, %c-1 : index
    %100 = arith.addi %98, %99 : index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf16, #hivm.address_space<gm>> -> memref<f16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [%93], sizes: [%100], strides: [1] : memref<f16, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %72 : memref<128xf16, #hivm.address_space<cbuf>> -> memref<f16, #hivm.address_space<cbuf>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [%100], strides: [1] : memref<f16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %60]
    %cast_11 = memref.cast %reinterpret_cast_5 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_12 = memref.cast %reinterpret_cast_10 : memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    call @load_gm_to_cbuf_1d_half(%cast_11, %cast_12, %9, %3) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i32, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %63]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %cast_13 = memref.cast %76 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_14 = memref.cast %74 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_15 = memref.cast %72 : memref<128xf16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_16 = memref.cast %70 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_with_half_bias_half_to_float(%cast_13, %cast_14, %24, %26, %26, %26, %cast_15, %cast_16, %16, %16, %63, %61, %16, %16, %16, %8, %0) : (memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %60]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_17 = memref.cast %70 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_18 = memref.cast %78 : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_17, %cast_18, %0, %6, %0, %7, %24, %8, %0, %5) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %63]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %101 = llvm.icmp "sge" %64, %51 : i32
    %102 = llvm.icmp "slt" %64, %49 : i32
    %103 = llvm.and %101, %102  : i1
    cf.cond_br %103, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %104 = llvm.srem %64, %arg5  : i32
    %105 = llvm.sub %23, %104 : i32
    %106 = llvm.intr.smin(%105, %23)  : (i32, i32) -> i32
    %107 = llvm.srem %104, %106  : i32
    %108 = llvm.add %104, %107 : i32
    %109 = llvm.sdiv %3, %106  : i32
    %110 = llvm.icmp "sge" %108, %3 : i32
    "llvm.intr.assume"(%110) : (i1) -> ()
    %111 = llvm.icmp "sge" %109, %3 : i32
    "llvm.intr.assume"(%111) : (i1) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %112 = llvm.add %57, %2 : i64
    memref.store %112, %alloca[%27] : memref<1xi64>
    %113 = arith.addi %55, %22 overflow<nsw> : i32
    cf.br ^bb1(%113 : i32)
  ^bb7:  // pred: ^bb1
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
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vcmp.gt.s.z"(%10, %25, %26) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<256xi1>
    %28 = "hivm_regbaseintrins.intr.hivm.vsel"(%10, %25, %27) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %29 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %30 = "hivm_regbaseintrins.intr.hivm.vsub.s.x"(%28, %8, %29) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %31 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %32 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"(%30, %31, %4, %4, %4) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<64x128xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index, index, index
    %c128_4 = arith.constant 128 : index
    %33 = arith.muli %14, %c128_4 : index
    %34 = arith.addi %33, %18 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_0 to offset: [%34], sizes: [64], strides: [1] : memref<f16, #hivm.address_space<ub>> to memref<64xf16, #map, #hivm.address_space<ub>>
    %35 = builtin.unrealized_conversion_cast %reinterpret_cast_5 : memref<64xf16, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %37 = llvm.bitcast %32 : vector<128xf16> to vector<64xf32>
    %38 = llvm.extractvalue %35[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.extractvalue %35[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %38[%39] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%37, %40, %4, %0, %4, %36) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %41 = arith.addi %15, %3 overflow<nsw> : i32
    cf.br ^bb3(%41 : i32)
  ^bb5:  // pred: ^bb3
    %42 = arith.addi %11, %2 overflow<nsw> : i32
    cf.br ^bb1(%42 : i32)
  ^bb6:  // pred: ^bb1
    return
  }
  func.func private @nd2nz_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @load_gm_to_cbuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i32, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_with_half_bias_half_to_float(memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @triton_tem_fused_addmm_relu_sub_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(81920 : i64) : i64
    %2 = llvm.mlir.constant(32768 : i64) : i64
    %3 = llvm.mlir.constant(49152 : i64) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(56 : i32) : i32
    %7 = llvm.mlir.constant(64 : index) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(128 : i32) : i32
    %10 = llvm.mlir.constant(128 : index) : i64
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.mlir.constant(28 : i32) : i32
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = builtin.unrealized_conversion_cast %14 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%15] : memref<1xi64>
    %16 = hivm.hir.pointer_cast(%2) : memref<64x128xf16, #hivm.address_space<ub>>
    %17 = builtin.unrealized_conversion_cast %16 : memref<64x128xf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = hivm.hir.pointer_cast(%1) : memref<64x128xf16, #hivm.address_space<ub>>
    %19 = builtin.unrealized_conversion_cast %18 : memref<64x128xf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = hivm.hir.pointer_cast(%3) : memref<64x128xf32, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = hivm.hir.pointer_cast(%0) : memref<64x128xf32, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<64x128xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mul %arg5, %arg6 : i32
    %25 = llvm.mul %24, %arg7 : i32
    %26 = hivm.hir.get_block_idx -> i64
    %27 = llvm.trunc %26 : i64 to i32
    %28 = hivm.hir.get_sub_block_idx -> i64
    %29 = builtin.unrealized_conversion_cast %28 : i64 to index
    %c64 = arith.constant 64 : index
    %30 = arith.muli %29, %c64 : index
    %31 = builtin.unrealized_conversion_cast %30 : index to i64
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    %32 = llvm.add %25, %6 : i32
    %33 = llvm.icmp "slt" %27, %25 : i32
    %34 = llvm.select %33, %32, %27 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    cf.br ^bb1(%27 : i32)
  ^bb1(%35: i32):  // 2 preds: ^bb0, ^bb6
    %36 = arith.cmpi slt, %35, %34 : i32
    cf.cond_br %36, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %37 = memref.load %alloca[%15] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %38 = llvm.urem %37, %5  : i64
    %39 = llvm.trunc %38 : i64 to i1
    %40 = llvm.xor %39, %4  : i1
    %41 = llvm.zext %40 : i1 to i64
    %42 = llvm.sub %35, %12 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %43 = llvm.icmp "sge" %35, %27 : i32
    %44 = llvm.icmp "slt" %35, %25 : i32
    %45 = llvm.and %43, %44  : i1
    cf.cond_br %45, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %46 = llvm.srem %35, %arg5  : i32
    %47 = llvm.sub %11, %46 : i32
    %48 = llvm.intr.smin(%47, %11)  : (i32, i32) -> i32
    %49 = llvm.srem %46, %48  : i32
    %50 = llvm.add %46, %49 : i32
    %51 = llvm.sdiv %8, %48  : i32
    %52 = llvm.icmp "sge" %50, %8 : i32
    "llvm.intr.assume"(%52) : (i1) -> ()
    %53 = llvm.icmp "sge" %51, %8 : i32
    "llvm.intr.assume"(%53) : (i1) -> ()
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %54 = llvm.icmp "sge" %42, %27 : i32
    %55 = llvm.icmp "slt" %42, %25 : i32
    %56 = llvm.and %54, %55  : i1
    cf.cond_br %56, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %57 = llvm.icmp "eq" %38, %13 : i64
    %58 = llvm.select %57, %19, %17 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %59 = builtin.unrealized_conversion_cast %58 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf16, #hivm.address_space<ub>>
    %60 = llvm.select %57, %23, %21 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %61 = builtin.unrealized_conversion_cast %60 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32, #hivm.address_space<ub>>
    %62 = llvm.srem %42, %arg5  : i32
    %63 = llvm.sub %11, %62 : i32
    %64 = llvm.intr.smin(%63, %11)  : (i32, i32) -> i32
    %65 = llvm.srem %62, %64  : i32
    %66 = llvm.add %62, %65 : i32
    %67 = llvm.sdiv %8, %64  : i32
    %68 = llvm.icmp "sge" %66, %8 : i32
    "llvm.intr.assume"(%68) : (i1) -> ()
    %69 = llvm.icmp "sge" %67, %8 : i32
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.mul %66, %9 : i32
    %71 = llvm.mul %67, %9 : i32
    %72 = llvm.sext %70 : i32 to i64
    %73 = builtin.unrealized_conversion_cast %72 : i64 to index
    %c128 = arith.constant 128 : index
    %74 = arith.addi %73, %c128 : index
    %75 = builtin.unrealized_conversion_cast %74 : index to i64
    %76 = llvm.sext %71 : i32 to i64
    %77 = builtin.unrealized_conversion_cast %76 : i64 to index
    %c128_0 = arith.constant 128 : index
    %78 = arith.addi %77, %c128_0 : index
    %79 = builtin.unrealized_conversion_cast %78 : index to i64
    %80 = llvm.intr.smax(%76, %10)  : (i64, i64) -> i64
    %c128_1 = arith.constant 128 : index
    %81 = arith.muli %73, %c128_1 : index
    %82 = arith.addi %77, %81 : index
    %83 = llvm.intr.smin(%79, %80)  : (i64, i64) -> i64
    %84 = builtin.unrealized_conversion_cast %83 : i64 to index
    %c-1 = arith.constant -1 : index
    %85 = arith.muli %77, %c-1 : index
    %86 = arith.addi %84, %85 : index
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %41]
    call @triton_tem_fused_addmm_relu_sub_0_mix_aiv_outlined_vf_0(%61, %59) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    %88 = llvm.intr.smax(%72, %10)  : (i64, i64) -> i64
    %89 = llvm.intr.smin(%75, %88)  : (i64, i64) -> i64
    %90 = builtin.unrealized_conversion_cast %89 : i64 to index
    %c-1_2 = arith.constant -1 : index
    %91 = arith.muli %73, %c-1_2 : index
    %92 = arith.addi %90, %91 : index
    %93 = builtin.unrealized_conversion_cast %92 : index to i64
    %94 = llvm.intr.smin(%93, %10)  : (i64, i64) -> i64
    %95 = llvm.intr.smax(%94, %14)  : (i64, i64) -> i64
    %96 = builtin.unrealized_conversion_cast %95 : i64 to index
    %97 = llvm.intr.smin(%87, %10)  : (i64, i64) -> i64
    %98 = llvm.intr.smax(%97, %14)  : (i64, i64) -> i64
    %99 = builtin.unrealized_conversion_cast %98 : i64 to index
    %100 = llvm.intr.smin(%31, %95)  : (i64, i64) -> i64
    %101 = builtin.unrealized_conversion_cast %100 : i64 to index
    %c-1_3 = arith.constant -1 : index
    %102 = arith.muli %101, %c-1_3 : index
    %103 = arith.addi %96, %102 : index
    %104 = builtin.unrealized_conversion_cast %103 : index to i64
    %105 = llvm.intr.smin(%104, %7)  : (i64, i64) -> i64
    %106 = builtin.unrealized_conversion_cast %105 : i64 to index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %59 : memref<64x128xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [%106, %99], strides: [128, 1] : memref<f16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %base_buffer_4, %offset_5, %sizes_6, %strides_7 = memref.extract_strided_metadata %arg4 : memref<?xf16, #hivm.address_space<gm>> -> memref<f16, #hivm.address_space<gm>>, index, index, index
    %c8192 = arith.constant 8192 : index
    %107 = arith.muli %29, %c8192 : index
    %108 = arith.addi %82, %107 : index
    %reinterpret_cast_8 = memref.reinterpret_cast %base_buffer_4 to offset: [%108], sizes: [%106, %99], strides: [128, 1] : memref<f16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast = memref.cast %reinterpret_cast : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %reinterpret_cast_8 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_half(%cast, %cast_9, %8) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_10 = memref.cast %reinterpret_cast : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_11 = memref.cast %reinterpret_cast_8 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_half(%cast_10, %cast_11, %8) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %41]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %109 = llvm.add %37, %13 : i64
    memref.store %109, %alloca[%15] : memref<1xi64>
    %110 = arith.addi %35, %12 overflow<nsw> : i32
    cf.br ^bb1(%110 : i32)
  ^bb7:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
