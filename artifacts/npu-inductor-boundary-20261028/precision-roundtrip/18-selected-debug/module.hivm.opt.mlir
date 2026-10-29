#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @triton_tem_fused__to_copy_bmm_2_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(262144 : i32) : i32
    %5 = llvm.mlir.constant(762880 : i32) : i32
    %6 = llvm.mlir.constant(512 : i32) : i32
    %7 = llvm.mlir.constant(4 : i32) : i32
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(1234 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(128 : index) : i64
    %12 = llvm.mlir.constant(1 : i8) : i8
    %13 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(false) : i1
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(131072 : i64) : i64
    %17 = llvm.mlir.constant(98304 : i64) : i64
    %18 = llvm.mlir.constant(65536 : i64) : i64
    %19 = llvm.mlir.constant(262144 : i64) : i64
    %20 = llvm.mlir.constant(196608 : i64) : i64
    %21 = llvm.mlir.constant(true) : i1
    %22 = llvm.mlir.constant(-1 : i64) : i64
    %23 = llvm.mlir.constant(3 : i64) : i64
    %24 = llvm.mlir.constant(2 : i64) : i64
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.mlir.constant(56 : i32) : i32
    %27 = llvm.mlir.constant(28 : i32) : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(256 : index) : i64
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    %31 = builtin.unrealized_conversion_cast %11 : i64 to index
    %32 = builtin.unrealized_conversion_cast %1 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%32] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca_0[%32] : memref<1xi64>
    %33 = hivm.hir.pointer_cast(%0) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %35 = hivm.hir.pointer_cast(%16) : memref<16x8x16x16xf32, #hivm.address_space<cc>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %37 = hivm.hir.pointer_cast(%0) : memref<64x256xf32, #hivm.address_space<ub>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<64x256xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = hivm.hir.pointer_cast(%17) : memref<64x256xf32, #hivm.address_space<ub>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<64x256xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = hivm.hir.pointer_cast(%18) : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %43 = hivm.hir.pointer_cast(%19) : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %45 = hivm.hir.pointer_cast(%0) : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
    %46 = builtin.unrealized_conversion_cast %45 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %47 = hivm.hir.pointer_cast(%20) : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
    %48 = builtin.unrealized_conversion_cast %47 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %49 = llvm.mul %arg6, %arg7 : i32
    %50 = llvm.mul %49, %arg8 : i32
    %51 = hivm.hir.get_block_idx -> i64
    %52 = llvm.trunc %51 : i64 to i32
    %53 = llvm.add %50, %26 : i32
    %54 = llvm.icmp "slt" %52, %50 : i32
    %55 = llvm.select %54, %53, %52 : i1, i32
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    cf.br ^bb1(%52 : i32)
  ^bb1(%56: i32):  // 2 preds: ^bb0, ^bb7
    %57 = arith.cmpi slt, %56, %55 : i32
    cf.cond_br %57, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    %58 = memref.load %alloca_0[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %59 = llvm.urem %58, %24  : i64
    %60 = llvm.icmp "eq" %59, %25 : i64
    %61 = llvm.select %60, %36, %34 : i1, !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %62 = builtin.unrealized_conversion_cast %61 : !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)> to memref<16x8x16x16xf32, #hivm.address_space<cc>>
    %63 = llvm.trunc %59 : i64 to i1
    %64 = llvm.xor %63, %21  : i1
    %65 = llvm.zext %64 : i1 to i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %66 = llvm.icmp "sge" %56, %52 : i32
    %67 = llvm.icmp "slt" %56, %50 : i32
    %68 = llvm.and %66, %67  : i1
    cf.cond_br %68, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %69 = llvm.select %60, %40, %38 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x256xf32, #hivm.address_space<ub>>
    %71 = llvm.srem %56, %arg6  : i32
    %72 = llvm.sdiv %56, %arg6  : i32
    %73 = llvm.srem %72, %arg7  : i32
    %74 = llvm.sdiv %71, %8  : i32
    %75 = llvm.sub %7, %74 : i32
    %76 = llvm.intr.smin(%75, %28)  : (i32, i32) -> i32
    %77 = llvm.srem %71, %76  : i32
    %78 = llvm.add %74, %77 : i32
    %79 = llvm.srem %71, %8  : i32
    %80 = llvm.sdiv %79, %76  : i32
    %81 = llvm.mul %78, %3 : i32
    %82 = llvm.mul %80, %2 : i32
    %83 = llvm.intr.smin(%82, %9)  : (i32, i32) -> i32
    %84 = llvm.mul %73, %4 : i32
    %85 = llvm.mul %73, %5 : i32
    %alloca_1 = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    memref.store %10, %alloca_1[] : memref<i32>
    %86 = llvm.sext %81 : i32 to i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    %88 = llvm.sext %84 : i32 to i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    %90 = llvm.sext %83 : i32 to i64
    %91 = builtin.unrealized_conversion_cast %90 : i64 to index
    %92 = llvm.sext %85 : i32 to i64
    %93 = builtin.unrealized_conversion_cast %92 : i64 to index
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %65]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    cf.br ^bb4(%10 : i32)
  ^bb4(%94: i32):  // 2 preds: ^bb3, ^bb5
    %95 = arith.cmpi slt, %94, %6 : i32
    cf.cond_br %95, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %96 = memref.load %alloca[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %97 = llvm.urem %96, %24  : i64
    %98 = llvm.icmp "eq" %97, %25 : i64
    %99 = llvm.select %98, %44, %42 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %100 = builtin.unrealized_conversion_cast %99 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<16x16x16x16xbf16, #hivm.address_space<cbuf>>
    %101 = llvm.select %98, %48, %46 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %102 = builtin.unrealized_conversion_cast %101 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<16x8x16x16xbf16, #hivm.address_space<cbuf>>
    %103 = llvm.trunc %97 : i64 to i1
    %104 = llvm.select %103, %24, %23 : i1, i64
    %105 = llvm.xor %103, %21  : i1
    %106 = llvm.zext %105 : i1 to i64
    %107 = llvm.sext %94 : i32 to i64
    %108 = builtin.unrealized_conversion_cast %107 : i64 to index
    %109 = arith.addi %108, %89 : index
    %c512 = arith.constant 512 : index
    %110 = arith.muli %87, %c512 : index
    %111 = arith.addi %109, %110 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%111], sizes: [128, 256], strides: [512, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %112 = arith.addi %91, %93 : index
    %c1490 = arith.constant 1490 : index
    %113 = arith.muli %108, %c1490 : index
    %114 = arith.addi %112, %113 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%114], sizes: [256, 256], strides: [1490, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<256x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %106]
    %cast = memref.cast %reinterpret_cast : memref<128x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %102 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast, %cast_3, %10) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %104]
    %cast_4 = memref.cast %reinterpret_cast_2 : memref<256x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %100 : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_4, %cast_5, %10) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %115 = memref.load %alloca_1[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<i32>
    %116 = llvm.icmp "eq" %115, %10 : i32
    %cast_6 = memref.cast %102 : memref<16x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %100 : memref<16x16x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_8 = memref.cast %62 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_bfloat16_t_to_float(%cast_6, %cast_7, %116, %31, %30, %30, %cast_8, %0, %25, %106, %104, %22, %22, %22, %15, %0) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    %117 = llvm.add %115, %28 : i32
    memref.store %117, %alloca_1[] : memref<i32>
    %118 = llvm.add %96, %25 : i64
    memref.store %118, %alloca[%32] : memref<1xi64>
    %119 = arith.addi %94, %2 overflow<nsw> : i32
    cf.br ^bb4(%119 : i32)
  ^bb6:  // pred: ^bb4
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %cast_9 = memref.cast %62 : memref<16x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_10 = memref.cast %70 : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(%cast_9, %cast_10, %0, %13, %0, %14, %21, %15, %0, %12) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %65]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb2, ^bb6
    %120 = llvm.add %58, %25 : i64
    memref.store %120, %alloca_0[%32] : memref<1xi64>
    %121 = arith.addi %56, %27 overflow<nsw> : i32
    cf.br ^bb1(%121 : i32)
  ^bb8:  // pred: ^bb1
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
  func.func @triton_tem_fused__to_copy_bmm_2_mix_aiv_outlined_vf_0(%arg0: memref<64x256xf32, #hivm.address_space<ub>>, %arg1: memref<64x256xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(256 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(64 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    cf.br ^bb1(%2 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb5
    %6 = arith.cmpi slt, %5, %3 : i32
    cf.cond_br %6, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %7 = llvm.sext %5 : i32 to i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    cf.br ^bb3(%2 : i32)
  ^bb3(%9: i32):  // 2 preds: ^bb2, ^bb4
    %10 = arith.cmpi slt, %9, %1 : i32
    cf.cond_br %10, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %11 = llvm.sext %9 : i32 to i64
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<64x256xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %c256 = arith.constant 256 : index
    %13 = arith.muli %8, %c256 : index
    %14 = arith.addi %13, %12 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%14], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %15 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.getelementptr %16[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%18, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %20 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %21 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%19, %20, %2, %2, %2) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<64x256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
    %c256_4 = arith.constant 256 : index
    %22 = arith.muli %8, %c256_4 : index
    %23 = arith.addi %22, %12 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_0 to offset: [%23], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, #map, #hivm.address_space<ub>>
    %24 = builtin.unrealized_conversion_cast %reinterpret_cast_5 : memref<64xbf16, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %26 = llvm.bitcast %21 : vector<128xbf16> to vector<64xf32>
    %27 = llvm.extractvalue %24[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.extractvalue %24[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%26, %29, %2, %0, %2, %25) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %30 = arith.addi %9, %3 overflow<nsw> : i32
    cf.br ^bb3(%30 : i32)
  ^bb5:  // pred: ^bb3
    %31 = arith.addi %5, %4 overflow<nsw> : i32
    cf.br ^bb1(%31 : i32)
  ^bb6:  // pred: ^bb1
    return
  }
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i64, f32, i64, i1, i1, i8, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @triton_tem_fused__to_copy_bmm_2_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(163840 : i64) : i64
    %2 = llvm.mlir.constant(65536 : i64) : i64
    %3 = llvm.mlir.constant(98304 : i64) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(56 : i32) : i32
    %7 = llvm.mlir.constant(256 : i32) : i32
    %8 = llvm.mlir.constant(128 : i32) : i32
    %9 = llvm.mlir.constant(762880 : i32) : i32
    %10 = llvm.mlir.constant(512 : i32) : i32
    %11 = llvm.mlir.constant(4 : i32) : i32
    %12 = llvm.mlir.constant(6 : i32) : i32
    %13 = llvm.mlir.constant(1234 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(28 : i32) : i32
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%19] : memref<1xi64>
    %20 = hivm.hir.pointer_cast(%2) : memref<64x256xbf16, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<64x256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = hivm.hir.pointer_cast(%1) : memref<64x256xbf16, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<64x256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = hivm.hir.pointer_cast(%3) : memref<64x256xf32, #hivm.address_space<ub>>
    %25 = builtin.unrealized_conversion_cast %24 : memref<64x256xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = hivm.hir.pointer_cast(%0) : memref<64x256xf32, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<64x256xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.mul %arg6, %arg7 : i32
    %29 = llvm.mul %28, %arg8 : i32
    %30 = hivm.hir.get_block_idx -> i64
    %31 = llvm.trunc %30 : i64 to i32
    %32 = hivm.hir.get_sub_block_idx -> i64
    %33 = builtin.unrealized_conversion_cast %32 : i64 to index
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    %34 = llvm.add %29, %6 : i32
    %35 = llvm.icmp "slt" %31, %29 : i32
    %36 = llvm.select %35, %34, %31 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    cf.br ^bb1(%31 : i32)
  ^bb1(%37: i32):  // 2 preds: ^bb0, ^bb9
    %38 = arith.cmpi slt, %37, %36 : i32
    cf.cond_br %38, ^bb2, ^bb10
  ^bb2:  // pred: ^bb1
    %39 = memref.load %alloca[%19] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %40 = llvm.urem %39, %5  : i64
    %41 = llvm.trunc %40 : i64 to i1
    %42 = llvm.xor %41, %4  : i1
    %43 = llvm.zext %42 : i1 to i64
    %44 = llvm.sub %37, %16 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %45 = llvm.icmp "sge" %37, %31 : i32
    %46 = llvm.icmp "slt" %37, %29 : i32
    %47 = llvm.and %45, %46  : i1
    cf.cond_br %47, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %alloca_0 = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    memref.store %14, %alloca_0[] : memref<i32>
    cf.br ^bb4(%14 : i32)
  ^bb4(%48: i32):  // 2 preds: ^bb3, ^bb5
    %49 = arith.cmpi slt, %48, %10 : i32
    cf.cond_br %49, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %50 = memref.load %alloca_0[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<i32>
    %51 = llvm.add %50, %15 : i32
    memref.store %51, %alloca_0[] : memref<i32>
    %52 = arith.addi %48, %7 overflow<nsw> : i32
    cf.br ^bb4(%52 : i32)
  ^bb6:  // pred: ^bb4
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb2, ^bb6
    %53 = llvm.icmp "sge" %44, %31 : i32
    %54 = llvm.icmp "slt" %44, %29 : i32
    %55 = llvm.and %53, %54  : i1
    cf.cond_br %55, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %56 = llvm.icmp "eq" %40, %17 : i64
    %57 = llvm.select %56, %23, %21 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %58 = builtin.unrealized_conversion_cast %57 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x256xbf16, #hivm.address_space<ub>>
    %59 = llvm.select %56, %27, %25 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<64x256xf32, #hivm.address_space<ub>>
    %61 = llvm.srem %44, %arg6  : i32
    %62 = llvm.sdiv %44, %arg6  : i32
    %63 = llvm.srem %62, %arg7  : i32
    %64 = llvm.sdiv %61, %12  : i32
    %65 = llvm.sub %11, %64 : i32
    %66 = llvm.intr.smin(%65, %15)  : (i32, i32) -> i32
    %67 = llvm.srem %61, %66  : i32
    %68 = llvm.add %64, %67 : i32
    %69 = llvm.srem %61, %12  : i32
    %70 = llvm.sdiv %69, %66  : i32
    %71 = llvm.mul %68, %8 : i32
    %72 = llvm.mul %70, %7 : i32
    %73 = llvm.intr.smin(%72, %13)  : (i32, i32) -> i32
    %74 = llvm.mul %63, %9 : i32
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %43]
    call @triton_tem_fused__to_copy_bmm_2_mix_aiv_outlined_vf_0(%60, %58) {hivm.vector_function, no_inline} : (memref<64x256xf32, #hivm.address_space<ub>>, memref<64x256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    %75 = llvm.sext %73 : i32 to i64
    %76 = builtin.unrealized_conversion_cast %75 : i64 to index
    %77 = llvm.sext %71 : i32 to i64
    %78 = builtin.unrealized_conversion_cast %77 : i64 to index
    %79 = llvm.sext %74 : i32 to i64
    %80 = builtin.unrealized_conversion_cast %79 : i64 to index
    %81 = arith.addi %76, %80 : index
    %c1490 = arith.constant 1490 : index
    %82 = arith.muli %78, %c1490 : index
    %83 = arith.addi %81, %82 : index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg4 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %c95360 = arith.constant 95360 : index
    %84 = arith.muli %33, %c95360 : index
    %85 = arith.addi %83, %84 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%85], sizes: [64, 256], strides: [1490, 1] : memref<bf16, #hivm.address_space<gm>> to memref<64x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast = memref.cast %58 : memref<64x256xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_1 = memref.cast %reinterpret_cast : memref<64x256xbf16, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_bfloat16_t(%cast, %cast_1, %14) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %43]
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %arg5 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %c95360_6 = arith.constant 95360 : index
    %86 = arith.muli %33, %c95360_6 : index
    %87 = arith.addi %83, %86 : index
    %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_2 to offset: [%87], sizes: [64, 256], strides: [1490, 1] : memref<f32, #hivm.address_space<gm>> to memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_8 = memref.cast %60 : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %reinterpret_cast_7 : memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_float(%cast_8, %cast_9, %14) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_10 = memref.cast %60 : memref<64x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_11 = memref.cast %reinterpret_cast_7 : memref<64x256xf32, strided<[1490, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_2d_float(%cast_10, %cast_11, %14) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 2
    cf.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %88 = llvm.add %39, %17 : i64
    memref.store %88, %alloca[%19] : memref<1xi64>
    %89 = arith.addi %37, %16 overflow<nsw> : i32
    cf.br ^bb1(%89 : i32)
  ^bb10:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
