module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @triton_tem_fused_addmm_0_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(false) : i1
    %6 = llvm.mlir.constant(16 : i64) : i64
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.mlir.constant(65536 : i64) : i64
    %9 = llvm.mlir.constant(33280 : i64) : i64
    %10 = llvm.mlir.constant(99328 : i64) : i64
    %11 = llvm.mlir.constant(512 : i64) : i64
    %12 = llvm.mlir.constant(66560 : i64) : i64
    %13 = llvm.mlir.constant(66048 : i64) : i64
    %14 = llvm.mlir.constant(-1 : i64) : i64
    %15 = llvm.mlir.constant(3 : i64) : i64
    %16 = llvm.mlir.constant(2 : i64) : i64
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.mlir.constant(56 : i32) : i32
    %19 = llvm.mlir.constant(28 : i32) : i32
    %20 = llvm.mlir.constant(1 : i32) : i32
    %21 = llvm.mlir.constant(true) : i1
    %22 = llvm.mlir.constant(128 : index) : i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = builtin.unrealized_conversion_cast %1 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%24] : memref<1xi64>
    %25 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %26 = builtin.unrealized_conversion_cast %25 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %27 = hivm.hir.pointer_cast(%8) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %28 = builtin.unrealized_conversion_cast %27 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %29 = hivm.hir.pointer_cast(%9) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %30 = builtin.unrealized_conversion_cast %29 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %31 = hivm.hir.pointer_cast(%10) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %32 = builtin.unrealized_conversion_cast %31 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %33 = hivm.hir.pointer_cast(%11) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %35 = hivm.hir.pointer_cast(%12) : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %37 = hivm.hir.pointer_cast(%13) : memref<128xf32, #hivm.address_space<cbuf>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = hivm.hir.pointer_cast(%0) : memref<128xf32, #hivm.address_space<cbuf>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.mul %arg5, %arg6 : i32
    %42 = llvm.mul %41, %arg7 : i32
    %43 = hivm.hir.get_block_idx -> i64
    %44 = llvm.trunc %43 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    %45 = llvm.add %42, %18 : i32
    %46 = llvm.icmp "slt" %44, %42 : i32
    %47 = llvm.select %46, %45, %44 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    cf.br ^bb1(%44 : i32)
  ^bb1(%48: i32):  // 2 preds: ^bb0, ^bb6
    %49 = arith.cmpi slt, %48, %47 : i32
    cf.cond_br %49, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %50 = memref.load %alloca[%24] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %51 = llvm.urem %50, %16  : i64
    %52 = llvm.trunc %51 : i64 to i1
    %53 = llvm.select %52, %16, %15 : i1, i64
    %54 = llvm.xor %52, %21  : i1
    %55 = llvm.zext %54 : i1 to i64
    %56 = llvm.sub %48, %19 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %57 = llvm.icmp "sge" %48, %44 : i32
    %58 = llvm.icmp "slt" %48, %42 : i32
    %59 = llvm.and %57, %58  : i1
    cf.cond_br %59, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %60 = llvm.srem %48, %arg5  : i32
    %61 = llvm.sub %20, %60 : i32
    %62 = llvm.intr.smin(%61, %20)  : (i32, i32) -> i32
    %63 = llvm.sdiv %2, %62  : i32
    %64 = llvm.icmp "sge" %63, %2 : i32
    "llvm.intr.assume"(%64) : (i1) -> ()
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %65 = llvm.icmp "sge" %56, %44 : i32
    %66 = llvm.icmp "slt" %56, %42 : i32
    %67 = llvm.and %65, %66  : i1
    cf.cond_br %67, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %68 = llvm.icmp "eq" %51, %17 : i64
    %69 = llvm.select %68, %28, %26 : i1, !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %71 = llvm.select %68, %32, %30 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %73 = llvm.select %68, %36, %34 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %74 = builtin.unrealized_conversion_cast %73 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %75 = llvm.select %68, %40, %38 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %76 = builtin.unrealized_conversion_cast %75 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf32, #hivm.address_space<cbuf>>
    %77 = llvm.srem %56, %arg5  : i32
    %78 = llvm.sub %20, %77 : i32
    %79 = llvm.intr.smin(%78, %20)  : (i32, i32) -> i32
    %80 = llvm.srem %77, %79  : i32
    %81 = llvm.add %77, %80 : i32
    %82 = llvm.sdiv %2, %79  : i32
    %83 = llvm.icmp "sge" %81, %2 : i32
    "llvm.intr.assume"(%83) : (i1) -> ()
    %84 = llvm.icmp "sge" %82, %2 : i32
    "llvm.intr.assume"(%84) : (i1) -> ()
    %85 = llvm.mul %81, %3 : i32
    %86 = llvm.mul %82, %3 : i32
    %87 = llvm.sext %85 : i32 to i64
    %88 = builtin.unrealized_conversion_cast %87 : i64 to index
    %c128 = arith.constant 128 : index
    %89 = arith.muli %88, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%89], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %55]
    %cast = memref.cast %reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %74 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast, %cast_0, %2) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %90 = llvm.sext %86 : i32 to i64
    %91 = builtin.unrealized_conversion_cast %90 : i64 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%91], sizes: [128, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %53]
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<128x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %72 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_bfloat16_t(%cast_2, %cast_3, %2) : (memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %c128_4 = arith.constant 128 : index
    %92 = arith.addi %91, %c128_4 : index
    %93 = builtin.unrealized_conversion_cast %92 : index to i64
    %94 = llvm.intr.smax(%90, %22)  : (i64, i64) -> i64
    %95 = llvm.intr.smin(%93, %94)  : (i64, i64) -> i64
    %96 = builtin.unrealized_conversion_cast %95 : i64 to index
    %c-1 = arith.constant -1 : index
    %97 = arith.muli %91, %c-1 : index
    %98 = arith.addi %96, %97 : index
    %99 = builtin.unrealized_conversion_cast %98 : index to i64
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %55]
    %cast_5 = memref.cast %74 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_6 = memref.cast %72 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_7 = memref.cast %76 : memref<128xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_8 = memref.cast %70 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_with_float_bias_bfloat16_t_to_float(%cast_5, %cast_6, %21, %23, %23, %23, %cast_7, %cast_8, %0, %17, %55, %53, %14, %14, %14, %7, %0) : (memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    %c128_9 = arith.constant 128 : index
    %100 = arith.muli %88, %c128_9 : index
    %101 = arith.addi %91, %100 : index
    %c128_10 = arith.constant 128 : index
    %102 = arith.addi %88, %c128_10 : index
    %103 = builtin.unrealized_conversion_cast %102 : index to i64
    %104 = llvm.intr.smax(%87, %22)  : (i64, i64) -> i64
    %105 = llvm.intr.smin(%103, %104)  : (i64, i64) -> i64
    %106 = builtin.unrealized_conversion_cast %105 : i64 to index
    %c-1_11 = arith.constant -1 : index
    %107 = arith.muli %88, %c-1_11 : index
    %108 = arith.addi %106, %107 : index
    %109 = builtin.unrealized_conversion_cast %108 : index to i64
    %110 = llvm.intr.smin(%109, %22)  : (i64, i64) -> i64
    %111 = llvm.intr.smax(%110, %1)  : (i64, i64) -> i64
    %112 = builtin.unrealized_conversion_cast %111 : i64 to index
    %113 = llvm.intr.smin(%99, %22)  : (i64, i64) -> i64
    %114 = llvm.intr.smax(%113, %1)  : (i64, i64) -> i64
    %115 = builtin.unrealized_conversion_cast %114 : i64 to index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %116 = arith.cmpi sle, %112, %c0 : index
    %117 = arith.subi %c0, %112 : index
    %118 = arith.subi %112, %c1 : index
    %119 = arith.select %116, %117, %118 : index
    %120 = arith.divsi %119, %c16 : index
    %121 = arith.subi %c0, %120 : index
    %122 = arith.addi %120, %c1 : index
    %123 = arith.select %116, %121, %122 : index
    %c16_12 = arith.constant 16 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %124 = arith.cmpi sle, %115, %c0_13 : index
    %125 = arith.subi %c0_13, %115 : index
    %126 = arith.subi %115, %c1_14 : index
    %127 = arith.select %124, %125, %126 : index
    %128 = arith.divsi %127, %c16_12 : index
    %129 = arith.subi %c0_13, %128 : index
    %130 = arith.addi %128, %c1_14 : index
    %131 = arith.select %124, %129, %130 : index
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %70 : memref<8x8x16x16xf32, #hivm.address_space<cc>> -> memref<f32, #hivm.address_space<cc>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [%131, %123, 16, 16], strides: [2048, 256, 16, 1] : memref<f32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>
    %base_buffer_16, %offset_17, %sizes_18, %strides_19 = memref.extract_strided_metadata %arg4 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_20 = memref.reinterpret_cast %base_buffer_16 to offset: [%101], sizes: [%112, %115], strides: [128, 1] : memref<bf16, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_FIX>]
    %cast_21 = memref.cast %reinterpret_cast_15 : memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_22 = memref.cast %reinterpret_cast_20 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm(%cast_21, %cast_22, %6, %4, %0, %5, %21, %7, %0) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, f32, i64, i1, i1, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %55]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %132 = llvm.add %50, %17 : i64
    memref.store %132, %alloca[%24] : memref<1xi64>
    %133 = arith.addi %48, %19 overflow<nsw> : i32
    cf.br ^bb1(%133 : i32)
  ^bb7:  // pred: ^bb1
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
  func.func @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_0(%arg0: memref<64xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_1(%arg0: memref<64xbf16, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(14 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = builtin.unrealized_conversion_cast %arg1 : memref<64xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%5, %2, %1, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%6, %7, %2) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%8, %10, %2, %0, %2, %9) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func private @nd2nz_bfloat16_t(memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_with_float_bias_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, f32, i64, i1, i1, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_cbuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @triton_tem_fused_addmm_0_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(384 : i64) : i64
    %3 = llvm.mlir.constant(66048 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(56 : i32) : i32
    %7 = llvm.mlir.constant(128 : i64) : i64
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.constant(128 : i32) : i32
    %11 = llvm.mlir.constant(128 : index) : i64
    %12 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(28 : i32) : i32
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%16] : memref<1xi64>
    %17 = hivm.hir.pointer_cast(%0) : memref<64xbf16, #hivm.address_space<ub>>
    %18 = builtin.unrealized_conversion_cast %17 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = hivm.hir.pointer_cast(%2) : memref<64xbf16, #hivm.address_space<ub>>
    %20 = builtin.unrealized_conversion_cast %19 : memref<64xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = hivm.hir.pointer_cast(%0) : memref<128xf32, #hivm.address_space<cbuf>>
    %22 = builtin.unrealized_conversion_cast %21 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = hivm.hir.pointer_cast(%3) : memref<128xf32, #hivm.address_space<cbuf>>
    %24 = builtin.unrealized_conversion_cast %23 : memref<128xf32, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.mul %arg5, %arg6 : i32
    %26 = llvm.mul %25, %arg7 : i32
    %27 = hivm.hir.get_block_idx -> i64
    %28 = llvm.trunc %27 : i64 to i32
    %29 = hivm.hir.get_sub_block_idx -> i64
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    %c64 = arith.constant 64 : index
    %31 = arith.muli %30, %c64 : index
    %32 = builtin.unrealized_conversion_cast %31 : index to i64
    %33 = llvm.add %26, %6 : i32
    %34 = llvm.icmp "slt" %28, %26 : i32
    %35 = llvm.select %34, %33, %28 : i1, i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%28 : i32)
  ^bb1(%36: i32):  // 2 preds: ^bb0, ^bb8
    %37 = arith.cmpi slt, %36, %35 : i32
    cf.cond_br %37, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %38 = memref.load %alloca[%16] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %39 = llvm.urem %38, %5  : i64
    %40 = llvm.icmp "eq" %39, %4 : i64
    %41 = llvm.select %40, %20, %18 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = builtin.unrealized_conversion_cast %41 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<64xbf16, #hivm.address_space<ub>>
    %43 = llvm.select %40, %24, %22 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = builtin.unrealized_conversion_cast %43 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf32, #hivm.address_space<cbuf>>
    %45 = llvm.trunc %39 : i64 to i1
    %46 = llvm.select %45, %4, %5 : i1, i64
    %47 = llvm.sub %36, %14 : i32
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %44 : memref<128xf32, #hivm.address_space<cbuf>> -> memref<f32, #hivm.address_space<cbuf>>, index, index, index
    %c64_0 = arith.constant 64 : index
    %48 = arith.muli %30, %c64_0 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%48], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<cbuf>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[48]
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[60]
    %49 = llvm.icmp "sge" %36, %28 : i32
    %50 = llvm.icmp "slt" %36, %26 : i32
    %51 = llvm.and %49, %50  : i1
    cf.cond_br %51, ^bb3, ^bb6
  ^bb3:  // pred: ^bb2
    %52 = llvm.srem %36, %arg5  : i32
    %53 = llvm.sub %13, %52 : i32
    %54 = llvm.intr.smin(%53, %13)  : (i32, i32) -> i32
    %55 = llvm.sdiv %9, %54  : i32
    %56 = llvm.icmp "sge" %55, %9 : i32
    "llvm.intr.assume"(%56) : (i1) -> ()
    %57 = llvm.mul %55, %10 : i32
    %58 = llvm.sext %57 : i32 to i64
    %59 = builtin.unrealized_conversion_cast %58 : i64 to index
    %c128 = arith.constant 128 : index
    %60 = arith.addi %59, %c128 : index
    %61 = builtin.unrealized_conversion_cast %60 : index to i64
    %62 = llvm.intr.smax(%58, %11)  : (i64, i64) -> i64
    %63 = llvm.intr.smin(%61, %62)  : (i64, i64) -> i64
    %64 = builtin.unrealized_conversion_cast %63 : i64 to index
    %c-1 = arith.constant -1 : index
    %65 = arith.muli %59, %c-1 : index
    %66 = arith.addi %64, %65 : index
    %67 = builtin.unrealized_conversion_cast %66 : index to i64
    %68 = llvm.icmp "slt" %67, %11 : i64
    %69 = llvm.intr.smin(%32, %67)  : (i64, i64) -> i64
    %70 = builtin.unrealized_conversion_cast %69 : i64 to index
    %c-1_1 = arith.constant -1 : index
    %71 = arith.muli %70, %c-1_1 : index
    %72 = arith.addi %71, %64 : index
    %c-1_2 = arith.constant -1 : index
    %73 = arith.muli %59, %c-1_2 : index
    %74 = arith.addi %72, %73 : index
    %75 = builtin.unrealized_conversion_cast %74 : index to i64
    %76 = llvm.intr.smin(%75, %8)  : (i64, i64) -> i64
    %77 = builtin.unrealized_conversion_cast %76 : i64 to index
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %c64_7 = arith.constant 64 : index
    %78 = arith.muli %30, %c64_7 : index
    %79 = arith.addi %59, %78 : index
    %reinterpret_cast_8 = memref.reinterpret_cast %base_buffer_3 to offset: [%79], sizes: [%77], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %42 : memref<64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [%77], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    cf.cond_br %68, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    call @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_0(%42) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %46]
    %cast = memref.cast %reinterpret_cast_8 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_14 = memref.cast %reinterpret_cast_13 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast, %cast_14, %1, %12, %16, %9) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %80 = hivm.hir.pointer_cast(%7) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @triton_tem_fused_addmm_0_mix_aiv_outlined_vf_1(%42, %80) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %46]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_15 = memref.cast %80 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_16 = memref.cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    call @copy_ubuf_to_cbuf_1d_float(%cast_15, %cast_16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb2, ^bb5
    %81 = llvm.icmp "sge" %47, %28 : i32
    %82 = llvm.icmp "slt" %47, %26 : i32
    %83 = llvm.and %81, %82  : i1
    cf.cond_br %83, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %84 = llvm.srem %47, %arg5  : i32
    %85 = llvm.sub %13, %84 : i32
    %86 = llvm.intr.smin(%85, %13)  : (i32, i32) -> i32
    %87 = llvm.srem %84, %86  : i32
    %88 = llvm.add %84, %87 : i32
    %89 = llvm.sdiv %9, %86  : i32
    %90 = llvm.icmp "sge" %88, %9 : i32
    "llvm.intr.assume"(%90) : (i1) -> ()
    %91 = llvm.icmp "sge" %89, %9 : i32
    "llvm.intr.assume"(%91) : (i1) -> ()
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %92 = llvm.add %38, %4 : i64
    memref.store %92, %alloca[%16] : memref<1xi64>
    %93 = arith.addi %36, %14 overflow<nsw> : i32
    cf.br ^bb1(%93 : i32)
  ^bb9:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
