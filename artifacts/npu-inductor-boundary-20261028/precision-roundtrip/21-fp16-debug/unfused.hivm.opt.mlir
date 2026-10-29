module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"<site-packages>/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIC>, ssbuffer.insertionOptimization} {
  func.func private @nd2nz_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @load_gm_to_cbuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i32, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_with_half_bias_half_to_float(memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_half_4d_to_2d_gm(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, f32, i64, i1, i1, i8, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func @triton_tem_fused_addmm_0(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "mix", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(28 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.mlir.constant(2 : i32) : i32
    %9 = llvm.mlir.constant(65536 : i64) : i64
    %10 = llvm.mlir.constant(131328 : i64) : i64
    %11 = llvm.mlir.constant(32768 : i64) : i64
    %12 = llvm.mlir.constant(98560 : i64) : i64
    %13 = llvm.mlir.constant(65792 : i64) : i64
    %14 = llvm.mlir.constant(-1 : i64) : i64
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.mlir.constant(3 : i64) : i64
    %17 = llvm.mlir.constant(5 : i64) : i64
    %18 = llvm.mlir.constant(4 : i64) : i64
    %19 = llvm.mlir.constant(2 : i64) : i64
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(128 : i32) : i32
    %22 = llvm.mlir.constant(128 : index) : i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = builtin.unrealized_conversion_cast %1 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%24] : memref<1xi64>
    %25 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %26 = builtin.unrealized_conversion_cast %25 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %27 = hivm.hir.pointer_cast(%9) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %28 = builtin.unrealized_conversion_cast %27 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %29 = hivm.hir.pointer_cast(%9) : memref<128xf16, #hivm.address_space<cbuf>>
    %30 = builtin.unrealized_conversion_cast %29 : memref<128xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = hivm.hir.pointer_cast(%10) : memref<128xf16, #hivm.address_space<cbuf>>
    %32 = builtin.unrealized_conversion_cast %31 : memref<128xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = hivm.hir.pointer_cast(%11) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %35 = hivm.hir.pointer_cast(%12) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %37 = hivm.hir.pointer_cast(%0) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %39 = hivm.hir.pointer_cast(%13) : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %41 = llvm.mul %arg5, %arg6 : i32
    %42 = llvm.mul %41, %arg7 : i32
    %43 = hivm.hir.get_block_idx -> i64
    %44 = llvm.trunc %43 : i64 to i32
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
    cf.br ^bb1(%44 : i32)
  ^bb1(%45: i32):  // 2 preds: ^bb0, ^bb2
    %46 = arith.cmpi slt, %45, %42 : i32
    cf.cond_br %46, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %47 = memref.load %alloca[%24] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %48 = llvm.urem %47, %19  : i64
    %49 = llvm.icmp "eq" %48, %15 : i64
    %50 = llvm.select %49, %28, %26 : i1, !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)>
    %51 = builtin.unrealized_conversion_cast %50 : !llvm.struct<(ptr<5>, ptr<5>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %52 = llvm.select %49, %32, %30 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)>
    %53 = builtin.unrealized_conversion_cast %52 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xf16, #hivm.address_space<cbuf>>
    %54 = llvm.select %49, %36, %34 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %55 = builtin.unrealized_conversion_cast %54 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %56 = llvm.select %49, %40, %38 : i1, !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)>
    %57 = builtin.unrealized_conversion_cast %56 : !llvm.struct<(ptr<2>, ptr<2>, i64, array<4 x i64>, array<4 x i64>)> to memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %58 = llvm.trunc %48 : i64 to i1
    %59 = llvm.select %58, %18, %17 : i1, i64
    %60 = llvm.select %58, %19, %16 : i1, i64
    %61 = llvm.xor %58, %4  : i1
    %62 = llvm.zext %61 : i1 to i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %63 = llvm.srem %45, %arg5  : i32
    %64 = llvm.sub %3, %63 : i32
    %65 = llvm.intr.smin(%64, %3)  : (i32, i32) -> i32
    %66 = llvm.srem %63, %65  : i32
    %67 = llvm.add %63, %66 : i32
    %68 = llvm.sdiv %20, %65  : i32
    %69 = llvm.icmp "sge" %67, %20 : i32
    "llvm.intr.assume"(%69) : (i1) -> ()
    %70 = llvm.icmp "sge" %68, %20 : i32
    "llvm.intr.assume"(%70) : (i1) -> ()
    %71 = llvm.mul %67, %21 : i32
    %72 = llvm.mul %68, %21 : i32
    %73 = llvm.sext %71 : i32 to i64
    %74 = builtin.unrealized_conversion_cast %73 : i64 to index
    %c128 = arith.constant 128 : index
    %75 = arith.muli %74, %c128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%75], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %62]
    %cast = memref.cast %reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %57 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_half(%cast, %cast_0, %20) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %76 = llvm.sext %72 : i32 to i64
    %77 = builtin.unrealized_conversion_cast %76 : i64 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%77], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %60]
    %cast_2 = memref.cast %reinterpret_cast_1 : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %55 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    call @nd2nz_half(%cast_2, %cast_3, %20) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i32) -> ()
    %c128_4 = arith.constant 128 : index
    %78 = arith.addi %77, %c128_4 : index
    %79 = builtin.unrealized_conversion_cast %78 : index to i64
    %80 = llvm.intr.smax(%76, %22)  : (i64, i64) -> i64
    %81 = llvm.intr.smin(%79, %80)  : (i64, i64) -> i64
    %82 = builtin.unrealized_conversion_cast %81 : i64 to index
    %c-1 = arith.constant -1 : index
    %83 = arith.muli %77, %c-1 : index
    %84 = arith.addi %82, %83 : index
    %85 = builtin.unrealized_conversion_cast %84 : index to i64
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf16, #hivm.address_space<gm>> -> memref<f16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [%77], sizes: [%84], strides: [1] : memref<f16, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %53 : memref<128xf16, #hivm.address_space<cbuf>> -> memref<f16, #hivm.address_space<cbuf>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [%84], strides: [1] : memref<f16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %59]
    %cast_11 = memref.cast %reinterpret_cast_5 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_12 = memref.cast %reinterpret_cast_10 : memref<?xf16, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    call @load_gm_to_cbuf_1d_half(%cast_11, %cast_12, %8, %20) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i32, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %62]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %cast_13 = memref.cast %57 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_14 = memref.cast %55 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_15 = memref.cast %53 : memref<128xf16, #hivm.address_space<cbuf>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
    %cast_16 = memref.cast %51 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    call @mma_tile_with_half_bias_half_to_float(%cast_13, %cast_14, %4, %23, %23, %23, %cast_15, %cast_16, %14, %14, %62, %60, %14, %14, %14, %7, %0) : (memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %59]
    %c128_17 = arith.constant 128 : index
    %86 = arith.muli %74, %c128_17 : index
    %87 = arith.addi %77, %86 : index
    %c128_18 = arith.constant 128 : index
    %88 = arith.addi %74, %c128_18 : index
    %89 = builtin.unrealized_conversion_cast %88 : index to i64
    %90 = llvm.intr.smax(%73, %22)  : (i64, i64) -> i64
    %91 = llvm.intr.smin(%89, %90)  : (i64, i64) -> i64
    %92 = builtin.unrealized_conversion_cast %91 : i64 to index
    %c-1_19 = arith.constant -1 : index
    %93 = arith.muli %74, %c-1_19 : index
    %94 = arith.addi %92, %93 : index
    %95 = builtin.unrealized_conversion_cast %94 : index to i64
    %96 = llvm.intr.smin(%95, %22)  : (i64, i64) -> i64
    %97 = llvm.intr.smax(%96, %1)  : (i64, i64) -> i64
    %98 = builtin.unrealized_conversion_cast %97 : i64 to index
    %99 = llvm.intr.smin(%85, %22)  : (i64, i64) -> i64
    %100 = llvm.intr.smax(%99, %1)  : (i64, i64) -> i64
    %101 = builtin.unrealized_conversion_cast %100 : i64 to index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %102 = arith.cmpi sle, %98, %c0 : index
    %103 = arith.subi %c0, %98 : index
    %104 = arith.subi %98, %c1 : index
    %105 = arith.select %102, %103, %104 : index
    %106 = arith.divsi %105, %c16 : index
    %107 = arith.subi %c0, %106 : index
    %108 = arith.addi %106, %c1 : index
    %109 = arith.select %102, %107, %108 : index
    %c16_20 = arith.constant 16 : index
    %c0_21 = arith.constant 0 : index
    %c1_22 = arith.constant 1 : index
    %110 = arith.cmpi sle, %101, %c0_21 : index
    %111 = arith.subi %c0_21, %101 : index
    %112 = arith.subi %101, %c1_22 : index
    %113 = arith.select %110, %111, %112 : index
    %114 = arith.divsi %113, %c16_20 : index
    %115 = arith.subi %c0_21, %114 : index
    %116 = arith.addi %114, %c1_22 : index
    %117 = arith.select %110, %115, %116 : index
    %base_buffer_23, %offset_24, %sizes_25:4, %strides_26:4 = memref.extract_strided_metadata %51 : memref<8x8x16x16xf32, #hivm.address_space<cc>> -> memref<f32, #hivm.address_space<cc>>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [%117, %109, 16, 16], strides: [2048, 256, 16, 1] : memref<f32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>>
    %base_buffer_28, %offset_29, %sizes_30, %strides_31 = memref.extract_strided_metadata %arg4 : memref<?xf16, #hivm.address_space<gm>> -> memref<f16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_32 = memref.reinterpret_cast %base_buffer_28 to offset: [%87], sizes: [%98, %101], strides: [128, 1] : memref<f16, #hivm.address_space<gm>> to memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_FIX>]
    %cast_33 = memref.cast %reinterpret_cast_27 : memref<?x?x16x16xf32, strided<[2048, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
    %cast_34 = memref.cast %reinterpret_cast_32 : memref<?x?xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    call @fixpipe_nz2nd_float_to_half_4d_to_2d_gm(%cast_33, %cast_34, %15, %5, %0, %6, %4, %7, %0) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, f32, i64, i1, i1, i8, i64) -> ()
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %62]
    hivm.hir.set_ctrl true at ctrl[60]
    %118 = llvm.add %47, %15 : i64
    memref.store %118, %alloca[%24] : memref<1xi64>
    %119 = arith.addi %45, %2 overflow<nsw> : i32
    cf.br ^bb1(%119 : i32)
  ^bb3:  // pred: ^bb1
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
}
