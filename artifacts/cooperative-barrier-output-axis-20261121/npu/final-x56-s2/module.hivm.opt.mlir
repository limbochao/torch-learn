module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @exchange_fused_0_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %6 = builtin.unrealized_conversion_cast %arg0 : memref<2xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"(%8, %3, %3, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
    %10 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    %11 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"(%10, %3, %2, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
    %12 = builtin.unrealized_conversion_cast %11 : vector<64xi32> to vector<1xi32>
    %13 = builtin.unrealized_conversion_cast %12 : vector<1xi32> to vector<i32>
    %14 = builtin.unrealized_conversion_cast %13 : vector<i32> to i32
    %15 = builtin.unrealized_conversion_cast %14 : i32 to vector<1xi32>
    %16 = builtin.unrealized_conversion_cast %15 : vector<1xi32> to vector<64xi32>
    %17 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%9, %7) : (vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %18 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %19 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%16, %17, %18) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %20 = builtin.unrealized_conversion_cast %19 : vector<64xi32> to vector<1xi32>
    %21 = builtin.unrealized_conversion_cast %20 : vector<1xi32> to i32
    %22 = builtin.unrealized_conversion_cast %21 : i32 to vector<i32>
    %23 = builtin.unrealized_conversion_cast %22 : vector<i32> to vector<1xi32>
    %24 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %25 = builtin.unrealized_conversion_cast %23 : vector<1xi32> to vector<64xi32>
    %26 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64s32"(%25, %26, %3, %0, %3, %24) : (vector<64xi32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t_nocache(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @exchange(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(160 : i64) : i64
    %3 = llvm.mlir.constant(64 : i64) : i64
    %4 = llvm.mlir.constant(32 : i64) : i64
    %5 = llvm.mlir.constant(128 : i64) : i64
    %6 = llvm.mlir.constant(96 : i64) : i64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(2 : i64) : i64
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.mlir.constant(10000 : i32) : i32
    %12 = llvm.mlir.constant(1000 : i32) : i32
    %13 = llvm.mlir.constant(8 : i32) : i32
    %14 = llvm.mlir.constant(4 : i32) : i32
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(1 : i32) : i32
    %17 = llvm.mlir.constant(56 : i32) : i32
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%19] : memref<1xi64>
    %20 = hivm.hir.pointer_cast(%3) : memref<i32, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %22 = hivm.hir.pointer_cast(%2) : memref<i32, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %24 = hivm.hir.pointer_cast(%4) : memref<2xi32, #hivm.address_space<ub>>
    %25 = builtin.unrealized_conversion_cast %24 : memref<2xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = hivm.hir.pointer_cast(%5) : memref<2xi32, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<2xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = hivm.hir.pointer_cast(%0) : memref<1xi32, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = hivm.hir.pointer_cast(%6) : memref<1xi32, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.mul %arg6, %arg7 : i32
    %33 = llvm.mul %32, %arg8 : i32
    %34 = hivm.hir.get_block_idx -> i64
    %35 = llvm.trunc %34 : i64 to i32
    %36 = llvm.mul %arg5, %12 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    cf.br ^bb1(%35 : i32)
  ^bb1(%37: i32):  // 2 preds: ^bb0, ^bb8
    %38 = arith.cmpi slt, %37, %33 : i32
    cf.cond_br %38, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %39 = llvm.srem %37, %arg6  : i32
    %40 = llvm.sdiv %37, %arg6  : i32
    %41 = llvm.srem %40, %arg7  : i32
    %42 = llvm.sdiv %37, %32  : i32
    %43 = llvm.srem %42, %arg8  : i32
    %44 = llvm.mul %43, %17 : i32
    %45 = llvm.add %41, %44 : i32
    %46 = llvm.srem %39, %14  : i32
    %47 = llvm.mul %46, %13 : i32
    cf.br ^bb3(%15, %15 : i32, i32)
  ^bb3(%48: i32, %49: i32):  // 2 preds: ^bb2, ^bb4
    %50 = arith.cmpi slt, %48, %47 : i32
    cf.cond_br %50, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %51 = llvm.add %49, %48 : i32
    %52 = arith.addi %48, %16 overflow<nsw> : i32
    cf.br ^bb3(%52, %51 : i32, i32)
  ^bb5:  // pred: ^bb3
    %53 = llvm.add %39, %16 : i32
    %54 = llvm.mul %45, %12 : i32
    %55 = llvm.add %53, %54 : i32
    %56 = llvm.mul %45, %10 : i32
    %57 = llvm.sext %56 : i32 to i64
    %58 = builtin.unrealized_conversion_cast %57 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%58], sizes: [2], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<2xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %59 = llvm.sext %39 : i32 to i64
    %60 = builtin.unrealized_conversion_cast %59 : i64 to index
    %61 = arith.addi %58, %60 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%61], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    cf.br ^bb6(%15 : i32)
  ^bb6(%62: i32):  // 2 preds: ^bb5, ^bb7
    %63 = arith.cmpi slt, %62, %14 : i32
    cf.cond_br %63, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %64 = memref.load %alloca[%19] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %65 = llvm.urem %64, %9  : i64
    %66 = llvm.icmp "eq" %65, %7 : i64
    %67 = llvm.select %66, %23, %21 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %68 = builtin.unrealized_conversion_cast %67 : !llvm.struct<(ptr<6>, ptr<6>, i64)> to memref<i32, #hivm.address_space<ub>>
    %69 = llvm.select %66, %27, %25 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<2xi32, #hivm.address_space<ub>>
    %71 = llvm.select %66, %31, %29 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32, #hivm.address_space<ub>>
    %73 = llvm.trunc %65 : i64 to i1
    %74 = llvm.select %73, %9, %8 : i1, i64
    %75 = llvm.xor %73, %1  : i1
    %76 = llvm.zext %75 : i1 to i64
    %77 = llvm.mul %62, %11 : i32
    %78 = llvm.add %55, %77 : i32
    %79 = llvm.add %78, %36 : i32
    %80 = llvm.add %79, %49 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %76]
    memref.store %80, %72[%19] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast = memref.cast %72 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_1 = memref.cast %reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %76]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_2 = memref.cast %reinterpret_cast : memref<2xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %70 : memref<2xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_2, %cast_3, %15, %15, %19, %15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %81 = llvm.mul %62, %17 : i32
    %82 = llvm.add %81, %45 : i32
    %83 = llvm.mul %82, %10 : i32
    %84 = llvm.sext %83 : i32 to i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %74]
    memref.store %15, %68[] : memref<i32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    call @exchange_fused_0_outlined_vf_0(%70, %68) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<i32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %base_buffer, %offset = memref.extract_strided_metadata %68 : memref<i32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
    %86 = arith.addi %85, %60 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_6 = memref.cast %reinterpret_cast_4 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_7 = memref.cast %reinterpret_cast_5 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_6, %cast_7, %15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %74]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    %87 = llvm.add %64, %7 : i64
    memref.store %87, %alloca[%19] : memref<1xi64>
    %88 = arith.addi %62, %16 overflow<nsw> : i32
    cf.br ^bb6(%88 : i32)
  ^bb8:  // pred: ^bb6
    hivm.hir.set_ctrl true at ctrl[60]
    %89 = arith.addi %37, %17 overflow<nsw> : i32
    cf.br ^bb1(%89 : i32)
  ^bb9:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
