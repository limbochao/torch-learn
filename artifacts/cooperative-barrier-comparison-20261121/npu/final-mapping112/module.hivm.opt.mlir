module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @exchange_fused_0_outlined_vf_0(%arg0: memref<128xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(128 : i32) : i32
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = builtin.unrealized_conversion_cast %arg1 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    %8 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"(%7, %3, %2, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
    %9 = builtin.unrealized_conversion_cast %8 : vector<64xi32> to vector<1xi32>
    %10 = builtin.unrealized_conversion_cast %9 : vector<1xi32> to vector<i32>
    %11 = builtin.unrealized_conversion_cast %10 : vector<i32> to vector<64xi32>
    cf.br ^bb1(%3, %11 : i32, vector<64xi32>)
  ^bb1(%12: i32, %13: vector<64xi32>):  // 2 preds: ^bb0, ^bb2
    %14 = arith.cmpi slt, %12, %4 : i32
    cf.cond_br %14, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %15 = llvm.sext %12 : i32 to i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %17 = builtin.unrealized_conversion_cast %13 : vector<64xi32> to vector<i32>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%16], sizes: [64], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %18 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %19[%20] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, i32
    %22 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"(%21, %3, %3, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xi32>
    %23 = builtin.unrealized_conversion_cast %17 : vector<i32> to i32
    %24 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %25 = builtin.unrealized_conversion_cast %23 : i32 to vector<1xi32>
    %26 = builtin.unrealized_conversion_cast %25 : vector<1xi32> to vector<64xi32>
    %27 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%22, %24) : (vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%26, %27, %28) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %30 = builtin.unrealized_conversion_cast %29 : vector<64xi32> to vector<1xi32>
    %31 = builtin.unrealized_conversion_cast %30 : vector<1xi32> to i32
    %32 = builtin.unrealized_conversion_cast %31 : i32 to vector<i32>
    %33 = builtin.unrealized_conversion_cast %32 : vector<i32> to vector<64xi32>
    %34 = arith.addi %12, %5 overflow<nsw> : i32
    cf.br ^bb1(%34, %33 : i32, vector<64xi32>)
  ^bb3:  // pred: ^bb1
    %35 = builtin.unrealized_conversion_cast %13 : vector<64xi32> to vector<i32>
    %36 = builtin.unrealized_conversion_cast %35 : vector<i32> to vector<1xi32>
    %37 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %38 = builtin.unrealized_conversion_cast %36 : vector<1xi32> to vector<64xi32>
    %39 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64s32"(%38, %39, %3, %0, %3, %37) : (vector<64xi32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @exchange_outlined_vf_0(%arg0: memref<128xi32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(64 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %5, %1) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    cf.br ^bb1(%4 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb2
    %8 = arith.cmpi slt, %7, %3 : i32
    cf.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.sext %7 : i32 to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%10], sizes: [64], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, i32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64s32"(%6, %15, %4, %0, %4, %12) : (vector<64xi32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = arith.addi %7, %2 overflow<nsw> : i32
    cf.br ^bb1(%16 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t_nocache(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @exchange(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.constant(1120 : i64) : i64
    %4 = llvm.mlir.constant(544 : i64) : i64
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.constant(608 : i64) : i64
    %7 = llvm.mlir.constant(576 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(3 : i64) : i64
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.mlir.constant(16 : i32) : i32
    %12 = llvm.mlir.constant(112 : i32) : i32
    %13 = llvm.mlir.constant(10000 : i32) : i32
    %14 = llvm.mlir.constant(1000 : i32) : i32
    %15 = llvm.mlir.constant(8 : i32) : i32
    %16 = llvm.mlir.constant(4 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(1 : i32) : i32
    %19 = llvm.mlir.constant(56 : i32) : i32
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = builtin.unrealized_conversion_cast %20 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%21] : memref<1xi64>
    %22 = hivm.hir.pointer_cast(%4) : memref<i32, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %24 = hivm.hir.pointer_cast(%3) : memref<i32, #hivm.address_space<ub>>
    %25 = builtin.unrealized_conversion_cast %24 : memref<i32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %26 = hivm.hir.pointer_cast(%5) : memref<128xi32, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<128xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = hivm.hir.pointer_cast(%6) : memref<128xi32, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<128xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = hivm.hir.pointer_cast(%0) : memref<1xi32, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = hivm.hir.pointer_cast(%7) : memref<1xi32, #hivm.address_space<ub>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.mul %arg6, %arg7 : i32
    %35 = llvm.mul %34, %arg8 : i32
    %36 = hivm.hir.get_block_idx -> i64
    %37 = llvm.trunc %36 : i64 to i32
    %38 = llvm.mul %arg5, %14 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    cf.br ^bb1(%37 : i32)
  ^bb1(%39: i32):  // 2 preds: ^bb0, ^bb8
    %40 = arith.cmpi slt, %39, %35 : i32
    cf.cond_br %40, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %41 = llvm.srem %39, %arg6  : i32
    %42 = llvm.sdiv %39, %arg6  : i32
    %43 = llvm.srem %42, %arg7  : i32
    %44 = llvm.sdiv %39, %34  : i32
    %45 = llvm.srem %44, %arg8  : i32
    %46 = llvm.add %43, %45 : i32
    %47 = llvm.srem %41, %16  : i32
    %48 = llvm.mul %47, %15 : i32
    cf.br ^bb3(%17, %17 : i32, i32)
  ^bb3(%49: i32, %50: i32):  // 2 preds: ^bb2, ^bb4
    %51 = arith.cmpi slt, %49, %48 : i32
    cf.cond_br %51, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %52 = llvm.add %50, %49 : i32
    %53 = arith.addi %49, %18 overflow<nsw> : i32
    cf.br ^bb3(%53, %52 : i32, i32)
  ^bb5:  // pred: ^bb3
    %54 = llvm.add %41, %18 : i32
    %55 = llvm.mul %46, %14 : i32
    %56 = llvm.add %54, %55 : i32
    %57 = llvm.mul %46, %12 : i32
    %58 = llvm.sext %57 : i32 to i64
    %59 = builtin.unrealized_conversion_cast %58 : i64 to index
    %60 = llvm.sext %41 : i32 to i64
    %61 = builtin.unrealized_conversion_cast %60 : i64 to index
    %62 = arith.addi %59, %61 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%62], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xi32, #hivm.address_space<gm>> -> memref<i32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_0 = memref.reinterpret_cast %base_buffer to offset: [%59], sizes: [112], strides: [1] : memref<i32, #hivm.address_space<gm>> to memref<112xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    cf.br ^bb6(%17 : i32)
  ^bb6(%63: i32):  // 2 preds: ^bb5, ^bb7
    %64 = arith.cmpi slt, %63, %11 : i32
    cf.cond_br %64, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %65 = memref.load %alloca[%21] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %66 = llvm.urem %65, %10  : i64
    %67 = llvm.icmp "eq" %66, %8 : i64
    %68 = llvm.select %67, %25, %23 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %69 = builtin.unrealized_conversion_cast %68 : !llvm.struct<(ptr<6>, ptr<6>, i64)> to memref<i32, #hivm.address_space<ub>>
    %70 = llvm.select %67, %29, %27 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %71 = builtin.unrealized_conversion_cast %70 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<128xi32, #hivm.address_space<ub>>
    %72 = llvm.select %67, %33, %31 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %73 = builtin.unrealized_conversion_cast %72 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32, #hivm.address_space<ub>>
    %74 = llvm.trunc %66 : i64 to i1
    %75 = llvm.select %74, %10, %9 : i1, i64
    %76 = llvm.xor %74, %2  : i1
    %77 = llvm.zext %76 : i1 to i64
    %78 = llvm.mul %63, %13 : i32
    %79 = llvm.add %56, %78 : i32
    %80 = llvm.add %79, %38 : i32
    %81 = llvm.add %80, %50 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %77]
    memref.store %81, %73[%21] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast = memref.cast %73 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_1 = memref.cast %reinterpret_cast : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast, %cast_1, %17) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %77]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    call @exchange_outlined_vf_0(%71) {hivm.vector_function, no_inline} : (memref<128xi32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %71 : memref<128xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [0], sizes: [112], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<112xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_7 = memref.cast %reinterpret_cast_0 : memref<112xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_8 = memref.cast %reinterpret_cast_6 : memref<112xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_7, %cast_8, %1, %17, %21, %17) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %82 = llvm.add %63, %46 : i32
    %83 = llvm.mul %82, %12 : i32
    %84 = llvm.sext %83 : i32 to i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %75]
    memref.store %17, %69[] : memref<i32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    call @exchange_fused_0_outlined_vf_0(%71, %69) {hivm.vector_function, no_inline} : (memref<128xi32, #hivm.address_space<ub>>, memref<i32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %base_buffer_9, %offset_10 = memref.extract_strided_metadata %69 : memref<i32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [1], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<1xi32, #hivm.address_space<ub>>
    %86 = arith.addi %85, %61 : index
    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_13 = memref.cast %reinterpret_cast_11 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_14 = memref.cast %reinterpret_cast_12 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_13, %cast_14, %17) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %75]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    %87 = llvm.add %65, %8 : i64
    memref.store %87, %alloca[%21] : memref<1xi64>
    %88 = arith.addi %63, %18 overflow<nsw> : i32
    cf.br ^bb6(%88 : i32)
  ^bb8:  // pred: ^bb6
    hivm.hir.set_ctrl true at ctrl[60]
    %89 = arith.addi %39, %19 overflow<nsw> : i32
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
