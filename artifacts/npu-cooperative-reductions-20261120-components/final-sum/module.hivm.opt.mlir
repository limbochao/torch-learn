module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @reduction_fused_0_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(1024 : i32) : i32
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = builtin.unrealized_conversion_cast %arg1 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%2, %7, %1) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    cf.br ^bb1(%3, %8 : i32, vector<64xf32>)
  ^bb1(%9: i32, %10: vector<64xf32>):  // 2 preds: ^bb0, ^bb2
    %11 = arith.cmpi slt, %9, %4 : i32
    cf.cond_br %11, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %12 = llvm.sext %9 : i32 to i64
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %14 = builtin.unrealized_conversion_cast %10 : vector<64xf32> to vector<f32>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%13], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %15 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.getelementptr %16[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%18, %3, %3, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %20 = builtin.unrealized_conversion_cast %14 : vector<f32> to f32
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = builtin.unrealized_conversion_cast %20 : f32 to vector<1xf32>
    %23 = builtin.unrealized_conversion_cast %22 : vector<1xf32> to vector<64xf32>
    %24 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%19, %21) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %25 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %26 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%23, %24, %25) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %27 = builtin.unrealized_conversion_cast %26 : vector<64xf32> to vector<1xf32>
    %28 = builtin.unrealized_conversion_cast %27 : vector<1xf32> to f32
    %29 = builtin.unrealized_conversion_cast %28 : f32 to vector<f32>
    %30 = builtin.unrealized_conversion_cast %29 : vector<f32> to vector<64xf32>
    %31 = arith.addi %9, %5 overflow<nsw> : i32
    cf.br ^bb1(%31, %30 : i32, vector<64xf32>)
  ^bb3:  // pred: ^bb1
    %32 = builtin.unrealized_conversion_cast %10 : vector<64xf32> to vector<f32>
    %33 = builtin.unrealized_conversion_cast %32 : vector<f32> to vector<1xf32>
    %34 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %35 = builtin.unrealized_conversion_cast %33 : vector<1xf32> to vector<64xf32>
    %36 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%35, %36, %3, %0, %3, %34) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @reduction_fused_1_outlined_vf_0(%arg0: memref<32xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(12 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %6 = builtin.unrealized_conversion_cast %arg0 : memref<32xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %7, %3) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %9 = builtin.unrealized_conversion_cast %8 : vector<64xf32> to vector<f32>
    %10 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %11 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%11, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %13 = builtin.unrealized_conversion_cast %9 : vector<f32> to f32
    %14 = builtin.unrealized_conversion_cast %13 : f32 to vector<1xf32>
    %15 = builtin.unrealized_conversion_cast %14 : vector<1xf32> to vector<64xf32>
    %16 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%12, %10) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %17 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %18 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%15, %16, %17) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %19 = builtin.unrealized_conversion_cast %18 : vector<64xf32> to vector<1xf32>
    %20 = builtin.unrealized_conversion_cast %19 : vector<1xf32> to f32
    %21 = builtin.unrealized_conversion_cast %20 : f32 to vector<f32>
    %22 = builtin.unrealized_conversion_cast %21 : vector<f32> to vector<1xf32>
    %23 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %24 = builtin.unrealized_conversion_cast %22 : vector<1xf32> to vector<64xf32>
    %25 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%24, %25, %2, %0, %2, %23) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @reduction_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(64 : i32) : i32
    %3 = llvm.mlir.constant(1024 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %7 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%5, %6, %1) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    cf.br ^bb1(%4 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb2
    %9 = arith.cmpi slt, %8, %3 : i32
    cf.cond_br %9, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %10 = llvm.sext %8 : i32 to i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.getelementptr %14[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%7, %16, %4, %0, %4, %13) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %17 = arith.addi %8, %2 overflow<nsw> : i32
    cf.br ^bb1(%17 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @reduction_outlined_vf_1(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<1024xf32, #hivm.address_space<ub>>, %arg2: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1024 : i32) : i32
    %3 = llvm.mlir.constant(64 : i32) : i32
    cf.br ^bb1(%1 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %2 : i32
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.sext %4 : i32 to i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%7], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %8 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%7], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %9 = builtin.unrealized_conversion_cast %reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%7], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %10 = builtin.unrealized_conversion_cast %reinterpret_cast_9 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.getelementptr %11[%12] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %14 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%13, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %15 = llvm.extractvalue %9[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %9[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.getelementptr %15[%16] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %18 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%17, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %19 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %20 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%14, %18, %19) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.getelementptr %22[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%20, %24, %1, %0, %1, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %25 = arith.addi %4, %3 overflow<nsw> : i32
    cf.br ^bb1(%25 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @reduction_outlined_vf_2(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(64 : i32) : i32
    %3 = llvm.mlir.constant(1024 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %7 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%5, %6, %1) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    cf.br ^bb1(%4 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb2
    %9 = arith.cmpi slt, %8, %3 : i32
    cf.cond_br %9, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %10 = llvm.sext %8 : i32 to i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %4) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.getelementptr %14[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%7, %16, %4, %0, %4, %13) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %17 = arith.addi %8, %2 overflow<nsw> : i32
    cf.br ^bb1(%17 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @reduction_infer_sync_block_lock_num_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_num_function>} {
    %0 = llvm.mlir.constant(8589934592 : i64) : i64
    return %0 : i64
  }
  func.func @reduction_infer_sync_block_lock_init_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_init_function>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    return %0 : i64
  }
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @sync_block_lock_unordered(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t_nocache(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @sync_block_unlock_unordered(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float_nocache(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @free_lock_var_unordered(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @reduction(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(8544 : i64) : i64
    %4 = llvm.mlir.constant(32 : i64) : i64
    %5 = llvm.mlir.constant(8352 : i64) : i64
    %6 = llvm.mlir.constant(12640 : i64) : i64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(2 : i64) : i64
    %9 = llvm.mlir.constant(8512 : i64) : i64
    %10 = llvm.mlir.constant(8384 : i64) : i64
    %11 = llvm.mlir.constant(8320 : i64) : i64
    %12 = llvm.mlir.constant(8288 : i64) : i64
    %13 = llvm.mlir.constant(8256 : i64) : i64
    %14 = llvm.mlir.constant(8224 : i64) : i64
    %15 = llvm.mlir.constant(4128 : i64) : i64
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(1024 : index) : i64
    %18 = llvm.mlir.constant(1048576 : index) : i64
    %19 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %20 = llvm.mlir.constant(1024 : i32) : i32
    %21 = llvm.mlir.constant(1048576 : i32) : i32
    %22 = llvm.mlir.constant(32 : i32) : i32
    %23 = llvm.mlir.constant(-2147483647 : i32) : i32
    %24 = llvm.mlir.constant(false) : i1
    %25 = llvm.mlir.constant(true) : i1
    %26 = llvm.mlir.constant(-2147483648 : i32) : i32
    %27 = llvm.mlir.constant(32768 : i32) : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%30] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca_0[%30] : memref<1xi64>
    %31 = hivm.hir.pointer_cast(%4) : memref<1024xf32, #hivm.address_space<ub>>
    %32 = builtin.unrealized_conversion_cast %31 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = hivm.hir.pointer_cast(%3) : memref<1024xf32, #hivm.address_space<ub>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = hivm.hir.pointer_cast(%5) : memref<1xi32, #hivm.address_space<ub>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = hivm.hir.pointer_cast(%6) : memref<1xi32, #hivm.address_space<ub>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = hivm.hir.get_block_num -> i64
    %40 = llvm.mul %39, %8 : i64
    %41 = llvm.add %40, %7 : i64
    %42 = llvm.mul %41, %2 : i64
    %43 = builtin.unrealized_conversion_cast %42 : i64 to index
    %view = memref.view %arg0[%43][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %44 = hivm.hir.get_block_idx -> i64
    %45 = llvm.trunc %44 : i64 to i32
    %46 = llvm.srem %45, %arg6  : i32
    %47 = llvm.sdiv %45, %arg6  : i32
    %48 = llvm.srem %47, %arg7  : i32
    %49 = hivm.hir.pointer_cast(%0) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %16, %49[%30] : memref<1xi32, #hivm.address_space<ub>>
    %50 = hivm.hir.pointer_cast(%15) : memref<1024xf32, #hivm.address_space<ub>>
    call @reduction_outlined_vf_2(%50) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    %51 = llvm.mul %46, %27 : i32
    %52 = llvm.add %46, %28 : i32
    %53 = llvm.mul %52, %27 : i32
    %54 = llvm.intr.smin(%53, %21)  : (i32, i32) -> i32
    %55 = llvm.mul %48, %21 : i32
    %56 = llvm.sext %55 : i32 to i64
    %57 = builtin.unrealized_conversion_cast %56 : i64 to index
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%51, %50 : i32, memref<1024xf32, #hivm.address_space<ub>>)
  ^bb1(%58: i32, %59: memref<1024xf32, #hivm.address_space<ub>>):  // 2 preds: ^bb0, ^bb4
    %60 = arith.cmpi slt, %58, %54 : i32
    cf.cond_br %60, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    %61 = memref.load %alloca_0[%30] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %62 = llvm.urem %61, %8  : i64
    %63 = llvm.icmp "eq" %62, %7 : i64
    %64 = llvm.select %63, %34, %32 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %65 = builtin.unrealized_conversion_cast %64 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1024xf32, #hivm.address_space<ub>>
    %66 = llvm.trunc %62 : i64 to i1
    %67 = llvm.select %66, %7, %8 : i1, i64
    %68 = llvm.sext %58 : i32 to i64
    %69 = builtin.unrealized_conversion_cast %68 : i64 to index
    %70 = arith.addi %57, %69 : index
    %c1024 = arith.constant 1024 : index
    %71 = arith.addi %69, %c1024 : index
    %72 = builtin.unrealized_conversion_cast %71 : index to i64
    %73 = llvm.intr.smax(%68, %18)  : (i64, i64) -> i64
    %74 = llvm.intr.smin(%72, %73)  : (i64, i64) -> i64
    %75 = builtin.unrealized_conversion_cast %74 : i64 to index
    %c-1 = arith.constant -1 : index
    %76 = arith.muli %69, %c-1 : index
    %77 = arith.addi %75, %76 : index
    %78 = builtin.unrealized_conversion_cast %77 : index to i64
    %79 = llvm.icmp "slt" %78, %17 : i64
    cf.cond_br %79, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    call @reduction_outlined_vf_0(%65) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%70], sizes: [%77], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %65 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [%77], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %67]
    %cast = memref.cast %reinterpret_cast : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast, %cast_6, %1, %19, %30, %16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %80 = hivm.hir.pointer_cast(%15) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduction_outlined_vf_1(%59, %65, %80) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %67]
    %81 = llvm.add %61, %7 : i64
    memref.store %81, %alloca_0[%30] : memref<1xi64>
    %82 = arith.addi %58, %20 overflow<nsw> : i32
    cf.br ^bb1(%82, %80 : i32, memref<1024xf32, #hivm.address_space<ub>>)
  ^bb5:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %83 = hivm.hir.pointer_cast(%14) : memref<f32, #hivm.address_space<ub>>
    call @reduction_fused_0_outlined_vf_0(%59, %83) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %84 = memref.load %83[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<f32, #hivm.address_space<ub>>
    %85 = llvm.mul %48, %22 : i32
    %86 = llvm.sext %85 : i32 to i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    %88 = hivm.hir.pointer_cast(%13) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %84, %88[%30] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %89 = llvm.sext %46 : i32 to i64
    %90 = builtin.unrealized_conversion_cast %89 : i64 to index
    %91 = arith.addi %87, %90 : index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%91], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_8 = memref.cast %88 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %reinterpret_cast_7 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_8, %cast_9, %16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %92 = llvm.icmp "eq" %46, %16 : i32
    cf.cond_br %92, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %93 = llvm.sub %23, %arg6 : i32
    cf.br ^bb8(%93 : i32)
  ^bb7:  // pred: ^bb5
    cf.br ^bb8(%28 : i32)
  ^bb8(%94: i32):  // 2 preds: ^bb6, ^bb7
    cf.br ^bb9
  ^bb9:  // pred: ^bb8
    %95 = hivm.hir.pointer_cast(%12) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %94, %95[%30] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %96 = llvm.sext %48 : i32 to i64
    %97 = builtin.unrealized_conversion_cast %96 : i64 to index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%97], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %98 = hivm.hir.pointer_cast(%11) : memref<1xi32, #hivm.address_space<ub>>
    %view_11 = memref.view %arg0[%30][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    %cast_12 = memref.cast %view_11 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock_unordered(%cast_12) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %cast_13 = memref.cast %reinterpret_cast_10 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_14 = memref.cast %98 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_13, %cast_14, %16, %16, %30, %16) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_15 = memref.cast %95 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_16 = memref.cast %reinterpret_cast_10 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_15, %cast_16, %28) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_17 = memref.cast %view_11 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock_unordered(%cast_17) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %99 = memref.load %98[%30] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    cf.br ^bb10(%24 : i1)
  ^bb10(%100: i1):  // 2 preds: ^bb9, ^bb11
    %101 = llvm.xor %100, %25  : i1
    cf.cond_br %101, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %102 = memref.load %alloca[%30] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %103 = llvm.urem %102, %8  : i64
    %104 = llvm.icmp "eq" %103, %7 : i64
    %105 = llvm.select %104, %38, %36 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %106 = builtin.unrealized_conversion_cast %105 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32, #hivm.address_space<ub>>
    %107 = llvm.trunc %103 : i64 to i1
    %108 = llvm.xor %107, %25  : i1
    %109 = llvm.zext %108 : i1 to i64
    %cast_18 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock_unordered(%cast_18) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %109]
    %cast_19 = memref.cast %reinterpret_cast_10 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_20 = memref.cast %106 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_19, %cast_20, %16, %16, %30, %16) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_21 = memref.cast %49 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_22 = memref.cast %reinterpret_cast_10 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_21, %cast_22, %28) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    %cast_23 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock_unordered(%cast_23) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %110 = memref.load %106[%30] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %109]
    %111 = llvm.xor %99, %110  : i32
    %112 = llvm.and %111, %26  : i32
    %113 = llvm.icmp "ne" %112, %16 : i32
    %114 = llvm.add %102, %7 : i64
    memref.store %114, %alloca[%30] : memref<1xi64>
    cf.br ^bb10(%113 : i1)
  ^bb12:  // pred: ^bb10
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %reinterpret_cast_24 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %115 = hivm.hir.pointer_cast(%10) : memref<32xf32, #hivm.address_space<ub>>
    %cast_25 = memref.cast %reinterpret_cast_24 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_26 = memref.cast %115 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_25, %cast_26, %16, %19, %30, %16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %116 = hivm.hir.pointer_cast(%10) : memref<f32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduction_fused_1_outlined_vf_0(%115, %116) {hivm.vector_function, no_inline} : (memref<32xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %117 = memref.load %116[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<f32, #hivm.address_space<ub>>
    cf.cond_br %92, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %118 = hivm.hir.pointer_cast(%9) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %117, %118[%30] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_27 = memref.reinterpret_cast %arg4 to offset: [%97], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_28 = memref.cast %118 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_29 = memref.cast %reinterpret_cast_27 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_28, %cast_29, %16) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    cf.br ^bb14
  ^bb14:  // 2 preds: ^bb12, ^bb13
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %cast_30 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var_unordered(%cast_30) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %cast_31 = memref.cast %view_11 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var_unordered(%cast_31) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    return
  }
}
