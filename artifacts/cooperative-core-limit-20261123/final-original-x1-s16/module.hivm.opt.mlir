module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 4 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 4 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 8 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/l30023782/miniforge3/envs/pta_213_case/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @reduce_fused_0_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<1024xf32, #hivm.address_space<ub>>, %arg2: memref<1024xf32, #hivm.address_space<ub>>, %arg3: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg3 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [%7], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast_14 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.getelementptr %12[%13] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %15 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%14, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %16 = llvm.extractvalue %9[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %9[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.getelementptr %16[%17] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %19 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%18, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %20 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.getelementptr %20[%21] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %23 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%22, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %24 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %25 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%19, %23, %24) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%15, %25, %26) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%27, %31, %1, %0, %1, %28) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %32 = arith.addi %4, %3 overflow<nsw> : i32
    cf.br ^bb1(%32 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @reduce_fused_1_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @reduce_fused_2_outlined_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %6 = builtin.unrealized_conversion_cast %arg0 : memref<16xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
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
  func.func @reduce_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @reduce_outlined_vf_1(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @reduce_outlined_vf_2(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @reduce_infer_sync_block_lock_num_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_num_function>} {
    %0 = llvm.mlir.constant(8589934592 : i64) : i64
    return %0 : i64
  }
  func.func @reduce_infer_sync_block_lock_init_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_sync_block_lock_init_function>} {
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
  func.func @reduce(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(16672 : i64) : i64
    %4 = llvm.mlir.constant(4128 : i64) : i64
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.constant(12576 : i64) : i64
    %7 = llvm.mlir.constant(12448 : i64) : i64
    %8 = llvm.mlir.constant(20768 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.mlir.constant(12544 : i64) : i64
    %12 = llvm.mlir.constant(12480 : i64) : i64
    %13 = llvm.mlir.constant(12416 : i64) : i64
    %14 = llvm.mlir.constant(12384 : i64) : i64
    %15 = llvm.mlir.constant(12352 : i64) : i64
    %16 = llvm.mlir.constant(12320 : i64) : i64
    %17 = llvm.mlir.constant(8224 : i64) : i64
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(1024 : index) : i64
    %20 = llvm.mlir.constant(32768 : index) : i64
    %21 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %22 = llvm.mlir.constant(1024 : i32) : i32
    %23 = llvm.mlir.constant(32768 : i32) : i32
    %24 = llvm.mlir.constant(16 : i32) : i32
    %25 = llvm.mlir.constant(-2147483647 : i32) : i32
    %26 = llvm.mlir.constant(false) : i1
    %27 = llvm.mlir.constant(true) : i1
    %28 = llvm.mlir.constant(-2147483648 : i32) : i32
    %29 = llvm.mlir.constant(2048 : i32) : i32
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.mlir.constant(0 : index) : i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%32] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca_0[%32] : memref<1xi64>
    %33 = hivm.hir.pointer_cast(%4) : memref<1024xf32, #hivm.address_space<ub>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = hivm.hir.pointer_cast(%3) : memref<1024xf32, #hivm.address_space<ub>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = hivm.hir.pointer_cast(%5) : memref<1024xf32, #hivm.address_space<ub>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = hivm.hir.pointer_cast(%6) : memref<1024xf32, #hivm.address_space<ub>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = hivm.hir.pointer_cast(%7) : memref<1xi32, #hivm.address_space<ub>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %43 = hivm.hir.pointer_cast(%8) : memref<1xi32, #hivm.address_space<ub>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<1xi32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %45 = hivm.hir.get_block_num -> i64
    %46 = llvm.mul %45, %10 : i64
    %47 = llvm.add %46, %9 : i64
    %48 = llvm.mul %47, %2 : i64
    %49 = builtin.unrealized_conversion_cast %48 : i64 to index
    %view = memref.view %arg0[%49][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %50 = llvm.mul %arg7, %arg8 : i32
    %51 = hivm.hir.get_block_idx -> i64
    %52 = llvm.trunc %51 : i64 to i32
    %53 = llvm.srem %52, %arg7  : i32
    %54 = llvm.sdiv %52, %arg7  : i32
    %55 = llvm.srem %54, %arg8  : i32
    %56 = llvm.sdiv %52, %50  : i32
    %57 = llvm.srem %56, %arg9  : i32
    %58 = hivm.hir.pointer_cast(%0) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %18, %58[%32] : memref<1xi32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%17) : memref<1024xf32, #hivm.address_space<ub>>
    call @reduce_outlined_vf_2(%59) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    %60 = llvm.add %55, %57 : i32
    %61 = llvm.mul %53, %29 : i32
    %62 = llvm.add %53, %30 : i32
    %63 = llvm.mul %62, %29 : i32
    %64 = llvm.intr.smin(%63, %23)  : (i32, i32) -> i32
    %65 = llvm.mul %60, %23 : i32
    %66 = llvm.sext %65 : i32 to i64
    %67 = builtin.unrealized_conversion_cast %66 : i64 to index
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%61, %59 : i32, memref<1024xf32, #hivm.address_space<ub>>)
  ^bb1(%68: i32, %69: memref<1024xf32, #hivm.address_space<ub>>):  // 2 preds: ^bb0, ^bb6
    %70 = arith.cmpi slt, %68, %64 : i32
    cf.cond_br %70, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %71 = memref.load %alloca_0[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %72 = llvm.urem %71, %10  : i64
    %73 = llvm.icmp "eq" %72, %9 : i64
    %74 = llvm.select %73, %36, %34 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %75 = builtin.unrealized_conversion_cast %74 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1024xf32, #hivm.address_space<ub>>
    %76 = llvm.select %73, %40, %38 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = builtin.unrealized_conversion_cast %76 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1024xf32, #hivm.address_space<ub>>
    %78 = llvm.trunc %72 : i64 to i1
    %79 = llvm.select %78, %9, %10 : i1, i64
    %80 = llvm.sext %68 : i32 to i64
    %81 = builtin.unrealized_conversion_cast %80 : i64 to index
    %82 = arith.addi %67, %81 : index
    %c1024 = arith.constant 1024 : index
    %83 = arith.addi %81, %c1024 : index
    %84 = builtin.unrealized_conversion_cast %83 : index to i64
    %85 = llvm.intr.smax(%80, %20)  : (i64, i64) -> i64
    %86 = llvm.intr.smin(%84, %85)  : (i64, i64) -> i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    %c-1 = arith.constant -1 : index
    %88 = arith.muli %81, %c-1 : index
    %89 = arith.addi %87, %88 : index
    %90 = builtin.unrealized_conversion_cast %89 : index to i64
    %91 = llvm.icmp "slt" %90, %19 : i64
    cf.cond_br %91, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    call @reduce_outlined_vf_0(%77) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%82], sizes: [%89], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_1, %offset_2, %sizes_3, %strides_4 = memref.extract_strided_metadata %77 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [%89], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %79]
    %cast = memref.cast %reinterpret_cast : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast, %cast_6, %1, %21, %32, %18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    cf.cond_br %91, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    call @reduce_outlined_vf_1(%75) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %arg3 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%82], sizes: [%89], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_12, %offset_13, %sizes_14, %strides_15 = memref.extract_strided_metadata %75 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [%89], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    %cast_17 = memref.cast %reinterpret_cast_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_18 = memref.cast %reinterpret_cast_16 : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast_17, %cast_18, %1, %21, %32, %18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %92 = hivm.hir.pointer_cast(%17) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_0_outlined_vf_0(%69, %77, %75, %92) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %79]
    %93 = llvm.add %71, %9 : i64
    memref.store %93, %alloca_0[%32] : memref<1xi64>
    %94 = arith.addi %68, %22 overflow<nsw> : i32
    cf.br ^bb1(%94, %92 : i32, memref<1024xf32, #hivm.address_space<ub>>)
  ^bb7:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %95 = hivm.hir.pointer_cast(%16) : memref<f32, #hivm.address_space<ub>>
    call @reduce_fused_1_outlined_vf_0(%69, %95) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %96 = memref.load %95[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<f32, #hivm.address_space<ub>>
    %97 = llvm.mul %60, %24 : i32
    %98 = llvm.sext %97 : i32 to i64
    %99 = builtin.unrealized_conversion_cast %98 : i64 to index
    %100 = hivm.hir.pointer_cast(%15) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %96, %100[%32] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %101 = llvm.sext %53 : i32 to i64
    %102 = builtin.unrealized_conversion_cast %101 : i64 to index
    %103 = arith.addi %99, %102 : index
    %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%103], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_20 = memref.cast %100 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_21 = memref.cast %reinterpret_cast_19 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_20, %cast_21, %18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %104 = llvm.icmp "eq" %53, %18 : i32
    cf.cond_br %104, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %105 = llvm.sub %25, %arg7 : i32
    cf.br ^bb10(%105 : i32)
  ^bb9:  // pred: ^bb7
    cf.br ^bb10(%30 : i32)
  ^bb10(%106: i32):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    %107 = hivm.hir.pointer_cast(%14) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %106, %107[%32] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %108 = llvm.sext %60 : i32 to i64
    %109 = builtin.unrealized_conversion_cast %108 : i64 to index
    %reinterpret_cast_22 = memref.reinterpret_cast %arg6 to offset: [%109], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %110 = hivm.hir.pointer_cast(%13) : memref<1xi32, #hivm.address_space<ub>>
    %view_23 = memref.view %arg0[%32][] : memref<?xi8, #hivm.address_space<gm>> to memref<1xi64, #hivm.address_space<gm>>
    %cast_24 = memref.cast %view_23 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock_unordered(%cast_24) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %cast_25 = memref.cast %reinterpret_cast_22 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_26 = memref.cast %110 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_25, %cast_26, %18, %18, %32, %18) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_27 = memref.cast %107 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_28 = memref.cast %reinterpret_cast_22 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_27, %cast_28, %30) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_29 = memref.cast %view_23 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock_unordered(%cast_29) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %111 = memref.load %110[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    cf.br ^bb12(%26 : i1)
  ^bb12(%112: i1):  // 2 preds: ^bb11, ^bb13
    %113 = llvm.xor %112, %27  : i1
    cf.cond_br %113, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %114 = memref.load %alloca[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %115 = llvm.urem %114, %10  : i64
    %116 = llvm.icmp "eq" %115, %9 : i64
    %117 = llvm.select %116, %44, %42 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %118 = builtin.unrealized_conversion_cast %117 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32, #hivm.address_space<ub>>
    %119 = llvm.trunc %115 : i64 to i1
    %120 = llvm.xor %119, %27  : i1
    %121 = llvm.zext %120 : i1 to i64
    %cast_30 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_lock_unordered(%cast_30) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %121]
    %cast_31 = memref.cast %reinterpret_cast_22 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_32 = memref.cast %118 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int32_t_nocache(%cast_31, %cast_32, %18, %18, %32, %18) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_33 = memref.cast %58 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_34 = memref.cast %reinterpret_cast_22 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int32_t(%cast_33, %cast_34, %30) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    %cast_35 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @sync_block_unlock_unordered(%cast_35) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %122 = memref.load %118[%32] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %121]
    %123 = llvm.xor %111, %122  : i32
    %124 = llvm.and %123, %28  : i32
    %125 = llvm.icmp "ne" %124, %18 : i32
    %126 = llvm.add %114, %9 : i64
    memref.store %126, %alloca[%32] : memref<1xi64>
    cf.br ^bb12(%125 : i1)
  ^bb14:  // pred: ^bb12
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %reinterpret_cast_36 = memref.reinterpret_cast %arg4 to offset: [%99], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %127 = hivm.hir.pointer_cast(%12) : memref<16xf32, #hivm.address_space<ub>>
    %cast_37 = memref.cast %reinterpret_cast_36 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_38 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_37, %cast_38, %18, %21, %32, %18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %128 = hivm.hir.pointer_cast(%12) : memref<f32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_2_outlined_vf_0(%127, %128) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %129 = memref.load %128[] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<f32, #hivm.address_space<ub>>
    cf.cond_br %104, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %130 = hivm.hir.pointer_cast(%11) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %129, %130[%32] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_39 = memref.reinterpret_cast %arg5 to offset: [%109], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_40 = memref.cast %130 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_41 = memref.cast %reinterpret_cast_39 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_40, %cast_41, %18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    cf.br ^bb16
  ^bb16:  // 2 preds: ^bb14, ^bb15
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    %cast_42 = memref.cast %view_23 : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var_unordered(%cast_42) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    %cast_43 = memref.cast %view : memref<1xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @free_lock_var_unordered(%cast_43) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>) -> ()
    return
  }
}
