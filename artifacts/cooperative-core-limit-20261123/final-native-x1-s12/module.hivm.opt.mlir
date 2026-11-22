#map = affine_map<(d0)[s0] -> (d0 + s0)>
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
  func.func @reduce_fused_1_outlined_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @reduce_outlined_vf_0(%arg0: memref<1x1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
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
  func.func @reduce_outlined_vf_1(%arg0: memref<1x1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
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
  func.func @reduce_outlined_vf_2(%arg0: memref<1x1024xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
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
  func.func @reduce_outlined_vf_3(%arg0: memref<1x1024xf32, #hivm.address_space<ub>>, %arg1: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(512 : i32) : i32
    %4 = llvm.mlir.constant(64 : i32) : i32
    %5 = llvm.mlir.constant(512 : index) : i64
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
    %cast = memref.cast %reinterpret_cast : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %6 = builtin.unrealized_conversion_cast %cast : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.getelementptr %7[%8] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %10 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%9, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [512], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 512>, #hivm.address_space<ub>>
    %cast_5 = memref.cast %reinterpret_cast_4 : memref<64xf32, strided<[1], offset: 512>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %cast_5 : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.getelementptr %12[%13] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %15 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%14, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    cf.br ^bb1(%4, %10, %15 : i32, vector<64xf32>, vector<64xf32>)
  ^bb1(%16: i32, %17: vector<64xf32>, %18: vector<64xf32>):  // 2 preds: ^bb0, ^bb2
    %19 = arith.cmpi slt, %16, %3 : i32
    cf.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %20 = llvm.sext %16 : i32 to i64
    %21 = builtin.unrealized_conversion_cast %20 : i64 to index
    %22 = llvm.add %20, %5 : i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [%21], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %24 = builtin.unrealized_conversion_cast %reinterpret_cast_10 : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %24[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.extractvalue %24[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%27, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %29 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %30 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%28, %17, %29) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %arg0 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%23], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %reinterpret_cast_15 : memref<64xf32, #map, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %35 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%34, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %36 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %37 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%35, %18, %36) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %38 = arith.addi %16, %4 overflow<nsw> : i32
    cf.br ^bb1(%38, %30, %37 : i32, vector<64xf32>, vector<64xf32>)
  ^bb3:  // pred: ^bb1
    %39 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %40 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%17, %18, %39) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %base_buffer_16, %offset_17, %sizes_18, %strides_19 = memref.extract_strided_metadata %arg1 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %41 = builtin.unrealized_conversion_cast %base_buffer_16 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %42 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %43 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%40, %42) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %44 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %45 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%43, %45, %2, %0, %2, %44) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func @reduce_outlined_vf_4(%arg0: memref<16xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = builtin.unrealized_conversion_cast %arg0 : memref<16xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %7 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %6, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %8 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %3) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
    %9 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%7, %9, %3, %0, %3, %8) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    return
  }
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float_nocache(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @reduce(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(20704 : i64) : i64
    %3 = llvm.mlir.constant(12384 : i64) : i64
    %4 = llvm.mlir.constant(12320 : i64) : i64
    %5 = llvm.mlir.constant(20640 : i64) : i64
    %6 = llvm.mlir.constant(12288 : i64) : i64
    %7 = llvm.mlir.constant(20608 : i64) : i64
    %8 = llvm.mlir.constant(4096 : i64) : i64
    %9 = llvm.mlir.constant(16512 : i64) : i64
    %10 = llvm.mlir.constant(12416 : i64) : i64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(8192 : i64) : i64
    %14 = llvm.mlir.constant(1024 : i32) : i32
    %15 = llvm.mlir.constant(1024 : index) : i64
    %16 = llvm.mlir.constant(32768 : index) : i64
    %17 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %18 = llvm.mlir.constant(32768 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(12 : i32) : i32
    %21 = llvm.mlir.constant(3072 : i32) : i32
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.mlir.constant(8 : i32) : i32
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%25] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca_0[%25] : memref<1xi64>
    %26 = hivm.hir.pointer_cast(%3) : memref<f32, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %28 = hivm.hir.pointer_cast(%2) : memref<f32, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %30 = hivm.hir.pointer_cast(%4) : memref<16xf32, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<16xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = hivm.hir.pointer_cast(%5) : memref<16xf32, #hivm.address_space<ub>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<16xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = hivm.hir.pointer_cast(%6) : memref<1xf32, #hivm.address_space<ub>>
    %35 = builtin.unrealized_conversion_cast %34 : memref<1xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = hivm.hir.pointer_cast(%7) : memref<1xf32, #hivm.address_space<ub>>
    %37 = builtin.unrealized_conversion_cast %36 : memref<1xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %38 = hivm.hir.pointer_cast(%8) : memref<1x1024xf32, #hivm.address_space<ub>>
    %39 = builtin.unrealized_conversion_cast %38 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = hivm.hir.pointer_cast(%9) : memref<1x1024xf32, #hivm.address_space<ub>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = hivm.hir.pointer_cast(%0) : memref<1x1024xf32, #hivm.address_space<ub>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %44 = hivm.hir.pointer_cast(%10) : memref<1x1024xf32, #hivm.address_space<ub>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mul %arg7, %arg8 : i32
    %47 = llvm.mul %46, %arg9 : i32
    %48 = hivm.hir.get_block_idx -> i64
    %49 = llvm.trunc %48 : i64 to i32
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%49 : i32)
  ^bb1(%50: i32):  // 2 preds: ^bb0, ^bb11
    %51 = arith.cmpi slt, %50, %47 : i32
    cf.cond_br %51, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %52 = memref.load %alloca_0[%25] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %53 = llvm.urem %52, %12  : i64
    %54 = llvm.icmp "eq" %53, %11 : i64
    %55 = llvm.select %54, %29, %27 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr<6>, ptr<6>, i64)> to memref<f32, #hivm.address_space<ub>>
    %57 = llvm.select %54, %33, %31 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %58 = builtin.unrealized_conversion_cast %57 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<16xf32, #hivm.address_space<ub>>
    %59 = llvm.select %54, %37, %35 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %61 = llvm.srem %50, %arg7  : i32
    %62 = llvm.sdiv %50, %arg7  : i32
    %63 = llvm.srem %62, %arg8  : i32
    %64 = llvm.sdiv %50, %46  : i32
    %65 = llvm.srem %64, %arg9  : i32
    %66 = hivm.hir.pointer_cast(%13) : memref<1x1024xf32, #hivm.address_space<ub>>
    call @reduce_outlined_vf_2(%66) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    %67 = llvm.add %63, %65 : i32
    %68 = llvm.mul %61, %21 : i32
    %69 = llvm.add %61, %22 : i32
    %70 = llvm.mul %69, %21 : i32
    %71 = llvm.intr.smin(%70, %18)  : (i32, i32) -> i32
    %72 = llvm.mul %67, %18 : i32
    %73 = llvm.sext %72 : i32 to i64
    %74 = builtin.unrealized_conversion_cast %73 : i64 to index
    cf.br ^bb3(%68, %66 : i32, memref<1x1024xf32, #hivm.address_space<ub>>)
  ^bb3(%75: i32, %76: memref<1x1024xf32, #hivm.address_space<ub>>):  // 2 preds: ^bb2, ^bb8
    %77 = arith.cmpi slt, %75, %71 : i32
    cf.cond_br %77, ^bb4, ^bb9
  ^bb4:  // pred: ^bb3
    %78 = memref.load %alloca[%25] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %79 = llvm.urem %78, %12  : i64
    %80 = llvm.icmp "eq" %79, %11 : i64
    %81 = llvm.select %80, %41, %39 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %82 = builtin.unrealized_conversion_cast %81 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<1x1024xf32, #hivm.address_space<ub>>
    %83 = llvm.select %80, %45, %43 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = builtin.unrealized_conversion_cast %83 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<1x1024xf32, #hivm.address_space<ub>>
    %85 = llvm.trunc %79 : i64 to i1
    %86 = llvm.select %85, %11, %12 : i1, i64
    %87 = llvm.sext %75 : i32 to i64
    %88 = builtin.unrealized_conversion_cast %87 : i64 to index
    %89 = arith.addi %74, %88 : index
    %c1024 = arith.constant 1024 : index
    %90 = arith.addi %88, %c1024 : index
    %91 = builtin.unrealized_conversion_cast %90 : index to i64
    %92 = llvm.intr.smax(%87, %16)  : (i64, i64) -> i64
    %93 = llvm.intr.smin(%91, %92)  : (i64, i64) -> i64
    %94 = builtin.unrealized_conversion_cast %93 : i64 to index
    %c-1 = arith.constant -1 : index
    %95 = arith.muli %88, %c-1 : index
    %96 = arith.addi %94, %95 : index
    %97 = builtin.unrealized_conversion_cast %96 : index to i64
    %98 = llvm.icmp "slt" %97, %15 : i64
    cf.cond_br %98, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    call @reduce_outlined_vf_0(%84) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%89], sizes: [%96], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %84 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [%96], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %86]
    %cast = memref.cast %reinterpret_cast : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast, %cast_6, %1, %17, %25, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    cf.cond_br %98, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    call @reduce_outlined_vf_1(%82) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %arg3 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%89], sizes: [%96], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %82 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [%96], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %cast_17 = memref.cast %reinterpret_cast_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_18 = memref.cast %reinterpret_cast_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast_17, %cast_18, %1, %17, %25, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %76 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %base_buffer_24, %offset_25, %sizes_26:2, %strides_27:2 = memref.extract_strided_metadata %82 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %base_buffer_29, %offset_30, %sizes_31:2, %strides_32:2 = memref.extract_strided_metadata %84 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %99 = hivm.hir.pointer_cast(%13) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_0_outlined_vf_0(%reinterpret_cast_23, %reinterpret_cast_33, %reinterpret_cast_28, %99) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %86]
    %base_buffer_34, %offset_35, %sizes_36, %strides_37 = memref.extract_strided_metadata %99 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_38 = memref.reinterpret_cast %base_buffer_34 to offset: [0], sizes: [1, 1024], strides: [1024, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1024xf32, #hivm.address_space<ub>>
    %100 = llvm.add %78, %11 : i64
    memref.store %100, %alloca[%25] : memref<1xi64>
    %101 = arith.addi %75, %14 overflow<nsw> : i32
    cf.br ^bb3(%101, %reinterpret_cast_38 : i32, memref<1x1024xf32, #hivm.address_space<ub>>)
  ^bb9:  // pred: ^bb3
    call @reduce_outlined_vf_3(%76, %60) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %102 = llvm.mul %67, %20 : i32
    %103 = llvm.sext %102 : i32 to i64
    %104 = builtin.unrealized_conversion_cast %103 : i64 to index
    %105 = llvm.sext %61 : i32 to i64
    %106 = builtin.unrealized_conversion_cast %105 : i64 to index
    %107 = arith.addi %104, %106 : index
    %reinterpret_cast_39 = memref.reinterpret_cast %arg4 to offset: [%107], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_40 = memref.cast %60 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_41 = memref.cast %reinterpret_cast_39 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_40, %cast_41, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    call @reduce_outlined_vf_4(%58) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %base_buffer_42, %offset_43, %sizes_44, %strides_45 = memref.extract_strided_metadata %arg4 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_46 = memref.reinterpret_cast %base_buffer_42 to offset: [%104], sizes: [12], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<12xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_47, %offset_48, %sizes_49, %strides_50 = memref.extract_strided_metadata %58 : memref<16xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_51 = memref.reinterpret_cast %base_buffer_47 to offset: [0], sizes: [12], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<12xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_52 = memref.cast %reinterpret_cast_46 : memref<12xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_53 = memref.cast %reinterpret_cast_51 : memref<12xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_52, %cast_53, %1, %17, %25, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_1_outlined_vf_0(%58, %56) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %base_buffer_54, %offset_55 = memref.extract_strided_metadata %56 : memref<f32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index
    %reinterpret_cast_56 = memref.reinterpret_cast %base_buffer_54 to offset: [0], sizes: [1], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
    %108 = llvm.icmp "eq" %61, %19 : i32
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    cf.cond_br %108, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %109 = llvm.sext %67 : i32 to i64
    %110 = builtin.unrealized_conversion_cast %109 : i64 to index
    %reinterpret_cast_57 = memref.reinterpret_cast %arg5 to offset: [%110], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %cast_58 = memref.cast %reinterpret_cast_56 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_59 = memref.cast %reinterpret_cast_57 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_58, %cast_59, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    cf.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    hivm.hir.set_ctrl true at ctrl[60]
    %111 = llvm.add %52, %11 : i64
    memref.store %111, %alloca_0[%25] : memref<1xi64>
    %112 = arith.addi %50, %23 overflow<nsw> : i32
    cf.br ^bb1(%112 : i32)
  ^bb12:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
