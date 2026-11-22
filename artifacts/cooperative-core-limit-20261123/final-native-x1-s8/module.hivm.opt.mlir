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
  func.func @reduce_fused_1_outlined_vf_0(%arg0: memref<8xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = builtin.unrealized_conversion_cast %arg1 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %5 = builtin.unrealized_conversion_cast %arg0 : memref<8xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %7 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%3, %6, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %8 = builtin.unrealized_conversion_cast %7 : vector<64xf32> to vector<f32>
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%0, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%10, %1, %1, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %12 = builtin.unrealized_conversion_cast %8 : vector<f32> to f32
    %13 = builtin.unrealized_conversion_cast %12 : f32 to vector<1xf32>
    %14 = builtin.unrealized_conversion_cast %13 : vector<1xf32> to vector<64xf32>
    %15 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%11, %9) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %16 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %17 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%14, %15, %16) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %18 = builtin.unrealized_conversion_cast %17 : vector<64xf32> to vector<1xf32>
    %19 = builtin.unrealized_conversion_cast %18 : vector<1xf32> to f32
    %20 = builtin.unrealized_conversion_cast %19 : f32 to vector<f32>
    %21 = builtin.unrealized_conversion_cast %20 : vector<f32> to vector<1xf32>
    %22 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %1) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %23 = builtin.unrealized_conversion_cast %21 : vector<1xf32> to vector<64xf32>
    %24 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<6>, ptr<6>, i64)> 
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%23, %24, %1, %0, %1, %22) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
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
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float_nocache(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @reduce(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(20640 : i64) : i64
    %3 = llvm.mlir.constant(12352 : i64) : i64
    %4 = llvm.mlir.constant(12320 : i64) : i64
    %5 = llvm.mlir.constant(20608 : i64) : i64
    %6 = llvm.mlir.constant(12288 : i64) : i64
    %7 = llvm.mlir.constant(20576 : i64) : i64
    %8 = llvm.mlir.constant(4096 : i64) : i64
    %9 = llvm.mlir.constant(16480 : i64) : i64
    %10 = llvm.mlir.constant(12384 : i64) : i64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(8192 : i64) : i64
    %14 = llvm.mlir.constant(1024 : i32) : i32
    %15 = llvm.mlir.constant(1024 : index) : i64
    %16 = llvm.mlir.constant(32768 : index) : i64
    %17 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %18 = llvm.mlir.constant(32768 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(4096 : i32) : i32
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(8 : i32) : i32
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = builtin.unrealized_conversion_cast %23 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%24] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca_0[%24] : memref<1xi64>
    %25 = hivm.hir.pointer_cast(%3) : memref<f32, #hivm.address_space<ub>>
    %26 = builtin.unrealized_conversion_cast %25 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %27 = hivm.hir.pointer_cast(%2) : memref<f32, #hivm.address_space<ub>>
    %28 = builtin.unrealized_conversion_cast %27 : memref<f32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %29 = hivm.hir.pointer_cast(%4) : memref<8xf32, #hivm.address_space<ub>>
    %30 = builtin.unrealized_conversion_cast %29 : memref<8xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = hivm.hir.pointer_cast(%5) : memref<8xf32, #hivm.address_space<ub>>
    %32 = builtin.unrealized_conversion_cast %31 : memref<8xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = hivm.hir.pointer_cast(%6) : memref<1xf32, #hivm.address_space<ub>>
    %34 = builtin.unrealized_conversion_cast %33 : memref<1xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = hivm.hir.pointer_cast(%7) : memref<1xf32, #hivm.address_space<ub>>
    %36 = builtin.unrealized_conversion_cast %35 : memref<1xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = hivm.hir.pointer_cast(%8) : memref<1x1024xf32, #hivm.address_space<ub>>
    %38 = builtin.unrealized_conversion_cast %37 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = hivm.hir.pointer_cast(%9) : memref<1x1024xf32, #hivm.address_space<ub>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = hivm.hir.pointer_cast(%0) : memref<1x1024xf32, #hivm.address_space<ub>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = hivm.hir.pointer_cast(%10) : memref<1x1024xf32, #hivm.address_space<ub>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<1x1024xf32, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.mul %arg7, %arg8 : i32
    %46 = llvm.mul %45, %arg9 : i32
    %47 = hivm.hir.get_block_idx -> i64
    %48 = llvm.trunc %47 : i64 to i32
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%48 : i32)
  ^bb1(%49: i32):  // 2 preds: ^bb0, ^bb11
    %50 = arith.cmpi slt, %49, %46 : i32
    cf.cond_br %50, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %51 = memref.load %alloca_0[%24] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %52 = llvm.urem %51, %12  : i64
    %53 = llvm.icmp "eq" %52, %11 : i64
    %54 = llvm.select %53, %28, %26 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64)>
    %55 = builtin.unrealized_conversion_cast %54 : !llvm.struct<(ptr<6>, ptr<6>, i64)> to memref<f32, #hivm.address_space<ub>>
    %56 = llvm.select %53, %32, %30 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = builtin.unrealized_conversion_cast %56 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<8xf32, #hivm.address_space<ub>>
    %58 = llvm.select %53, %36, %34 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %59 = builtin.unrealized_conversion_cast %58 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %60 = llvm.srem %49, %arg7  : i32
    %61 = llvm.sdiv %49, %arg7  : i32
    %62 = llvm.srem %61, %arg8  : i32
    %63 = llvm.sdiv %49, %45  : i32
    %64 = llvm.srem %63, %arg9  : i32
    %65 = hivm.hir.pointer_cast(%13) : memref<1x1024xf32, #hivm.address_space<ub>>
    call @reduce_outlined_vf_2(%65) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    %66 = llvm.add %62, %64 : i32
    %67 = llvm.mul %60, %20 : i32
    %68 = llvm.add %60, %21 : i32
    %69 = llvm.mul %68, %20 : i32
    %70 = llvm.intr.smin(%69, %18)  : (i32, i32) -> i32
    %71 = llvm.mul %66, %18 : i32
    %72 = llvm.sext %71 : i32 to i64
    %73 = builtin.unrealized_conversion_cast %72 : i64 to index
    cf.br ^bb3(%67, %65 : i32, memref<1x1024xf32, #hivm.address_space<ub>>)
  ^bb3(%74: i32, %75: memref<1x1024xf32, #hivm.address_space<ub>>):  // 2 preds: ^bb2, ^bb8
    %76 = arith.cmpi slt, %74, %70 : i32
    cf.cond_br %76, ^bb4, ^bb9
  ^bb4:  // pred: ^bb3
    %77 = memref.load %alloca[%24] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %78 = llvm.urem %77, %12  : i64
    %79 = llvm.icmp "eq" %78, %11 : i64
    %80 = llvm.select %79, %40, %38 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %81 = builtin.unrealized_conversion_cast %80 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<1x1024xf32, #hivm.address_space<ub>>
    %82 = llvm.select %79, %44, %42 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)>
    %83 = builtin.unrealized_conversion_cast %82 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<2 x i64>, array<2 x i64>)> to memref<1x1024xf32, #hivm.address_space<ub>>
    %84 = llvm.trunc %78 : i64 to i1
    %85 = llvm.select %84, %11, %12 : i1, i64
    %86 = llvm.sext %74 : i32 to i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    %88 = arith.addi %73, %87 : index
    %c1024 = arith.constant 1024 : index
    %89 = arith.addi %87, %c1024 : index
    %90 = builtin.unrealized_conversion_cast %89 : index to i64
    %91 = llvm.intr.smax(%86, %16)  : (i64, i64) -> i64
    %92 = llvm.intr.smin(%90, %91)  : (i64, i64) -> i64
    %93 = builtin.unrealized_conversion_cast %92 : i64 to index
    %c-1 = arith.constant -1 : index
    %94 = arith.muli %87, %c-1 : index
    %95 = arith.addi %93, %94 : index
    %96 = builtin.unrealized_conversion_cast %95 : index to i64
    %97 = llvm.icmp "slt" %96, %15 : i64
    cf.cond_br %97, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    call @reduce_outlined_vf_0(%83) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%88], sizes: [%95], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %83 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [0], sizes: [%95], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %85]
    %cast = memref.cast %reinterpret_cast : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_6 = memref.cast %reinterpret_cast_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast, %cast_6, %1, %17, %24, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    cf.cond_br %97, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    call @reduce_outlined_vf_1(%81) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %arg3 : memref<?xf32, #hivm.address_space<gm>> -> memref<f32, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%88], sizes: [%95], strides: [1] : memref<f32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_12, %offset_13, %sizes_14:2, %strides_15:2 = memref.extract_strided_metadata %81 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [%95], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %cast_17 = memref.cast %reinterpret_cast_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_18 = memref.cast %reinterpret_cast_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float(%cast_17, %cast_18, %1, %17, %24, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %75 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %base_buffer_24, %offset_25, %sizes_26:2, %strides_27:2 = memref.extract_strided_metadata %81 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %base_buffer_29, %offset_30, %sizes_31:2, %strides_32:2 = memref.extract_strided_metadata %83 : memref<1x1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
    %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [0], sizes: [1024], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1024xf32, #hivm.address_space<ub>>
    %98 = hivm.hir.pointer_cast(%13) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_0_outlined_vf_0(%reinterpret_cast_23, %reinterpret_cast_33, %reinterpret_cast_28, %98) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %85]
    %base_buffer_34, %offset_35, %sizes_36, %strides_37 = memref.extract_strided_metadata %98 : memref<1024xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_38 = memref.reinterpret_cast %base_buffer_34 to offset: [0], sizes: [1, 1024], strides: [1024, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1024xf32, #hivm.address_space<ub>>
    %99 = llvm.add %77, %11 : i64
    memref.store %99, %alloca[%24] : memref<1xi64>
    %100 = arith.addi %74, %14 overflow<nsw> : i32
    cf.br ^bb3(%100, %reinterpret_cast_38 : i32, memref<1x1024xf32, #hivm.address_space<ub>>)
  ^bb9:  // pred: ^bb3
    call @reduce_outlined_vf_3(%75, %59) {hivm.vector_function, no_inline} : (memref<1x1024xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %101 = llvm.mul %66, %22 : i32
    %102 = llvm.sext %101 : i32 to i64
    %103 = builtin.unrealized_conversion_cast %102 : i64 to index
    %104 = llvm.sext %60 : i32 to i64
    %105 = builtin.unrealized_conversion_cast %104 : i64 to index
    %106 = arith.addi %103, %105 : index
    %reinterpret_cast_39 = memref.reinterpret_cast %arg4 to offset: [%106], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_40 = memref.cast %59 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_41 = memref.cast %reinterpret_cast_39 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_40, %cast_41, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>
    %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%103], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    %cast_43 = memref.cast %reinterpret_cast_42 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_44 = memref.cast %57 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_float_nocache(%cast_43, %cast_44, %19, %17, %24, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @reduce_fused_1_outlined_vf_0(%57, %55) {hivm.vector_function, no_inline} : (memref<8xf32, #hivm.address_space<ub>>, memref<f32, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %base_buffer_45, %offset_46 = memref.extract_strided_metadata %55 : memref<f32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index
    %reinterpret_cast_47 = memref.reinterpret_cast %base_buffer_45 to offset: [0], sizes: [1], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
    %107 = llvm.icmp "eq" %60, %19 : i32
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    cf.cond_br %107, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %108 = llvm.sext %66 : i32 to i64
    %109 = builtin.unrealized_conversion_cast %108 : i64 to index
    %reinterpret_cast_48 = memref.reinterpret_cast %arg5 to offset: [%109], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %cast_49 = memref.cast %reinterpret_cast_47 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_50 = memref.cast %reinterpret_cast_48 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_float(%cast_49, %cast_50, %19) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    cf.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    hivm.hir.set_ctrl true at ctrl[60]
    %110 = llvm.add %51, %11 : i64
    memref.store %110, %alloca_0[%24] : memref<1xi64>
    %111 = arith.addi %49, %22 overflow<nsw> : i32
    cf.br ^bb1(%111 : i32)
  ^bb12:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
