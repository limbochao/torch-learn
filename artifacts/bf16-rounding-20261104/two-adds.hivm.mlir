module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @two_adds_fused_0_outlined_vf_0(%arg0: memref<256xbf16, #hivm.address_space<ub>>, %arg1: memref<256xbf16, #hivm.address_space<ub>>, %arg2: memref<256xbf16, #hivm.address_space<ub>>, %arg3: memref<256xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(14 : i32) : i32
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(256 : i32) : i32
    %5 = llvm.mlir.constant(128 : i32) : i32
    cf.br ^bb1(%3 : i32)
  ^bb1(%6: i32):  // 2 preds: ^bb0, ^bb2
    %7 = arith.cmpi slt, %6, %4 : i32
    cf.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = llvm.sext %6 : i32 to i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%9], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %10 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%9], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast_4 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%9], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast_9 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg3 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [%9], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %13 = builtin.unrealized_conversion_cast %reinterpret_cast_14 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.getelementptr %14[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %17 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%16, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %18 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.getelementptr %18[%19] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %21 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%20, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %22 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.getelementptr %22[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %25 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%24, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%25, %26, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%21, %28, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %30 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %31 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%17, %30, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %32 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %33 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%31, %29, %32) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %34 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %35 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%33, %27, %34) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %36 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %37 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%35, %36, %3, %3, %3) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %38 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %39 = llvm.bitcast %37 : vector<128xbf16> to vector<64xf32>
    %40 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%39, %42, %3, %0, %3, %38) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %43 = llvm.add %8, %2 : i64
    %44 = builtin.unrealized_conversion_cast %43 : i64 to index
    %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [%44], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %45 = builtin.unrealized_conversion_cast %reinterpret_cast_19 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %arg1 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [%44], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %46 = builtin.unrealized_conversion_cast %reinterpret_cast_24 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_25, %offset_26, %sizes_27, %strides_28 = memref.extract_strided_metadata %arg2 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_29 = memref.reinterpret_cast %base_buffer_25 to offset: [%44], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %47 = builtin.unrealized_conversion_cast %reinterpret_cast_29 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_30, %offset_31, %sizes_32, %strides_33 = memref.extract_strided_metadata %arg3 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_34 = memref.reinterpret_cast %base_buffer_30 to offset: [%44], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %48 = builtin.unrealized_conversion_cast %reinterpret_cast_34 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %49 = llvm.extractvalue %45[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.extractvalue %45[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %52 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%51, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %53 = llvm.extractvalue %46[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.extractvalue %46[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.getelementptr %53[%54] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %56 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%55, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %57 = llvm.extractvalue %47[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.extractvalue %47[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %59 = llvm.getelementptr %57[%58] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %60 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%59, %3, %1, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %61 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %62 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%60, %61, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %63 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %64 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%56, %63, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %65 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %66 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%52, %65, %3) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %67 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %68 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%66, %64, %67) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %69 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %70 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%68, %62, %69) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %71 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %72 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%70, %71, %3, %3, %3) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %73 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %74 = llvm.bitcast %72 : vector<128xbf16> to vector<64xf32>
    %75 = llvm.extractvalue %48[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %76 = llvm.extractvalue %48[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.getelementptr %75[%76] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%74, %77, %3, %0, %3, %73) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %78 = arith.addi %6, %5 overflow<nsw> : i32
    cf.br ^bb1(%78 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @two_adds_outlined_vf_0(%arg0: memref<256xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(128 : i32) : i32
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %5, %0) : (bf16, vector<256xi1>, i32) -> vector<128xbf16>
    cf.br ^bb1(%3 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb2
    %8 = arith.cmpi slt, %7, %2 : i32
    cf.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.sext %7 : i32 to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%10], sizes: [128], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%6, %15, %3, %0, %3, %12) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = arith.addi %7, %1 overflow<nsw> : i32
    cf.br ^bb1(%16 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @two_adds_outlined_vf_1(%arg0: memref<256xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(128 : i32) : i32
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %5, %0) : (bf16, vector<256xi1>, i32) -> vector<128xbf16>
    cf.br ^bb1(%3 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb2
    %8 = arith.cmpi slt, %7, %2 : i32
    cf.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.sext %7 : i32 to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%10], sizes: [128], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%6, %15, %3, %0, %3, %12) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = arith.addi %7, %1 overflow<nsw> : i32
    cf.br ^bb1(%16 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @two_adds_outlined_vf_2(%arg0: memref<256xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(128 : i32) : i32
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %5, %0) : (bf16, vector<256xi1>, i32) -> vector<128xbf16>
    cf.br ^bb1(%3 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb2
    %8 = arith.cmpi slt, %7, %2 : i32
    cf.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.sext %7 : i32 to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%10], sizes: [128], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %11 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%3, %3) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.extractvalue %11[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128bf16"(%6, %15, %3, %0, %3, %12) : (vector<128xbf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %16 = arith.addi %7, %1 overflow<nsw> : i32
    cf.br ^bb1(%16 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.noinline, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, llvm.emit_c_interface}
  func.func @two_adds(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(3584 : i64) : i64
    %5 = llvm.mlir.constant(1536 : i64) : i64
    %6 = llvm.mlir.constant(1024 : i64) : i64
    %7 = llvm.mlir.constant(3072 : i64) : i64
    %8 = llvm.mlir.constant(512 : i64) : i64
    %9 = llvm.mlir.constant(2560 : i64) : i64
    %10 = llvm.mlir.constant(2048 : i64) : i64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(256 : i32) : i32
    %14 = llvm.mlir.constant(256 : index) : i64
    %15 = llvm.mlir.constant(16641 : index) : i64
    %16 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %17 = llvm.mlir.constant(56 : i32) : i32
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%19] : memref<1xi64>
    %20 = hivm.hir.pointer_cast(%5) : memref<256xbf16, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = hivm.hir.pointer_cast(%4) : memref<256xbf16, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = hivm.hir.pointer_cast(%6) : memref<256xbf16, #hivm.address_space<ub>>
    %25 = builtin.unrealized_conversion_cast %24 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = hivm.hir.pointer_cast(%7) : memref<256xbf16, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = hivm.hir.pointer_cast(%8) : memref<256xbf16, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = hivm.hir.pointer_cast(%9) : memref<256xbf16, #hivm.address_space<ub>>
    %31 = builtin.unrealized_conversion_cast %30 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = hivm.hir.pointer_cast(%0) : memref<256xbf16, #hivm.address_space<ub>>
    %33 = builtin.unrealized_conversion_cast %32 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = hivm.hir.pointer_cast(%10) : memref<256xbf16, #hivm.address_space<ub>>
    %35 = builtin.unrealized_conversion_cast %34 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.mul %arg6, %arg7 : i32
    %37 = llvm.mul %36, %arg8 : i32
    %38 = hivm.hir.get_block_idx -> i64
    %39 = llvm.trunc %38 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%39 : i32)
  ^bb1(%40: i32):  // 2 preds: ^bb0, ^bb8
    %41 = arith.cmpi slt, %40, %37 : i32
    cf.cond_br %41, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %42 = memref.load %alloca[%19] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %43 = llvm.urem %42, %12  : i64
    %44 = llvm.icmp "eq" %43, %11 : i64
    %45 = llvm.select %44, %23, %21 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = builtin.unrealized_conversion_cast %45 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %47 = llvm.select %44, %27, %25 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %49 = llvm.select %44, %31, %29 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = builtin.unrealized_conversion_cast %49 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %51 = llvm.select %44, %35, %33 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = builtin.unrealized_conversion_cast %51 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %53 = llvm.trunc %43 : i64 to i1
    %54 = llvm.select %53, %11, %12 : i1, i64
    %55 = llvm.xor %53, %3  : i1
    %56 = llvm.zext %55 : i1 to i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %57 = llvm.srem %40, %arg6  : i32
    %58 = llvm.mul %57, %13 : i32
    %59 = llvm.sext %58 : i32 to i64
    %60 = builtin.unrealized_conversion_cast %59 : i64 to index
    %c256 = arith.constant 256 : index
    %61 = arith.addi %60, %c256 : index
    %62 = builtin.unrealized_conversion_cast %61 : index to i64
    %63 = llvm.intr.smax(%59, %15)  : (i64, i64) -> i64
    %64 = llvm.intr.smin(%62, %63)  : (i64, i64) -> i64
    %65 = builtin.unrealized_conversion_cast %64 : i64 to index
    %c-1 = arith.constant -1 : index
    %66 = arith.muli %60, %c-1 : index
    %67 = arith.addi %65, %66 : index
    %68 = builtin.unrealized_conversion_cast %67 : index to i64
    %69 = llvm.icmp "slt" %68, %14 : i64
    cf.cond_br %69, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    call @two_adds_outlined_vf_0(%52) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%60], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %52 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %54]
    %cast = memref.cast %reinterpret_cast : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %reinterpret_cast_4 : memref<?xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast, %cast_5, %2, %16, %19, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    cf.cond_br %69, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    call @two_adds_outlined_vf_1(%50) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [%60], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %50 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, #hivm.address_space<ub>>
    %cast_16 = memref.cast %reinterpret_cast_10 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_17 = memref.cast %reinterpret_cast_15 : memref<?xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast_16, %cast_17, %2, %16, %19, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    cf.cond_br %69, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    call @two_adds_outlined_vf_2(%48) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %arg4 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%60], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %48 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [0], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, #hivm.address_space<ub>>
    %cast_28 = memref.cast %reinterpret_cast_22 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_29 = memref.cast %reinterpret_cast_27 : memref<?xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast_28, %cast_29, %2, %16, %19, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %56]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @two_adds_fused_0_outlined_vf_0(%52, %50, %48, %46) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>, memref<256xbf16, #hivm.address_space<ub>>, memref<256xbf16, #hivm.address_space<ub>>, memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %54]
    %base_buffer_30, %offset_31, %sizes_32, %strides_33 = memref.extract_strided_metadata %46 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_34 = memref.reinterpret_cast %base_buffer_30 to offset: [0], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    %base_buffer_35, %offset_36, %sizes_37, %strides_38 = memref.extract_strided_metadata %arg5 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [%60], sizes: [%67], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_40 = memref.cast %reinterpret_cast_34 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_41 = memref.cast %reinterpret_cast_39 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_bfloat16_t(%cast_40, %cast_41, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %56]
    hivm.hir.set_ctrl true at ctrl[60]
    %70 = llvm.add %42, %11 : i64
    memref.store %70, %alloca[%19] : memref<1xi64>
    %71 = arith.addi %40, %17 overflow<nsw> : i32
    cf.br ^bb1(%71 : i32)
  ^bb9:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
