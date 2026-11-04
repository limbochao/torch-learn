module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.c310.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.c310.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c310.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c310.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @add_sigmoid_mul_fused_0_outlined_vf_0(%arg0: memref<256xbf16, #hivm.address_space<ub>>, %arg1: memref<256xbf16, #hivm.address_space<ub>>, %arg2: memref<256xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(14 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(256 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%8, %9, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    cf.br ^bb1(%5 : i32)
  ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb2
    %12 = arith.cmpi slt, %11, %4 : i32
    cf.cond_br %12, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %13 = llvm.sext %11 : i32 to i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%14], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %15 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg1 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%14], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %16 = builtin.unrealized_conversion_cast %reinterpret_cast_4 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg2 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%14], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %17 = builtin.unrealized_conversion_cast %reinterpret_cast_9 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %15[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.getelementptr %18[%19] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %21 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%20, %5, %1, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %22 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.getelementptr %22[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %25 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%24, %5, %1, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%25, %26, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%21, %28, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %30 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %31 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%29, %27, %30) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %32 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %33 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%31, %7, %32) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %34 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %35 = "hivm_regbaseintrins.intr.hivm.vexp.x"(%33, %34) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %36 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %37 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%35, %8, %36) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %38 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %39 = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%10, %37, %38) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %40 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %41 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%39, %40, %5, %5, %5) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %42 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %43 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%41, %42, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %44 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %45 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%31, %43, %44) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %46 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %47 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%45, %46, %5, %5, %5) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %48 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %49 = llvm.bitcast %47 : vector<128xbf16> to vector<64xf32>
    %50 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.extractvalue %17[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.getelementptr %50[%51] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%49, %52, %5, %0, %5, %48) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %53 = llvm.add %13, %6 : i64
    %54 = builtin.unrealized_conversion_cast %53 : i64 to index
    %base_buffer_10, %offset_11, %sizes_12, %strides_13 = memref.extract_strided_metadata %arg0 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_14 = memref.reinterpret_cast %base_buffer_10 to offset: [%54], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %55 = builtin.unrealized_conversion_cast %reinterpret_cast_14 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %arg1 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [%54], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %56 = builtin.unrealized_conversion_cast %reinterpret_cast_19 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %arg2 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [%54], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %57 = builtin.unrealized_conversion_cast %reinterpret_cast_24 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.extractvalue %55[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %59 = llvm.extractvalue %55[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %60 = llvm.getelementptr %58[%59] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %61 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%60, %5, %1, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %62 = llvm.extractvalue %56[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %63 = llvm.extractvalue %56[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    %65 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%64, %5, %1, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xbf16>
    %66 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %67 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%65, %66, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %68 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %69 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%61, %68, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %70 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %71 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%69, %67, %70) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %72 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %73 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%71, %7, %72) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %74 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %75 = "hivm_regbaseintrins.intr.hivm.vexp.x"(%73, %74) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %76 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %77 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%75, %8, %76) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %78 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %79 = "hivm_regbaseintrins.intr.hivm.vdiv.s.x"(%10, %77, %78) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %80 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %81 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%79, %80, %5, %5, %5) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %82 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %83 = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"(%81, %82, %5) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
    %84 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %85 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%71, %83, %84) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %86 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %87 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%85, %86, %5, %5, %5) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
    %88 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %89 = llvm.bitcast %87 : vector<128xbf16> to vector<64xf32>
    %90 = llvm.extractvalue %57[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %91 = llvm.extractvalue %57[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, bf16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%89, %92, %5, %0, %5, %88) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %93 = arith.addi %11, %3 overflow<nsw> : i32
    cf.br ^bb1(%93 : i32)
  ^bb3:  // pred: ^bb1
    return
  }
  func.func @add_sigmoid_mul_outlined_vf_0(%arg0: memref<256xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @add_sigmoid_mul_outlined_vf_1(%arg0: memref<256xbf16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @add_sigmoid_mul(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.mlir.constant(2560 : i64) : i64
    %5 = llvm.mlir.constant(1024 : i64) : i64
    %6 = llvm.mlir.constant(512 : i64) : i64
    %7 = llvm.mlir.constant(2048 : i64) : i64
    %8 = llvm.mlir.constant(1536 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.mlir.constant(256 : i32) : i32
    %12 = llvm.mlir.constant(256 : index) : i64
    %13 = llvm.mlir.constant(16641 : index) : i64
    %14 = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    %15 = llvm.mlir.constant(56 : i32) : i32
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %0, %alloca[%17] : memref<1xi64>
    %18 = hivm.hir.pointer_cast(%5) : memref<256xbf16, #hivm.address_space<ub>>
    %19 = builtin.unrealized_conversion_cast %18 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = hivm.hir.pointer_cast(%4) : memref<256xbf16, #hivm.address_space<ub>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = hivm.hir.pointer_cast(%6) : memref<256xbf16, #hivm.address_space<ub>>
    %23 = builtin.unrealized_conversion_cast %22 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %24 = hivm.hir.pointer_cast(%7) : memref<256xbf16, #hivm.address_space<ub>>
    %25 = builtin.unrealized_conversion_cast %24 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = hivm.hir.pointer_cast(%0) : memref<256xbf16, #hivm.address_space<ub>>
    %27 = builtin.unrealized_conversion_cast %26 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = hivm.hir.pointer_cast(%8) : memref<256xbf16, #hivm.address_space<ub>>
    %29 = builtin.unrealized_conversion_cast %28 : memref<256xbf16, #hivm.address_space<ub>> to !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.mul %arg5, %arg6 : i32
    %31 = llvm.mul %30, %arg7 : i32
    %32 = hivm.hir.get_block_idx -> i64
    %33 = llvm.trunc %32 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    cf.br ^bb1(%33 : i32)
  ^bb1(%34: i32):  // 2 preds: ^bb0, ^bb6
    %35 = arith.cmpi slt, %34, %31 : i32
    cf.cond_br %35, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %36 = memref.load %alloca[%17] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
    %37 = llvm.urem %36, %10  : i64
    %38 = llvm.icmp "eq" %37, %9 : i64
    %39 = llvm.select %38, %21, %19 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %41 = llvm.select %38, %25, %23 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %42 = builtin.unrealized_conversion_cast %41 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %43 = llvm.select %38, %29, %27 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %44 = builtin.unrealized_conversion_cast %43 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> to memref<256xbf16, #hivm.address_space<ub>>
    %45 = llvm.trunc %37 : i64 to i1
    %46 = llvm.select %45, %9, %10 : i1, i64
    %47 = llvm.xor %45, %3  : i1
    %48 = llvm.zext %47 : i1 to i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %49 = llvm.srem %34, %arg5  : i32
    %50 = llvm.mul %49, %11 : i32
    %51 = llvm.sext %50 : i32 to i64
    %52 = builtin.unrealized_conversion_cast %51 : i64 to index
    %c256 = arith.constant 256 : index
    %53 = arith.addi %52, %c256 : index
    %54 = builtin.unrealized_conversion_cast %53 : index to i64
    %55 = llvm.intr.smax(%51, %13)  : (i64, i64) -> i64
    %56 = llvm.intr.smin(%54, %55)  : (i64, i64) -> i64
    %57 = builtin.unrealized_conversion_cast %56 : i64 to index
    %c-1 = arith.constant -1 : index
    %58 = arith.muli %52, %c-1 : index
    %59 = arith.addi %57, %58 : index
    %60 = builtin.unrealized_conversion_cast %59 : index to i64
    %61 = llvm.icmp "slt" %60, %12 : i64
    cf.cond_br %61, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    call @add_sigmoid_mul_outlined_vf_0(%44) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%52], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %44 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [0], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %46]
    %cast = memref.cast %reinterpret_cast : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_5 = memref.cast %reinterpret_cast_4 : memref<?xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast, %cast_5, %2, %14, %17, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    cf.cond_br %61, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    call @add_sigmoid_mul_outlined_vf_1(%42) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %arg3 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [%52], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %42 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, #hivm.address_space<ub>>
    %cast_16 = memref.cast %reinterpret_cast_10 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_17 = memref.cast %reinterpret_cast_15 : memref<?xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_bfloat16_t(%cast_16, %cast_17, %2, %14, %17, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %48]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @add_sigmoid_mul_fused_0_outlined_vf_0(%44, %42, %40) {hivm.vector_function, no_inline} : (memref<256xbf16, #hivm.address_space<ub>>, memref<256xbf16, #hivm.address_space<ub>>, memref<256xbf16, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %46]
    %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %40 : memref<256xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [0], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    %base_buffer_23, %offset_24, %sizes_25, %strides_26 = memref.extract_strided_metadata %arg4 : memref<?xbf16, #hivm.address_space<gm>> -> memref<bf16, #hivm.address_space<gm>>, index, index, index
    %reinterpret_cast_27 = memref.reinterpret_cast %base_buffer_23 to offset: [%52], sizes: [%59], strides: [1] : memref<bf16, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    %cast_28 = memref.cast %reinterpret_cast_22 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_29 = memref.cast %reinterpret_cast_27 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_bfloat16_t(%cast_28, %cast_29, %1) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %48]
    hivm.hir.set_ctrl true at ctrl[60]
    %62 = llvm.add %36, %9 : i64
    memref.store %62, %alloca[%17] : memref<1xi64>
    %63 = arith.addi %34, %15 overflow<nsw> : i32
    cf.br ^bb1(%63 : i32)
  ^bb7:  // pred: ^bb1
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
