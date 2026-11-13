module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @raw_numerics(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.693147182 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1152 = arith.constant 1152 : index
    %c128 = arith.constant 128 : index
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    %3 = arith.muli %arg12, %c128_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<128xf32>
    %5 = arith.addi %4, %c128 : index
    %6 = arith.maxsi %4, %c1152 : index
    %7 = arith.minsi %5, %6 : index
    %8 = arith.subi %7, %4 : index
    %9 = arith.cmpi slt, %8, %c128 : index
    scf.if %9 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<128xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_2 = memref.subview %alloc[0] [%8] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<128xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<128xf32>
    scf.if %9 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_4 : memref<128xf32>)
    } {hivm.unlikely_condition}
    %subview_5 = memref.subview %reinterpret_cast_3[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0] [%8] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_5, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %11 = bufferization.to_tensor %alloc_4 restrict writable : memref<128xf32>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %10[0] [%8] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_8 = memref.subview %reinterpret_cast_7[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %12 = arith.addf %10, %10 : tensor<128xf32>
    %extracted_slice_10 = tensor.extract_slice %12[0] [%8] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_11 = memref.subview %reinterpret_cast_9[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_10 in writable %subview_11 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_12 = memref.reinterpret_cast %arg6 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %13 = arith.mulf %10, %1 : tensor<128xf32>
    %extracted_slice_13 = tensor.extract_slice %13[0] [%8] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_14 = memref.subview %reinterpret_cast_12[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_13 in writable %subview_14 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %alloc_16 = memref.alloc() : memref<128xf32>
    %14 = bufferization.to_tensor %alloc_16 restrict writable : memref<128xf32>
    %15 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "exp2_fp32"} "exp2_fp32" ins(%11 : tensor<128xf32>) outs(%14 : tensor<128xf32>) -> tensor<128xf32>
    %extracted_slice_17 = tensor.extract_slice %15[0] [%8] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_18 = memref.subview %reinterpret_cast_15[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_17 in writable %subview_18 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_19 = memref.reinterpret_cast %arg8 to offset: [%4], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %16 = arith.mulf %11, %2 : tensor<128xf32>
    %alloc_20 = memref.alloc() : memref<128xf32>
    %17 = bufferization.to_tensor %alloc_20 restrict writable : memref<128xf32>
    %18 = hivm.hir.custom {arg_attrs = [], bitcode = "", hivm.inline_mode = #hivm.inline_mode<always_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "exp_fp32"} "exp_fp32" ins(%16 : tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
    %extracted_slice_21 = tensor.extract_slice %18[0] [%8] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_22 = memref.subview %reinterpret_cast_19[0] [%8] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_21 in writable %subview_22 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
