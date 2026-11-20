; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_fused_0_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
  %3 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  %4 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %3, i32 1)
  br label %5

5:                                                ; preds = %9, %2
  %6 = phi i32 [ %17, %9 ], [ 0, %2 ]
  %7 = phi <64 x float> [ %16, %9 ], [ %4, %2 ]
  %8 = icmp slt i32 %6, 1024
  br i1 %8, label %9, label %18

9:                                                ; preds = %5
  %10 = sext i32 %6 to i64
  %11 = getelementptr float, ptr addrspace(6) %0, i64 %10
  %12 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %11, i32 0, i32 0, i32 0)
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %12, <256 x i1> %13)
  %15 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %16 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %7, <64 x float> %14, <256 x i1> %15)
  %17 = add nsw i32 %6, 64
  br label %5

18:                                               ; preds = %5
  %19 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %7, ptr addrspace(6) %1, i32 0, i32 5, i32 0, <256 x i1> %19)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_fused_1_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
  %3 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  %4 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %3, i32 1)
  %5 = call <256 x i1> @llvm.hivm.pge.b32(i32 12, i32 0)
  %6 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %0, i32 0, i32 0, i32 0)
  %7 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %6, <256 x i1> %5)
  %8 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %9 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %4, <64 x float> %7, <256 x i1> %8)
  %10 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %9, ptr addrspace(6) %1, i32 0, i32 5, i32 0, <256 x i1> %10)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_outlined_vf_0(ptr addrspace(6) noalias %0) #0 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %2, i32 1)
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp slt i32 %5, 1024
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %10 = getelementptr float, ptr addrspace(6) %0, i64 %8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %3, ptr addrspace(6) %10, i32 0, i32 2, i32 0, <256 x i1> %9)
  %11 = add nsw i32 %5, 64
  br label %4

12:                                               ; preds = %4
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_outlined_vf_1(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2) #0 {
  br label %4

4:                                                ; preds = %7, %3
  %5 = phi i32 [ %17, %7 ], [ 0, %3 ]
  %6 = icmp slt i32 %5, 1024
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64
  %9 = getelementptr float, ptr addrspace(6) %0, i64 %8
  %10 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %9, i32 0, i32 0, i32 0)
  %11 = getelementptr float, ptr addrspace(6) %1, i64 %8
  %12 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %11, i32 0, i32 0, i32 0)
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %10, <64 x float> %12, <256 x i1> %13)
  %15 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %16 = getelementptr float, ptr addrspace(6) %2, i64 %8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %14, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %15)
  %17 = add nsw i32 %5, 64
  br label %4

18:                                               ; preds = %4
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_outlined_vf_2(ptr addrspace(6) noalias %0) #0 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %2, i32 1)
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp slt i32 %5, 1024
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %10 = getelementptr float, ptr addrspace(6) %0, i64 %8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %3, ptr addrspace(6) %10, i32 0, i32 2, i32 0, <256 x i1> %9)
  %11 = add nsw i32 %5, 64
  br label %4

12:                                               ; preds = %4
  ret void
}

; Function Attrs: noinline
define private void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12, i32 %13) #1 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr %20, ptr %26, i32 %10, float %11, i64 %12, i32 %13)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #1 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr, ptr, i32) #1

; Function Attrs: noinline
define private void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #1 {
  %6 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %7 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %6, ptr addrspace(1) %1, 1
  %8 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %10, ptr %11, align 8
  call void @_mlir_ciface_sync_block_lock_unordered(ptr %11)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_sync_block_lock_unordered(ptr) #1

; Function Attrs: noinline
define private void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, i32 %11, i64 %12, i32 %13) #1 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_int32_t_nocache(ptr %20, ptr %26, i32 %10, i32 %11, i64 %12, i32 %13)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_int32_t_nocache(ptr, ptr, i32, i32, i64, i32) #1

; Function Attrs: noinline
define private void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #1 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_int32_t(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_int32_t(ptr, ptr, i32) #1

; Function Attrs: noinline
define private void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #1 {
  %6 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %7 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %6, ptr addrspace(1) %1, 1
  %8 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %10, ptr %11, align 8
  call void @_mlir_ciface_sync_block_unlock_unordered(ptr %11)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_sync_block_unlock_unordered(ptr) #1

; Function Attrs: noinline
define private void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12, i32 %13) #1 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float_nocache(ptr %20, ptr %26, i32 %10, float %11, i64 %12, i32 %13)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float_nocache(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: noinline
define private void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #1 {
  %6 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %7 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %6, ptr addrspace(1) %1, 1
  %8 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %10, ptr %11, align 8
  call void @_mlir_ciface_free_lock_var_unordered(ptr %11)
  ret void
}

; Function Attrs: noinline
declare dso_local void @_mlir_ciface_free_lock_var_unordered(ptr) #1

define dso_local void @reduction(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, i32 %6, i32 %7, i32 %8) #2 {
  %10 = alloca i64, i64 1, align 8
  store i64 0, ptr %10, align 4
  %11 = alloca i64, i64 1, align 8
  store i64 0, ptr %11, align 4
  %12 = call i64 @llvm.hivm.GET.BLOCK.NUM()
  %13 = mul i64 %12, 2
  %14 = add i64 %13, 1
  %15 = mul i64 %14, 64
  %16 = getelementptr i8, ptr addrspace(1) %0, i64 %15
  %17 = call i64 @llvm.hivm.GET.CTRL()
  %18 = call i64 @llvm.hivm.SBITSET0(i64 %17, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %18)
  %19 = call i64 @llvm.hivm.GET.CTRL()
  %20 = call i64 @llvm.hivm.SBITSET1(i64 %19, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %20)
  %21 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %22 = trunc i64 %21 to i32
  %23 = srem i32 %22, %6
  %24 = sdiv i32 %22, %6
  %25 = srem i32 %24, %7
  store i32 0, ptr addrspace(6) null, align 4
  call ptc_simdvf void @reduction_outlined_vf_2(ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)))
  %26 = mul i32 %23, 32768
  %27 = add i32 %23, 1
  %28 = mul i32 %27, 32768
  %29 = call i32 @llvm.smin.i32(i32 %28, i32 1048576)
  %30 = mul i32 %25, 1048576
  %31 = sext i32 %30 to i64
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %32

32:                                               ; preds = %53, %9
  %33 = phi i32 [ %59, %53 ], [ %26, %9 ]
  %34 = phi { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } [ { ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %53 ], [ { ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %9 ]
  %35 = icmp slt i32 %33, %29
  br i1 %35, label %36, label %60

36:                                               ; preds = %32
  %37 = load i64, ptr %11, align 4
  %38 = urem i64 %37, 2
  %39 = icmp eq i64 %38, 1
  %40 = select i1 %39, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 8544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8544 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }
  %41 = trunc i64 %38 to i1
  %42 = select i1 %41, i64 1, i64 2
  %43 = sext i32 %33 to i64
  %44 = add i64 %31, %43
  %45 = add i64 %43, 1024
  %46 = call i64 @llvm.smax.i64(i64 %43, i64 1048576)
  %47 = call i64 @llvm.smin.i64(i64 %45, i64 %46)
  %48 = mul i64 %43, -1
  %49 = add i64 %47, %48
  %50 = icmp slt i64 %49, 1024
  br i1 %50, label %51, label %53

51:                                               ; preds = %36
  %52 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 1
  call ptc_simdvf void @reduction_outlined_vf_0(ptr addrspace(6) %52)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %53

53:                                               ; preds = %51, %36
  %54 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 0
  %55 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %42)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %44, i64 %49, i64 1, ptr addrspace(6) %54, ptr addrspace(6) %55, i64 0, i64 %49, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %56 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, 1
  %57 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 1
  call ptc_simdvf void @reduction_outlined_vf_1(ptr addrspace(6) %56, ptr addrspace(6) %57, ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %42)
  %58 = add i64 %37, 1
  store i64 %58, ptr %11, align 4
  %59 = add nsw i32 %33, 1024
  br label %32

60:                                               ; preds = %32
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  %61 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, 1
  call ptc_simdvf void @reduction_fused_0_outlined_vf_0(ptr addrspace(6) %61, ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %62 = load float, ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), align 4
  %63 = mul i32 %25, 32
  %64 = sext i32 %63 to i64
  store float %62, ptr addrspace(6) inttoptr (i64 8256 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %65 = sext i32 %23 to i64
  %66 = add i64 %64, %65
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 8256 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8256 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %66, i64 1, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 6)
  %67 = icmp eq i32 %23, 0
  br i1 %67, label %68, label %70

68:                                               ; preds = %60
  %69 = sub i32 -2147483647, %6
  br label %71

70:                                               ; preds = %60
  br label %71

71:                                               ; preds = %68, %70
  %72 = phi i32 [ 1, %70 ], [ %69, %68 ]
  br label %73

73:                                               ; preds = %71
  store i32 %72, ptr addrspace(6) inttoptr (i64 8288 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %74 = sext i32 %25 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %74, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 8320 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8320 to ptr addrspace(6)), i64 0, i64 1, i64 1, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) inttoptr (i64 8288 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8288 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %74, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %75 = load i32, ptr addrspace(6) inttoptr (i64 8320 to ptr addrspace(6)), align 4
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  br label %76

76:                                               ; preds = %79, %73
  %77 = phi i1 [ %96, %79 ], [ false, %73 ]
  %78 = xor i1 %77, true
  br i1 %78, label %79, label %98

79:                                               ; preds = %76
  %80 = load i64, ptr %10, align 4
  %81 = urem i64 %80, 2
  %82 = icmp eq i64 %81, 1
  %83 = select i1 %82, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 8352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8352 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %84 = trunc i64 %81 to i1
  %85 = xor i1 %84, true
  %86 = zext i1 %85 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %16, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 0, i64 4, i64 %86)
  %87 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 0
  %88 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 1
  %89 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 2
  %90 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 3, 0
  %91 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 4, 0
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %74, i64 1, i64 1, ptr addrspace(6) %87, ptr addrspace(6) %88, i64 %89, i64 %90, i64 %91, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %74, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %16, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %92 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %83, 1
  %93 = load i32, ptr addrspace(6) %92, align 4
  call void @llvm.hivm.SET.FLAG.REG(i64 0, i64 4, i64 %86)
  %94 = xor i32 %75, %93
  %95 = and i32 %94, -2147483648
  %96 = icmp ne i32 %95, 0
  %97 = add i64 %80, 1
  store i64 %97, ptr %10, align 4
  br label %76

98:                                               ; preds = %76
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %64, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call ptc_simdvf void @reduction_fused_1_outlined_vf_0(ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %99 = load float, ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), align 4
  br i1 %67, label %100, label %101

100:                                              ; preds = %98
  store float %99, ptr addrspace(6) inttoptr (i64 8512 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 8512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8512 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %74, i64 1, i64 1, i32 0)
  br label %101

101:                                              ; preds = %100, %98
  %102 = call i64 @llvm.hivm.GET.CTRL()
  %103 = call i64 @llvm.hivm.SBITSET1(i64 %102, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %103)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %16, i64 0, i64 1, i64 1)
  call void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  ret void
}

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #3

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #5

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #4

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.NUM() #4

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #4

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #8

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smax.i64(i64, i64) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smin.i64(i64, i64) #8

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #3 = { nounwind }
attributes #4 = { nounwind readnone  }
attributes #5 = { nounwind writeonly  }
attributes #6 = { nounwind readonly  }
attributes #7 = { nounwind  inaccessiblememonly }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @reduction, !"kernel", i32 1}

