; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduce_fused_0_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2, ptr addrspace(6) noalias %3) #0 {
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i32 [ %22, %8 ], [ 0, %4 ]
  %7 = icmp slt i32 %6, 1024
  br i1 %7, label %8, label %23

8:                                                ; preds = %5
  %9 = sext i32 %6 to i64
  %10 = getelementptr float, ptr addrspace(6) %0, i64 %9
  %11 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %10, i32 0, i32 0, i32 0)
  %12 = getelementptr float, ptr addrspace(6) %1, i64 %9
  %13 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %12, i32 0, i32 0, i32 0)
  %14 = getelementptr float, ptr addrspace(6) %2, i64 %9
  %15 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %14, i32 0, i32 0, i32 0)
  %16 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %17 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %13, <64 x float> %15, <256 x i1> %16)
  %18 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %19 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %11, <64 x float> %17, <256 x i1> %18)
  %20 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %21 = getelementptr float, ptr addrspace(6) %3, i64 %9
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %19, ptr addrspace(6) %21, i32 0, i32 2, i32 0, <256 x i1> %20)
  %22 = add nsw i32 %6, 64
  br label %5

23:                                               ; preds = %5
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduce_fused_1_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
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
define internal ptc_simdvf void @reduce_fused_2_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
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
define internal ptc_simdvf void @reduce_outlined_vf_0(ptr addrspace(6) noalias %0) #0 {
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
define internal ptc_simdvf void @reduce_outlined_vf_1(ptr addrspace(6) noalias %0) #0 {
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
define internal ptc_simdvf void @reduce_outlined_vf_2(ptr addrspace(6) noalias %0) #0 {
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

define dso_local void @reduce(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i32 %7, i32 %8, i32 %9) #2 {
  %11 = alloca i64, i64 1, align 8
  store i64 0, ptr %11, align 4
  %12 = alloca i64, i64 1, align 8
  store i64 0, ptr %12, align 4
  %13 = call i64 @llvm.hivm.GET.BLOCK.NUM()
  %14 = mul i64 %13, 2
  %15 = add i64 %14, 1
  %16 = mul i64 %15, 64
  %17 = getelementptr i8, ptr addrspace(1) %0, i64 %16
  %18 = call i64 @llvm.hivm.GET.CTRL()
  %19 = call i64 @llvm.hivm.SBITSET0(i64 %18, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %19)
  %20 = call i64 @llvm.hivm.GET.CTRL()
  %21 = call i64 @llvm.hivm.SBITSET1(i64 %20, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %21)
  %22 = mul i32 %7, %8
  %23 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %24 = trunc i64 %23 to i32
  %25 = srem i32 %24, %7
  %26 = sdiv i32 %24, %7
  %27 = srem i32 %26, %8
  %28 = sdiv i32 %24, %22
  %29 = srem i32 %28, %9
  store i32 0, ptr addrspace(6) null, align 4
  call ptc_simdvf void @reduce_outlined_vf_2(ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)))
  %30 = mul i32 %29, 2
  %31 = add i32 %27, %30
  %32 = mul i32 %25, 1024
  %33 = add i32 %25, 1
  %34 = mul i32 %33, 1024
  %35 = call i32 @llvm.smin.i32(i32 %34, i32 32768)
  %36 = mul i32 %31, 32768
  %37 = sext i32 %36 to i64
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %38

38:                                               ; preds = %65, %10
  %39 = phi i32 [ %72, %65 ], [ %32, %10 ]
  %40 = phi { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } [ { ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %65 ], [ { ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %10 ]
  %41 = icmp slt i32 %39, %35
  br i1 %41, label %42, label %73

42:                                               ; preds = %38
  %43 = load i64, ptr %12, align 4
  %44 = urem i64 %43, 2
  %45 = icmp eq i64 %44, 1
  %46 = select i1 %45, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 16736 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16736 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }
  %47 = select i1 %45, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }
  %48 = trunc i64 %44 to i1
  %49 = select i1 %48, i64 1, i64 2
  %50 = sext i32 %39 to i64
  %51 = add i64 %37, %50
  %52 = add i64 %50, 1024
  %53 = call i64 @llvm.smax.i64(i64 %50, i64 32768)
  %54 = call i64 @llvm.smin.i64(i64 %52, i64 %53)
  %55 = mul i64 %50, -1
  %56 = add i64 %54, %55
  %57 = icmp slt i64 %56, 1024
  br i1 %57, label %58, label %60

58:                                               ; preds = %42
  %59 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %47, 1
  call ptc_simdvf void @reduce_outlined_vf_0(ptr addrspace(6) %59)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %60

60:                                               ; preds = %58, %42
  %61 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %47, 0
  %62 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %47, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %49)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %51, i64 %56, i64 1, ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 %56, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  br i1 %57, label %63, label %65

63:                                               ; preds = %60
  %64 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  call ptc_simdvf void @reduce_outlined_vf_1(ptr addrspace(6) %64)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %65

65:                                               ; preds = %63, %60
  %66 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 0
  %67 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %51, i64 %56, i64 1, ptr addrspace(6) %66, ptr addrspace(6) %67, i64 0, i64 %56, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 1
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %47, 1
  %70 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  call ptc_simdvf void @reduce_fused_0_outlined_vf_0(ptr addrspace(6) %68, ptr addrspace(6) %69, ptr addrspace(6) %70, ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %49)
  %71 = add i64 %43, 1
  store i64 %71, ptr %12, align 4
  %72 = add nsw i32 %39, 1024
  br label %38

73:                                               ; preds = %38
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  %74 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, 1
  call ptc_simdvf void @reduce_fused_1_outlined_vf_0(ptr addrspace(6) %74, ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %75 = load float, ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)), align 4
  %76 = mul i32 %31, 32
  %77 = sext i32 %76 to i64
  store float %75, ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %78 = sext i32 %25 to i64
  %79 = add i64 %77, %78
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %79, i64 1, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 6)
  %80 = icmp eq i32 %25, 0
  br i1 %80, label %81, label %83

81:                                               ; preds = %73
  %82 = sub i32 -2147483647, %7
  br label %84

83:                                               ; preds = %73
  br label %84

84:                                               ; preds = %81, %83
  %85 = phi i32 [ 1, %83 ], [ %82, %81 ]
  br label %86

86:                                               ; preds = %84
  store i32 %85, ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %87 = sext i32 %31 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %87, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), i64 0, i64 1, i64 1, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %87, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %88 = load i32, ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), align 4
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  br label %89

89:                                               ; preds = %92, %86
  %90 = phi i1 [ %109, %92 ], [ false, %86 ]
  %91 = xor i1 %90, true
  br i1 %91, label %92, label %111

92:                                               ; preds = %89
  %93 = load i64, ptr %11, align 4
  %94 = urem i64 %93, 2
  %95 = icmp eq i64 %94, 1
  %96 = select i1 %95, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 20832 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 20832 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12448 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12448 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %97 = trunc i64 %94 to i1
  %98 = xor i1 %97, true
  %99 = zext i1 %98 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %17, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 0, i64 4, i64 %99)
  %100 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 0
  %101 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 1
  %102 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 2
  %103 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 3, 0
  %104 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 4, 0
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %87, i64 1, i64 1, ptr addrspace(6) %100, ptr addrspace(6) %101, i64 %102, i64 %103, i64 %104, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %87, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %17, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %105 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %96, 1
  %106 = load i32, ptr addrspace(6) %105, align 4
  call void @llvm.hivm.SET.FLAG.REG(i64 0, i64 4, i64 %99)
  %107 = xor i32 %88, %106
  %108 = and i32 %107, -2147483648
  %109 = icmp ne i32 %108, 0
  %110 = add i64 %93, 1
  store i64 %110, ptr %11, align 4
  br label %89

111:                                              ; preds = %89
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %77, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call ptc_simdvf void @reduce_fused_2_outlined_vf_0(ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %112 = load float, ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), align 4
  br i1 %80, label %113, label %114

113:                                              ; preds = %111
  store float %112, ptr addrspace(6) inttoptr (i64 12608 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 12608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12608 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %87, i64 1, i64 1, i32 0)
  br label %114

114:                                              ; preds = %113, %111
  %115 = call i64 @llvm.hivm.GET.CTRL()
  %116 = call i64 @llvm.hivm.SBITSET1(i64 %115, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %116)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %17, i64 0, i64 1, i64 1)
  ret void
}

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #3

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.NUM() #5

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #5

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #8

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smax.i64(i64, i64) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smin.i64(i64, i64) #8

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #3 = { nounwind readonly  }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nounwind  inaccessiblememonly }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @reduce, !"kernel", i32 1}

