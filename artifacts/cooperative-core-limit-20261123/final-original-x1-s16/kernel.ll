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
  %5 = call <256 x i1> @llvm.hivm.pge.b32(i32 6, i32 0)
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
  %30 = add i32 %27, %29
  %31 = mul i32 %25, 2048
  %32 = add i32 %25, 1
  %33 = mul i32 %32, 2048
  %34 = call i32 @llvm.smin.i32(i32 %33, i32 32768)
  %35 = mul i32 %30, 32768
  %36 = sext i32 %35 to i64
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %37

37:                                               ; preds = %64, %10
  %38 = phi i32 [ %71, %64 ], [ %31, %10 ]
  %39 = phi { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } [ { ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %64 ], [ { ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, %10 ]
  %40 = icmp slt i32 %38, %34
  br i1 %40, label %41, label %72

41:                                               ; preds = %37
  %42 = load i64, ptr %12, align 4
  %43 = urem i64 %42, 2
  %44 = icmp eq i64 %43, 1
  %45 = select i1 %44, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 16672 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16672 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4128 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }
  %46 = select i1 %44, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12576 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12576 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1024], [1 x i64] [i64 1] }
  %47 = trunc i64 %43 to i1
  %48 = select i1 %47, i64 1, i64 2
  %49 = sext i32 %38 to i64
  %50 = add i64 %36, %49
  %51 = add i64 %49, 1024
  %52 = call i64 @llvm.smax.i64(i64 %49, i64 32768)
  %53 = call i64 @llvm.smin.i64(i64 %51, i64 %52)
  %54 = mul i64 %49, -1
  %55 = add i64 %53, %54
  %56 = icmp slt i64 %55, 1024
  br i1 %56, label %57, label %59

57:                                               ; preds = %41
  %58 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  call ptc_simdvf void @reduce_outlined_vf_0(ptr addrspace(6) %58)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %59

59:                                               ; preds = %57, %41
  %60 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 0
  %61 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %48)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %50, i64 %55, i64 1, ptr addrspace(6) %60, ptr addrspace(6) %61, i64 0, i64 %55, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  br i1 %56, label %62, label %64

62:                                               ; preds = %59
  %63 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %45, 1
  call ptc_simdvf void @reduce_outlined_vf_1(ptr addrspace(6) %63)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %64

64:                                               ; preds = %62, %59
  %65 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %45, 0
  %66 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %45, 1
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %50, i64 %55, i64 1, ptr addrspace(6) %65, ptr addrspace(6) %66, i64 0, i64 %55, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %67 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, 1
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %46, 1
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %45, 1
  call ptc_simdvf void @reduce_fused_0_outlined_vf_0(ptr addrspace(6) %67, ptr addrspace(6) %68, ptr addrspace(6) %69, ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %48)
  %70 = add i64 %42, 1
  store i64 %70, ptr %12, align 4
  %71 = add nsw i32 %38, 1024
  br label %37

72:                                               ; preds = %37
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  %73 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, 1
  call ptc_simdvf void @reduce_fused_1_outlined_vf_0(ptr addrspace(6) %73, ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %74 = load float, ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)), align 4
  %75 = mul i32 %30, 16
  %76 = sext i32 %75 to i64
  store float %74, ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %77 = sext i32 %25 to i64
  %78 = add i64 %76, %77
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12352 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %78, i64 1, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 6)
  %79 = icmp eq i32 %25, 0
  br i1 %79, label %80, label %82

80:                                               ; preds = %72
  %81 = sub i32 -2147483647, %7
  br label %83

82:                                               ; preds = %72
  br label %83

83:                                               ; preds = %80, %82
  %84 = phi i32 [ 1, %82 ], [ %81, %80 ]
  br label %85

85:                                               ; preds = %83
  store i32 %84, ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  %86 = sext i32 %30 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %86, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), i64 0, i64 1, i64 1, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %86, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %87 = load i32, ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), align 4
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  br label %88

88:                                               ; preds = %91, %85
  %89 = phi i1 [ %108, %91 ], [ false, %85 ]
  %90 = xor i1 %89, true
  br i1 %90, label %91, label %110

91:                                               ; preds = %88
  %92 = load i64, ptr %11, align 4
  %93 = urem i64 %92, 2
  %94 = icmp eq i64 %93, 1
  %95 = select i1 %94, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 20768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 20768 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12448 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12448 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %96 = trunc i64 %93 to i1
  %97 = xor i1 %96, true
  %98 = zext i1 %97 to i64
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %17, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 0, i64 4, i64 %98)
  %99 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 0
  %100 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 1
  %101 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 2
  %102 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 3, 0
  %103 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 4, 0
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %86, i64 1, i64 1, ptr addrspace(6) %99, ptr addrspace(6) %100, i64 %101, i64 %102, i64 %103, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1, i64 1, ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %86, i64 1, i64 1, i32 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %17, i64 0, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  %104 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %95, 1
  %105 = load i32, ptr addrspace(6) %104, align 4
  call void @llvm.hivm.SET.FLAG.REG(i64 0, i64 4, i64 %98)
  %106 = xor i32 %87, %105
  %107 = and i32 %106, -2147483648
  %108 = icmp ne i32 %107, 0
  %109 = add i64 %92, 1
  store i64 %109, ptr %11, align 4
  br label %88

110:                                              ; preds = %88
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %76, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), i64 0, i64 16, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call ptc_simdvf void @reduce_fused_2_outlined_vf_0(ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  %111 = load float, ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), align 4
  br i1 %79, label %112, label %113

112:                                              ; preds = %110
  store float %111, ptr addrspace(6) inttoptr (i64 12544 to ptr addrspace(6)), align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 12544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12544 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %86, i64 1, i64 1, i32 0)
  br label %113

113:                                              ; preds = %112, %110
  %114 = call i64 @llvm.hivm.GET.CTRL()
  %115 = call i64 @llvm.hivm.SBITSET1(i64 %114, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %115)
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

