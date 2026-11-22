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

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduce_outlined_vf_3(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
  %3 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %0, i32 0, i32 0, i32 0)
  %4 = getelementptr float, ptr addrspace(6) %0, i32 512
  %5 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %4, i32 0, i32 0, i32 0)
  br label %6

6:                                                ; preds = %11, %2
  %7 = phi i32 [ %22, %11 ], [ 64, %2 ]
  %8 = phi <64 x float> [ %17, %11 ], [ %3, %2 ]
  %9 = phi <64 x float> [ %21, %11 ], [ %5, %2 ]
  %10 = icmp slt i32 %7, 512
  br i1 %10, label %11, label %23

11:                                               ; preds = %6
  %12 = sext i32 %7 to i64
  %13 = add i64 %12, 512
  %14 = getelementptr float, ptr addrspace(6) %0, i64 %12
  %15 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %14, i32 0, i32 0, i32 0)
  %16 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %17 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %15, <64 x float> %8, <256 x i1> %16)
  %18 = getelementptr float, ptr addrspace(6) %0, i64 %13
  %19 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %18, i32 0, i32 0, i32 0)
  %20 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %21 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %19, <64 x float> %9, <256 x i1> %20)
  %22 = add nsw i32 %7, 64
  br label %6

23:                                               ; preds = %6
  %24 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %25 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %8, <64 x float> %9, <256 x i1> %24)
  %26 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %27 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %25, <256 x i1> %26)
  %28 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %27, ptr addrspace(6) %1, i32 0, i32 5, i32 0, <256 x i1> %28)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduce_outlined_vf_4(ptr addrspace(6) noalias %0) #0 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %2, i32 1)
  %4 = call <256 x i1> @llvm.hivm.pge.b32(i32 6, i32 0)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %3, ptr addrspace(6) %0, i32 0, i32 2, i32 0, <256 x i1> %4)
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

define dso_local void @reduce(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i32 %7, i32 %8, i32 %9) #2 {
  %11 = alloca i64, i64 1, align 8
  store i64 0, ptr %11, align 4
  %12 = alloca i64, i64 1, align 8
  store i64 0, ptr %12, align 4
  %13 = mul i32 %7, %8
  %14 = mul i32 %13, %9
  %15 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %16 = trunc i64 %15 to i32
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %17

17:                                               ; preds = %100, %10
  %18 = phi i32 [ %104, %100 ], [ %16, %10 ]
  %19 = icmp slt i32 %18, %14
  br i1 %19, label %20, label %105

20:                                               ; preds = %17
  %21 = load i64, ptr %12, align 4
  %22 = urem i64 %21, 2
  %23 = icmp eq i64 %22, 1
  %24 = select i1 %23, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 20704 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 20704 to ptr addrspace(6)), i64 0 }, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12384 to ptr addrspace(6)), i64 0 }
  %25 = select i1 %23, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 20640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 20640 to ptr addrspace(6)), i64 0, [1 x i64] [i64 16], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12320 to ptr addrspace(6)), i64 0, [1 x i64] [i64 16], [1 x i64] [i64 1] }
  %26 = select i1 %23, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 20608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 20608 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12288 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12288 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %27 = call i64 @llvm.hivm.GET.CTRL()
  %28 = call i64 @llvm.hivm.SBITSET0(i64 %27, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %28)
  %29 = call i64 @llvm.hivm.GET.CTRL()
  %30 = call i64 @llvm.hivm.SBITSET1(i64 %29, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %30)
  %31 = srem i32 %18, %7
  %32 = sdiv i32 %18, %7
  %33 = srem i32 %32, %8
  %34 = sdiv i32 %18, %13
  %35 = srem i32 %34, %9
  call ptc_simdvf void @reduce_outlined_vf_2(ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)))
  %36 = add i32 %33, %35
  %37 = mul i32 %31, 3072
  %38 = add i32 %31, 1
  %39 = mul i32 %38, 3072
  %40 = call i32 @llvm.smin.i32(i32 %39, i32 32768)
  %41 = mul i32 %36, 32768
  %42 = sext i32 %41 to i64
  br label %43

43:                                               ; preds = %70, %20
  %44 = phi i32 [ %77, %70 ], [ %37, %20 ]
  %45 = phi { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, %70 ], [ { ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, %20 ]
  %46 = icmp slt i32 %44, %40
  br i1 %46, label %47, label %78

47:                                               ; preds = %43
  %48 = load i64, ptr %11, align 4
  %49 = urem i64 %48, 2
  %50 = icmp eq i64 %49, 1
  %51 = select i1 %50, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) inttoptr (i64 16512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16512 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }
  %52 = select i1 %50, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12416 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) null, ptr addrspace(6) null, i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }
  %53 = trunc i64 %49 to i1
  %54 = select i1 %53, i64 1, i64 2
  %55 = sext i32 %44 to i64
  %56 = add i64 %42, %55
  %57 = add i64 %55, 1024
  %58 = call i64 @llvm.smax.i64(i64 %55, i64 32768)
  %59 = call i64 @llvm.smin.i64(i64 %57, i64 %58)
  %60 = mul i64 %55, -1
  %61 = add i64 %59, %60
  %62 = icmp slt i64 %61, 1024
  br i1 %62, label %63, label %65

63:                                               ; preds = %47
  %64 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %52, 1
  call ptc_simdvf void @reduce_outlined_vf_0(ptr addrspace(6) %64)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %65

65:                                               ; preds = %63, %47
  %66 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %52, 0
  %67 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %52, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %54)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %56, i64 %61, i64 1, ptr addrspace(6) %66, ptr addrspace(6) %67, i64 0, i64 %61, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  br i1 %62, label %68, label %70

68:                                               ; preds = %65
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %51, 1
  call ptc_simdvf void @reduce_outlined_vf_1(ptr addrspace(6) %69)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %70

70:                                               ; preds = %68, %65
  %71 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %51, 0
  %72 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %51, 1
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %56, i64 %61, i64 1, ptr addrspace(6) %71, ptr addrspace(6) %72, i64 0, i64 %61, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  %73 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, 1
  %74 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %51, 1
  %75 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %52, 1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call ptc_simdvf void @reduce_fused_0_outlined_vf_0(ptr addrspace(6) %73, ptr addrspace(6) %75, ptr addrspace(6) %74, ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %54)
  %76 = add i64 %48, 1
  store i64 %76, ptr %11, align 4
  %77 = add nsw i32 %44, 1024
  br label %43

78:                                               ; preds = %43
  %79 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, 1
  %80 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 1
  call ptc_simdvf void @reduce_outlined_vf_3(ptr addrspace(6) %79, ptr addrspace(6) %80)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %81 = mul i32 %36, 12
  %82 = sext i32 %81 to i64
  %83 = sext i32 %31 to i64
  %84 = add i64 %82, %83
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  %85 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 0
  %86 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 1
  %87 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 2
  %88 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 3, 0
  %89 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, 4, 0
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %85, ptr addrspace(6) %86, i64 %87, i64 %88, i64 %89, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %84, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  %90 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 1
  call ptc_simdvf void @reduce_outlined_vf_4(ptr addrspace(6) %90)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  %91 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 0
  %92 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %82, i64 12, i64 1, ptr addrspace(6) %91, ptr addrspace(6) %92, i64 0, i64 12, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %93 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 1
  %94 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %24, 1
  call ptc_simdvf void @reduce_fused_1_outlined_vf_0(ptr addrspace(6) %93, ptr addrspace(6) %94)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %95 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %24, 0
  %96 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %24, 1
  %97 = icmp eq i32 %31, 0
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  br i1 %97, label %98, label %100

98:                                               ; preds = %78
  %99 = sext i32 %36 to i64
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %95, ptr addrspace(6) %96, i64 0, i64 1, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %99, i64 1, i64 1, i32 0)
  br label %100

100:                                              ; preds = %98, %78
  %101 = call i64 @llvm.hivm.GET.CTRL()
  %102 = call i64 @llvm.hivm.SBITSET1(i64 %101, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %102)
  %103 = add i64 %21, 1
  store i64 %103, ptr %12, align 4
  %104 = add nsw i32 %18, 8
  br label %17

105:                                              ; preds = %17
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.BARRIER(i64 6)
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
declare i64 @llvm.hivm.GET.BLOCK.IDX() #5

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i32 @llvm.smin.i32(i32, i32) #8

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64, i64) #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smax.i64(i64, i64) #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn readnone 
declare i64 @llvm.smin.i64(i64, i64) #8

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

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

