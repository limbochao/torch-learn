; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_fused_0_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
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

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @reduction_outlined_vf_3(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
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

define dso_local void @reduction(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, i32 %6, i32 %7, i32 %8) #2 {
  %10 = alloca i64, i64 1, align 8
  store i64 0, ptr %10, align 4
  %11 = alloca i64, i64 1, align 8
  store i64 0, ptr %11, align 4
  %12 = mul i32 %6, %7
  %13 = mul i32 %12, %8
  %14 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %15 = trunc i64 %14 to i32
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  br label %16

16:                                               ; preds = %91, %9
  %17 = phi i32 [ %95, %91 ], [ %15, %9 ]
  %18 = icmp slt i32 %17, %13
  br i1 %18, label %19, label %96

19:                                               ; preds = %16
  %20 = load i64, ptr %11, align 4
  %21 = urem i64 %20, 2
  %22 = icmp eq i64 %21, 1
  %23 = select i1 %22, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12640 to ptr addrspace(6)), i64 0 }, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 8352 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8352 to ptr addrspace(6)), i64 0 }
  %24 = select i1 %22, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12512 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12512 to ptr addrspace(6)), i64 0, [1 x i64] [i64 32], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8224 to ptr addrspace(6)), i64 0, [1 x i64] [i64 32], [1 x i64] [i64 1] }
  %25 = select i1 %22, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12480 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %26 = call i64 @llvm.hivm.GET.CTRL()
  %27 = call i64 @llvm.hivm.SBITSET0(i64 %26, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %27)
  %28 = call i64 @llvm.hivm.GET.CTRL()
  %29 = call i64 @llvm.hivm.SBITSET1(i64 %28, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %29)
  %30 = srem i32 %17, %6
  %31 = sdiv i32 %17, %6
  %32 = srem i32 %31, %7
  call ptc_simdvf void @reduction_outlined_vf_2(ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)))
  %33 = mul i32 %30, 32768
  %34 = add i32 %30, 1
  %35 = mul i32 %34, 32768
  %36 = call i32 @llvm.smin.i32(i32 %35, i32 1048576)
  %37 = mul i32 %32, 1048576
  %38 = sext i32 %37 to i64
  br label %39

39:                                               ; preds = %60, %19
  %40 = phi i32 [ %66, %60 ], [ %33, %19 ]
  %41 = phi { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, %60 ], [ { ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, %19 ]
  %42 = icmp slt i32 %40, %36
  br i1 %42, label %43, label %67

43:                                               ; preds = %39
  %44 = load i64, ptr %10, align 4
  %45 = urem i64 %44, 2
  %46 = icmp eq i64 %45, 1
  %47 = select i1 %46, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8384 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } { ptr addrspace(6) null, ptr addrspace(6) null, i64 0, [2 x i64] [i64 1, i64 1024], [2 x i64] [i64 1024, i64 1] }
  %48 = trunc i64 %45 to i1
  %49 = select i1 %48, i64 1, i64 2
  %50 = sext i32 %40 to i64
  %51 = add i64 %38, %50
  %52 = add i64 %50, 1024
  %53 = call i64 @llvm.smax.i64(i64 %50, i64 1048576)
  %54 = call i64 @llvm.smin.i64(i64 %52, i64 %53)
  %55 = mul i64 %50, -1
  %56 = add i64 %54, %55
  %57 = icmp slt i64 %56, 1024
  br i1 %57, label %58, label %60

58:                                               ; preds = %43
  %59 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, 1
  call ptc_simdvf void @reduction_outlined_vf_0(ptr addrspace(6) %59)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %60

60:                                               ; preds = %58, %43
  %61 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, 0
  %62 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %49)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %51, i64 %56, i64 1, ptr addrspace(6) %61, ptr addrspace(6) %62, i64 0, i64 %56, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %63 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, 1
  %64 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, 1
  call ptc_simdvf void @reduction_outlined_vf_1(ptr addrspace(6) %63, ptr addrspace(6) %64, ptr addrspace(6) inttoptr (i64 4096 to ptr addrspace(6)))
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %49)
  %65 = add i64 %44, 1
  store i64 %65, ptr %10, align 4
  %66 = add nsw i32 %40, 1024
  br label %39

67:                                               ; preds = %39
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, 1
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 1
  call ptc_simdvf void @reduction_outlined_vf_3(ptr addrspace(6) %68, ptr addrspace(6) %69)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %70 = mul i32 %32, 32
  %71 = sext i32 %70 to i64
  %72 = sext i32 %30 to i64
  %73 = add i64 %71, %72
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  %74 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 0
  %75 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 1
  %76 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 2
  %77 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 3, 0
  %78 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, 4, 0
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %74, ptr addrspace(6) %75, i64 %76, i64 %77, i64 %78, ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %73, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  %79 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 0
  %80 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 1
  %81 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 2
  %82 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 3, 0
  %83 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 4, 0
  call void @load_gm_to_ubuf_1d_float_nocache(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %71, i64 32, i64 1, ptr addrspace(6) %79, ptr addrspace(6) %80, i64 %81, i64 %82, i64 %83, i32 0, float 0.000000e+00, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  %84 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, 1
  %85 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %23, 1
  call ptc_simdvf void @reduction_fused_0_outlined_vf_0(ptr addrspace(6) %84, ptr addrspace(6) %85)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %86 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %23, 0
  %87 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %23, 1
  %88 = icmp eq i32 %30, 0
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  br i1 %88, label %89, label %91

89:                                               ; preds = %67
  %90 = sext i32 %32 to i64
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %86, ptr addrspace(6) %87, i64 0, i64 1, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %90, i64 1, i64 1, i32 0)
  br label %91

91:                                               ; preds = %89, %67
  %92 = call i64 @llvm.hivm.GET.CTRL()
  %93 = call i64 @llvm.hivm.SBITSET1(i64 %92, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %93)
  %94 = add i64 %20, 1
  store i64 %94, ptr %11, align 4
  %95 = add nsw i32 %17, 56
  br label %16

96:                                               ; preds = %16
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #3

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32) #4

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #5

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #4

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
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #3

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #3 = { nounwind }
attributes #4 = { nounwind readnone  }
attributes #5 = { nounwind readonly  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nounwind  inaccessiblememonly }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn readnone  }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @reduction, !"kernel", i32 1}

