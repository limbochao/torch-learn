; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @exchange_fused_0_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
  %3 = call <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) %1, i32 0, i32 3, i32 0)
  br label %4

4:                                                ; preds = %8, %2
  %5 = phi i32 [ %16, %8 ], [ 0, %2 ]
  %6 = phi <64 x i32> [ %15, %8 ], [ %3, %2 ]
  %7 = icmp slt i32 %5, 128
  br i1 %7, label %8, label %17

8:                                                ; preds = %4
  %9 = sext i32 %5 to i64
  %10 = getelementptr i32, ptr addrspace(6) %0, i64 %9
  %11 = call <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) %10, i32 0, i32 0, i32 0)
  %12 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %13 = call <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v64i32(<64 x i32> %11, <256 x i1> %12)
  %14 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %15 = call <64 x i32> @llvm.hivm.vadd.s.x.v64i32(<64 x i32> %6, <64 x i32> %13, <256 x i1> %14)
  %16 = add nsw i32 %5, 64
  br label %4

17:                                               ; preds = %4
  %18 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  call void @llvm.hivm.vstsx1.v64s32(<64 x i32> %6, ptr addrspace(6) %1, i32 0, i32 5, i32 0, <256 x i1> %18)
  ret void
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @exchange_outlined_vf_0(ptr addrspace(6) noalias %0) #0 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %2, i32 1)
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp slt i32 %5, 128
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %10 = getelementptr i32, ptr addrspace(6) %0, i64 %8
  call void @llvm.hivm.vstsx1.v64s32(<64 x i32> %3, ptr addrspace(6) %10, i32 0, i32 2, i32 0, <256 x i1> %9)
  %11 = add nsw i32 %5, 64
  br label %4

12:                                               ; preds = %4
  ret void
}

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

define dso_local void @exchange(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8) #2 {
  %10 = alloca i64, i64 1, align 8
  store i64 0, ptr %10, align 4
  %11 = mul i32 %6, %7
  %12 = mul i32 %11, %8
  %13 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %14 = trunc i64 %13 to i32
  %15 = mul i32 %5, 1000
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 0, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 0, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 0, i64 3)
  br label %16

16:                                               ; preds = %85, %9
  %17 = phi i32 [ %88, %85 ], [ %14, %9 ]
  %18 = icmp slt i32 %17, %12
  br i1 %18, label %19, label %89

19:                                               ; preds = %16
  %20 = call i64 @llvm.hivm.GET.CTRL()
  %21 = call i64 @llvm.hivm.SBITSET0(i64 %20, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %21)
  %22 = call i64 @llvm.hivm.GET.CTRL()
  %23 = call i64 @llvm.hivm.SBITSET1(i64 %22, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %23)
  %24 = srem i32 %17, %6
  %25 = sdiv i32 %17, %6
  %26 = srem i32 %25, %7
  %27 = sdiv i32 %17, %11
  %28 = srem i32 %27, %8
  %29 = add i32 %26, %28
  %30 = srem i32 %24, 4
  %31 = mul i32 %30, 8
  br label %32

32:                                               ; preds = %36, %19
  %33 = phi i32 [ %38, %36 ], [ 0, %19 ]
  %34 = phi i32 [ %37, %36 ], [ 0, %19 ]
  %35 = icmp slt i32 %33, %31
  br i1 %35, label %36, label %39

36:                                               ; preds = %32
  %37 = add i32 %34, %33
  %38 = add nsw i32 %33, 1
  br label %32

39:                                               ; preds = %32
  %40 = add i32 %24, 1
  %41 = mul i32 %29, 1000
  %42 = add i32 %40, %41
  %43 = mul i32 %29, 112
  %44 = sext i32 %43 to i64
  %45 = sext i32 %24 to i64
  %46 = add i64 %44, %45
  br label %47

47:                                               ; preds = %50, %39
  %48 = phi i32 [ %84, %50 ], [ 0, %39 ]
  %49 = icmp slt i32 %48, 16
  br i1 %49, label %50, label %85

50:                                               ; preds = %47
  %51 = load i64, ptr %10, align 4
  %52 = urem i64 %51, 2
  %53 = icmp eq i64 %52, 1
  %54 = select i1 %53, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 1120 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 1120 to ptr addrspace(6)), i64 0 }, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 544 to ptr addrspace(6)), i64 0 }
  %55 = select i1 %53, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 608 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 608 to ptr addrspace(6)), i64 0, [1 x i64] [i64 128], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, [1 x i64] [i64 128], [1 x i64] [i64 1] }
  %56 = select i1 %53, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 576 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 576 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) null, ptr addrspace(6) null, i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %57 = trunc i64 %52 to i1
  %58 = select i1 %57, i64 2, i64 3
  %59 = xor i1 %57, true
  %60 = zext i1 %59 to i64
  %61 = mul i32 %48, 10000
  %62 = add i32 %42, %61
  %63 = add i32 %62, %15
  %64 = add i32 %63, %34
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 0, i64 %60)
  %65 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 1
  store i32 %64, ptr addrspace(6) %65, align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  %66 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 0
  %67 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 1
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 2
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 3, 0
  %70 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 4, 0
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %66, ptr addrspace(6) %67, i64 %68, i64 %69, i64 %70, ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %46, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 0, i64 %60)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  %71 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, 1
  call ptc_simdvf void @exchange_outlined_vf_0(ptr addrspace(6) %71)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  %72 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, 0
  %73 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, 1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %44, i64 112, i64 1, ptr addrspace(6) %72, ptr addrspace(6) %73, i64 0, i64 112, i64 1, i32 2, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  %74 = add i32 %48, %29
  %75 = mul i32 %74, 112
  %76 = sext i32 %75 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 0, i64 %58)
  %77 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %54, 1
  store i32 0, ptr addrspace(6) %77, align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 0)
  %78 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, 1
  %79 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %54, 1
  call ptc_simdvf void @exchange_fused_0_outlined_vf_0(ptr addrspace(6) %78, ptr addrspace(6) %79)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %80 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %54, 0
  %81 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %54, 1
  %82 = add i64 %76, %45
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %80, ptr addrspace(6) %81, i64 0, i64 1, i64 1, ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %82, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 0, i64 %58)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  %83 = add i64 %51, 1
  store i64 %83, ptr %10, align 4
  %84 = add nsw i32 %48, 1
  br label %47

85:                                               ; preds = %47
  %86 = call i64 @llvm.hivm.GET.CTRL()
  %87 = call i64 @llvm.hivm.SBITSET1(i64 %86, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %87)
  %88 = add nsw i32 %17, 56
  br label %16

89:                                               ; preds = %16
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 3)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind readonly 
declare <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #3

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64s32(<64 x i32>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v64i32(<64 x i32>, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vadd.s.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32) #6

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #6

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #6

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #6

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64, i64) #7

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #3 = { nounwind readonly  }
attributes #4 = { nounwind }
attributes #5 = { nounwind writeonly  }
attributes #6 = { nounwind readnone  }
attributes #7 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @exchange, !"kernel", i32 1}

