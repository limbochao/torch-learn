; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @exchange_fused_0_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1) #0 {
  %3 = call <256 x i1> @llvm.hivm.pge.b32(i32 2, i32 0)
  %4 = call <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) %0, i32 0, i32 0, i32 0)
  %5 = call <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) %1, i32 0, i32 3, i32 0)
  %6 = call <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v64i32(<64 x i32> %4, <256 x i1> %3)
  %7 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %8 = call <64 x i32> @llvm.hivm.vadd.s.x.v64i32(<64 x i32> %5, <64 x i32> %6, <256 x i1> %7)
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 1, i32 0)
  call void @llvm.hivm.vstsx1.v64s32(<64 x i32> %8, ptr addrspace(6) %1, i32 0, i32 5, i32 0, <256 x i1> %9)
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

16:                                               ; preds = %89, %9
  %17 = phi i32 [ %92, %89 ], [ %14, %9 ]
  %18 = icmp slt i32 %17, %12
  br i1 %18, label %19, label %93

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
  %29 = mul i32 %28, 56
  %30 = add i32 %26, %29
  %31 = srem i32 %24, 4
  %32 = mul i32 %31, 8
  br label %33

33:                                               ; preds = %37, %19
  %34 = phi i32 [ %39, %37 ], [ 0, %19 ]
  %35 = phi i32 [ %38, %37 ], [ 0, %19 ]
  %36 = icmp slt i32 %34, %32
  br i1 %36, label %37, label %40

37:                                               ; preds = %33
  %38 = add i32 %35, %34
  %39 = add nsw i32 %34, 1
  br label %33

40:                                               ; preds = %33
  %41 = add i32 %24, 1
  %42 = mul i32 %30, 1000
  %43 = add i32 %41, %42
  %44 = mul i32 %30, 2
  %45 = sext i32 %44 to i64
  %46 = sext i32 %24 to i64
  %47 = add i64 %45, %46
  br label %48

48:                                               ; preds = %51, %40
  %49 = phi i32 [ %88, %51 ], [ 0, %40 ]
  %50 = icmp slt i32 %49, 4
  br i1 %50, label %51, label %89

51:                                               ; preds = %48
  %52 = load i64, ptr %10, align 4
  %53 = urem i64 %52, 2
  %54 = icmp eq i64 %53, 1
  %55 = select i1 %54, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 160 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 160 to ptr addrspace(6)), i64 0 }, { ptr addrspace(6), ptr addrspace(6), i64 } { ptr addrspace(6) inttoptr (i64 64 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 64 to ptr addrspace(6)), i64 0 }
  %56 = select i1 %54, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 128 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2], [1 x i64] [i64 1] }
  %57 = select i1 %54, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 96 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 96 to ptr addrspace(6)), i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) null, ptr addrspace(6) null, i64 0, [1 x i64] [i64 1], [1 x i64] [i64 1] }
  %58 = trunc i64 %53 to i1
  %59 = select i1 %58, i64 2, i64 3
  %60 = xor i1 %58, true
  %61 = zext i1 %60 to i64
  %62 = mul i32 %49, 10000
  %63 = add i32 %43, %62
  %64 = add i32 %63, %15
  %65 = add i32 %64, %35
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 0, i64 %61)
  %66 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 1
  store i32 %65, ptr addrspace(6) %66, align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  %67 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 0
  %68 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 1
  %69 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 2
  %70 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 3, 0
  %71 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %57, 4, 0
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %67, ptr addrspace(6) %68, i64 %69, i64 %70, i64 %71, ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %47, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 0, i64 %61)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  %72 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 0
  %73 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 1
  %74 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 2
  %75 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 3, 0
  %76 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 4, 0
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %45, i64 2, i64 1, ptr addrspace(6) %72, ptr addrspace(6) %73, i64 %74, i64 %75, i64 %76, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  %77 = mul i32 %49, 56
  %78 = add i32 %77, %30
  %79 = mul i32 %78, 2
  %80 = sext i32 %79 to i64
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 0, i64 %59)
  %81 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %55, 1
  store i32 0, ptr addrspace(6) %81, align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 0)
  %82 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %56, 1
  %83 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %55, 1
  call ptc_simdvf void @exchange_fused_0_outlined_vf_0(ptr addrspace(6) %82, ptr addrspace(6) %83)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %84 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %55, 0
  %85 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64 } %55, 1
  %86 = add i64 %80, %46
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %84, ptr addrspace(6) %85, i64 0, i64 1, i64 1, ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %86, i64 1, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 0, i64 %59)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
  %87 = add i64 %52, 1
  store i64 %87, ptr %10, align 4
  %88 = add nsw i32 %49, 1
  br label %48

89:                                               ; preds = %48
  %90 = call i64 @llvm.hivm.GET.CTRL()
  %91 = call i64 @llvm.hivm.SBITSET1(i64 %90, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %91)
  %92 = add nsw i32 %17, 56
  br label %16

93:                                               ; preds = %16
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 0, i64 3)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #3

; Function Attrs: nounwind readonly 
declare <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #4

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v64i32(<64 x i32>, <256 x i1>) #5

; Function Attrs: nounwind readnone 
declare <64 x i32> @llvm.hivm.vadd.s.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64s32(<64 x i32>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #5

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

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

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64, i64) #7

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #3 = { nounwind }
attributes #4 = { nounwind readonly  }
attributes #5 = { nounwind readnone  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @exchange, !"kernel", i32 1}

