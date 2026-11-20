; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noinline
define private void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #0 {
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
declare dso_local void @_mlir_ciface_sync_block_lock_unordered(ptr) #0

; Function Attrs: noinline
define private void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, i32 %11, i64 %12, i32 %13) #0 {
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
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_int32_t_nocache(ptr, ptr, i32, i32, i64, i32) #0

; Function Attrs: noinline
define private void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
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
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_int32_t(ptr, ptr, i32) #0

; Function Attrs: noinline
define private void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #0 {
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
declare dso_local void @_mlir_ciface_sync_block_unlock_unordered(ptr) #0

; Function Attrs: noinline
define private void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4) #0 {
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
declare dso_local void @_mlir_ciface_free_lock_var_unordered(ptr) #0

define dso_local void @atomic_component(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, i32 %5, i32 %6) #1 {
  %8 = call i64 @llvm.hivm.GET.CTRL()
  %9 = call i64 @llvm.hivm.SBITSET0(i64 %8, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %9)
  %10 = call i64 @llvm.hivm.GET.CTRL()
  %11 = call i64 @llvm.hivm.SBITSET1(i64 %10, i64 48)
  call void @llvm.hivm.SET.CTRL(i64 %11)
  %12 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %13 = trunc i64 %12 to i32
  %14 = srem i32 %13, %4
  store i32 1, ptr addrspace(6) null, align 4
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @sync_block_lock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  call void @load_gm_to_ubuf_1d_int32_t_nocache(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 0, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, i64 1, i64 1, i32 0, i32 0, i64 0, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %2, i64 0, i64 1, i64 1, i32 1)
  call void @sync_block_unlock_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  %15 = sext i32 %14 to i64
  call void @store_ubuf_to_gm_1d_int32_t(ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32 to ptr addrspace(6)), i64 0, i64 1, i64 1, ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %15, i64 1, i64 1, i32 0)
  %16 = call i64 @llvm.hivm.GET.CTRL()
  %17 = call i64 @llvm.hivm.SBITSET1(i64 %16, i64 60)
  call void @llvm.hivm.SET.CTRL(i64 %17)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @free_lock_var_unordered(ptr addrspace(1) %0, ptr addrspace(1) %0, i64 0, i64 1, i64 1)
  ret void
}

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #2

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #2

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #3

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #2

attributes #0 = { noinline "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #1 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" "vec-only-kernel" }
attributes #2 = { nounwind  inaccessiblememonly }
attributes #3 = { nounwind readnone  }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @atomic_component, !"kernel", i32 1}

