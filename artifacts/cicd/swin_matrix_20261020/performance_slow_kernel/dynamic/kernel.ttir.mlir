module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @triton_per_fused__unsafe_view_clone_native_lay_44(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<50176> : tensor<1x1x1x1xi64>
    %cst_0 = arith.constant dense<25088> : tensor<1x1x1x1xi64>
    %cst_1 = arith.constant dense<2> : tensor<1x1x1x1xi64>
    %cst_2 = arith.constant dense<3584> : tensor<1x1x1x1xi64>
    %cst_3 = arith.constant dense<512> : tensor<1x1x1x1xi64>
    %cst_4 = arith.constant dense<7> : tensor<1x1x1x1xi64>
    %cst_5 = arith.constant dense<14> : tensor<1x1x1x1xi64>
    %c14_i32 = arith.constant 14 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst_6 = arith.constant dense<0> : tensor<1x1x1x1xi64>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst_7 = arith.constant dense<5.120000e+02> : tensor<4x1x1x512xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<4x1x1x1xf32>
    %cst_9 = arith.constant dense<196> : tensor<4x1x1x1xi32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<4x1x1x512xf32>
    %cst_11 = arith.constant dense<100352> : tensor<4x1x1x1xi32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x1x1x512xf32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg10 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = arith.addi %arg10, %c3_i32 : i32
    %4 = arith.divsi %3, %c4_i32 : i32
    %5 = tt.get_program_id y : i32
    %6 = arith.muli %5, %arg11 : i32
    %7 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %8 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %9 = tt.expand_dims %8 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %10 = tt.expand_dims %9 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %11 = arith.addi %arg10, %1 : i32
    %12 = arith.minsi %11, %arg6 : i32
    %13 = tt.splat %12 : i32 -> tensor<4x1x1x1xi32>
    %14 = arith.addi %arg11, %6 : i32
    %15 = arith.minsi %14, %arg7 : i32
    %16 = tt.expand_dims %7 {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<1x512xi32> -> tensor<1x1x512xi32>
    %18 = tt.expand_dims %17 {axis = 2 : i32} : tensor<1x1x512xi32> -> tensor<1x1x1x512xi32>
    %19 = tt.splat %arg9 : i32 -> tensor<1x1x1x512xi32>
    %20 = arith.cmpi slt, %18, %19 : tensor<1x1x1x512xi32>
    %21 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x1x1x512x!tt.ptr<f32>>
    %22 = tt.addptr %21, %18 : tensor<1x1x1x512x!tt.ptr<f32>>, tensor<1x1x1x512xi32>
    %23 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x1x512x!tt.ptr<f32>>
    %24 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<4x1x1x1x!tt.ptr<f32>>
    %25 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<4x1x1x1x!tt.ptr<f32>>
    %26 = arith.extsi %18 : tensor<1x1x1x512xi32> to tensor<1x1x1x512xi64>
    %27 = tt.broadcast %26 : tensor<1x1x1x512xi64> -> tensor<4x1x1x512xi64>
    %28 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x1x1x512x!tt.ptr<f32>>
    scf.for %arg12 = %c0_i32 to %4 step %c1_i32  : i32 {
      %29 = arith.muli %arg12, %c4_i32 : i32
      %30 = arith.addi %1, %29 : i32
      %31 = tt.splat %30 : i32 -> tensor<4x1x1x1xi32>
      %32 = arith.addi %31, %10 : tensor<4x1x1x1xi32>
      %33 = arith.cmpi slt, %32, %13 : tensor<4x1x1x1xi32>
      %34 = arith.muli %32, %cst_11 : tensor<4x1x1x1xi32>
      %35 = tt.broadcast %34 : tensor<4x1x1x1xi32> -> tensor<4x1x1x512xi32>
      %36 = tt.broadcast %33 : tensor<4x1x1x1xi1> -> tensor<4x1x1x512xi1>
      %37 = arith.muli %32, %cst_9 : tensor<4x1x1x1xi32>
      %38 = arith.extsi %34 : tensor<4x1x1x1xi32> to tensor<4x1x1x1xi64>
      %39 = tt.broadcast %38 : tensor<4x1x1x1xi64> -> tensor<4x1x1x512xi64>
      scf.for %arg13 = %c0_i32 to %arg11 step %c1_i32  : i32 {
        %40 = arith.addi %6, %arg13 : i32
        %41 = arith.cmpi slt, %40, %15 : i32
        %42 = tt.splat %41 : i1 -> tensor<1x1x1x1xi1>
        %43 = tt.addptr %arg1, %40 : !tt.ptr<i64>, i32
        %44 = tt.splat %43 : !tt.ptr<i64> -> tensor<1x1x1x1x!tt.ptr<i64>>
        %45 = arith.muli %40, %c7168_i32 : i32
        %46 = tt.splat %45 : i32 -> tensor<1x1x1x512xi32>
        %47 = tt.splat %41 : i1 -> tensor<1x1x1x512xi1>
        %48 = arith.muli %40, %c14_i32 : i32
        scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32  : i32 {
          %49 = arith.cmpi slt, %arg14, %arg8 : i32
          %50 = tt.splat %49 : i1 -> tensor<1x1x1x1xi1>
          %51 = tt.load %44, %42, %cst_6 : tensor<1x1x1x1x!tt.ptr<i64>>
          %52 = tt.addptr %arg1, %arg14 : !tt.ptr<i64>, i32
          %53 = tt.splat %52 : !tt.ptr<i64> -> tensor<1x1x1x1x!tt.ptr<i64>>
          %54 = tt.load %53, %50, %cst_6 : tensor<1x1x1x1x!tt.ptr<i64>>
          %55 = tt.load %22, %20, %cst_12 : tensor<1x1x1x512x!tt.ptr<f32>>
          %56 = arith.muli %arg14, %c512_i32 : i32
          %57 = tt.splat %56 : i32 -> tensor<1x1x1x512xi32>
          %58 = arith.addi %18, %57 : tensor<1x1x1x512xi32>
          %59 = arith.addi %58, %46 : tensor<1x1x1x512xi32>
          %60 = tt.broadcast %59 : tensor<1x1x1x512xi32> -> tensor<4x1x1x512xi32>
          %61 = arith.addi %60, %35 : tensor<4x1x1x512xi32>
          %62 = tt.addptr %23, %61 : tensor<4x1x1x512x!tt.ptr<f32>>, tensor<4x1x1x512xi32>
          %63 = tt.splat %49 : i1 -> tensor<1x1x1x512xi1>
          %64 = arith.andi %20, %63 : tensor<1x1x1x512xi1>
          %65 = arith.andi %64, %47 : tensor<1x1x1x512xi1>
          %66 = tt.broadcast %65 : tensor<1x1x1x512xi1> -> tensor<4x1x1x512xi1>
          %67 = arith.andi %66, %36 : tensor<4x1x1x512xi1>
          %68 = tt.load %62, %67, %cst_10 : tensor<4x1x1x512x!tt.ptr<f32>>
          %69 = arith.addi %arg14, %48 : i32
          %70 = tt.splat %69 : i32 -> tensor<4x1x1x1xi32>
          %71 = arith.addi %70, %37 : tensor<4x1x1x1xi32>
          %72 = tt.addptr %24, %71 : tensor<4x1x1x1x!tt.ptr<f32>>, tensor<4x1x1x1xi32>
          %73 = arith.andi %49, %41 : i1
          %74 = tt.splat %73 : i1 -> tensor<4x1x1x1xi1>
          %75 = arith.andi %74, %33 : tensor<4x1x1x1xi1>
          %76 = tt.load %72, %75, %cst_8 : tensor<4x1x1x1x!tt.ptr<f32>>
          %77 = tt.addptr %25, %71 : tensor<4x1x1x1x!tt.ptr<f32>>, tensor<4x1x1x1xi32>
          %78 = tt.load %77, %75, %cst_8 : tensor<4x1x1x1x!tt.ptr<f32>>
          %79 = arith.addi %51, %cst_5 : tensor<1x1x1x1xi64>
          %80 = arith.cmpi slt, %51, %cst_6 : tensor<1x1x1x1xi64>
          %81 = arith.select %80, %79, %51 : tensor<1x1x1x1xi1>, tensor<1x1x1x1xi64>
          %82 = arith.addi %54, %cst_5 : tensor<1x1x1x1xi64>
          %83 = arith.cmpi slt, %54, %cst_6 : tensor<1x1x1x1xi64>
          %84 = arith.select %83, %82, %54 : tensor<1x1x1x1xi1>, tensor<1x1x1x1xi64>
          %85 = arith.remsi %84, %cst_4 : tensor<1x1x1x1xi64>
          %86 = arith.muli %85, %cst_3 : tensor<1x1x1x1xi64>
          %87 = tt.broadcast %86 : tensor<1x1x1x1xi64> -> tensor<4x1x1x512xi64>
          %88 = arith.addi %27, %87 : tensor<4x1x1x512xi64>
          %89 = arith.remsi %81, %cst_4 : tensor<1x1x1x1xi64>
          %90 = arith.muli %89, %cst_2 : tensor<1x1x1x1xi64>
          %91 = tt.broadcast %90 : tensor<1x1x1x1xi64> -> tensor<4x1x1x512xi64>
          %92 = arith.addi %88, %91 : tensor<4x1x1x512xi64>
          %93 = arith.divsi %84, %cst_4 : tensor<1x1x1x1xi64>
          %94 = arith.remsi %93, %cst_1 : tensor<1x1x1x1xi64>
          %95 = arith.muli %94, %cst_0 : tensor<1x1x1x1xi64>
          %96 = tt.broadcast %95 : tensor<1x1x1x1xi64> -> tensor<4x1x1x512xi64>
          %97 = arith.addi %92, %96 : tensor<4x1x1x512xi64>
          %98 = arith.divsi %81, %cst_4 : tensor<1x1x1x1xi64>
          %99 = arith.remsi %98, %cst_1 : tensor<1x1x1x1xi64>
          %100 = arith.muli %99, %cst : tensor<1x1x1x1xi64>
          %101 = tt.broadcast %100 : tensor<1x1x1x1xi64> -> tensor<4x1x1x512xi64>
          %102 = arith.addi %97, %101 : tensor<4x1x1x512xi64>
          %103 = arith.addi %102, %39 : tensor<4x1x1x512xi64>
          %104 = tt.addptr %28, %103 : tensor<4x1x1x512x!tt.ptr<f32>>, tensor<4x1x1x512xi64>
          %105 = tt.load %104, %67, %cst_10 : tensor<4x1x1x512x!tt.ptr<f32>>
          %106 = tt.broadcast %55 : tensor<1x1x1x512xf32> -> tensor<4x1x1x512xf32>
          %107 = arith.mulf %105, %106 : tensor<4x1x1x512xf32>
          %108 = "tt.reduce"(%107) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %123 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %123 : f32
          }) : (tensor<4x1x1x512xf32>) -> tensor<4x1x1xf32>
          %109 = tt.reshape %108 : tensor<4x1x1xf32> -> tensor<4x1x1x1xf32>
          %110 = tt.broadcast %76 : tensor<4x1x1x1xf32> -> tensor<4x1x1x512xf32>
          %111 = arith.subf %68, %110 : tensor<4x1x1x512xf32>
          %112 = tt.broadcast %78 : tensor<4x1x1x1xf32> -> tensor<4x1x1x512xf32>
          %113 = arith.mulf %111, %112 : tensor<4x1x1x512xf32>
          %114 = arith.mulf %107, %113 : tensor<4x1x1x512xf32>
          %115 = "tt.reduce"(%114) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %123 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %123 : f32
          }) : (tensor<4x1x1x512xf32>) -> tensor<4x1x1xf32>
          %116 = tt.reshape %115 : tensor<4x1x1xf32> -> tensor<4x1x1x1xf32>
          %117 = arith.mulf %107, %cst_7 : tensor<4x1x1x512xf32>
          %118 = tt.broadcast %109 : tensor<4x1x1x1xf32> -> tensor<4x1x1x512xf32>
          %119 = arith.subf %117, %118 : tensor<4x1x1x512xf32>
          %120 = tt.broadcast %116 : tensor<4x1x1x1xf32> -> tensor<4x1x1x512xf32>
          %121 = arith.mulf %113, %120 : tensor<4x1x1x512xf32>
          %122 = arith.subf %119, %121 : tensor<4x1x1x512xf32>
          tt.store %62, %122, %67 : tensor<4x1x1x512x!tt.ptr<f32>>
        }
      }
    }
    tt.return
  }
}
