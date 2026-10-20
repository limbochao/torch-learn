module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @triton_per_fused__unsafe_view_clone_native_lay_42(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<50176> : tensor<1x1x1x1xi64>
    %cst_0 = arith.constant dense<2> : tensor<1x1x1x1xi64>
    %cst_1 = arith.constant dense<25088> : tensor<1x1x4x1xi64>
    %cst_2 = arith.constant dense<2> : tensor<1x1x4x1xi64>
    %cst_3 = arith.constant dense<3584> : tensor<1x1x1x1xi64>
    %cst_4 = arith.constant dense<7> : tensor<1x1x1x1xi64>
    %cst_5 = arith.constant dense<512> : tensor<1x1x4x1xi64>
    %cst_6 = arith.constant dense<7> : tensor<1x1x4x1xi64>
    %cst_7 = arith.constant dense<14> : tensor<1x1x4x1xi64>
    %cst_8 = arith.constant dense<14> : tensor<1x1x1x1xi64>
    %c196_i32 = arith.constant 196 : i32
    %c14_i32 = arith.constant 14 : i32
    %c100352_i32 = arith.constant 100352 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %cst_9 = arith.constant dense<0> : tensor<1x1x4x1xi64>
    %cst_10 = arith.constant dense<0> : tensor<1x1x1x1xi64>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst_11 = arith.constant dense<5.120000e+02> : tensor<1x1x4x512xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x1x4x1xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<1x1x4x512xf32>
    %cst_14 = arith.constant dense<512> : tensor<1x1x4x1xi32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<1x1x1x512xf32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg10 : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %2, %arg11 : i32
    %4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %5 = arith.addi %arg8, %c3_i32 : i32
    %6 = arith.divsi %5, %c4_i32 : i32
    %7 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %8 = arith.addi %arg10, %1 : i32
    %9 = arith.minsi %8, %arg6 : i32
    %10 = arith.addi %arg11, %3 : i32
    %11 = arith.minsi %10, %arg7 : i32
    %12 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.expand_dims %13 {axis = 3 : i32} : tensor<1x1x4xi32> -> tensor<1x1x4x1xi32>
    %15 = tt.splat %arg8 : i32 -> tensor<1x1x4x1xi32>
    %16 = tt.expand_dims %7 {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<1x512xi32> -> tensor<1x1x512xi32>
    %18 = tt.expand_dims %17 {axis = 2 : i32} : tensor<1x1x512xi32> -> tensor<1x1x1x512xi32>
    %19 = tt.splat %arg9 : i32 -> tensor<1x1x1x512xi32>
    %20 = arith.cmpi slt, %18, %19 : tensor<1x1x1x512xi32>
    %21 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x1x4x1x!tt.ptr<i64>>
    %22 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x1x1x512x!tt.ptr<f32>>
    %23 = tt.addptr %22, %18 : tensor<1x1x1x512x!tt.ptr<f32>>, tensor<1x1x1x512xi32>
    %24 = tt.broadcast %18 : tensor<1x1x1x512xi32> -> tensor<1x1x4x512xi32>
    %25 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1x4x512x!tt.ptr<f32>>
    %26 = tt.broadcast %20 : tensor<1x1x1x512xi1> -> tensor<1x1x4x512xi1>
    %27 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<1x1x4x1x!tt.ptr<f32>>
    %28 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x1x4x1x!tt.ptr<f32>>
    %29 = arith.extsi %18 : tensor<1x1x1x512xi32> to tensor<1x1x1x512xi64>
    %30 = tt.broadcast %29 : tensor<1x1x1x512xi64> -> tensor<1x1x4x512xi64>
    %31 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x1x4x512x!tt.ptr<f32>>
    scf.for %arg12 = %c0_i32 to %arg10 step %c1_i32  : i32 {
      %32 = arith.addi %1, %arg12 : i32
      %33 = arith.cmpi slt, %32, %9 : i32
      %34 = arith.muli %32, %c100352_i32 : i32
      %35 = tt.splat %34 : i32 -> tensor<1x1x4x512xi32>
      %36 = tt.splat %33 : i1 -> tensor<1x1x4x512xi1>
      %37 = arith.muli %32, %c196_i32 : i32
      %38 = tt.splat %37 : i32 -> tensor<1x1x4x1xi32>
      %39 = tt.splat %33 : i1 -> tensor<1x1x4x1xi1>
      %40 = arith.extsi %34 : i32 to i64
      %41 = tt.splat %40 : i64 -> tensor<1x1x4x512xi64>
      scf.for %arg13 = %c0_i32 to %arg11 step %c1_i32  : i32 {
        %42 = arith.addi %3, %arg13 : i32
        %43 = arith.cmpi slt, %42, %11 : i32
        %44 = tt.splat %43 : i1 -> tensor<1x1x1x1xi1>
        %45 = tt.addptr %arg1, %42 : !tt.ptr<i64>, i32
        %46 = tt.splat %45 : !tt.ptr<i64> -> tensor<1x1x1x1x!tt.ptr<i64>>
        %47 = arith.muli %42, %c7168_i32 : i32
        %48 = tt.splat %47 : i32 -> tensor<1x1x4x512xi32>
        %49 = tt.splat %43 : i1 -> tensor<1x1x4x512xi1>
        %50 = arith.muli %42, %c14_i32 : i32
        %51 = tt.splat %50 : i32 -> tensor<1x1x4x1xi32>
        %52 = tt.splat %43 : i1 -> tensor<1x1x4x1xi1>
        scf.for %arg14 = %c0_i32 to %6 step %c1_i32  : i32 {
          %53 = arith.muli %arg14, %c4_i32 : i32
          %54 = tt.splat %53 : i32 -> tensor<1x1x4x1xi32>
          %55 = arith.addi %54, %14 : tensor<1x1x4x1xi32>
          %56 = arith.cmpi slt, %55, %15 : tensor<1x1x4x1xi32>
          %57 = tt.load %46, %44, %cst_10 : tensor<1x1x1x1x!tt.ptr<i64>>
          %58 = tt.addptr %21, %55 : tensor<1x1x4x1x!tt.ptr<i64>>, tensor<1x1x4x1xi32>
          %59 = tt.load %58, %56, %cst_9 : tensor<1x1x4x1x!tt.ptr<i64>>
          %60 = tt.load %23, %20, %cst_15 : tensor<1x1x1x512x!tt.ptr<f32>>
          %61 = arith.muli %55, %cst_14 : tensor<1x1x4x1xi32>
          %62 = tt.broadcast %61 : tensor<1x1x4x1xi32> -> tensor<1x1x4x512xi32>
          %63 = arith.addi %24, %62 : tensor<1x1x4x512xi32>
          %64 = arith.addi %63, %48 : tensor<1x1x4x512xi32>
          %65 = arith.addi %64, %35 : tensor<1x1x4x512xi32>
          %66 = tt.addptr %25, %65 : tensor<1x1x4x512x!tt.ptr<f32>>, tensor<1x1x4x512xi32>
          %67 = tt.broadcast %56 : tensor<1x1x4x1xi1> -> tensor<1x1x4x512xi1>
          %68 = arith.andi %26, %67 : tensor<1x1x4x512xi1>
          %69 = arith.andi %68, %49 : tensor<1x1x4x512xi1>
          %70 = arith.andi %69, %36 : tensor<1x1x4x512xi1>
          %71 = tt.load %66, %70, %cst_13 : tensor<1x1x4x512x!tt.ptr<f32>>
          %72 = arith.addi %55, %51 : tensor<1x1x4x1xi32>
          %73 = arith.addi %72, %38 : tensor<1x1x4x1xi32>
          %74 = tt.addptr %27, %73 : tensor<1x1x4x1x!tt.ptr<f32>>, tensor<1x1x4x1xi32>
          %75 = arith.andi %56, %52 : tensor<1x1x4x1xi1>
          %76 = arith.andi %75, %39 : tensor<1x1x4x1xi1>
          %77 = tt.load %74, %76, %cst_12 : tensor<1x1x4x1x!tt.ptr<f32>>
          %78 = tt.addptr %28, %73 : tensor<1x1x4x1x!tt.ptr<f32>>, tensor<1x1x4x1xi32>
          %79 = tt.load %78, %76, %cst_12 : tensor<1x1x4x1x!tt.ptr<f32>>
          %80 = arith.addi %57, %cst_8 : tensor<1x1x1x1xi64>
          %81 = arith.cmpi slt, %57, %cst_10 : tensor<1x1x1x1xi64>
          %82 = arith.select %81, %80, %57 : tensor<1x1x1x1xi1>, tensor<1x1x1x1xi64>
          %83 = arith.addi %59, %cst_7 : tensor<1x1x4x1xi64>
          %84 = arith.cmpi slt, %59, %cst_9 : tensor<1x1x4x1xi64>
          %85 = arith.select %84, %83, %59 : tensor<1x1x4x1xi1>, tensor<1x1x4x1xi64>
          %86 = arith.remsi %85, %cst_6 : tensor<1x1x4x1xi64>
          %87 = arith.muli %86, %cst_5 : tensor<1x1x4x1xi64>
          %88 = tt.broadcast %87 : tensor<1x1x4x1xi64> -> tensor<1x1x4x512xi64>
          %89 = arith.addi %30, %88 : tensor<1x1x4x512xi64>
          %90 = arith.remsi %82, %cst_4 : tensor<1x1x1x1xi64>
          %91 = arith.muli %90, %cst_3 : tensor<1x1x1x1xi64>
          %92 = tt.broadcast %91 : tensor<1x1x1x1xi64> -> tensor<1x1x4x512xi64>
          %93 = arith.addi %89, %92 : tensor<1x1x4x512xi64>
          %94 = arith.divsi %85, %cst_6 : tensor<1x1x4x1xi64>
          %95 = arith.remsi %94, %cst_2 : tensor<1x1x4x1xi64>
          %96 = arith.muli %95, %cst_1 : tensor<1x1x4x1xi64>
          %97 = tt.broadcast %96 : tensor<1x1x4x1xi64> -> tensor<1x1x4x512xi64>
          %98 = arith.addi %93, %97 : tensor<1x1x4x512xi64>
          %99 = arith.divsi %82, %cst_4 : tensor<1x1x1x1xi64>
          %100 = arith.remsi %99, %cst_0 : tensor<1x1x1x1xi64>
          %101 = arith.muli %100, %cst : tensor<1x1x1x1xi64>
          %102 = tt.broadcast %101 : tensor<1x1x1x1xi64> -> tensor<1x1x4x512xi64>
          %103 = arith.addi %98, %102 : tensor<1x1x4x512xi64>
          %104 = arith.addi %103, %41 : tensor<1x1x4x512xi64>
          %105 = tt.addptr %31, %104 : tensor<1x1x4x512x!tt.ptr<f32>>, tensor<1x1x4x512xi64>
          %106 = tt.load %105, %70, %cst_13 : tensor<1x1x4x512x!tt.ptr<f32>>
          %107 = tt.broadcast %60 : tensor<1x1x1x512xf32> -> tensor<1x1x4x512xf32>
          %108 = arith.mulf %106, %107 : tensor<1x1x4x512xf32>
          %109 = "tt.reduce"(%108) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %124 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %124 : f32
          }) : (tensor<1x1x4x512xf32>) -> tensor<1x1x4xf32>
          %110 = tt.reshape %109 : tensor<1x1x4xf32> -> tensor<1x1x4x1xf32>
          %111 = tt.broadcast %77 : tensor<1x1x4x1xf32> -> tensor<1x1x4x512xf32>
          %112 = arith.subf %71, %111 : tensor<1x1x4x512xf32>
          %113 = tt.broadcast %79 : tensor<1x1x4x1xf32> -> tensor<1x1x4x512xf32>
          %114 = arith.mulf %112, %113 : tensor<1x1x4x512xf32>
          %115 = arith.mulf %108, %114 : tensor<1x1x4x512xf32>
          %116 = "tt.reduce"(%115) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %124 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %124 : f32
          }) : (tensor<1x1x4x512xf32>) -> tensor<1x1x4xf32>
          %117 = tt.reshape %116 : tensor<1x1x4xf32> -> tensor<1x1x4x1xf32>
          %118 = arith.mulf %108, %cst_11 : tensor<1x1x4x512xf32>
          %119 = tt.broadcast %110 : tensor<1x1x4x1xf32> -> tensor<1x1x4x512xf32>
          %120 = arith.subf %118, %119 : tensor<1x1x4x512xf32>
          %121 = tt.broadcast %117 : tensor<1x1x4x1xf32> -> tensor<1x1x4x512xf32>
          %122 = arith.mulf %114, %121 : tensor<1x1x4x512xf32>
          %123 = arith.subf %120, %122 : tensor<1x1x4x512xf32>
          tt.store %66, %123, %70 : tensor<1x1x4x512x!tt.ptr<f32>>
        }
      }
    }
    tt.return
  }
}
