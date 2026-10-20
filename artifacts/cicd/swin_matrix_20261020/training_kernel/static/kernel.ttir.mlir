module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @triton_per_fused__unsafe_view_clone_native_lay_42(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<50176> : tensor<1x2x1x1xi64>
    %cst_0 = arith.constant dense<2> : tensor<1x2x1x1xi64>
    %cst_1 = arith.constant dense<25088> : tensor<1x1x2x1xi64>
    %cst_2 = arith.constant dense<2> : tensor<1x1x2x1xi64>
    %cst_3 = arith.constant dense<3584> : tensor<1x2x1x1xi64>
    %cst_4 = arith.constant dense<7> : tensor<1x2x1x1xi64>
    %cst_5 = arith.constant dense<512> : tensor<1x1x2x1xi64>
    %cst_6 = arith.constant dense<7> : tensor<1x1x2x1xi64>
    %cst_7 = arith.constant dense<14> : tensor<1x1x2x1xi64>
    %cst_8 = arith.constant dense<14> : tensor<1x2x1x1xi64>
    %c196_i32 = arith.constant 196 : i32
    %c100352_i32 = arith.constant 100352 : i32
    %cst_9 = arith.constant dense<0> : tensor<1x1x2x1xi64>
    %cst_10 = arith.constant dense<0> : tensor<1x2x1x1xi64>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_11 = arith.constant dense<5.120000e+02> : tensor<1x2x2x512xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x2x2x1xf32>
    %cst_13 = arith.constant dense<14> : tensor<1x2x1x1xi32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<1x2x2x512xf32>
    %cst_15 = arith.constant dense<7168> : tensor<1x2x1x1xi32>
    %cst_16 = arith.constant dense<512> : tensor<1x1x2x1xi32>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<1x1x1x512xf32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg10 : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %2, %arg11 : i32
    %4 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %5 = arith.addi %arg11, %c1_i32 : i32
    %6 = arith.divsi %5, %c2_i32 : i32
    %7 = arith.addi %arg8, %c1_i32 : i32
    %8 = arith.divsi %7, %c2_i32 : i32
    %9 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %10 = arith.addi %arg10, %1 : i32
    %11 = arith.minsi %10, %arg6 : i32
    %12 = tt.expand_dims %4 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %13 = tt.expand_dims %12 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
    %14 = tt.expand_dims %13 {axis = 3 : i32} : tensor<1x2x1xi32> -> tensor<1x2x1x1xi32>
    %15 = arith.addi %arg11, %3 : i32
    %16 = arith.minsi %15, %arg7 : i32
    %17 = tt.splat %16 : i32 -> tensor<1x2x1x1xi32>
    %18 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
    %19 = tt.expand_dims %18 {axis = 3 : i32} : tensor<1x1x2xi32> -> tensor<1x1x2x1xi32>
    %20 = tt.splat %arg8 : i32 -> tensor<1x1x2x1xi32>
    %21 = tt.expand_dims %9 {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<1x512xi32> -> tensor<1x1x512xi32>
    %23 = tt.expand_dims %22 {axis = 2 : i32} : tensor<1x1x512xi32> -> tensor<1x1x1x512xi32>
    %24 = tt.splat %arg9 : i32 -> tensor<1x1x1x512xi32>
    %25 = arith.cmpi slt, %23, %24 : tensor<1x1x1x512xi32>
    %26 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x2x1x1x!tt.ptr<i64>>
    %27 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x1x2x1x!tt.ptr<i64>>
    %28 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x1x1x512x!tt.ptr<f32>>
    %29 = tt.addptr %28, %23 : tensor<1x1x1x512x!tt.ptr<f32>>, tensor<1x1x1x512xi32>
    %30 = tt.broadcast %23 : tensor<1x1x1x512xi32> -> tensor<1x1x2x512xi32>
    %31 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x2x2x512x!tt.ptr<f32>>
    %32 = tt.broadcast %25 : tensor<1x1x1x512xi1> -> tensor<1x1x2x512xi1>
    %33 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<1x2x2x1x!tt.ptr<f32>>
    %34 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x2x2x1x!tt.ptr<f32>>
    %35 = arith.extsi %23 : tensor<1x1x1x512xi32> to tensor<1x1x1x512xi64>
    %36 = tt.broadcast %35 : tensor<1x1x1x512xi64> -> tensor<1x2x2x512xi64>
    %37 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x2x2x512x!tt.ptr<f32>>
    scf.for %arg12 = %c0_i32 to %arg10 step %c1_i32  : i32 {
      %38 = arith.addi %1, %arg12 : i32
      %39 = arith.cmpi slt, %38, %11 : i32
      %40 = arith.muli %38, %c100352_i32 : i32
      %41 = tt.splat %40 : i32 -> tensor<1x2x2x512xi32>
      %42 = tt.splat %39 : i1 -> tensor<1x2x2x512xi1>
      %43 = arith.muli %38, %c196_i32 : i32
      %44 = tt.splat %43 : i32 -> tensor<1x2x2x1xi32>
      %45 = tt.splat %39 : i1 -> tensor<1x2x2x1xi1>
      %46 = arith.extsi %40 : i32 to i64
      %47 = tt.splat %46 : i64 -> tensor<1x2x2x512xi64>
      scf.for %arg13 = %c0_i32 to %6 step %c1_i32  : i32 {
        %48 = arith.muli %arg13, %c2_i32 : i32
        %49 = arith.addi %3, %48 : i32
        %50 = tt.splat %49 : i32 -> tensor<1x2x1x1xi32>
        %51 = arith.addi %50, %14 : tensor<1x2x1x1xi32>
        %52 = arith.cmpi slt, %51, %17 : tensor<1x2x1x1xi32>
        %53 = tt.addptr %26, %51 : tensor<1x2x1x1x!tt.ptr<i64>>, tensor<1x2x1x1xi32>
        %54 = arith.muli %51, %cst_15 : tensor<1x2x1x1xi32>
        %55 = tt.broadcast %54 : tensor<1x2x1x1xi32> -> tensor<1x2x2x512xi32>
        %56 = tt.broadcast %52 : tensor<1x2x1x1xi1> -> tensor<1x2x2x512xi1>
        %57 = arith.muli %51, %cst_13 : tensor<1x2x1x1xi32>
        %58 = tt.broadcast %57 : tensor<1x2x1x1xi32> -> tensor<1x2x2x1xi32>
        %59 = tt.broadcast %52 : tensor<1x2x1x1xi1> -> tensor<1x2x2x1xi1>
        scf.for %arg14 = %c0_i32 to %8 step %c1_i32  : i32 {
          %60 = arith.muli %arg14, %c2_i32 : i32
          %61 = tt.splat %60 : i32 -> tensor<1x1x2x1xi32>
          %62 = arith.addi %61, %19 : tensor<1x1x2x1xi32>
          %63 = arith.cmpi slt, %62, %20 : tensor<1x1x2x1xi32>
          %64 = tt.load %53, %52, %cst_10 : tensor<1x2x1x1x!tt.ptr<i64>>
          %65 = tt.addptr %27, %62 : tensor<1x1x2x1x!tt.ptr<i64>>, tensor<1x1x2x1xi32>
          %66 = tt.load %65, %63, %cst_9 : tensor<1x1x2x1x!tt.ptr<i64>>
          %67 = tt.load %29, %25, %cst_17 : tensor<1x1x1x512x!tt.ptr<f32>>
          %68 = arith.muli %62, %cst_16 : tensor<1x1x2x1xi32>
          %69 = tt.broadcast %68 : tensor<1x1x2x1xi32> -> tensor<1x1x2x512xi32>
          %70 = arith.addi %30, %69 : tensor<1x1x2x512xi32>
          %71 = tt.broadcast %70 : tensor<1x1x2x512xi32> -> tensor<1x2x2x512xi32>
          %72 = arith.addi %71, %55 : tensor<1x2x2x512xi32>
          %73 = arith.addi %72, %41 : tensor<1x2x2x512xi32>
          %74 = tt.addptr %31, %73 : tensor<1x2x2x512x!tt.ptr<f32>>, tensor<1x2x2x512xi32>
          %75 = tt.broadcast %63 : tensor<1x1x2x1xi1> -> tensor<1x1x2x512xi1>
          %76 = arith.andi %32, %75 : tensor<1x1x2x512xi1>
          %77 = tt.broadcast %76 : tensor<1x1x2x512xi1> -> tensor<1x2x2x512xi1>
          %78 = arith.andi %77, %56 : tensor<1x2x2x512xi1>
          %79 = arith.andi %78, %42 : tensor<1x2x2x512xi1>
          %80 = tt.load %74, %79, %cst_14 : tensor<1x2x2x512x!tt.ptr<f32>>
          %81 = tt.broadcast %62 : tensor<1x1x2x1xi32> -> tensor<1x2x2x1xi32>
          %82 = arith.addi %81, %58 : tensor<1x2x2x1xi32>
          %83 = arith.addi %82, %44 : tensor<1x2x2x1xi32>
          %84 = tt.addptr %33, %83 : tensor<1x2x2x1x!tt.ptr<f32>>, tensor<1x2x2x1xi32>
          %85 = tt.broadcast %63 : tensor<1x1x2x1xi1> -> tensor<1x2x2x1xi1>
          %86 = arith.andi %85, %59 : tensor<1x2x2x1xi1>
          %87 = arith.andi %86, %45 : tensor<1x2x2x1xi1>
          %88 = tt.load %84, %87, %cst_12 : tensor<1x2x2x1x!tt.ptr<f32>>
          %89 = tt.addptr %34, %83 : tensor<1x2x2x1x!tt.ptr<f32>>, tensor<1x2x2x1xi32>
          %90 = tt.load %89, %87, %cst_12 : tensor<1x2x2x1x!tt.ptr<f32>>
          %91 = arith.addi %64, %cst_8 : tensor<1x2x1x1xi64>
          %92 = arith.cmpi slt, %64, %cst_10 : tensor<1x2x1x1xi64>
          %93 = arith.select %92, %91, %64 : tensor<1x2x1x1xi1>, tensor<1x2x1x1xi64>
          %94 = arith.addi %66, %cst_7 : tensor<1x1x2x1xi64>
          %95 = arith.cmpi slt, %66, %cst_9 : tensor<1x1x2x1xi64>
          %96 = arith.select %95, %94, %66 : tensor<1x1x2x1xi1>, tensor<1x1x2x1xi64>
          %97 = arith.remsi %96, %cst_6 : tensor<1x1x2x1xi64>
          %98 = arith.muli %97, %cst_5 : tensor<1x1x2x1xi64>
          %99 = tt.broadcast %98 : tensor<1x1x2x1xi64> -> tensor<1x2x2x512xi64>
          %100 = arith.addi %36, %99 : tensor<1x2x2x512xi64>
          %101 = arith.remsi %93, %cst_4 : tensor<1x2x1x1xi64>
          %102 = arith.muli %101, %cst_3 : tensor<1x2x1x1xi64>
          %103 = tt.broadcast %102 : tensor<1x2x1x1xi64> -> tensor<1x2x2x512xi64>
          %104 = arith.addi %100, %103 : tensor<1x2x2x512xi64>
          %105 = arith.divsi %96, %cst_6 : tensor<1x1x2x1xi64>
          %106 = arith.remsi %105, %cst_2 : tensor<1x1x2x1xi64>
          %107 = arith.muli %106, %cst_1 : tensor<1x1x2x1xi64>
          %108 = tt.broadcast %107 : tensor<1x1x2x1xi64> -> tensor<1x2x2x512xi64>
          %109 = arith.addi %104, %108 : tensor<1x2x2x512xi64>
          %110 = arith.divsi %93, %cst_4 : tensor<1x2x1x1xi64>
          %111 = arith.remsi %110, %cst_0 : tensor<1x2x1x1xi64>
          %112 = arith.muli %111, %cst : tensor<1x2x1x1xi64>
          %113 = tt.broadcast %112 : tensor<1x2x1x1xi64> -> tensor<1x2x2x512xi64>
          %114 = arith.addi %109, %113 : tensor<1x2x2x512xi64>
          %115 = arith.addi %114, %47 : tensor<1x2x2x512xi64>
          %116 = tt.addptr %37, %115 : tensor<1x2x2x512x!tt.ptr<f32>>, tensor<1x2x2x512xi64>
          %117 = tt.load %116, %79, %cst_14 : tensor<1x2x2x512x!tt.ptr<f32>>
          %118 = tt.broadcast %67 : tensor<1x1x1x512xf32> -> tensor<1x2x2x512xf32>
          %119 = arith.mulf %117, %118 : tensor<1x2x2x512xf32>
          %120 = "tt.reduce"(%119) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %135 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %135 : f32
          }) : (tensor<1x2x2x512xf32>) -> tensor<1x2x2xf32>
          %121 = tt.reshape %120 : tensor<1x2x2xf32> -> tensor<1x2x2x1xf32>
          %122 = tt.broadcast %88 : tensor<1x2x2x1xf32> -> tensor<1x2x2x512xf32>
          %123 = arith.subf %80, %122 : tensor<1x2x2x512xf32>
          %124 = tt.broadcast %90 : tensor<1x2x2x1xf32> -> tensor<1x2x2x512xf32>
          %125 = arith.mulf %123, %124 : tensor<1x2x2x512xf32>
          %126 = arith.mulf %119, %125 : tensor<1x2x2x512xf32>
          %127 = "tt.reduce"(%126) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %135 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %135 : f32
          }) : (tensor<1x2x2x512xf32>) -> tensor<1x2x2xf32>
          %128 = tt.reshape %127 : tensor<1x2x2xf32> -> tensor<1x2x2x1xf32>
          %129 = arith.mulf %119, %cst_11 : tensor<1x2x2x512xf32>
          %130 = tt.broadcast %121 : tensor<1x2x2x1xf32> -> tensor<1x2x2x512xf32>
          %131 = arith.subf %129, %130 : tensor<1x2x2x512xf32>
          %132 = tt.broadcast %128 : tensor<1x2x2x1xf32> -> tensor<1x2x2x512xf32>
          %133 = arith.mulf %125, %132 : tensor<1x2x2x512xf32>
          %134 = arith.subf %131, %133 : tensor<1x2x2x512xf32>
          tt.store %74, %134, %79 : tensor<1x2x2x512x!tt.ptr<f32>>
        }
      }
    }
    tt.return
  }
}
