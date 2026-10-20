module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @triton_per_fused__unsafe_view_clone_native_lay_44(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<50176> : tensor<1x2x1x1xi64>
    %cst_0 = arith.constant dense<2> : tensor<1x2x1x1xi64>
    %cst_1 = arith.constant dense<25088> : tensor<1x1x1x1xi64>
    %cst_2 = arith.constant dense<2> : tensor<1x1x1x1xi64>
    %cst_3 = arith.constant dense<3584> : tensor<1x2x1x1xi64>
    %cst_4 = arith.constant dense<7> : tensor<1x2x1x1xi64>
    %cst_5 = arith.constant dense<512> : tensor<1x1x1x1xi64>
    %cst_6 = arith.constant dense<7> : tensor<1x1x1x1xi64>
    %cst_7 = arith.constant dense<14> : tensor<1x1x1x1xi64>
    %cst_8 = arith.constant dense<14> : tensor<1x2x1x1xi64>
    %c512_i32 = arith.constant 512 : i32
    %cst_9 = arith.constant dense<0> : tensor<1x1x1x1xi64>
    %cst_10 = arith.constant dense<0> : tensor<1x2x1x1xi64>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_11 = arith.constant dense<5.120000e+02> : tensor<2x2x1x512xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<2x2x1x1xf32>
    %cst_13 = arith.constant dense<196> : tensor<2x1x1x1xi32>
    %cst_14 = arith.constant dense<14> : tensor<1x2x1x1xi32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<2x2x1x512xf32>
    %cst_16 = arith.constant dense<100352> : tensor<2x1x1x1xi32>
    %cst_17 = arith.constant dense<7168> : tensor<1x2x1x1xi32>
    %cst_18 = arith.constant dense<0.000000e+00> : tensor<1x1x1x512xf32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg10 : i32
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %3 = arith.addi %arg10, %c1_i32 : i32
    %4 = arith.divsi %3, %c2_i32 : i32
    %5 = tt.get_program_id y : i32
    %6 = arith.muli %5, %arg11 : i32
    %7 = arith.addi %arg11, %c1_i32 : i32
    %8 = arith.divsi %7, %c2_i32 : i32
    %9 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %10 = tt.expand_dims %2 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %11 = tt.expand_dims %10 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %12 = tt.expand_dims %11 {axis = 3 : i32} : tensor<2x1x1xi32> -> tensor<2x1x1x1xi32>
    %13 = arith.addi %arg10, %1 : i32
    %14 = arith.minsi %13, %arg6 : i32
    %15 = tt.splat %14 : i32 -> tensor<2x1x1x1xi32>
    %16 = tt.expand_dims %2 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %17 = tt.expand_dims %16 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
    %18 = tt.expand_dims %17 {axis = 3 : i32} : tensor<1x2x1xi32> -> tensor<1x2x1x1xi32>
    %19 = arith.addi %arg11, %6 : i32
    %20 = arith.minsi %19, %arg7 : i32
    %21 = tt.splat %20 : i32 -> tensor<1x2x1x1xi32>
    %22 = tt.expand_dims %9 {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<1x512xi32> -> tensor<1x1x512xi32>
    %24 = tt.expand_dims %23 {axis = 2 : i32} : tensor<1x1x512xi32> -> tensor<1x1x1x512xi32>
    %25 = tt.splat %arg9 : i32 -> tensor<1x1x1x512xi32>
    %26 = arith.cmpi slt, %24, %25 : tensor<1x1x1x512xi32>
    %27 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x2x1x1x!tt.ptr<i64>>
    %28 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x1x1x512x!tt.ptr<f32>>
    %29 = tt.addptr %28, %24 : tensor<1x1x1x512x!tt.ptr<f32>>, tensor<1x1x1x512xi32>
    %30 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x1x512x!tt.ptr<f32>>
    %31 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<2x2x1x1x!tt.ptr<f32>>
    %32 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<2x2x1x1x!tt.ptr<f32>>
    %33 = arith.extsi %24 : tensor<1x1x1x512xi32> to tensor<1x1x1x512xi64>
    %34 = tt.broadcast %33 : tensor<1x1x1x512xi64> -> tensor<2x2x1x512xi64>
    %35 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x2x1x512x!tt.ptr<f32>>
    scf.for %arg12 = %c0_i32 to %4 step %c1_i32  : i32 {
      %36 = arith.muli %arg12, %c2_i32 : i32
      %37 = arith.addi %1, %36 : i32
      %38 = tt.splat %37 : i32 -> tensor<2x1x1x1xi32>
      %39 = arith.addi %38, %12 : tensor<2x1x1x1xi32>
      %40 = arith.cmpi slt, %39, %15 : tensor<2x1x1x1xi32>
      %41 = arith.muli %39, %cst_16 : tensor<2x1x1x1xi32>
      %42 = tt.broadcast %41 : tensor<2x1x1x1xi32> -> tensor<2x2x1x512xi32>
      %43 = tt.broadcast %40 : tensor<2x1x1x1xi1> -> tensor<2x2x1x512xi1>
      %44 = arith.muli %39, %cst_13 : tensor<2x1x1x1xi32>
      %45 = tt.broadcast %44 : tensor<2x1x1x1xi32> -> tensor<2x2x1x1xi32>
      %46 = tt.broadcast %40 : tensor<2x1x1x1xi1> -> tensor<2x2x1x1xi1>
      %47 = arith.extsi %41 : tensor<2x1x1x1xi32> to tensor<2x1x1x1xi64>
      %48 = tt.broadcast %47 : tensor<2x1x1x1xi64> -> tensor<2x2x1x512xi64>
      scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
        %49 = arith.muli %arg13, %c2_i32 : i32
        %50 = arith.addi %6, %49 : i32
        %51 = tt.splat %50 : i32 -> tensor<1x2x1x1xi32>
        %52 = arith.addi %51, %18 : tensor<1x2x1x1xi32>
        %53 = arith.cmpi slt, %52, %21 : tensor<1x2x1x1xi32>
        %54 = tt.addptr %27, %52 : tensor<1x2x1x1x!tt.ptr<i64>>, tensor<1x2x1x1xi32>
        %55 = arith.muli %52, %cst_17 : tensor<1x2x1x1xi32>
        %56 = tt.broadcast %55 : tensor<1x2x1x1xi32> -> tensor<1x2x1x512xi32>
        %57 = tt.broadcast %53 : tensor<1x2x1x1xi1> -> tensor<1x2x1x512xi1>
        %58 = arith.muli %52, %cst_14 : tensor<1x2x1x1xi32>
        scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32  : i32 {
          %59 = arith.cmpi slt, %arg14, %arg8 : i32
          %60 = tt.splat %59 : i1 -> tensor<1x1x1x1xi1>
          %61 = tt.load %54, %53, %cst_10 : tensor<1x2x1x1x!tt.ptr<i64>>
          %62 = tt.addptr %arg1, %arg14 : !tt.ptr<i64>, i32
          %63 = tt.splat %62 : !tt.ptr<i64> -> tensor<1x1x1x1x!tt.ptr<i64>>
          %64 = tt.load %63, %60, %cst_9 : tensor<1x1x1x1x!tt.ptr<i64>>
          %65 = tt.load %29, %26, %cst_18 : tensor<1x1x1x512x!tt.ptr<f32>>
          %66 = arith.muli %arg14, %c512_i32 : i32
          %67 = tt.splat %66 : i32 -> tensor<1x1x1x512xi32>
          %68 = arith.addi %24, %67 : tensor<1x1x1x512xi32>
          %69 = tt.broadcast %68 : tensor<1x1x1x512xi32> -> tensor<1x2x1x512xi32>
          %70 = arith.addi %69, %56 : tensor<1x2x1x512xi32>
          %71 = tt.broadcast %70 : tensor<1x2x1x512xi32> -> tensor<2x2x1x512xi32>
          %72 = arith.addi %71, %42 : tensor<2x2x1x512xi32>
          %73 = tt.addptr %30, %72 : tensor<2x2x1x512x!tt.ptr<f32>>, tensor<2x2x1x512xi32>
          %74 = tt.splat %59 : i1 -> tensor<1x1x1x512xi1>
          %75 = arith.andi %26, %74 : tensor<1x1x1x512xi1>
          %76 = tt.broadcast %75 : tensor<1x1x1x512xi1> -> tensor<1x2x1x512xi1>
          %77 = arith.andi %76, %57 : tensor<1x2x1x512xi1>
          %78 = tt.broadcast %77 : tensor<1x2x1x512xi1> -> tensor<2x2x1x512xi1>
          %79 = arith.andi %78, %43 : tensor<2x2x1x512xi1>
          %80 = tt.load %73, %79, %cst_15 : tensor<2x2x1x512x!tt.ptr<f32>>
          %81 = tt.splat %arg14 : i32 -> tensor<1x2x1x1xi32>
          %82 = arith.addi %81, %58 : tensor<1x2x1x1xi32>
          %83 = tt.broadcast %82 : tensor<1x2x1x1xi32> -> tensor<2x2x1x1xi32>
          %84 = arith.addi %83, %45 : tensor<2x2x1x1xi32>
          %85 = tt.addptr %31, %84 : tensor<2x2x1x1x!tt.ptr<f32>>, tensor<2x2x1x1xi32>
          %86 = tt.splat %59 : i1 -> tensor<1x2x1x1xi1>
          %87 = arith.andi %86, %53 : tensor<1x2x1x1xi1>
          %88 = tt.broadcast %87 : tensor<1x2x1x1xi1> -> tensor<2x2x1x1xi1>
          %89 = arith.andi %88, %46 : tensor<2x2x1x1xi1>
          %90 = tt.load %85, %89, %cst_12 : tensor<2x2x1x1x!tt.ptr<f32>>
          %91 = tt.addptr %32, %84 : tensor<2x2x1x1x!tt.ptr<f32>>, tensor<2x2x1x1xi32>
          %92 = tt.load %91, %89, %cst_12 : tensor<2x2x1x1x!tt.ptr<f32>>
          %93 = arith.addi %61, %cst_8 : tensor<1x2x1x1xi64>
          %94 = arith.cmpi slt, %61, %cst_10 : tensor<1x2x1x1xi64>
          %95 = arith.select %94, %93, %61 : tensor<1x2x1x1xi1>, tensor<1x2x1x1xi64>
          %96 = arith.addi %64, %cst_7 : tensor<1x1x1x1xi64>
          %97 = arith.cmpi slt, %64, %cst_9 : tensor<1x1x1x1xi64>
          %98 = arith.select %97, %96, %64 : tensor<1x1x1x1xi1>, tensor<1x1x1x1xi64>
          %99 = arith.remsi %98, %cst_6 : tensor<1x1x1x1xi64>
          %100 = arith.muli %99, %cst_5 : tensor<1x1x1x1xi64>
          %101 = tt.broadcast %100 : tensor<1x1x1x1xi64> -> tensor<2x2x1x512xi64>
          %102 = arith.addi %34, %101 : tensor<2x2x1x512xi64>
          %103 = arith.remsi %95, %cst_4 : tensor<1x2x1x1xi64>
          %104 = arith.muli %103, %cst_3 : tensor<1x2x1x1xi64>
          %105 = tt.broadcast %104 : tensor<1x2x1x1xi64> -> tensor<2x2x1x512xi64>
          %106 = arith.addi %102, %105 : tensor<2x2x1x512xi64>
          %107 = arith.divsi %98, %cst_6 : tensor<1x1x1x1xi64>
          %108 = arith.remsi %107, %cst_2 : tensor<1x1x1x1xi64>
          %109 = arith.muli %108, %cst_1 : tensor<1x1x1x1xi64>
          %110 = tt.broadcast %109 : tensor<1x1x1x1xi64> -> tensor<2x2x1x512xi64>
          %111 = arith.addi %106, %110 : tensor<2x2x1x512xi64>
          %112 = arith.divsi %95, %cst_4 : tensor<1x2x1x1xi64>
          %113 = arith.remsi %112, %cst_0 : tensor<1x2x1x1xi64>
          %114 = arith.muli %113, %cst : tensor<1x2x1x1xi64>
          %115 = tt.broadcast %114 : tensor<1x2x1x1xi64> -> tensor<2x2x1x512xi64>
          %116 = arith.addi %111, %115 : tensor<2x2x1x512xi64>
          %117 = arith.addi %116, %48 : tensor<2x2x1x512xi64>
          %118 = tt.addptr %35, %117 : tensor<2x2x1x512x!tt.ptr<f32>>, tensor<2x2x1x512xi64>
          %119 = tt.load %118, %79, %cst_15 : tensor<2x2x1x512x!tt.ptr<f32>>
          %120 = tt.broadcast %65 : tensor<1x1x1x512xf32> -> tensor<2x2x1x512xf32>
          %121 = arith.mulf %119, %120 : tensor<2x2x1x512xf32>
          %122 = "tt.reduce"(%121) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %137 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %137 : f32
          }) : (tensor<2x2x1x512xf32>) -> tensor<2x2x1xf32>
          %123 = tt.reshape %122 : tensor<2x2x1xf32> -> tensor<2x2x1x1xf32>
          %124 = tt.broadcast %90 : tensor<2x2x1x1xf32> -> tensor<2x2x1x512xf32>
          %125 = arith.subf %80, %124 : tensor<2x2x1x512xf32>
          %126 = tt.broadcast %92 : tensor<2x2x1x1xf32> -> tensor<2x2x1x512xf32>
          %127 = arith.mulf %125, %126 : tensor<2x2x1x512xf32>
          %128 = arith.mulf %121, %127 : tensor<2x2x1x512xf32>
          %129 = "tt.reduce"(%128) <{axis = 3 : i32}> ({
          ^bb0(%arg15: f32, %arg16: f32):
            %137 = arith.addf %arg15, %arg16 : f32
            tt.reduce.return %137 : f32
          }) : (tensor<2x2x1x512xf32>) -> tensor<2x2x1xf32>
          %130 = tt.reshape %129 : tensor<2x2x1xf32> -> tensor<2x2x1x1xf32>
          %131 = arith.mulf %121, %cst_11 : tensor<2x2x1x512xf32>
          %132 = tt.broadcast %123 : tensor<2x2x1x1xf32> -> tensor<2x2x1x512xf32>
          %133 = arith.subf %131, %132 : tensor<2x2x1x512xf32>
          %134 = tt.broadcast %130 : tensor<2x2x1x1xf32> -> tensor<2x2x1x512xf32>
          %135 = arith.mulf %127, %134 : tensor<2x2x1x512xf32>
          %136 = arith.subf %133, %135 : tensor<2x2x1x512xf32>
          tt.store %73, %136, %79 : tensor<2x2x1x512x!tt.ptr<f32>>
        }
      }
    }
    tt.return
  }
}
