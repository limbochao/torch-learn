class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64]", arg1_1: "f32[64]", arg2_1: "f32[64, 64]", arg3_1: "f32[64, 64]", arg4_1: "f32[64]", arg5_1: "f32[64, 64]", arg6_1: "f32[64, 64]", arg7_1: "f32[64]", arg8_1: "f32[64, 64]", arg9_1: "f32[64, 64]", arg10_1: "f32[64]", arg11_1: "f32[64, 64]", arg12_1: "f32[64, 64]", arg13_1: "f32[64]", arg14_1: "f32[64, 64]", arg15_1: "f32[64, 64]", arg16_1: "f32[64]", arg17_1: "f32[64, 64]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch/optim/sgd.py:379 in _single_tensor_sgd, code: param.add_(grad, alpha=-lr)
        add: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg0_1, arg9_1, alpha = -0.01);  arg9_1 = None
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(arg1_1, arg10_1, alpha = -0.01);  arg10_1 = None
        add_2: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg2_1, arg11_1, alpha = -0.01);  arg11_1 = None
        add_3: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg3_1, arg12_1, alpha = -0.01);  arg12_1 = None
        add_4: "f32[64]" = torch.ops.aten.add.Tensor(arg4_1, arg13_1, alpha = -0.01);  arg13_1 = None
        add_5: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg5_1, arg14_1, alpha = -0.01);  arg14_1 = None
        add_6: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg6_1, arg15_1, alpha = -0.01);  arg15_1 = None
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(arg7_1, arg16_1, alpha = -0.01);  arg16_1 = None
        add_8: "f32[64, 64]" = torch.ops.aten.add.Tensor(arg8_1, arg17_1, alpha = -0.01);  arg17_1 = None
        copy_: "f32[64, 64]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = copy__1 = None
        copy__2: "f32[64, 64]" = torch.ops.aten.copy_.default(arg2_1, add_2);  arg2_1 = add_2 = copy__2 = None
        copy__3: "f32[64, 64]" = torch.ops.aten.copy_.default(arg3_1, add_3);  arg3_1 = add_3 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(arg4_1, add_4);  arg4_1 = add_4 = copy__4 = None
        copy__5: "f32[64, 64]" = torch.ops.aten.copy_.default(arg5_1, add_5);  arg5_1 = add_5 = copy__5 = None
        copy__6: "f32[64, 64]" = torch.ops.aten.copy_.default(arg6_1, add_6);  arg6_1 = add_6 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(arg7_1, add_7);  arg7_1 = add_7 = copy__7 = None
        copy__8: "f32[64, 64]" = torch.ops.aten.copy_.default(arg8_1, add_8);  arg8_1 = add_8 = copy__8 = None
        return ()
