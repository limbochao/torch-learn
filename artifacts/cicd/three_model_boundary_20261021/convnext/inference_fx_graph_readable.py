class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[80, 3, 4, 4]", arg1_1: "f32[80]", arg2_1: "f32[8, 3, 288, 288]", arg3_1: "f32[80]", arg4_1: "f32[80]", arg5_1: "f32[80, 1, 7, 7]", arg6_1: "f32[80]", arg7_1: "f32[80]", arg8_1: "f32[80]", arg9_1: "f32[320, 80, 1, 1]", arg10_1: "f32[320]", arg11_1: "f32[320]", arg12_1: "f32[320]", arg13_1: "f32[80, 320, 1, 1]", arg14_1: "f32[80]", arg15_1: "f32[80, 1, 7, 7]", arg16_1: "f32[80]", arg17_1: "f32[80]", arg18_1: "f32[80]", arg19_1: "f32[320, 80, 1, 1]", arg20_1: "f32[320]", arg21_1: "f32[320]", arg22_1: "f32[320]", arg23_1: "f32[80, 320, 1, 1]", arg24_1: "f32[80]", arg25_1: "f32[80]", arg26_1: "f32[80]", arg27_1: "f32[160, 80, 2, 2]", arg28_1: "f32[160]", arg29_1: "f32[160, 1, 7, 7]", arg30_1: "f32[160]", arg31_1: "f32[160]", arg32_1: "f32[160]", arg33_1: "f32[640, 160, 1, 1]", arg34_1: "f32[640]", arg35_1: "f32[640]", arg36_1: "f32[640]", arg37_1: "f32[160, 640, 1, 1]", arg38_1: "f32[160]", arg39_1: "f32[160, 1, 7, 7]", arg40_1: "f32[160]", arg41_1: "f32[160]", arg42_1: "f32[160]", arg43_1: "f32[640, 160, 1, 1]", arg44_1: "f32[640]", arg45_1: "f32[640]", arg46_1: "f32[640]", arg47_1: "f32[160, 640, 1, 1]", arg48_1: "f32[160]", arg49_1: "f32[160]", arg50_1: "f32[160]", arg51_1: "f32[320, 160, 2, 2]", arg52_1: "f32[320]", arg53_1: "f32[320, 1, 7, 7]", arg54_1: "f32[320]", arg55_1: "f32[320]", arg56_1: "f32[320]", arg57_1: "f32[1280, 320, 1, 1]", arg58_1: "f32[1280]", arg59_1: "f32[1280]", arg60_1: "f32[1280]", arg61_1: "f32[320, 1280, 1, 1]", arg62_1: "f32[320]", arg63_1: "f32[320, 1, 7, 7]", arg64_1: "f32[320]", arg65_1: "f32[320]", arg66_1: "f32[320]", arg67_1: "f32[1280, 320, 1, 1]", arg68_1: "f32[1280]", arg69_1: "f32[1280]", arg70_1: "f32[1280]", arg71_1: "f32[320, 1280, 1, 1]", arg72_1: "f32[320]", arg73_1: "f32[320, 1, 7, 7]", arg74_1: "f32[320]", arg75_1: "f32[320]", arg76_1: "f32[320]", arg77_1: "f32[1280, 320, 1, 1]", arg78_1: "f32[1280]", arg79_1: "f32[1280]", arg80_1: "f32[1280]", arg81_1: "f32[320, 1280, 1, 1]", arg82_1: "f32[320]", arg83_1: "f32[320, 1, 7, 7]", arg84_1: "f32[320]", arg85_1: "f32[320]", arg86_1: "f32[320]", arg87_1: "f32[1280, 320, 1, 1]", arg88_1: "f32[1280]", arg89_1: "f32[1280]", arg90_1: "f32[1280]", arg91_1: "f32[320, 1280, 1, 1]", arg92_1: "f32[320]", arg93_1: "f32[320, 1, 7, 7]", arg94_1: "f32[320]", arg95_1: "f32[320]", arg96_1: "f32[320]", arg97_1: "f32[1280, 320, 1, 1]", arg98_1: "f32[1280]", arg99_1: "f32[1280]", arg100_1: "f32[1280]", arg101_1: "f32[320, 1280, 1, 1]", arg102_1: "f32[320]", arg103_1: "f32[320, 1, 7, 7]", arg104_1: "f32[320]", arg105_1: "f32[320]", arg106_1: "f32[320]", arg107_1: "f32[1280, 320, 1, 1]", arg108_1: "f32[1280]", arg109_1: "f32[1280]", arg110_1: "f32[1280]", arg111_1: "f32[320, 1280, 1, 1]", arg112_1: "f32[320]", arg113_1: "f32[320, 1, 7, 7]", arg114_1: "f32[320]", arg115_1: "f32[320]", arg116_1: "f32[320]", arg117_1: "f32[1280, 320, 1, 1]", arg118_1: "f32[1280]", arg119_1: "f32[1280]", arg120_1: "f32[1280]", arg121_1: "f32[320, 1280, 1, 1]", arg122_1: "f32[320]", arg123_1: "f32[320, 1, 7, 7]", arg124_1: "f32[320]", arg125_1: "f32[320]", arg126_1: "f32[320]", arg127_1: "f32[1280, 320, 1, 1]", arg128_1: "f32[1280]", arg129_1: "f32[1280]", arg130_1: "f32[1280]", arg131_1: "f32[320, 1280, 1, 1]", arg132_1: "f32[320]", arg133_1: "f32[320]", arg134_1: "f32[320]", arg135_1: "f32[640, 320, 2, 2]", arg136_1: "f32[640]", arg137_1: "f32[640, 1, 7, 7]", arg138_1: "f32[640]", arg139_1: "f32[640]", arg140_1: "f32[640]", arg141_1: "f32[2560, 640, 1, 1]", arg142_1: "f32[2560]", arg143_1: "f32[2560]", arg144_1: "f32[2560]", arg145_1: "f32[640, 2560, 1, 1]", arg146_1: "f32[640]", arg147_1: "f32[640, 1, 7, 7]", arg148_1: "f32[640]", arg149_1: "f32[640]", arg150_1: "f32[640]", arg151_1: "f32[2560, 640, 1, 1]", arg152_1: "f32[2560]", arg153_1: "f32[2560]", arg154_1: "f32[2560]", arg155_1: "f32[640, 2560, 1, 1]", arg156_1: "f32[640]", arg157_1: "f32[640]", arg158_1: "f32[640]", arg159_1: "f32[1000, 640]", arg160_1: "f32[1000]"):
        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:610 in forward_features, code: x = self.stem(x)
        convolution: "f32[8, 80, 72, 72]" = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[8, 72, 72, 80]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm = torch.ops.aten.native_layer_norm.default(permute, [80], arg3_1, arg4_1, 1e-06);  permute = arg3_1 = arg4_1 = None
        getitem: "f32[8, 72, 72, 80]" = native_layer_norm[0];  native_layer_norm = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_1: "f32[8, 80, 72, 72]" = torch.ops.aten.permute.default(getitem, [0, 3, 1, 2]);  getitem = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_1: "f32[8, 80, 72, 72]" = torch.ops.aten.convolution.default(permute_1, arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg5_1 = arg6_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "f32[8, 72, 72, 80]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(permute_2, [80], arg7_1, arg8_1, 1e-06);  permute_2 = arg7_1 = arg8_1 = None
        getitem_3: "f32[8, 72, 72, 80]" = native_layer_norm_1[0];  native_layer_norm_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_3: "f32[8, 80, 72, 72]" = torch.ops.aten.permute.default(getitem_3, [0, 3, 1, 2]);  getitem_3 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_2: "f32[8, 320, 72, 72]" = torch.ops.aten.convolution.default(permute_3, arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_3 = arg9_1 = arg10_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu: "f32[8, 320, 72, 72]" = torch.ops.aten.gelu.default(convolution_2);  convolution_2 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_1: "f32[8, 320, 72, 72]" = torch.ops.aten.pow.Tensor_Scalar(gelu, 2)
        sum_1: "f32[8, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [2, 3], True);  pow_1 = None
        pow_2: "f32[8, 320, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_2, [1], True)
        add: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        div: "f32[8, 320, 1, 1]" = torch.ops.aten.div.Tensor(pow_2, add);  pow_2 = add = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg11_1, [1, -1, 1, 1]);  arg11_1 = None
        view_1: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg12_1, [1, -1, 1, 1]);  arg12_1 = None
        mul: "f32[8, 320, 72, 72]" = torch.ops.aten.mul.Tensor(gelu, div);  div = None
        addcmul: "f32[8, 320, 72, 72]" = torch.ops.aten.addcmul.default(view, view_1, mul);  view = view_1 = mul = None
        add_1: "f32[8, 320, 72, 72]" = torch.ops.aten.add.Tensor(gelu, addcmul);  gelu = addcmul = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_3: "f32[8, 80, 72, 72]" = torch.ops.aten.convolution.default(add_1, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_1 = arg13_1 = arg14_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_2: "f32[8, 80, 72, 72]" = torch.ops.aten.add.Tensor(convolution_3, permute_1);  convolution_3 = permute_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_4: "f32[8, 80, 72, 72]" = torch.ops.aten.convolution.default(add_2, arg15_1, arg16_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80);  arg15_1 = arg16_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "f32[8, 72, 72, 80]" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_2 = torch.ops.aten.native_layer_norm.default(permute_4, [80], arg17_1, arg18_1, 1e-06);  permute_4 = arg17_1 = arg18_1 = None
        getitem_6: "f32[8, 72, 72, 80]" = native_layer_norm_2[0];  native_layer_norm_2 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_5: "f32[8, 80, 72, 72]" = torch.ops.aten.permute.default(getitem_6, [0, 3, 1, 2]);  getitem_6 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_5: "f32[8, 320, 72, 72]" = torch.ops.aten.convolution.default(permute_5, arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_5 = arg19_1 = arg20_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_1: "f32[8, 320, 72, 72]" = torch.ops.aten.gelu.default(convolution_5);  convolution_5 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_3: "f32[8, 320, 72, 72]" = torch.ops.aten.pow.Tensor_Scalar(gelu_1, 2)
        sum_2: "f32[8, 320, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_3, [2, 3], True);  pow_3 = None
        pow_4: "f32[8, 320, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_1: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_4, [1], True)
        add_3: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        div_1: "f32[8, 320, 1, 1]" = torch.ops.aten.div.Tensor(pow_4, add_3);  pow_4 = add_3 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_2: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg21_1, [1, -1, 1, 1]);  arg21_1 = None
        view_3: "f32[1, 320, 1, 1]" = torch.ops.aten.view.default(arg22_1, [1, -1, 1, 1]);  arg22_1 = None
        mul_1: "f32[8, 320, 72, 72]" = torch.ops.aten.mul.Tensor(gelu_1, div_1);  div_1 = None
        addcmul_1: "f32[8, 320, 72, 72]" = torch.ops.aten.addcmul.default(view_2, view_3, mul_1);  view_2 = view_3 = mul_1 = None
        add_4: "f32[8, 320, 72, 72]" = torch.ops.aten.add.Tensor(gelu_1, addcmul_1);  gelu_1 = addcmul_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_6: "f32[8, 80, 72, 72]" = torch.ops.aten.convolution.default(add_4, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_4 = arg23_1 = arg24_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_5: "f32[8, 80, 72, 72]" = torch.ops.aten.add.Tensor(convolution_6, add_2);  convolution_6 = add_2 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_6: "f32[8, 72, 72, 80]" = torch.ops.aten.permute.default(add_5, [0, 2, 3, 1]);  add_5 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_3 = torch.ops.aten.native_layer_norm.default(permute_6, [80], arg25_1, arg26_1, 1e-06);  permute_6 = arg25_1 = arg26_1 = None
        getitem_9: "f32[8, 72, 72, 80]" = native_layer_norm_3[0];  native_layer_norm_3 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_7: "f32[8, 80, 72, 72]" = torch.ops.aten.permute.default(getitem_9, [0, 3, 1, 2]);  getitem_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:303 in forward, code: x = self.downsample(x)
        convolution_7: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(permute_7, arg27_1, arg28_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_7 = arg27_1 = arg28_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_8: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(convolution_7, arg29_1, arg30_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg29_1 = arg30_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "f32[8, 36, 36, 160]" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_4 = torch.ops.aten.native_layer_norm.default(permute_8, [160], arg31_1, arg32_1, 1e-06);  permute_8 = arg31_1 = arg32_1 = None
        getitem_12: "f32[8, 36, 36, 160]" = native_layer_norm_4[0];  native_layer_norm_4 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_9: "f32[8, 160, 36, 36]" = torch.ops.aten.permute.default(getitem_12, [0, 3, 1, 2]);  getitem_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_9: "f32[8, 640, 36, 36]" = torch.ops.aten.convolution.default(permute_9, arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_9 = arg33_1 = arg34_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_2: "f32[8, 640, 36, 36]" = torch.ops.aten.gelu.default(convolution_9);  convolution_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_5: "f32[8, 640, 36, 36]" = torch.ops.aten.pow.Tensor_Scalar(gelu_2, 2)
        sum_3: "f32[8, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_5, [2, 3], True);  pow_5 = None
        pow_6: "f32[8, 640, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_2: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_6, [1], True)
        add_6: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        div_2: "f32[8, 640, 1, 1]" = torch.ops.aten.div.Tensor(pow_6, add_6);  pow_6 = add_6 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_4: "f32[1, 640, 1, 1]" = torch.ops.aten.view.default(arg35_1, [1, -1, 1, 1]);  arg35_1 = None
        view_5: "f32[1, 640, 1, 1]" = torch.ops.aten.view.default(arg36_1, [1, -1, 1, 1]);  arg36_1 = None
        mul_2: "f32[8, 640, 36, 36]" = torch.ops.aten.mul.Tensor(gelu_2, div_2);  div_2 = None
        addcmul_2: "f32[8, 640, 36, 36]" = torch.ops.aten.addcmul.default(view_4, view_5, mul_2);  view_4 = view_5 = mul_2 = None
        add_7: "f32[8, 640, 36, 36]" = torch.ops.aten.add.Tensor(gelu_2, addcmul_2);  gelu_2 = addcmul_2 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_10: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(add_7, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_7 = arg37_1 = arg38_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_8: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(convolution_10, convolution_7);  convolution_10 = convolution_7 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_11: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(add_8, arg39_1, arg40_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160);  arg39_1 = arg40_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "f32[8, 36, 36, 160]" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_5 = torch.ops.aten.native_layer_norm.default(permute_10, [160], arg41_1, arg42_1, 1e-06);  permute_10 = arg41_1 = arg42_1 = None
        getitem_15: "f32[8, 36, 36, 160]" = native_layer_norm_5[0];  native_layer_norm_5 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_11: "f32[8, 160, 36, 36]" = torch.ops.aten.permute.default(getitem_15, [0, 3, 1, 2]);  getitem_15 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_12: "f32[8, 640, 36, 36]" = torch.ops.aten.convolution.default(permute_11, arg43_1, arg44_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_11 = arg43_1 = arg44_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_3: "f32[8, 640, 36, 36]" = torch.ops.aten.gelu.default(convolution_12);  convolution_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_7: "f32[8, 640, 36, 36]" = torch.ops.aten.pow.Tensor_Scalar(gelu_3, 2)
        sum_4: "f32[8, 640, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_7, [2, 3], True);  pow_7 = None
        pow_8: "f32[8, 640, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_3: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_8, [1], True)
        add_9: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        div_3: "f32[8, 640, 1, 1]" = torch.ops.aten.div.Tensor(pow_8, add_9);  pow_8 = add_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_6: "f32[1, 640, 1, 1]" = torch.ops.aten.view.default(arg45_1, [1, -1, 1, 1]);  arg45_1 = None
        view_7: "f32[1, 640, 1, 1]" = torch.ops.aten.view.default(arg46_1, [1, -1, 1, 1]);  arg46_1 = None
        mul_3: "f32[8, 640, 36, 36]" = torch.ops.aten.mul.Tensor(gelu_3, div_3);  div_3 = None
        addcmul_3: "f32[8, 640, 36, 36]" = torch.ops.aten.addcmul.default(view_6, view_7, mul_3);  view_6 = view_7 = mul_3 = None
        add_10: "f32[8, 640, 36, 36]" = torch.ops.aten.add.Tensor(gelu_3, addcmul_3);  gelu_3 = addcmul_3 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_13: "f32[8, 160, 36, 36]" = torch.ops.aten.convolution.default(add_10, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_10 = arg47_1 = arg48_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_11: "f32[8, 160, 36, 36]" = torch.ops.aten.add.Tensor(convolution_13, add_8);  convolution_13 = add_8 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_12: "f32[8, 36, 36, 160]" = torch.ops.aten.permute.default(add_11, [0, 2, 3, 1]);  add_11 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_6 = torch.ops.aten.native_layer_norm.default(permute_12, [160], arg49_1, arg50_1, 1e-06);  permute_12 = arg49_1 = arg50_1 = None
        getitem_18: "f32[8, 36, 36, 160]" = native_layer_norm_6[0];  native_layer_norm_6 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_13: "f32[8, 160, 36, 36]" = torch.ops.aten.permute.default(getitem_18, [0, 3, 1, 2]);  getitem_18 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:303 in forward, code: x = self.downsample(x)
        convolution_14: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(permute_13, arg51_1, arg52_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_13 = arg51_1 = arg52_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_15: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(convolution_14, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg53_1 = arg54_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_7 = torch.ops.aten.native_layer_norm.default(permute_14, [320], arg55_1, arg56_1, 1e-06);  permute_14 = arg55_1 = arg56_1 = None
        getitem_21: "f32[8, 18, 18, 320]" = native_layer_norm_7[0];  native_layer_norm_7 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_15: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_21, [0, 3, 1, 2]);  getitem_21 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_16: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_15, arg57_1, arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_15 = arg57_1 = arg58_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_4: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_16);  convolution_16 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_9: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_4, 2)
        sum_5: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_9, [2, 3], True);  pow_9 = None
        pow_10: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_4: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_10, [1], True)
        add_12: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        div_4: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_10, add_12);  pow_10 = add_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_8: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg59_1, [1, -1, 1, 1]);  arg59_1 = None
        view_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg60_1, [1, -1, 1, 1]);  arg60_1 = None
        mul_4: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_4, div_4);  div_4 = None
        addcmul_4: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_8, view_9, mul_4);  view_8 = view_9 = mul_4 = None
        add_13: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_4, addcmul_4);  gelu_4 = addcmul_4 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_17: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_13, arg61_1, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_13 = arg61_1 = arg62_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_14: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_17, convolution_14);  convolution_17 = convolution_14 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_18: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_14, arg63_1, arg64_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg63_1 = arg64_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_8 = torch.ops.aten.native_layer_norm.default(permute_16, [320], arg65_1, arg66_1, 1e-06);  permute_16 = arg65_1 = arg66_1 = None
        getitem_24: "f32[8, 18, 18, 320]" = native_layer_norm_8[0];  native_layer_norm_8 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_17: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_24, [0, 3, 1, 2]);  getitem_24 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_19: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_17, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_17 = arg67_1 = arg68_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_5: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_19);  convolution_19 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_11: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_5, 2)
        sum_6: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_11, [2, 3], True);  pow_11 = None
        pow_12: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_6, 0.5);  sum_6 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_5: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_12, [1], True)
        add_15: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        div_5: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_12, add_15);  pow_12 = add_15 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_10: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg69_1, [1, -1, 1, 1]);  arg69_1 = None
        view_11: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg70_1, [1, -1, 1, 1]);  arg70_1 = None
        mul_5: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_5, div_5);  div_5 = None
        addcmul_5: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_10, view_11, mul_5);  view_10 = view_11 = mul_5 = None
        add_16: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_5, addcmul_5);  gelu_5 = addcmul_5 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_20: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_16, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_16 = arg71_1 = arg72_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_17: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_20, add_14);  convolution_20 = add_14 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_21: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_17, arg73_1, arg74_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg73_1 = arg74_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_9 = torch.ops.aten.native_layer_norm.default(permute_18, [320], arg75_1, arg76_1, 1e-06);  permute_18 = arg75_1 = arg76_1 = None
        getitem_27: "f32[8, 18, 18, 320]" = native_layer_norm_9[0];  native_layer_norm_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_19: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_27, [0, 3, 1, 2]);  getitem_27 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_22: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_19, arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_19 = arg77_1 = arg78_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_6: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_22);  convolution_22 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_13: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_6, 2)
        sum_7: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_13, [2, 3], True);  pow_13 = None
        pow_14: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_6: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_14, [1], True)
        add_18: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        div_6: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_14, add_18);  pow_14 = add_18 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg79_1, [1, -1, 1, 1]);  arg79_1 = None
        view_13: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg80_1, [1, -1, 1, 1]);  arg80_1 = None
        mul_6: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_6, div_6);  div_6 = None
        addcmul_6: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_12, view_13, mul_6);  view_12 = view_13 = mul_6 = None
        add_19: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_6, addcmul_6);  gelu_6 = addcmul_6 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_23: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_19, arg81_1, arg82_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_19 = arg81_1 = arg82_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_20: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_23, add_17);  convolution_23 = add_17 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_24: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_20, arg83_1, arg84_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg83_1 = arg84_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_10 = torch.ops.aten.native_layer_norm.default(permute_20, [320], arg85_1, arg86_1, 1e-06);  permute_20 = arg85_1 = arg86_1 = None
        getitem_30: "f32[8, 18, 18, 320]" = native_layer_norm_10[0];  native_layer_norm_10 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_21: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_30, [0, 3, 1, 2]);  getitem_30 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_25: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_21, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_21 = arg87_1 = arg88_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_7: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_25);  convolution_25 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_15: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_7, 2)
        sum_8: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_15, [2, 3], True);  pow_15 = None
        pow_16: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_7: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_16, [1], True)
        add_21: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        div_7: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_16, add_21);  pow_16 = add_21 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_14: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg89_1, [1, -1, 1, 1]);  arg89_1 = None
        view_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg90_1, [1, -1, 1, 1]);  arg90_1 = None
        mul_7: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_7, div_7);  div_7 = None
        addcmul_7: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_14, view_15, mul_7);  view_14 = view_15 = mul_7 = None
        add_22: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_7, addcmul_7);  gelu_7 = addcmul_7 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_26: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_22, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_22 = arg91_1 = arg92_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_23: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_26, add_20);  convolution_26 = add_20 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_27: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_23, arg93_1, arg94_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg93_1 = arg94_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_11 = torch.ops.aten.native_layer_norm.default(permute_22, [320], arg95_1, arg96_1, 1e-06);  permute_22 = arg95_1 = arg96_1 = None
        getitem_33: "f32[8, 18, 18, 320]" = native_layer_norm_11[0];  native_layer_norm_11 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_23: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_33, [0, 3, 1, 2]);  getitem_33 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_28: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_23, arg97_1, arg98_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_23 = arg97_1 = arg98_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_8: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_28);  convolution_28 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_17: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_8, 2)
        sum_9: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_17, [2, 3], True);  pow_17 = None
        pow_18: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_9, 0.5);  sum_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_8: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_18, [1], True)
        add_24: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        div_8: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_18, add_24);  pow_18 = add_24 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_16: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg99_1, [1, -1, 1, 1]);  arg99_1 = None
        view_17: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg100_1, [1, -1, 1, 1]);  arg100_1 = None
        mul_8: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_8, div_8);  div_8 = None
        addcmul_8: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_16, view_17, mul_8);  view_16 = view_17 = mul_8 = None
        add_25: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_8, addcmul_8);  gelu_8 = addcmul_8 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_29: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_25, arg101_1, arg102_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_25 = arg101_1 = arg102_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_26: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_29, add_23);  convolution_29 = add_23 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_30: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_26, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg103_1 = arg104_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_12 = torch.ops.aten.native_layer_norm.default(permute_24, [320], arg105_1, arg106_1, 1e-06);  permute_24 = arg105_1 = arg106_1 = None
        getitem_36: "f32[8, 18, 18, 320]" = native_layer_norm_12[0];  native_layer_norm_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_25: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_36, [0, 3, 1, 2]);  getitem_36 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_31: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_25, arg107_1, arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_25 = arg107_1 = arg108_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_9: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_31);  convolution_31 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_19: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_9, 2)
        sum_10: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_19, [2, 3], True);  pow_19 = None
        pow_20: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_9: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_20, [1], True)
        add_27: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        div_9: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_20, add_27);  pow_20 = add_27 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_18: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg109_1, [1, -1, 1, 1]);  arg109_1 = None
        view_19: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg110_1, [1, -1, 1, 1]);  arg110_1 = None
        mul_9: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_9, div_9);  div_9 = None
        addcmul_9: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_18, view_19, mul_9);  view_18 = view_19 = mul_9 = None
        add_28: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_9, addcmul_9);  gelu_9 = addcmul_9 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_32: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_28, arg111_1, arg112_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_28 = arg111_1 = arg112_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_29: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_32, add_26);  convolution_32 = add_26 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_33: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_29, arg113_1, arg114_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg113_1 = arg114_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_13 = torch.ops.aten.native_layer_norm.default(permute_26, [320], arg115_1, arg116_1, 1e-06);  permute_26 = arg115_1 = arg116_1 = None
        getitem_39: "f32[8, 18, 18, 320]" = native_layer_norm_13[0];  native_layer_norm_13 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_27: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_39, [0, 3, 1, 2]);  getitem_39 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_34: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_27, arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_27 = arg117_1 = arg118_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_10: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_34);  convolution_34 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_21: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_10, 2)
        sum_11: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_21, [2, 3], True);  pow_21 = None
        pow_22: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_10: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_22, [1], True)
        add_30: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        div_10: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_22, add_30);  pow_22 = add_30 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_20: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg119_1, [1, -1, 1, 1]);  arg119_1 = None
        view_21: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg120_1, [1, -1, 1, 1]);  arg120_1 = None
        mul_10: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_10, div_10);  div_10 = None
        addcmul_10: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_20, view_21, mul_10);  view_20 = view_21 = mul_10 = None
        add_31: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_10, addcmul_10);  gelu_10 = addcmul_10 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_35: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_31, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_31 = arg121_1 = arg122_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_32: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_35, add_29);  convolution_35 = add_29 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_36: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_32, arg123_1, arg124_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320);  arg123_1 = arg124_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_14 = torch.ops.aten.native_layer_norm.default(permute_28, [320], arg125_1, arg126_1, 1e-06);  permute_28 = arg125_1 = arg126_1 = None
        getitem_42: "f32[8, 18, 18, 320]" = native_layer_norm_14[0];  native_layer_norm_14 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_29: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_42, [0, 3, 1, 2]);  getitem_42 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_37: "f32[8, 1280, 18, 18]" = torch.ops.aten.convolution.default(permute_29, arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_29 = arg127_1 = arg128_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_11: "f32[8, 1280, 18, 18]" = torch.ops.aten.gelu.default(convolution_37);  convolution_37 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_23: "f32[8, 1280, 18, 18]" = torch.ops.aten.pow.Tensor_Scalar(gelu_11, 2)
        sum_12: "f32[8, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_23, [2, 3], True);  pow_23 = None
        pow_24: "f32[8, 1280, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_12, 0.5);  sum_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_11: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_24, [1], True)
        add_33: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        div_11: "f32[8, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_24, add_33);  pow_24 = add_33 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_22: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg129_1, [1, -1, 1, 1]);  arg129_1 = None
        view_23: "f32[1, 1280, 1, 1]" = torch.ops.aten.view.default(arg130_1, [1, -1, 1, 1]);  arg130_1 = None
        mul_11: "f32[8, 1280, 18, 18]" = torch.ops.aten.mul.Tensor(gelu_11, div_11);  div_11 = None
        addcmul_11: "f32[8, 1280, 18, 18]" = torch.ops.aten.addcmul.default(view_22, view_23, mul_11);  view_22 = view_23 = mul_11 = None
        add_34: "f32[8, 1280, 18, 18]" = torch.ops.aten.add.Tensor(gelu_11, addcmul_11);  gelu_11 = addcmul_11 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_38: "f32[8, 320, 18, 18]" = torch.ops.aten.convolution.default(add_34, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_34 = arg131_1 = arg132_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_35: "f32[8, 320, 18, 18]" = torch.ops.aten.add.Tensor(convolution_38, add_32);  convolution_38 = add_32 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_30: "f32[8, 18, 18, 320]" = torch.ops.aten.permute.default(add_35, [0, 2, 3, 1]);  add_35 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_15 = torch.ops.aten.native_layer_norm.default(permute_30, [320], arg133_1, arg134_1, 1e-06);  permute_30 = arg133_1 = arg134_1 = None
        getitem_45: "f32[8, 18, 18, 320]" = native_layer_norm_15[0];  native_layer_norm_15 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_31: "f32[8, 320, 18, 18]" = torch.ops.aten.permute.default(getitem_45, [0, 3, 1, 2]);  getitem_45 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:303 in forward, code: x = self.downsample(x)
        convolution_39: "f32[8, 640, 9, 9]" = torch.ops.aten.convolution.default(permute_31, arg135_1, arg136_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_31 = arg135_1 = arg136_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_40: "f32[8, 640, 9, 9]" = torch.ops.aten.convolution.default(convolution_39, arg137_1, arg138_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg137_1 = arg138_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "f32[8, 9, 9, 640]" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_16 = torch.ops.aten.native_layer_norm.default(permute_32, [640], arg139_1, arg140_1, 1e-06);  permute_32 = arg139_1 = arg140_1 = None
        getitem_48: "f32[8, 9, 9, 640]" = native_layer_norm_16[0];  native_layer_norm_16 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_33: "f32[8, 640, 9, 9]" = torch.ops.aten.permute.default(getitem_48, [0, 3, 1, 2]);  getitem_48 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_41: "f32[8, 2560, 9, 9]" = torch.ops.aten.convolution.default(permute_33, arg141_1, arg142_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_33 = arg141_1 = arg142_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_12: "f32[8, 2560, 9, 9]" = torch.ops.aten.gelu.default(convolution_41);  convolution_41 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_25: "f32[8, 2560, 9, 9]" = torch.ops.aten.pow.Tensor_Scalar(gelu_12, 2)
        sum_13: "f32[8, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_25, [2, 3], True);  pow_25 = None
        pow_26: "f32[8, 2560, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_12: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_26, [1], True)
        add_36: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        div_12: "f32[8, 2560, 1, 1]" = torch.ops.aten.div.Tensor(pow_26, add_36);  pow_26 = add_36 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_24: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg143_1, [1, -1, 1, 1]);  arg143_1 = None
        view_25: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg144_1, [1, -1, 1, 1]);  arg144_1 = None
        mul_12: "f32[8, 2560, 9, 9]" = torch.ops.aten.mul.Tensor(gelu_12, div_12);  div_12 = None
        addcmul_12: "f32[8, 2560, 9, 9]" = torch.ops.aten.addcmul.default(view_24, view_25, mul_12);  view_24 = view_25 = mul_12 = None
        add_37: "f32[8, 2560, 9, 9]" = torch.ops.aten.add.Tensor(gelu_12, addcmul_12);  gelu_12 = addcmul_12 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_42: "f32[8, 640, 9, 9]" = torch.ops.aten.convolution.default(add_37, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_37 = arg145_1 = arg146_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_38: "f32[8, 640, 9, 9]" = torch.ops.aten.add.Tensor(convolution_42, convolution_39);  convolution_42 = convolution_39 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:201 in forward, code: x = self.conv_dw(x)
        convolution_43: "f32[8, 640, 9, 9]" = torch.ops.aten.convolution.default(add_38, arg147_1, arg148_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640);  arg147_1 = arg148_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "f32[8, 9, 9, 640]" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_17 = torch.ops.aten.native_layer_norm.default(permute_34, [640], arg149_1, arg150_1, 1e-06);  permute_34 = arg149_1 = arg150_1 = None
        getitem_51: "f32[8, 9, 9, 640]" = native_layer_norm_17[0];  native_layer_norm_17 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_35: "f32[8, 640, 9, 9]" = torch.ops.aten.permute.default(getitem_51, [0, 3, 1, 2]);  getitem_51 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        convolution_44: "f32[8, 2560, 9, 9]" = torch.ops.aten.convolution.default(permute_35, arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_35 = arg151_1 = arg152_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        gelu_13: "f32[8, 2560, 9, 9]" = torch.ops.aten.gelu.default(convolution_44);  convolution_44 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_27: "f32[8, 2560, 9, 9]" = torch.ops.aten.pow.Tensor_Scalar(gelu_13, 2)
        sum_14: "f32[8, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_27, [2, 3], True);  pow_27 = None
        pow_28: "f32[8, 2560, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_13: "f32[8, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_28, [1], True)
        add_39: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        div_13: "f32[8, 2560, 1, 1]" = torch.ops.aten.div.Tensor(pow_28, add_39);  pow_28 = add_39 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        view_26: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg153_1, [1, -1, 1, 1]);  arg153_1 = None
        view_27: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg154_1, [1, -1, 1, 1]);  arg154_1 = None
        mul_13: "f32[8, 2560, 9, 9]" = torch.ops.aten.mul.Tensor(gelu_13, div_13);  div_13 = None
        addcmul_13: "f32[8, 2560, 9, 9]" = torch.ops.aten.addcmul.default(view_26, view_27, mul_13);  view_26 = view_27 = mul_13 = None
        add_40: "f32[8, 2560, 9, 9]" = torch.ops.aten.add.Tensor(gelu_13, addcmul_13);  gelu_13 = addcmul_13 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        convolution_45: "f32[8, 640, 9, 9]" = torch.ops.aten.convolution.default(add_40, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_40 = arg155_1 = arg156_1 = None

        # File: <env>/lib/python3.11/site-packages/timm/models/convnext.py:213 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_41: "f32[8, 640, 9, 9]" = torch.ops.aten.add.Tensor(convolution_45, add_38);  convolution_45 = add_38 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_14: "f32[8, 640, 1, 1]" = torch.ops.aten.mean.dim(add_41, [-1, -2], True);  add_41 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_36: "f32[8, 1, 1, 640]" = torch.ops.aten.permute.default(mean_14, [0, 2, 3, 1]);  mean_14 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        native_layer_norm_18 = torch.ops.aten.native_layer_norm.default(permute_36, [640], arg157_1, arg158_1, 1e-06);  permute_36 = arg157_1 = arg158_1 = None
        getitem_54: "f32[8, 1, 1, 640]" = native_layer_norm_18[0];  native_layer_norm_18 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_37: "f32[8, 640, 1, 1]" = torch.ops.aten.permute.default(getitem_54, [0, 3, 1, 2]);  getitem_54 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/classifier.py:237 in forward, code: x = self.flatten(x)
        view_28: "f32[8, 640]" = torch.ops.aten.view.default(permute_37, [8, 640]);  permute_37 = None

        # File: <env>/lib/python3.11/site-packages/timm/layers/classifier.py:242 in forward, code: x = self.fc(x)
        permute_38: "f32[640, 1000]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg160_1, view_28, permute_38);  arg160_1 = view_28 = permute_38 = None
        return (addmm,)
