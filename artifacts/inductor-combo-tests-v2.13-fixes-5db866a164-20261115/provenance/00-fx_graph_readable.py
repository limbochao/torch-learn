class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[20, 20]", arg2_1: "f32[10, 10]"):
        # File: <workspace> in forward, code: a1 = torch.nn.functional.relu(a)
        relu: "f32[10, 10]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None

        # File: <workspace> in forward, code: b1 = torch.nn.functional.sigmoid(b)
        sigmoid: "f32[20, 20]" = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None

        # File: <workspace> in forward, code: c1 = torch.nn.functional.tanh(c)
        tanh: "f32[10, 10]" = torch.ops.aten.tanh.default(arg2_1);  arg2_1 = None
        return (relu, sigmoid, tanh)
