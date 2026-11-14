class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s27)", arg1_1: "Sym(s53)", arg2_1: "Sym(s0)", arg3_1: "f32[1, s27, s53, s0]"):
        # File: /home/l30023782/PTA_213/combo-case-9a86e68-20261114/Community-Use-Case-Testing/v2.13.0/inductor_npu/test_torchinductor.py:18207 in fn, code: x.mean(),
        mean: "f32[]" = torch.ops.aten.mean.default(arg3_1)

        # File: /home/l30023782/PTA_213/combo-case-9a86e68-20261114/Community-Use-Case-Testing/v2.13.0/inductor_npu/test_torchinductor.py:18208 in fn, code: torch.mean(x, -2, keepdim=True),
        mean_1: "f32[1, s27, 1, s0]" = torch.ops.aten.mean.dim(arg3_1, [-2], True);  arg3_1 = None
        return (mean, mean_1)
