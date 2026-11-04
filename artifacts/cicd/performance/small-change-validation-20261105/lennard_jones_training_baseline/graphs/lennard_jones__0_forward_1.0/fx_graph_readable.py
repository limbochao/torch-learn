class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 1]", primals_2: "f32[16]", primals_3: "f32[4096, 1]", primals_4: "f32[16, 16]", primals_5: "f32[16]", primals_6: "f32[16, 16]", primals_7: "f32[16]", primals_8: "f32[16, 16]", primals_9: "f32[16]", primals_10: "f32[1, 16]", primals_11: "f32[1]"):
        # File: <home>/PTA_213/torch_bench/pytorch_new/benchmarks/torchbench/torchbench.py:615 in torch_dynamo_resume_in_forward_and_backward_pass_at_613, code: pred = mod(*cloned_inputs)
        permute: "f32[1, 16]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[4096, 16]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        tanh: "f32[4096, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        permute_1: "f32[16, 16]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[4096, 16]" = torch.ops.aten.addmm.default(primals_5, tanh, permute_1);  primals_5 = permute_1 = None
        tanh_1: "f32[4096, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        permute_2: "f32[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_2: "f32[4096, 16]" = torch.ops.aten.addmm.default(primals_7, tanh_1, permute_2);  primals_7 = permute_2 = None
        tanh_2: "f32[4096, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        permute_3: "f32[16, 16]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_3: "f32[4096, 16]" = torch.ops.aten.addmm.default(primals_9, tanh_2, permute_3);  primals_9 = permute_3 = None
        tanh_3: "f32[4096, 16]" = torch.ops.aten.tanh.default(addmm_3);  addmm_3 = None
        permute_4: "f32[16, 1]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_4: "f32[4096, 1]" = torch.ops.aten.addmm.default(primals_11, tanh_3, permute_4);  primals_11 = permute_4 = None
        return (addmm_4, primals_3, primals_4, primals_6, primals_8, primals_10, tanh, tanh_1, tanh_2, tanh_3)
