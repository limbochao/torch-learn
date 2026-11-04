class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 199978]", arg1_1: "i64[2, 10000]"):
        # File: <home>/miniforge3/envs/pta_213_cicd/lib/python3.11/site-packages/torch_geometric/utils/loop.py:370 in torch_dynamo_resume_in_add_remaining_self_loops_at_370, code: edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
        cat: "i64[2, 209978]" = torch.ops.aten.cat.default([arg0_1, arg1_1], 1);  arg0_1 = arg1_1 = None
        return (cat,)
