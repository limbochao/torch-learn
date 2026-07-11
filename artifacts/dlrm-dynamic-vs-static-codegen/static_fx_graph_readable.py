class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[128, 1000]", arg1_1: "f32[5000, 8]", arg2_1: "f32[6144, 1000]", arg3_1: "f32[6144]", arg4_1: "f32[128, 1000]", arg5_1: "f32[6144, 6144]", arg6_1: "f32[6144]", arg7_1: "f32[6144, 6144]", arg8_1: "f32[6144]", arg9_1: "f32[6144, 6144]", arg10_1: "f32[6144]", arg11_1: "f32[6144, 6144]", arg12_1: "f32[6144]", arg13_1: "f32[6144, 6144]", arg14_1: "f32[6144]", arg15_1: "f32[3072, 6144]", arg16_1: "f32[3072]", arg17_1: "f32[3072, 3072]", arg18_1: "f32[3072]", arg19_1: "f32[3072, 3072]", arg20_1: "f32[3072]", arg21_1: "f32[1536, 3072]", arg22_1: "f32[1536]", arg23_1: "f32[1536, 1536]", arg24_1: "f32[1536]", arg25_1: "f32[1536, 1536]", arg26_1: "f32[1536]", arg27_1: "f32[8, 1536]", arg28_1: "f32[8]", arg29_1: "f32[1, 166472]", arg30_1: "f32[1]"):
        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_1: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 20)
        embedding: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_1);  slice_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_1: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding, [1]);  embedding = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_2: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 20, 40)
        embedding_1: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_2);  slice_2 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_2: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_1, [1]);  embedding_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_3: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 40, 60)
        embedding_2: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_3);  slice_3 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_3: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_2, [1]);  embedding_2 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_4: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 60, 80)
        embedding_3: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_4);  slice_4 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_4: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_3, [1]);  embedding_3 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_5: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 80, 100)
        embedding_4: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_5);  slice_5 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_5: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_4, [1]);  embedding_4 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_6: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 100, 120)
        embedding_5: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_6);  slice_6 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_6: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_5, [1]);  embedding_5 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_7: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 120, 140)
        embedding_6: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_7);  slice_7 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_7: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_6, [1]);  embedding_6 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_8: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 140, 160)
        embedding_7: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_8);  slice_8 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_8: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_7, [1]);  embedding_7 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_9: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 160, 180)
        embedding_8: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_9);  slice_9 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_9: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_8, [1]);  embedding_8 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_10: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 180, 200)
        embedding_9: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_10);  slice_10 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_10: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_9, [1]);  embedding_9 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_11: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 200, 220)
        embedding_10: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_11);  slice_11 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_11: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_10, [1]);  embedding_10 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_12: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 220, 240)
        embedding_11: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_12);  slice_12 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_12: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_11, [1]);  embedding_11 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_13: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 240, 260)
        embedding_12: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_13);  slice_13 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_13: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_12, [1]);  embedding_12 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_14: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 260, 280)
        embedding_13: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_14);  slice_14 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_14: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_13, [1]);  embedding_13 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_15: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 280, 300)
        embedding_14: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_15);  slice_15 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_15: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_14, [1]);  embedding_14 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_16: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 300, 320)
        embedding_15: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_16);  slice_16 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_16: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_15, [1]);  embedding_15 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_17: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 320, 340)
        embedding_16: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_17);  slice_17 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_17: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_16, [1]);  embedding_16 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_18: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 340, 360)
        embedding_17: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_18);  slice_18 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_18: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_17, [1]);  embedding_17 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_19: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 360, 380)
        embedding_18: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_19);  slice_19 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_19: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_18, [1]);  embedding_18 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_20: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 380, 400)
        embedding_19: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_20);  slice_20 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_20: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_19, [1]);  embedding_19 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_21: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 400, 420)
        embedding_20: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_21);  slice_21 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_21: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_20, [1]);  embedding_20 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_22: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 420, 440)
        embedding_21: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_22);  slice_22 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_22: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_21, [1]);  embedding_21 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_23: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 440, 460)
        embedding_22: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_23);  slice_23 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_23: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_22, [1]);  embedding_22 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_24: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 460, 480)
        embedding_23: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_24);  slice_24 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_24: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_23, [1]);  embedding_23 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_25: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 480, 500)
        embedding_24: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_25);  slice_25 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_25: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_24, [1]);  embedding_24 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_26: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 500, 520)
        embedding_25: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_26);  slice_26 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_26: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_25, [1]);  embedding_25 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_27: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 520, 540)
        embedding_26: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_27);  slice_27 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_27: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_26, [1]);  embedding_26 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_28: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 540, 560)
        embedding_27: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_28);  slice_28 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_28: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_27, [1]);  embedding_27 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_29: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 560, 580)
        embedding_28: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_29);  slice_29 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_29: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_28, [1]);  embedding_28 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_30: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 580, 600)
        embedding_29: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_30);  slice_30 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_30: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_29, [1]);  embedding_29 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_31: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 600, 620)
        embedding_30: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_31);  slice_31 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_31: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_30, [1]);  embedding_30 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_32: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 620, 640)
        embedding_31: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_32);  slice_32 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_32: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_31, [1]);  embedding_31 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_33: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 640, 660)
        embedding_32: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_33);  slice_33 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_33: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_32, [1]);  embedding_32 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_34: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 660, 680)
        embedding_33: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_34);  slice_34 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_34: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_33, [1]);  embedding_33 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_35: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 680, 700)
        embedding_34: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_35);  slice_35 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_35: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_34, [1]);  embedding_34 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_36: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 700, 720)
        embedding_35: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_36);  slice_36 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_36: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_35, [1]);  embedding_35 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_37: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 720, 740)
        embedding_36: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_37);  slice_37 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_37: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_36, [1]);  embedding_36 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_38: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 740, 760)
        embedding_37: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_38);  slice_38 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_38: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_37, [1]);  embedding_37 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_39: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 760, 780)
        embedding_38: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_39);  slice_39 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_39: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_38, [1]);  embedding_38 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_40: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 780, 800)
        embedding_39: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_40);  slice_40 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_40: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_39, [1]);  embedding_39 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_41: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 800, 820)
        embedding_40: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_41);  slice_41 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_41: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_40, [1]);  embedding_40 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_42: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 820, 840)
        embedding_41: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_42);  slice_42 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_42: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_41, [1]);  embedding_41 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_43: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 840, 860)
        embedding_42: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_43);  slice_43 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_43: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_42, [1]);  embedding_42 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_44: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 860, 880)
        embedding_43: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_44);  slice_44 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_44: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_43, [1]);  embedding_43 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_45: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 880, 900)
        embedding_44: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_45);  slice_45 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_45: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_44, [1]);  embedding_44 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_46: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 900, 920)
        embedding_45: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_46);  slice_46 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_46: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_45, [1]);  embedding_45 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_47: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 920, 940)
        embedding_46: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_47);  slice_47 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_47: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_46, [1]);  embedding_46 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_48: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 940, 960)
        embedding_47: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_48);  slice_48 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_48: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_47, [1]);  embedding_47 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_49: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 960, 980)
        embedding_48: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_49);  slice_49 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_49: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_48, [1]);  embedding_48 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:97 in embedding_layer, code: emb = self.feature_embedding(feat_ids[:, start:end])
        slice_50: "i64[128, 20]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 980, 1000);  arg0_1 = None
        embedding_49: "f32[128, 20, 8]" = torch.ops.aten.embedding.default(arg1_1, slice_50);  arg1_1 = slice_50 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:98 in embedding_layer, code: group_sum = torch.sum(emb, dim=1)
        sum_50: "f32[128, 8]" = torch.ops.aten.sum.dim_IntList(embedding_49, [1]);  embedding_49 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:100 in embedding_layer, code: out = torch.cat(group_sums, dim=1)
        cat: "f32[128, 400]" = torch.ops.aten.cat.default([sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7, sum_8, sum_9, sum_10, sum_11, sum_12, sum_13, sum_14, sum_15, sum_16, sum_17, sum_18, sum_19, sum_20, sum_21, sum_22, sum_23, sum_24, sum_25, sum_26, sum_27, sum_28, sum_29, sum_30, sum_31, sum_32, sum_33, sum_34, sum_35, sum_36, sum_37, sum_38, sum_39, sum_40, sum_41, sum_42, sum_43, sum_44, sum_45, sum_46, sum_47, sum_48, sum_49, sum_50], 1);  sum_1 = sum_2 = sum_3 = sum_4 = sum_5 = sum_6 = sum_7 = sum_8 = sum_9 = sum_10 = sum_11 = sum_12 = sum_13 = sum_14 = sum_15 = sum_16 = sum_17 = sum_18 = sum_19 = sum_20 = sum_21 = sum_22 = sum_23 = sum_24 = sum_25 = sum_26 = sum_27 = sum_28 = sum_29 = sum_30 = sum_31 = sum_32 = sum_33 = sum_34 = sum_35 = sum_36 = sum_37 = sum_38 = sum_39 = sum_40 = sum_41 = sum_42 = sum_43 = sum_44 = sum_45 = sum_46 = sum_47 = sum_48 = sum_49 = sum_50 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:128 in forward, code: dense_embedding = self.deep_layers(feat_vals)
        permute: "f32[1000, 6144]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg3_1, arg4_1, permute);  arg3_1 = arg4_1 = permute = None
        relu: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "f32[6144, 6144]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg6_1, relu, permute_1);  arg6_1 = relu = permute_1 = None
        relu_1: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_2: "f32[6144, 6144]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg8_1, relu_1, permute_2);  arg8_1 = relu_1 = permute_2 = None
        relu_2: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        permute_3: "f32[6144, 6144]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg10_1, relu_2, permute_3);  arg10_1 = relu_2 = permute_3 = None
        relu_3: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        permute_4: "f32[6144, 6144]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_4: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg12_1, relu_3, permute_4);  arg12_1 = relu_3 = permute_4 = None
        relu_4: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        permute_5: "f32[6144, 6144]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_5: "f32[128, 6144]" = torch.ops.aten.addmm.default(arg14_1, relu_4, permute_5);  arg14_1 = relu_4 = permute_5 = None
        relu_5: "f32[128, 6144]" = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        permute_6: "f32[6144, 3072]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_6: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg16_1, relu_5, permute_6);  arg16_1 = relu_5 = permute_6 = None
        relu_6: "f32[128, 3072]" = torch.ops.aten.relu.default(addmm_6);  addmm_6 = None
        permute_7: "f32[3072, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_7: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg18_1, relu_6, permute_7);  arg18_1 = relu_6 = permute_7 = None
        relu_7: "f32[128, 3072]" = torch.ops.aten.relu.default(addmm_7);  addmm_7 = None
        permute_8: "f32[3072, 3072]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_8: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg20_1, relu_7, permute_8);  arg20_1 = relu_7 = permute_8 = None
        relu_8: "f32[128, 3072]" = torch.ops.aten.relu.default(addmm_8);  addmm_8 = None
        permute_9: "f32[3072, 1536]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_9: "f32[128, 1536]" = torch.ops.aten.addmm.default(arg22_1, relu_8, permute_9);  arg22_1 = relu_8 = permute_9 = None
        relu_9: "f32[128, 1536]" = torch.ops.aten.relu.default(addmm_9);  addmm_9 = None
        permute_10: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_10: "f32[128, 1536]" = torch.ops.aten.addmm.default(arg24_1, relu_9, permute_10);  arg24_1 = relu_9 = permute_10 = None
        relu_10: "f32[128, 1536]" = torch.ops.aten.relu.default(addmm_10);  addmm_10 = None
        permute_11: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_11: "f32[128, 1536]" = torch.ops.aten.addmm.default(arg26_1, relu_10, permute_11);  arg26_1 = relu_10 = permute_11 = None
        relu_11: "f32[128, 1536]" = torch.ops.aten.relu.default(addmm_11);  addmm_11 = None
        permute_12: "f32[1536, 8]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_12: "f32[128, 8]" = torch.ops.aten.addmm.default(arg28_1, relu_11, permute_12);  arg28_1 = relu_11 = permute_12 = None
        relu_12: "f32[128, 8]" = torch.ops.aten.relu.default(addmm_12);  addmm_12 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:130 in forward, code: torch.cat([dense_embedding, sparse_embedding], dim=1)
        cat_1: "f32[128, 408]" = torch.ops.aten.cat.default([relu_12, cat], 1);  cat = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:107 in dot_interaction, code: inputs = inputs.unsqueeze(-1)
        unsqueeze: "f32[128, 408, 1]" = torch.ops.aten.unsqueeze.default(cat_1, -1);  cat_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:109 in dot_interaction, code: inputs, inputs.transpose(1, 2)
        permute_13: "f32[128, 1, 408]" = torch.ops.aten.permute.default(unsqueeze, [0, 2, 1])

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:108 in dot_interaction, code: xactions = torch.matmul(
        expand: "f32[128, 408, 1]" = torch.ops.aten.expand.default(unsqueeze, [128, 408, 1]);  unsqueeze = None
        expand_1: "f32[128, 1, 408]" = torch.ops.aten.expand.default(permute_13, [128, 1, 408]);  permute_13 = None
        bmm: "f32[128, 408, 408]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:111 in dot_interaction, code: ones = torch.ones_like(xactions, dtype = torch.float32)
        full_default: "f32[128, 408, 408]" = torch.ops.aten.full.default([128, 408, 408], 1, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:112 in dot_interaction, code: upper_tri_mask = torch.triu(ones, diagonal=0)
        iota: "i64[408]" = torch.ops.prims.iota.default(408, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 408]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[408]" = torch.ops.prims.iota.default(408, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_2: "i64[408, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[408, 408]" = torch.ops.aten.sub.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        ge: "b8[408, 408]" = torch.ops.aten.ge.Scalar(sub, 0);  sub = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[128, 408, 408]" = torch.ops.aten.where.self(ge, full_default, full_default_1);  ge = full_default = full_default_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:114 in dot_interaction, code: condition = upper_tri_mask.to(bool),
        convert_element_type: "b8[128, 408, 408]" = torch.ops.prims.convert_element_type.default(where, torch.bool);  where = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:115 in dot_interaction, code: input = torch.zeros_like(xactions),
        full_default_2: "f32[128, 408, 408]" = torch.ops.aten.full.default([128, 408, 408], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:113 in dot_interaction, code: activations = torch.where(
        where_1: "f32[128, 408, 408]" = torch.ops.aten.where.self(convert_element_type, full_default_2, bmm);  convert_element_type = full_default_2 = bmm = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:118 in dot_interaction, code: activations = torch.reshape(
        view_3: "f32[128, 166464]" = torch.ops.aten.view.default(where_1, [128, 166464]);  where_1 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:133 in forward, code: torch.concat([dense_embedding, interaction_output], dim=-1)
        cat_2: "f32[128, 166472]" = torch.ops.aten.cat.default([relu_12, view_3], -1);  relu_12 = view_3 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:132 in forward, code: pred = self.predict_layers(
        permute_14: "f32[166472, 1]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_13: "f32[128, 1]" = torch.ops.aten.addmm.default(arg30_1, cat_2, permute_14);  arg30_1 = cat_2 = permute_14 = None

        # File: <workspace>/rec_model_zoo_pytorch/feature_interaction/dlrm.py:135 in forward, code: return {"ctr": torch.sigmoid(pred).squeeze(dim=1)}
        sigmoid: "f32[128, 1]" = torch.ops.aten.sigmoid.default(addmm_13);  addmm_13 = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(sigmoid, 1);  sigmoid = None
        return (squeeze,)
