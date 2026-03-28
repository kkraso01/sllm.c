// Artifact-independent unit tests for experimental top-1 MoE + expert-local memory.
#define TESTING
#include "../../train_gpt2_moe_experimental.c"

static void reset_moe_weights(GPT2* model) {
    MoEState* moe = &model->moe;
    int E = model->moe_config.moe_num_experts;
    int C = model->config.channels;
    int S = model->moe_config.moe_expert_memory_slots;
    int D = model->moe_config.moe_expert_memory_dim;
    memset(moe->router_w, 0, (size_t)E * C * sizeof(float));
    memset(moe->router_b, 0, (size_t)E * sizeof(float));
    memset(moe->expert_fcw, 0, (size_t)E * (4 * C) * C * sizeof(float));
    memset(moe->expert_fcb, 0, (size_t)E * (4 * C) * sizeof(float));
    memset(moe->expert_fcprojw, 0, (size_t)E * C * (4 * C) * sizeof(float));
    memset(moe->expert_fcprojb, 0, (size_t)E * C * sizeof(float));
    memset(moe->expert_memory, 0, (size_t)E * S * D * sizeof(float));
    memset(moe->expert_memory_keys, 0, (size_t)E * S * D * sizeof(float));
    memset(moe->memory_query_proj, 0, (size_t)E * D * C * sizeof(float));
    memset(moe->memory_write_proj, 0, (size_t)E * D * C * sizeof(float));
    memset(moe->memory_slot_to_hidden, 0, (size_t)E * C * D * sizeof(float));
}

static GPT2 make_test_model(int E, int C, int S, int D) {
    GPT2 model;
    memset(&model, 0, sizeof(model));
    GPT2Config cfg = {.max_seq_len = 8, .vocab_size = 32, .padded_vocab_size = 32, .num_layers = 1, .num_heads = 1, .channels = C};
    gpt2_build_from_synthetic(&model, cfg);
    model.moe_config.use_moe = 1;
    model.moe_config.moe_num_experts = E;
    model.moe_config.moe_topk = 1;
    model.moe_config.moe_apply_every_n_layers = 1;
    model.moe_config.moe_expert_memory_slots = S;
    model.moe_config.moe_expert_memory_dim = D;
    model.moe_config.moe_memory_update_rate = 0.5f;
    model.moe_config.moe_memory_fusion_scale = 1.0f;
    gpt2_init_moe_state(&model, 1, 4);
    reset_moe_weights(&model);
    return model;
}

int main(void) {
    {
        GPT2 model = make_test_model(2, 4, 2, 2);
        // Route by sign of hidden[0].
        model.moe.router_w[0 * 4 + 0] = 2.0f;
        model.moe.router_w[1 * 4 + 0] = -2.0f;
        float ln2[4 * 4] = {
            1,0,0,0,
           -1,0,0,0,
            3,0,0,0,
           -2,0,0,0,
        };
        float out[4 * 4];
        memset(out, 0, sizeof(out));
        moe_forward_top1_expert_local(&model, ln2, out, 1, 4, 4);
        for (int bt = 0; bt < 4; bt++) {
            assert(model.moe.selected_expert[bt] >= 0 && model.moe.selected_expert[bt] < 2);
        }
        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(2, 4, 2, 2);
        // Force expert selection via router bias and give distinct expert outputs.
        model.moe.router_b[0] = 5.0f;
        model.moe.expert_fcprojb[0 * 4 + 0] = 1.0f;
        float ln2[4] = {0};
        float out_a[4] = {0};
        moe_forward_top1_expert_local(&model, ln2, out_a, 1, 1, 4);
        assert(model.moe.selected_expert[0] == 0);

        model.moe.router_b[0] = -5.0f;
        model.moe.router_b[1] = 5.0f;
        model.moe.expert_fcprojb[1 * 4 + 0] = -3.0f;
        float out_b[4] = {0};
        moe_forward_top1_expert_local(&model, ln2, out_b, 1, 1, 4);
        assert(model.moe.selected_expert[0] == 1);
        assert(out_a[0] != out_b[0]); // experts differ on same input
        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(2, 4, 2, 2);
        // Read-path activation: memory contributes to output.
        model.moe.router_b[0] = 3.0f;
        model.moe.memory_slot_to_hidden[(0 * 4 + 0) * 2 + 0] = 1.0f; // c0 <- d0
        model.moe.expert_memory[(0 * 2 + 0) * 2 + 0] = 4.0f;         // slot0,d0
        model.moe.expert_memory_keys[(0 * 2 + 0) * 2 + 0] = 1.0f;
        model.moe.memory_query_proj[(0 * 2 + 0) * 4 + 0] = 1.0f;      // q0 <- u0
        model.moe.expert_fcprojb[0 * 4 + 0] = 2.0f;                   // u0 = 2
        float ln2[4] = {0};
        float out[4] = {0};
        moe_forward_top1_expert_local(&model, ln2, out, 1, 1, 4);
        assert(out[0] > 2.0f); // fused with memory read

        // Write-path isolation: selected expert mutates, other expert unchanged.
        float other_before = model.moe.expert_memory[(1 * 2 + 0) * 2 + 0];
        float self_before = model.moe.expert_memory[(0 * 2 + 0) * 2 + 0];
        model.moe.memory_write_proj[(0 * 2 + 0) * 4 + 0] = 1.0f;
        moe_forward_top1_expert_local(&model, ln2, out, 1, 1, 4);
        float other_after = model.moe.expert_memory[(1 * 2 + 0) * 2 + 0];
        float self_after = model.moe.expert_memory[(0 * 2 + 0) * 2 + 0];
        assert(other_before == other_after);
        assert(self_before != self_after);

        // Repeated writes still must not mutate non-selected expert.
        for (int i = 0; i < 8; i++) {
            moe_forward_top1_expert_local(&model, ln2, out, 1, 1, 4);
        }
        assert(model.moe.expert_memory[(1 * 2 + 1) * 2 + 1] == 0.0f);
        for (int i = 0; i < 4; i++) { assert(isfinite(out[i])); }
        gpt2_free(&model);
    }

    printf("moe_expert_memory tests passed\n");
    return 0;
}
