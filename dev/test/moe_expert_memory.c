// Artifact-independent unit tests for experimental top-2 MoE + expert-local memory.
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
    model.moe_config.moe_topk = 2;
    model.moe_config.moe_apply_every_n_layers = 1;
    model.moe_config.moe_expert_memory_slots = S;
    model.moe_config.moe_expert_memory_dim = D;
    model.moe_config.moe_memory_update_rate = 0.5f;
    model.moe_config.moe_memory_fusion_scale = 1.0f;
    model.moe_config.moe_router_temperature = 1.0f;
    model.moe_config.moe_load_balance_coef = 0.0f;
    gpt2_init_moe_state(&model, 1, 8);
    reset_moe_weights(&model);
    return model;
}

static int approx(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    {
        GPT2 model = make_test_model(3, 2, 1, 1);
        model.moe_config.moe_load_balance_coef = 0.5f;
        model.moe_config.moe_memory_fusion_scale = 0.0f;
        model.moe.router_b[0] = 2.0f;
        model.moe.router_b[1] = 0.0f;
        model.moe.router_b[2] = -4.0f;
        float ln2[4] = {0};
        float out[4] = {0};
        moe_reset_batch_stats(&model);
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 2, 2);

        int E = model.moe_config.moe_num_experts;
        int N = model.moe.layer_active_tokens[0];
        assert(N == 2);
        float denom = expf(2.0f) + expf(0.0f) + expf(-4.0f);
        float p0 = expf(2.0f) / denom;
        float p1 = expf(0.0f) / denom;
        float p2 = expf(-4.0f) / denom;
        assert(approx(model.moe.layer_importance_sum[0 * E + 0], 2.0f * p0, 1e-5f));
        assert(approx(model.moe.layer_importance_sum[0 * E + 1], 2.0f * p1, 1e-5f));
        assert(approx(model.moe.layer_importance_sum[0 * E + 2], 2.0f * p2, 1e-5f));
        assert(approx(model.moe.layer_load_count[0 * E + 0], 2.0f, 1e-6f));
        assert(approx(model.moe.layer_load_count[0 * E + 1], 2.0f, 1e-6f));
        assert(approx(model.moe.layer_load_count[0 * E + 2], 0.0f, 1e-6f));

        float expected_loss = model.moe_config.moe_load_balance_coef * (float)E * (p0 + p1);
        assert(isfinite(model.moe.layer_balance_loss[0]));
        assert(model.moe.layer_balance_loss[0] >= 0.0f);
        assert(approx(model.moe.layer_balance_loss[0], expected_loss, 1e-5f));
        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(3, 2, 1, 1);
        model.moe_config.moe_memory_fusion_scale = 0.0f;
        model.moe.router_b[0] = 2.0f;
        model.moe.router_b[1] = 0.0f;
        model.moe.router_b[2] = -4.0f;
        float ln2[2] = {0.0f, 0.0f};
        float out[2] = {0};
        float dout[2] = {0.0f, 0.0f}; // isolate aux-load-balance gradient path
        float dln2[2] = {0.0f, 0.0f};

        moe_reset_batch_stats(&model);
        model.moe_config.moe_load_balance_coef = 0.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 1, 2);
        moe_zero_param_grads(&model);
        moe_backward_topk_expert_local(&model, 0, ln2, dout, dln2, 1, 1, 2);
        float grad_no_aux = fabsf(model.moe.d_router_b[0]) + fabsf(model.moe.d_router_b[1]) + fabsf(model.moe.d_router_b[2]);
        assert(approx(grad_no_aux, 0.0f, 1e-8f));

        moe_reset_batch_stats(&model);
        model.moe_config.moe_load_balance_coef = 0.5f;
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 1, 2);
        moe_zero_param_grads(&model);
        memset(dln2, 0, sizeof(dln2));
        moe_backward_topk_expert_local(&model, 0, ln2, dout, dln2, 1, 1, 2);
        float grad_with_aux = fabsf(model.moe.d_router_b[0]) + fabsf(model.moe.d_router_b[1]) + fabsf(model.moe.d_router_b[2]);
        assert(grad_with_aux > 1e-7f);

        float router_before = model.moe.router_b[0];
        model.grads_memory = (float*)calloc(model.num_parameters, sizeof(float));
        model.grads_acts_memory = (float*)calloc(model.num_activations, sizeof(float));
        gpt2_update(&model, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
        assert(!approx(model.moe.router_b[0], router_before, 1e-12f));
        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(4, 4, 2, 2);
        // Routing by hidden[0]: logits descend with expert id.
        model.moe.router_w[0 * 4 + 0] = 3.0f;
        model.moe.router_w[1 * 4 + 0] = 1.5f;
        model.moe.router_w[2 * 4 + 0] = -1.0f;
        model.moe.router_w[3 * 4 + 0] = -2.0f;
        float ln2[4 * 4] = {
            1,0,0,0,
            2,0,0,0,
            -1,0,0,0,
            3,0,0,0,
        };
        float out[4 * 4];
        memset(out, 0, sizeof(out));
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 4, 4);
        for (int bt = 0; bt < 4; bt++) {
            int* sel = model.moe.selected_expert + bt * 2;
            float* w = model.moe.selected_expert_weight + bt * 2;
            assert(sel[0] >= 0 && sel[0] < 4);
            assert(sel[1] >= 0 && sel[1] < 4);
            assert(sel[0] != sel[1]);
            assert(approx(w[0] + w[1], 1.0f, 1e-5f));
            assert(w[0] >= 0.0f && w[1] >= 0.0f);
        }
        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(3, 4, 1, 1);
        model.moe_config.moe_memory_fusion_scale = 0.0f; // isolate weighted expert output combine

        // Constant per-expert outputs through fcproj biases.
        model.moe.expert_fcprojb[0 * 4 + 0] = 10.0f;
        model.moe.expert_fcprojb[1 * 4 + 0] = -6.0f;
        model.moe.expert_fcprojb[2 * 4 + 0] = 100.0f; // should never contribute when not selected

        float ln2[4] = {0};
        float out_a[4] = {0};
        float out_b[4] = {0};
        float out_c[4] = {0};

        // Case A: expert 0 favored over expert 1.
        model.moe.router_b[0] = 3.0f;
        model.moe.router_b[1] = 2.0f;
        model.moe.router_b[2] = -10.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out_a, 1, 1, 4);
        int* sel_a = model.moe.selected_expert;
        assert((sel_a[0] == 0 || sel_a[1] == 0) && (sel_a[0] == 1 || sel_a[1] == 1));

        // Case B: flip preference; output must change.
        model.moe.router_b[0] = 2.0f;
        model.moe.router_b[1] = 3.0f;
        model.moe.router_b[2] = -10.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out_b, 1, 1, 4);
        assert(!approx(out_a[0], out_b[0], 1e-4f));

        // Case C: expert 0 dominates strongly.
        model.moe.router_b[0] = 8.0f;
        model.moe.router_b[1] = 1.0f;
        model.moe.router_b[2] = -10.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out_c, 1, 1, 4);
        assert(fabsf(out_c[0] - 10.0f) < 0.05f);

        // Non-selected expert (2) does not contribute despite huge bias output.
        assert(fabsf(out_a[0]) < 20.0f);
        assert(fabsf(out_b[0]) < 20.0f);
        assert(fabsf(out_c[0]) < 20.0f);

        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(3, 2, 1, 1);
        model.moe_config.moe_memory_update_rate = 0.5f;
        model.moe_config.moe_memory_fusion_scale = 0.0f;

        // Make u[0] = 2 for all experts, and write[d0] = u[0].
        for (int e = 0; e < 3; e++) {
            model.moe.expert_fcprojb[e * 2 + 0] = 2.0f;
            model.moe.memory_write_proj[(e * 1 + 0) * 2 + 0] = 1.0f;
            model.moe.memory_query_proj[(e * 1 + 0) * 2 + 0] = 1.0f;
            model.moe.expert_memory_keys[(e * 1 + 0) * 1 + 0] = 1.0f;
        }

        float ln2[2] = {0};
        float out[2] = {0};

        // First pass: experts 0 and 1 selected, expert 2 unselected.
        model.moe.router_b[0] = 3.0f;
        model.moe.router_b[1] = 2.0f;
        model.moe.router_b[2] = -10.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 1, 2);
        float mem0_a = model.moe.expert_memory[(0 * 1 + 0) * 1 + 0];
        float mem1_a = model.moe.expert_memory[(1 * 1 + 0) * 1 + 0];
        float mem2_a = model.moe.expert_memory[(2 * 1 + 0) * 1 + 0];
        assert(mem0_a > 0.0f && mem1_a > 0.0f);
        assert(approx(mem2_a, 0.0f, 1e-7f));

        // Reset memory and rerun with reversed router preference to verify weighted write scaling.
        model.moe.expert_memory[(0 * 1 + 0) * 1 + 0] = 0.0f;
        model.moe.expert_memory[(1 * 1 + 0) * 1 + 0] = 0.0f;
        model.moe.expert_memory[(2 * 1 + 0) * 1 + 0] = 0.0f;
        model.moe.router_b[0] = 1.0f;
        model.moe.router_b[1] = 4.0f;
        model.moe.router_b[2] = -10.0f;
        moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 1, 2);
        float mem0_b = model.moe.expert_memory[(0 * 1 + 0) * 1 + 0];
        float mem1_b = model.moe.expert_memory[(1 * 1 + 0) * 1 + 0];
        float mem2_b = model.moe.expert_memory[(2 * 1 + 0) * 1 + 0];
        assert(mem1_b > mem0_b); // higher router weight => larger write magnitude
        assert(approx(mem2_b, 0.0f, 1e-7f));

        // Finite/stable forward behavior over repeated writes.
        for (int i = 0; i < 32; i++) {
            moe_forward_topk_expert_local(&model, 0, ln2, out, 1, 1, 2);
            assert(isfinite(out[0]) && isfinite(out[1]));
            for (int e = 0; e < 3; e++) {
                assert(isfinite(model.moe.expert_memory[(e * 1 + 0) * 1 + 0]));
            }
        }

        gpt2_free(&model);
    }

    {
        GPT2 model = make_test_model(3, 2, 1, 1);
        model.moe_config.moe_memory_fusion_scale = 0.0f;
        model.moe.router_b[0] = 3.0f;
        model.moe.router_b[1] = 2.0f;
        model.moe.router_b[2] = -10.0f;
        model.moe.expert_fcprojb[0 * 2 + 0] = 1.0f;
        model.moe.expert_fcprojb[1 * 2 + 0] = -2.0f;
        model.moe.expert_fcprojb[2 * 2 + 0] = 4.0f;

        float ln2[2] = {0.0f, 0.0f};
        float out_before[2] = {0};
        float out_after[2] = {0};
        float dout[2] = {1.0f, 0.0f};
        float dln2[2] = {0};

        moe_forward_topk_expert_local(&model, 0, ln2, out_before, 1, 1, 2);
        int sel0 = model.moe.selected_expert[0];
        int sel1 = model.moe.selected_expert[1];
        int non_selected = 3 - (sel0 + sel1); // valid for experts {0,1,2}
        assert(sel0 != sel1);
        assert(non_selected >= 0 && non_selected < 3);
        moe_zero_param_grads(&model);
        moe_backward_topk_expert_local(&model, 0, ln2, dout, dln2, 1, 1, 2);

        assert(fabsf(model.moe.d_expert_fcprojb[sel0 * 2 + 0]) > 1e-7f);
        assert(fabsf(model.moe.d_expert_fcprojb[sel1 * 2 + 0]) > 1e-7f);
        assert(approx(model.moe.d_expert_fcprojb[non_selected * 2 + 0], 0.0f, 1e-8f));
        assert(fabsf(model.moe.d_router_b[sel0]) > 1e-7f || fabsf(model.moe.d_router_b[sel1]) > 1e-7f);
        assert(approx(model.moe.d_router_b[non_selected], 0.0f, 1e-8f));

        float sel_bias_before = model.moe.expert_fcprojb[sel0 * 2 + 0];
        float nonsel_bias_before = model.moe.expert_fcprojb[non_selected * 2 + 0];
        float router_before = model.moe.router_b[sel0];
        model.grads_memory = (float*)calloc(model.num_parameters, sizeof(float));
        model.grads_acts_memory = (float*)calloc(model.num_activations, sizeof(float));
        gpt2_update(&model, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);

        assert(!approx(model.moe.expert_fcprojb[sel0 * 2 + 0], sel_bias_before, 1e-10f));
        assert(approx(model.moe.expert_fcprojb[non_selected * 2 + 0], nonsel_bias_before, 1e-10f));
        assert(!approx(model.moe.router_b[sel0], router_before, 1e-10f));

        moe_forward_topk_expert_local(&model, 0, ln2, out_after, 1, 1, 2);
        assert(!approx(out_before[0], out_after[0], 1e-8f));

        gpt2_free(&model);
    }

    {
        GPT2 model;
        memset(&model, 0, sizeof(model));
        GPT2Config cfg = {.max_seq_len = 8, .vocab_size = 32, .padded_vocab_size = 32, .num_layers = 1, .num_heads = 1, .channels = 8};
        gpt2_build_from_synthetic(&model, cfg);
        model.moe_config.use_moe = 0;
        int B = 1, T = 2;
        int inputs[2] = {1, 2};
        int targets[2] = {2, 3};
        gpt2_forward(&model, inputs, targets, B, T);
        assert(approx(model.moe_aux_loss, 0.0f, 1e-12f));
        gpt2_free(&model);
    }

    printf("moe_expert_memory top-2 tests passed\n");
    return 0;
}
