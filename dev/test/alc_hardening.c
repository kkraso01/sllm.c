// Artifact-independent ALC hardening tests:
// - gradient finite-difference checks (representative subset)
// - EMA update correctness
// - persistence identity across save/load into fresh model
// - long-run boundedness/stability stress checks

#define TESTING
#include "../../train_gpt2.c"

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) { m = d; }
    }
    return m;
}

static int all_finite(const float* x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) { return 0; }
    }
    return 1;
}

static float fused_loss_for_gradcheck(GPT2* model, const float* hidden_in, int B, int T) {
    int C = model->config.channels;
    int BT = B * T;
    float* hidden = (float*)mallocCheck((size_t)BT * C * sizeof(float));
    memcpy(hidden, hidden_in, (size_t)BT * C * sizeof(float));
    alc_forward_read_and_fuse(model, hidden, B, T, 0);
    float loss = 0.0f;
    for (int i = 0; i < BT * C; i++) {
        loss += hidden[i];
    }
    free(hidden);
    return loss;
}

static void test_gradient_fd(void) {
    GPT2 model;
    memset(&model, 0, sizeof(model));
    model.config.channels = 4;
    model.config.num_layers = 1;
    model.alc_config = (ALCConfig){
        .use_alc = 1,
        .alc_num_slots = 3,
        .alc_slot_dim = 3,
        .alc_key_dim = 3,
        .alc_update_rate = 0.5f,
        .alc_fusion_mode = ALC_FUSION_GATED,
        .alc_update_mode = ALC_UPDATE_ALWAYS,
        .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f,
    };
    int B = 1;
    int T = 2;
    int C = model.config.channels;
    int BT = B * T;
    gpt2_init_alc_state(&model, B, T);
    alc_ensure_layer_traces(&model, B, T);
    alc_ensure_grad_buffers(&model);

    // force stable routing margin
    memset(model.alc.slot_keys, 0, (size_t)model.alc_config.alc_num_slots * model.alc_config.alc_key_dim * sizeof(float));
    model.alc.slot_keys[0] = 2.0f;
    model.alc.slot_keys[model.alc_config.alc_key_dim + 0] = 0.2f;
    model.alc.slot_keys[2 * model.alc_config.alc_key_dim + 0] = -0.2f;
    for (int i = 0; i < model.alc_config.alc_num_slots * model.alc_config.alc_slot_dim; i++) {
        model.alc.slots[i] = 0.05f * (float)(i + 1);
    }

    float hidden_in[2 * 4] = {
        0.2f, -0.1f, 0.05f, 0.3f,
        -0.15f, 0.11f, 0.08f, -0.04f,
    };

    float hidden[2 * 4];
    memcpy(hidden, hidden_in, sizeof(hidden));
    alc_forward_read_and_fuse(&model, hidden, B, T, 0);

    float d_hidden_out[2 * 4];
    for (int i = 0; i < BT * C; i++) { d_hidden_out[i] = 1.0f; }
    alc_zero_param_grads(&model.alc, C, model.alc_config.alc_slot_dim);
    alc_backward_fuse_and_accumulate(&model, 0, d_hidden_out, B, T);

    assert(all_finite(model.alc.d_slot_to_hidden, (size_t)C * model.alc_config.alc_slot_dim));
    assert(all_finite(model.alc.d_gate_h, (size_t)C));
    assert(all_finite(model.alc.d_gate_a, (size_t)C));
    assert(all_finite(model.alc.d_gate_b, (size_t)C));
    int any_gate_nonzero = 0;
    for (int i = 0; i < C; i++) {
        if (fabsf(model.alc.d_gate_h[i]) > 1e-7f || fabsf(model.alc.d_gate_a[i]) > 1e-7f || fabsf(model.alc.d_gate_b[i]) > 1e-7f) {
            any_gate_nonzero = 1;
            break;
        }
    }
    assert(any_gate_nonzero);

    // representative subset finite-difference checks
    const float eps = 1e-3f;
    const float rel_tol = 2e-2f;
    const float abs_tol = 2e-3f;

    int row = 0;
    int col = 0;
    size_t idx = (size_t)row * model.alc_config.alc_slot_dim + col;
    float base = model.alc.slot_to_hidden[idx];
    model.alc.slot_to_hidden[idx] = base + eps;
    float lpos = fused_loss_for_gradcheck(&model, hidden_in, B, T);
    model.alc.slot_to_hidden[idx] = base - eps;
    float lneg = fused_loss_for_gradcheck(&model, hidden_in, B, T);
    model.alc.slot_to_hidden[idx] = base;
    float g_num = (lpos - lneg) / (2.0f * eps);
    float g_ana = model.alc.d_slot_to_hidden[idx];
    float err = fabsf(g_num - g_ana);
    float denom = fabsf(g_num) + fabsf(g_ana) + 1e-6f;
    assert(err <= abs_tol || (err / denom) <= rel_tol);

    for (int cidx = 0; cidx < 2; cidx++) {
        float* tensors[3] = { model.alc.gate_h, model.alc.gate_a, model.alc.gate_b };
        float* grads[3] = { model.alc.d_gate_h, model.alc.d_gate_a, model.alc.d_gate_b };
        for (int t = 0; t < 3; t++) {
            float b = tensors[t][cidx];
            tensors[t][cidx] = b + eps;
            float lp = fused_loss_for_gradcheck(&model, hidden_in, B, T);
            tensors[t][cidx] = b - eps;
            float ln = fused_loss_for_gradcheck(&model, hidden_in, B, T);
            tensors[t][cidx] = b;
            float gn = (lp - ln) / (2.0f * eps);
            float ga = grads[t][cidx];
            float e = fabsf(gn - ga);
            float d = fabsf(gn) + fabsf(ga) + 1e-6f;
            assert(e <= abs_tol || (e / d) <= rel_tol);
        }
    }

    gpt2_free(&model);
}

static void test_ema_update_rules(void) {
    GPT2 model;
    memset(&model, 0, sizeof(model));
    model.config.channels = 2;
    model.config.num_layers = 1;
    model.alc_config = (ALCConfig){
        .use_alc = 1,
        .alc_num_slots = 3,
        .alc_slot_dim = 2,
        .alc_key_dim = 2,
        .alc_update_rate = 0.5f,
        .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS,
        .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f,
    };
    int B = 1, T = 1;
    gpt2_init_alc_state(&model, B, T);

    // set write_proj to identity from hidden -> slot
    memset(model.alc.write_proj, 0, (size_t)model.alc_config.alc_slot_dim * model.config.channels * sizeof(float));
    model.alc.write_proj[0 * model.config.channels + 0] = 1.0f;
    model.alc.write_proj[1 * model.config.channels + 1] = 1.0f;
    memset(model.alc.slots, 0, (size_t)model.alc_config.alc_num_slots * model.alc_config.alc_slot_dim * sizeof(float));
    memset(model.alc.slot_keys, 0, (size_t)model.alc_config.alc_num_slots * model.alc_config.alc_key_dim * sizeof(float));

    float hidden[2] = { 2.0f, -4.0f };
    model.alc.selected_slots[0] = 1;
    model.alc.query_buffer[0] = 1.0f;
    model.alc.query_buffer[1] = -3.0f;

    // eta = 0 => no change
    model.alc_config.alc_update_rate = 0.0f;
    float before_slots[6];
    float before_keys[6];
    memcpy(before_slots, model.alc.slots, sizeof(before_slots));
    memcpy(before_keys, model.alc.slot_keys, sizeof(before_keys));
    alc_write_update(&model, hidden, B, T);
    assert(max_abs_diff(before_slots, model.alc.slots, 6) == 0.0f);
    assert(max_abs_diff(before_keys, model.alc.slot_keys, 6) == 0.0f);

    // eta = 1 => exact overwrite on selected slot only
    model.alc_config.alc_update_rate = 1.0f;
    alc_write_update(&model, hidden, B, T);
    assert(model.alc.slots[1 * 2 + 0] == hidden[0]);
    assert(model.alc.slots[1 * 2 + 1] == hidden[1]);
    assert(model.alc.slot_keys[1 * 2 + 0] == model.alc.query_buffer[0]);
    assert(model.alc.slot_keys[1 * 2 + 1] == model.alc.query_buffer[1]);
    // non-selected slots unchanged
    assert(model.alc.slots[0] == 0.0f && model.alc.slots[1] == 0.0f);
    assert(model.alc.slots[4] == 0.0f && model.alc.slots[5] == 0.0f);

    // small eta repeated writes converge to write vector
    memset(model.alc.slots, 0, sizeof(before_slots));
    model.alc_config.alc_update_rate = 0.1f;
    for (int i = 0; i < 400; i++) {
        alc_write_update(&model, hidden, B, T);
    }
    assert(fabsf(model.alc.slots[1 * 2 + 0] - hidden[0]) < 1e-3f);
    assert(fabsf(model.alc.slots[1 * 2 + 1] - hidden[1]) < 1e-3f);
    assert(model.alc.slots[0] == 0.0f && model.alc.slots[1] == 0.0f);
    assert(model.alc.slots[4] == 0.0f && model.alc.slots[5] == 0.0f);

    gpt2_free(&model);
}

static void fill_inputs_targets(int* inputs, int* targets, int n, int V) {
    for (int i = 0; i < n; i++) {
        inputs[i] = i % V;
        targets[i] = (i + 1) % V;
    }
}

static void test_persistence_identity(void) {
    GPT2Config cfg = {
        .max_seq_len = 8,
        .vocab_size = 32,
        .padded_vocab_size = 32,
        .num_layers = 1,
        .num_heads = 2,
        .channels = 8,
    };
    GPT2 m1, m2;
    gpt2_build_from_synthetic(&m1, cfg);
    gpt2_build_from_synthetic(&m2, cfg);

    ALCConfig alc = m1.alc_config;
    alc.use_alc = 1;
    alc.alc_num_slots = 5;
    alc.alc_slot_dim = 6;
    alc.alc_key_dim = 4;
    alc.alc_fusion_mode = ALC_FUSION_GATED;
    alc.alc_update_mode = ALC_UPDATE_ALWAYS;
    gpt2_set_alc_config(&m1, alc);
    gpt2_set_alc_config(&m2, alc);

    int B = 2;
    int T = 4;
    int n = B * T;
    int* inputs = (int*)mallocCheck((size_t)n * sizeof(int));
    int* targets = (int*)mallocCheck((size_t)n * sizeof(int));
    fill_inputs_targets(inputs, targets, n, cfg.vocab_size);

    gpt2_forward(&m1, inputs, targets, B, T);
    const char* state_path = "dev/test/.alc_hardening_state.bin";
    assert(gpt2_save_alc_state(&m1, state_path) == 1);
    assert(gpt2_load_alc_state(&m2, state_path, B, T) == 1);

    int C = cfg.channels;
    int S = alc.alc_num_slots;
    int D = alc.alc_slot_dim;
    int K = alc.alc_key_dim;
    assert(max_abs_diff(m1.alc.query_proj, m2.alc.query_proj, (size_t)K * C) == 0.0f);
    assert(max_abs_diff(m1.alc.write_proj, m2.alc.write_proj, (size_t)D * C) == 0.0f);
    assert(max_abs_diff(m1.alc.slot_to_hidden, m2.alc.slot_to_hidden, (size_t)C * D) == 0.0f);
    assert(max_abs_diff(m1.alc.gate_h, m2.alc.gate_h, (size_t)C) == 0.0f);
    assert(max_abs_diff(m1.alc.gate_a, m2.alc.gate_a, (size_t)C) == 0.0f);
    assert(max_abs_diff(m1.alc.gate_b, m2.alc.gate_b, (size_t)C) == 0.0f);
    assert(max_abs_diff(m1.alc.slot_keys, m2.alc.slot_keys, (size_t)S * K) == 0.0f);
    assert(max_abs_diff(m1.alc.slots, m2.alc.slots, (size_t)S * D) == 0.0f);

    gpt2_forward(&m1, inputs, targets, B, T);
    gpt2_forward(&m2, inputs, targets, B, T);
    assert(max_abs_diff(m1.acts.logits, m2.acts.logits, (size_t)B * T * cfg.padded_vocab_size) == 0.0f);

    remove(state_path);
    free(inputs);
    free(targets);
    gpt2_free(&m1);
    gpt2_free(&m2);
}

static void test_long_run_stability(void) {
    GPT2 model;
    GPT2Config cfg = {
        .max_seq_len = 8,
        .vocab_size = 64,
        .padded_vocab_size = 64,
        .num_layers = 1,
        .num_heads = 2,
        .channels = 12,
    };
    gpt2_build_from_synthetic(&model, cfg);
    ALCConfig alc = model.alc_config;
    alc.use_alc = 1;
    alc.alc_num_slots = 8;
    alc.alc_slot_dim = 10;
    alc.alc_key_dim = 6;
    alc.alc_fusion_mode = ALC_FUSION_GATED;
    alc.alc_update_mode = ALC_UPDATE_ALWAYS;
    alc.alc_update_rate = 0.05f;
    gpt2_set_alc_config(&model, alc);

    int B = 2;
    int T = 8;
    int n = B * T;
    int* inputs = (int*)mallocCheck((size_t)n * sizeof(int));
    int* targets = (int*)mallocCheck((size_t)n * sizeof(int));
    fill_inputs_targets(inputs, targets, n, cfg.vocab_size);

    const int steps = 800;
    float max_slot_norm = 0.0f;
    float max_key_norm = 0.0f;
    float max_fused_hidden_norm = 0.0f;
    float gate_min = 1.0f;
    float gate_max = 0.0f;
    double gate_sum = 0.0;
    long long gate_count = 0;

    for (int step = 0; step < steps; step++) {
        gpt2_forward(&model, inputs, targets, B, T);

        int S = alc.alc_num_slots;
        int D = alc.alc_slot_dim;
        int K = alc.alc_key_dim;
        int C = cfg.channels;
        int BT = B * T;
        for (int s = 0; s < S; s++) {
            float sn = alc_l2_norm(model.alc.slots + (size_t)s * D, D);
            float kn = alc_l2_norm(model.alc.slot_keys + (size_t)s * K, K);
            if (sn > max_slot_norm) { max_slot_norm = sn; }
            if (kn > max_key_norm) { max_key_norm = kn; }
        }
        for (int bt = 0; bt < BT; bt++) {
            const int slot = model.alc.selected_slots[bt];
            assert(slot >= 0 && slot < S);
            float* h = model.acts.residual3 + (size_t)bt * C;
            float hn = alc_l2_norm(h, C);
            if (hn > max_fused_hidden_norm) { max_fused_hidden_norm = hn; }
            const float* hpre = model.alc.hidden_pre_layers + (size_t)bt * C;
            const float* r = model.alc.retrieved_layers + (size_t)bt * C;
            for (int c = 0; c < C; c++) {
                float logit = model.alc.gate_h[c] * hpre[c] + model.alc.gate_a[c] * r[c] + model.alc.gate_b[c];
                float g = alc_sigmoid(alc_clamp(logit, -40.0f, 40.0f));
                if (g < gate_min) { gate_min = g; }
                if (g > gate_max) { gate_max = g; }
                gate_sum += g;
                gate_count++;
            }
        }
        assert(all_finite(model.alc.slots, (size_t)S * D));
        assert(all_finite(model.alc.slot_keys, (size_t)S * K));
        assert(all_finite(model.acts.residual3, (size_t)BT * C));
    }

    const float slot_norm_threshold = 1e4f;
    const float key_norm_threshold = 1e4f;
    const float fused_norm_threshold = 1e4f;
    assert(max_slot_norm < slot_norm_threshold);
    assert(max_key_norm < key_norm_threshold);
    assert(max_fused_hidden_norm < fused_norm_threshold);
    assert(gate_count > 0);
    assert(gate_min >= 0.0f && gate_max <= 1.0f);

    // selection histogram should show non-degenerate routing
    int hit_slots = 0;
    for (int s = 0; s < alc.alc_num_slots; s++) {
        if (model.alc.slot_hit_counts[s] > 0) { hit_slots++; }
    }
    assert(hit_slots >= 2);

    printf("stress metrics: max_slot_norm=%.5f max_key_norm=%.5f max_fused_hidden_norm=%.5f gate_mean=%.5f gate_min=%.5f gate_max=%.5f\n",
           max_slot_norm, max_key_norm, max_fused_hidden_norm,
           (float)(gate_sum / (double)gate_count), gate_min, gate_max);

    free(inputs);
    free(targets);
    gpt2_free(&model);
}

int main(void) {
    test_gradient_fd();
    test_ema_update_rules();
    test_persistence_identity();
    test_long_run_stability();
    printf("ALC hardening tests passed\n");
    return 0;
}
