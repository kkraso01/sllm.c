// Artifact-independent tiny GPT-2 + ALC end-to-end validation.
// Exercises gpt2_forward/gpt2_backward/gpt2_update through the integrated block path.

#define TESTING
#include "../../train_gpt2.c"

static void fill_tiny_batch(int* inputs, int* targets, int B, int T, int V) {
    for (int i = 0; i < B * T; i++) {
        inputs[i] = i % V;
        targets[i] = (i + 1) % V;
    }
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) { m = d; }
    }
    return m;
}

int main(void) {
    GPT2 model;
    GPT2Config cfg = {
        .max_seq_len = 8,
        .vocab_size = 64,
        .padded_vocab_size = 64,
        .num_layers = 2,
        .num_heads = 2,
        .channels = 16,
    };
    int B = 2;
    int T = 8;
    gpt2_build_from_synthetic(&model, cfg);

    int* inputs = (int*)mallocCheck((size_t)B * T * sizeof(int));
    int* targets = (int*)mallocCheck((size_t)B * T * sizeof(int));
    fill_tiny_batch(inputs, targets, B, T, cfg.vocab_size);

    // baseline (ALC disabled)
    float wte_before = model.params.wte[0];
    gpt2_forward(&model, inputs, targets, B, T);
    float baseline_logit0 = model.acts.logits[0];
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    float baseline_loss = model.mean_loss;

    // ALC-enabled run
    ALCConfig alc = model.alc_config;
    alc.use_alc = 1;
    alc.alc_num_slots = 8;
    alc.alc_slot_dim = 12;
    alc.alc_key_dim = 8;
    alc.alc_fusion_mode = ALC_FUSION_GATED;
    alc.alc_update_mode = ALC_UPDATE_ALWAYS;
    alc.alc_apply_every_n_layers = 1;
    gpt2_set_alc_config(&model, alc);

    // With ALC on, behavior should diverge from ALC-off behavior under identical inputs.
    gpt2_forward(&model, inputs, targets, B, T);
    float alc_logit0 = model.acts.logits[0];
    assert(fabsf(alc_logit0 - baseline_logit0) > 1e-7f);
    float slot_to_hidden_before = model.alc.slot_to_hidden[0];
    float gate_h_before = model.alc.gate_h[0];
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    assert(model.alc.d_slot_to_hidden != NULL);
    assert(model.alc.d_gate_h != NULL);
    assert(model.alc.d_slot_to_hidden[0] != 0.0f || model.alc.d_gate_h[0] != 0.0f);
    gpt2_update(&model, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, 2);
    assert(model.params.wte[0] != wte_before);
    assert(model.alc.slot_to_hidden[0] != slot_to_hidden_before || model.alc.gate_h[0] != gate_h_before);
    float alc_loss = model.mean_loss;

    // Deterministic write/update should alter future behavior in a stateful way.
    gpt2_forward(&model, inputs, targets, B, T);
    float logit_before_write = model.acts.logits[0];
    gpt2_forward(&model, inputs, targets, B, T);
    float logit_after_write = model.acts.logits[0];
    assert(fabsf(logit_after_write - logit_before_write) > 1e-8f);

    // ALC persistence round-trip validation
    const char* alc_state_file = "dev/test/.tiny_alc_state.bin";
    int saved = gpt2_save_alc_state(&model, alc_state_file);
    assert(saved == 1);
    int C = cfg.channels;
    int S = alc.alc_num_slots;
    int D = alc.alc_slot_dim;
    int K = alc.alc_key_dim;
    size_t n_query_proj = (size_t)K * C;
    size_t n_write_proj = (size_t)D * C;
    size_t n_slot_to_hidden = (size_t)C * D;
    size_t n_slots = (size_t)S * D;
    size_t n_slot_keys = (size_t)S * K;
    float* query_proj_copy = (float*)mallocCheck(n_query_proj * sizeof(float));
    float* write_proj_copy = (float*)mallocCheck(n_write_proj * sizeof(float));
    float* slot_to_hidden_copy = (float*)mallocCheck(n_slot_to_hidden * sizeof(float));
    float* slots_copy = (float*)mallocCheck(n_slots * sizeof(float));
    float* slot_keys_copy = (float*)mallocCheck(n_slot_keys * sizeof(float));
    memcpy(query_proj_copy, model.alc.query_proj, n_query_proj * sizeof(float));
    memcpy(write_proj_copy, model.alc.write_proj, n_write_proj * sizeof(float));
    memcpy(slot_to_hidden_copy, model.alc.slot_to_hidden, n_slot_to_hidden * sizeof(float));
    memcpy(slots_copy, model.alc.slots, n_slots * sizeof(float));
    memcpy(slot_keys_copy, model.alc.slot_keys, n_slot_keys * sizeof(float));
    memset(model.alc.slots, 0, (size_t)alc.alc_num_slots * alc.alc_slot_dim * sizeof(float));
    memset(model.alc.slot_keys, 0, (size_t)alc.alc_num_slots * alc.alc_key_dim * sizeof(float));
    memset(model.alc.slot_to_hidden, 0, (size_t)cfg.channels * alc.alc_slot_dim * sizeof(float));
    int loaded = gpt2_load_alc_state(&model, alc_state_file, B, T);
    assert(loaded == 1);
    assert(max_abs_diff(model.alc.query_proj, query_proj_copy, n_query_proj) == 0.0f);
    assert(max_abs_diff(model.alc.write_proj, write_proj_copy, n_write_proj) == 0.0f);
    assert(max_abs_diff(model.alc.slot_to_hidden, slot_to_hidden_copy, n_slot_to_hidden) == 0.0f);
    assert(max_abs_diff(model.alc.slots, slots_copy, n_slots) == 0.0f);
    assert(max_abs_diff(model.alc.slot_keys, slot_keys_copy, n_slot_keys) == 0.0f);
    gpt2_forward(&model, inputs, targets, B, T);
    assert(isfinite(model.mean_loss));
    remove(alc_state_file);
    free(query_proj_copy);
    free(write_proj_copy);
    free(slot_to_hidden_copy);
    free(slots_copy);
    free(slot_keys_copy);

    printf("tiny e2e baseline loss: %.6f\n", baseline_loss);
    printf("tiny e2e ALC loss: %.6f\n", alc_loss);
    printf("tiny ALC e2e test passed\n");

    gpt2_free(&model);
    free(inputs);
    free(targets);
    return 0;
}
