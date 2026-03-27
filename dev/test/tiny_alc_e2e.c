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
    gpt2_forward(&model, inputs, targets, B, T);
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

    gpt2_forward(&model, inputs, targets, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, 2);
    float alc_loss = model.mean_loss;

    // ALC persistence round-trip validation
    const char* alc_state_file = "dev/test/.tiny_alc_state.bin";
    int saved = gpt2_save_alc_state(&model, alc_state_file);
    assert(saved == 1);
    float slot_before = model.alc.slots[0];
    memset(model.alc.slots, 0, (size_t)alc.alc_num_slots * alc.alc_slot_dim * sizeof(float));
    int loaded = gpt2_load_alc_state(&model, alc_state_file, B, T);
    assert(loaded == 1);
    assert(model.alc.slots[0] == slot_before);
    remove(alc_state_file);

    printf("tiny e2e baseline loss: %.6f\n", baseline_loss);
    printf("tiny e2e ALC loss: %.6f\n", alc_loss);
    printf("tiny ALC e2e test passed\n");

    gpt2_free(&model);
    free(inputs);
    free(targets);
    return 0;
}
