// Minimal smoke test for the in-model Adaptive Learning Component (ALC).
// This test intentionally includes train_gpt2.c directly in TESTING mode so
// static ALC helpers can be exercised without checkpoint artifacts.
#define TESTING
#include "../../train_gpt2.c"

int main(void) {
    GPT2 model;
    memset(&model, 0, sizeof(model));
    model.config.channels = 8;
    model.config.num_layers = 2;

    model.alc_config = (ALCConfig){
        .use_alc = 1,
        .alc_num_slots = 4,
        .alc_slot_dim = 6,
        .alc_key_dim = 5,
        .alc_update_rate = 0.5f,
        .alc_fusion_mode = ALC_FUSION_GATED,
        .alc_update_mode = ALC_UPDATE_ALWAYS,
        .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f,
    };

    int B = 2;
    int T = 3;
    int C = model.config.channels;
    gpt2_init_alc_state(&model, B, T);

    float hidden[2 * 3 * 8];
    for (int i = 0; i < B * T * C; i++) {
        hidden[i] = 0.01f * (float)(i + 1);
    }

    // Read/fuse path should populate selected slots.
    alc_forward_read_and_fuse(&model, hidden, B, T);
    for (int bt = 0; bt < B * T; bt++) {
        assert(model.alc.selected_slots[bt] >= 0);
        assert(model.alc.selected_slots[bt] < model.alc_config.alc_num_slots);
    }

    // Write path should update at least one selected slot value.
    int first_slot = model.alc.selected_slots[0];
    float before = model.alc.slots[first_slot * model.alc_config.alc_slot_dim + 0];
    alc_write_update(&model, hidden, B, T);
    float after = model.alc.slots[first_slot * model.alc_config.alc_slot_dim + 0];
    assert(before != after);

    gpt2_free(&model);
    printf("ALC smoke test passed\n");
    return 0;
}
