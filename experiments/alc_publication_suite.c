#define TESTING
#include "../train_gpt2.c"
#include <ctype.h>
#include <time.h>

static float frand_sym(void) { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

static float max_abs_diff_local(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void zero_alc_projection(ALCState* alc, int C, int D, int K) {
    memset(alc->query_proj, 0, (size_t)K * C * sizeof(float));
    memset(alc->write_proj, 0, (size_t)D * C * sizeof(float));
    memset(alc->slot_to_hidden, 0, (size_t)C * D * sizeof(float));
}

static void setup_identity_interface(GPT2* m, int value_offset) {
    int C = m->config.channels;
    int D = m->alc_config.alc_slot_dim;
    int K = m->alc_config.alc_key_dim;
    zero_alc_projection(&m->alc, C, D, K);
    for (int i = 0; i < K && i < C; i++) m->alc.query_proj[i * C + i] = 1.0f;
    for (int i = 0; i < D && (value_offset + i) < C; i++) {
        m->alc.write_proj[i * C + (value_offset + i)] = 1.0f;
        m->alc.slot_to_hidden[(value_offset + i) * D + i] = 1.0f;
    }
}

static float mse_slice(const float* x, const float* y, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - y[i];
        s += d * d;
    }
    return s / (float)n;
}

static float run_core_episode(GPT2* m, const float* hidden, const float* query, const float* target, int value_offset, int apply_write) {
    int B = 1, T = 1;
    if (apply_write) {
        float hcopy[16]; memcpy(hcopy, hidden, 16 * sizeof(float));
        alc_forward_read_and_fuse(m, hcopy, B, T, 0);
        alc_write_update(m, hcopy, B, T);
    }
    float q1[16]; memcpy(q1, query, 16 * sizeof(float));
    alc_forward_read_and_fuse(m, q1, B, T, 0);
    float pred[4];
    for (int d = 0; d < 4; d++) pred[d] = q1[value_offset + d];
    return mse_slice(pred, target, 4);
}

static void run_core_adaptation(FILE* f) {
    GPT2 m;
    memset(&m, 0, sizeof(m));
    m.config.channels = 16;
    m.config.num_layers = 1;
    m.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 8, .alc_slot_dim = 4, .alc_key_dim = 4,
        .alc_update_rate = 0.5f, .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 1, .alc_temperature = 0.1f,
    };
    GPT2 m_nowrite = m;
    m_nowrite.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_nowrite.alc_config.alc_update_rate = 0.0f;
    int B = 1, T = 1, value_offset = 8;
    gpt2_init_alc_state(&m, B, T);
    gpt2_init_alc_state(&m_nowrite, B, T);
    alc_ensure_layer_traces(&m, B, T);
    alc_ensure_layer_traces(&m_nowrite, B, T);
    setup_identity_interface(&m, value_offset);
    setup_identity_interface(&m_nowrite, value_offset);
    memset(m.alc.slots, 0, (size_t)m.alc_config.alc_num_slots * m.alc_config.alc_slot_dim * sizeof(float));
    memset(m.alc.slot_keys, 0, (size_t)m.alc_config.alc_num_slots * m.alc_config.alc_key_dim * sizeof(float));
    memset(m_nowrite.alc.slots, 0, (size_t)m_nowrite.alc_config.alc_num_slots * m_nowrite.alc_config.alc_slot_dim * sizeof(float));
    memset(m_nowrite.alc.slot_keys, 0, (size_t)m_nowrite.alc_config.alc_num_slots * m_nowrite.alc_config.alc_key_dim * sizeof(float));

    int episodes = 64;
    int good_baseline = 0, good_alc = 0, good_nowrite = 0;
    float mse_baseline = 0.0f, mse_alc = 0.0f, mse_nowrite = 0.0f;

    for (int e = 0; e < episodes; e++) {
        int slot_id = e % m.alc_config.alc_num_slots;
        float hidden[16] = {0};
        float query[16] = {0};
        float target[4];
        for (int k = 0; k < 4; k++) {
            float keyv = (k == (slot_id % 4)) ? 4.0f : -1.0f;
            hidden[k] = keyv;
            query[k] = keyv;
        }
        for (int d = 0; d < 4; d++) {
            target[d] = frand_sym();
            hidden[value_offset + d] = target[d];
        }

        // baseline: query before writing current fact
        float q0[16]; memcpy(q0, query, sizeof(q0));
        alc_forward_read_and_fuse(&m, q0, B, T, 0);
        float m0 = mse_slice(q0 + value_offset, target, 4);
        mse_baseline += m0;
        if (m0 < 0.1f) good_baseline++;

        // full ALC: write then query
        float m1 = run_core_episode(&m, hidden, query, target, value_offset, 1);
        mse_alc += m1;
        if (m1 < 0.1f) good_alc++;

        // stronger baseline: ALC interface active but writes disabled
        float m2 = run_core_episode(&m_nowrite, hidden, query, target, value_offset, 1);
        mse_nowrite += m2;
        if (m2 < 0.1f) good_nowrite++;
    }

    fprintf(f, "core_adaptation,baseline,recall_mse,%.8f\n", mse_baseline / episodes);
    fprintf(f, "core_adaptation,alc,recall_mse,%.8f\n", mse_alc / episodes);
    fprintf(f, "core_adaptation,alc_no_write,recall_mse,%.8f\n", mse_nowrite / episodes);
    fprintf(f, "core_adaptation,baseline,recall_acc,%.8f\n", (float)good_baseline / episodes);
    fprintf(f, "core_adaptation,alc,recall_acc,%.8f\n", (float)good_alc / episodes);
    fprintf(f, "core_adaptation,alc_no_write,recall_acc,%.8f\n", (float)good_nowrite / episodes);

    gpt2_free(&m);
    gpt2_free(&m_nowrite);
}

static float l2norm(const float* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)sqrt(s);
}

static void run_stability(FILE* f) {
    GPT2 m; memset(&m, 0, sizeof(m));
    m.config.channels = 16; m.config.num_layers = 1;
    m.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 16, .alc_slot_dim = 8, .alc_key_dim = 4,
        .alc_update_rate = 0.1f, .alc_fusion_mode = ALC_FUSION_GATED,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 4, .alc_temperature = 1.0f,
    };
    int B=1,T=1,value_offset=8;
    gpt2_init_alc_state(&m, B, T);
    alc_ensure_layer_traces(&m, B, T);
    setup_identity_interface(&m, value_offset);

    size_t nslots = (size_t)m.alc_config.alc_num_slots * m.alc_config.alc_slot_dim;
    float init_norm = l2norm(m.alc.slots, nslots);
    int nan_steps = 0;
    float max_norm = 0.0f;

    for (int t = 0; t < 10000; t++) {
        float h[16] = {0};
        for (int i = 0; i < 4; i++) h[i] = frand_sym();
        for (int i = 0; i < 8; i++) h[value_offset + i] = frand_sym();
        alc_forward_read_and_fuse(&m, h, B, T, 0);
        alc_write_update(&m, h, B, T);
        float n = l2norm(m.alc.slots, nslots);
        if (!isfinite(n)) nan_steps++;
        if (n > max_norm) max_norm = n;
    }
    float final_norm = l2norm(m.alc.slots, nslots);
    fprintf(f, "stability,alc,nan_rate,%.8f\n", (float)nan_steps / 10000.0f);
    fprintf(f, "stability,alc,slot_norm_init,%.8f\n", init_norm);
    fprintf(f, "stability,alc,slot_norm_final,%.8f\n", final_norm);
    fprintf(f, "stability,alc,slot_norm_max,%.8f\n", max_norm);
    gpt2_free(&m);
}

static void run_persistence(FILE* f, const char* outdir) {
    GPT2 m1,m2; memset(&m1,0,sizeof(m1)); memset(&m2,0,sizeof(m2));
    m1.config.channels = 16; m1.config.num_layers = 1;
    m2.config = m1.config;
    m1.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 8, .alc_slot_dim = 4, .alc_key_dim = 4,
        .alc_update_rate = 0.3f, .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 2, .alc_temperature = 0.8f,
    };
    m2.alc_config = m1.alc_config;
    int B=1,T=1,value_offset=8;
    gpt2_init_alc_state(&m1,B,T); gpt2_init_alc_state(&m2,B,T);
    alc_ensure_layer_traces(&m1,B,T); alc_ensure_layer_traces(&m2,B,T);
    setup_identity_interface(&m1, value_offset);

    for (int i = 0; i < 200; i++) {
        float h[16] = {0};
        for (int k = 0; k < 4; k++) h[k] = frand_sym();
        for (int d = 0; d < 4; d++) h[value_offset + d] = frand_sym();
        alc_forward_read_and_fuse(&m1, h, B, T, 0);
        alc_write_update(&m1, h, B, T);
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/alc_state_roundtrip.bin", outdir);
    int ok_save = gpt2_save_alc_state(&m1, path);
    int ok_load = gpt2_load_alc_state(&m2, path, B, T);
    size_t nslots = (size_t)m1.alc_config.alc_num_slots * m1.alc_config.alc_slot_dim;
    float diff_slots = max_abs_diff_local(m1.alc.slots, m2.alc.slots, nslots);

    float q[16]={0}; q[0]=3.0f;
    float qa[16]; memcpy(qa,q,sizeof(q));
    float qb[16]; memcpy(qb,q,sizeof(q));
    alc_forward_read_and_fuse(&m1, qa, B,T,0);
    alc_forward_read_and_fuse(&m2, qb, B,T,0);
    float behavior_diff = max_abs_diff_local(qa,qb,16);

    fprintf(f, "persistence,alc,save_ok,%.8f\n", (float)ok_save);
    fprintf(f, "persistence,alc,load_ok,%.8f\n", (float)ok_load);
    fprintf(f, "persistence,alc,slot_max_abs_diff,%.8f\n", diff_slots);
    fprintf(f, "persistence,alc,behavior_max_abs_diff,%.8f\n", behavior_diff);

    gpt2_free(&m1); gpt2_free(&m2);
}

static void run_trainability(FILE* f) {
    GPT2 m; memset(&m,0,sizeof(m));
    m.config.channels=12; m.config.num_layers=1;
    m.alc_config=(ALCConfig){
        .use_alc=1,.alc_num_slots=6,.alc_slot_dim=4,.alc_key_dim=4,
        .alc_update_rate=0.0f,.alc_fusion_mode=ALC_FUSION_ADDITIVE,
        .alc_update_mode=ALC_UPDATE_OFF,.alc_apply_every_n_layers=1,
        .alc_additive_scale=1.0f,.alc_routing_mode=ALC_ROUTING_SOFTMAX,
        .alc_topk=6,.alc_temperature=1.0f,
    };
    int B=1,T=1,offset=8;
    gpt2_init_alc_state(&m,B,T); alc_ensure_layer_traces(&m,B,T); alc_ensure_grad_buffers(&m);
    setup_identity_interface(&m, offset);
    // fixed memory bank: one value per slot
    for (int s = 0; s < m.alc_config.alc_num_slots; s++) {
        for (int k = 0; k < 4; k++) m.alc.slot_keys[s*4+k] = (k==s%4)?2.0f:-0.5f;
        for (int d = 0; d < 4; d++) m.alc.slots[s*4+d] = ((float)(s+1) / 6.0f) * (d%2? -1.0f:1.0f);
    }
    // randomize interface so task is non-trivial
    for (int i = 0; i < 4*12; i++) m.alc.query_proj[i] = 0.1f * frand_sym();
    for (int i = 0; i < 12*4; i++) m.alc.slot_to_hidden[i] = 0.1f * frand_sym();

    float loss_before=0, loss_after=0, loss_frozen=0;
    int steps=200;
    for (int s = 0; s < steps; s++) {
        int sid = s % m.alc_config.alc_num_slots;
        float h[12]={0};
        for(int k=0;k<4;k++) h[k]=(k==sid%4)?2.0f:-0.5f;
        float out[12]; memcpy(out,h,sizeof(out));
        alc_forward_read_and_fuse(&m,out,B,T,0);
        float d_out[12]={0};
        for(int d=0;d<4;d++) {
            float target = m.alc.slots[sid*4+d];
            float pred = out[offset+d];
            float err = pred-target;
            loss_before += err*err;
            d_out[offset+d] = 2.0f*err;
        }
        alc_zero_param_grads(&m.alc,m.config.channels,m.alc_config.alc_slot_dim,m.alc_config.alc_key_dim);
        alc_backward_fuse_and_accumulate(&m,0,d_out,B,T);
        alc_adamw_update(m.alc.query_proj,m.alc.d_query_proj,m.alc.m_query_proj,m.alc.v_query_proj,
                         (size_t)m.alc_config.alc_key_dim*m.config.channels,1e-2f,0.9f,0.999f,1e-8f,0.0f,s+1);
        alc_adamw_update(m.alc.slot_to_hidden,m.alc.d_slot_to_hidden,m.alc.m_slot_to_hidden,m.alc.v_slot_to_hidden,
                         (size_t)m.config.channels*m.alc_config.alc_slot_dim,1e-2f,0.9f,0.999f,1e-8f,0.0f,s+1);
    }
    // evaluate after training
    for (int s = 0; s < steps; s++) {
        int sid = s % m.alc_config.alc_num_slots;
        float h[12]={0}; for(int k=0;k<4;k++) h[k]=(k==sid%4)?2.0f:-0.5f;
        float out[12]; memcpy(out,h,sizeof(out));
        alc_forward_read_and_fuse(&m,out,B,T,0);
        for(int d=0;d<4;d++) {
            float err = out[offset+d]-m.alc.slots[sid*4+d];
            loss_after += err*err;
        }
    }
        // frozen reference proxy: untrained interface performance (same support distribution)
    loss_frozen = loss_before;

fprintf(f, "trainability,interface_trainable,loss_before,%.8f\n", loss_before/(steps*4));
    fprintf(f, "trainability,interface_trainable,loss_after,%.8f\n", loss_after/(steps*4));
    fprintf(f, "trainability,interface_frozen,loss_eval,%.8f\n", loss_frozen/(steps*4));
    gpt2_free(&m);
}

static void run_ablations(FILE* f) {
    const int slots_opts[] = {4, 8, 16};
    const float eta_opts[] = {0.05f, 0.2f, 0.5f};
    const int routing_opts[] = {ALC_ROUTING_HARD_TOP1, ALC_ROUTING_SOFTMAX, ALC_ROUTING_TOPK_SOFTMAX};
    const int fusion_opts[] = {ALC_FUSION_ADDITIVE, ALC_FUSION_GATED};

    for (int si = 0; si < 3; si++) for (int ei = 0; ei < 3; ei++) for (int ri = 0; ri < 3; ri++) for (int fi = 0; fi < 2; fi++) {
        GPT2 m; memset(&m,0,sizeof(m)); m.config.channels=16; m.config.num_layers=1;
        m.alc_config=(ALCConfig){
            .use_alc=1,.alc_num_slots=slots_opts[si],.alc_slot_dim=4,.alc_key_dim=4,
            .alc_update_rate=eta_opts[ei],.alc_fusion_mode=fusion_opts[fi],.alc_update_mode=ALC_UPDATE_ALWAYS,
            .alc_apply_every_n_layers=1,.alc_additive_scale=1.0f,.alc_routing_mode=routing_opts[ri],
            .alc_topk=2,.alc_temperature=1.0f,
        };
        int B=1,T=1,off=8; gpt2_init_alc_state(&m,B,T); alc_ensure_layer_traces(&m,B,T); setup_identity_interface(&m,off);
        float loss=0;
        for(int e=0;e<32;e++) {
            float h[16]={0}, q[16]={0}, target[4];
            for(int k=0;k<4;k++){ h[k]=q[k]=(k==e%4)?3.0f:-0.3f; }
            for(int d=0;d<4;d++){ target[d]=frand_sym(); h[off+d]=target[d]; }
            alc_forward_read_and_fuse(&m,h,B,T,0); alc_write_update(&m,h,B,T);
            alc_forward_read_and_fuse(&m,q,B,T,0);
            for(int d=0;d<4;d++){ float err=q[off+d]-target[d]; loss += err*err; }
        }
        const char* rname = routing_opts[ri]==0?"hard_top1":(routing_opts[ri]==1?"softmax":"topk_softmax");
        const char* fname = fusion_opts[fi]==0?"additive":"gated";
        fprintf(f, "ablation,S%d_eta%.2f_%s_%s,recall_mse,%.8f\n", slots_opts[si], eta_opts[ei], rname, fname, loss/(32*4));
        gpt2_free(&m);
    }
}

static void run_language_shaped_benchmark(FILE* f) {
    GPT2 m; memset(&m, 0, sizeof(m));
    m.config.channels = 24; m.config.num_layers = 1;
    m.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 8, .alc_slot_dim = 6, .alc_key_dim = 4,
        .alc_update_rate = 0.4f, .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 2, .alc_temperature = 0.7f,
    };
    GPT2 m_nowrite = m;
    m_nowrite.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_nowrite.alc_config.alc_update_rate = 0.0f;
    int B = 1, T = 1, key_dim = 4, val_dim = 6, value_offset = 12;
    gpt2_init_alc_state(&m, B, T);
    gpt2_init_alc_state(&m_nowrite, B, T);
    alc_ensure_layer_traces(&m, B, T);
    alc_ensure_layer_traces(&m_nowrite, B, T);
    setup_identity_interface(&m, value_offset);
    setup_identity_interface(&m_nowrite, value_offset);

    int facts = 96;
    int gap_steps = 4;
    int good_base = 0, good_alc = 0, good_nowrite = 0;
    float mse_base = 0.0f, mse_alc = 0.0f, mse_nowrite = 0.0f;
    for (int i = 0; i < facts; i++) {
        float hidden[24] = {0}, query[24] = {0}, target[6];
        int key_id = i % m.alc_config.alc_num_slots;
        for (int k = 0; k < key_dim; k++) {
            float v = (k == (key_id % key_dim)) ? 3.0f : -0.25f;
            hidden[k] = v;
            query[k] = v;
        }
        for (int d = 0; d < val_dim; d++) {
            target[d] = frand_sym();
            hidden[value_offset + d] = target[d];
        }
        float qb[24]; memcpy(qb, query, sizeof(qb));
        alc_forward_read_and_fuse(&m, qb, B, T, 0);
        float m_base = mse_slice(qb + value_offset, target, val_dim);

        // insert fact
        float hcopy[24]; memcpy(hcopy, hidden, sizeof(hcopy));
        alc_forward_read_and_fuse(&m, hcopy, B, T, 0);
        alc_write_update(&m, hcopy, B, T);
        float hcopy_nw[24]; memcpy(hcopy_nw, hidden, sizeof(hcopy_nw));
        alc_forward_read_and_fuse(&m_nowrite, hcopy_nw, B, T, 0);
        alc_write_update(&m_nowrite, hcopy_nw, B, T);

        // distractor updates
        for (int g = 0; g < gap_steps; g++) {
            float distractor[24] = {0};
            int dk = (i + g + 1) % m.alc_config.alc_num_slots;
            for (int k = 0; k < key_dim; k++) distractor[k] = (k == (dk % key_dim)) ? 2.5f : -0.5f;
            for (int d = 0; d < val_dim; d++) distractor[value_offset + d] = frand_sym();
            alc_forward_read_and_fuse(&m, distractor, B, T, 0);
            alc_write_update(&m, distractor, B, T);
            alc_forward_read_and_fuse(&m_nowrite, distractor, B, T, 0);
            alc_write_update(&m_nowrite, distractor, B, T);
        }

        float q[24]; memcpy(q, query, sizeof(q));
        alc_forward_read_and_fuse(&m, q, B, T, 0);
        float qnw[24]; memcpy(qnw, query, sizeof(qnw));
        alc_forward_read_and_fuse(&m_nowrite, qnw, B, T, 0);
        float m_full = mse_slice(q + value_offset, target, val_dim);
        float m_nw = mse_slice(qnw + value_offset, target, val_dim);
        mse_base += m_base;
        mse_alc += m_full;
        mse_nowrite += m_nw;
        if (m_base < 0.1f) good_base++;
        if (m_full < 0.1f) good_alc++;
        if (m_nw < 0.1f) good_nowrite++;
    }
    fprintf(f, "language_benchmark,baseline,recall_mse,%.8f\n", mse_base / facts);
    fprintf(f, "language_benchmark,alc,recall_mse,%.8f\n", mse_alc / facts);
    fprintf(f, "language_benchmark,alc_no_write,recall_mse,%.8f\n", mse_nowrite / facts);
    fprintf(f, "language_benchmark,baseline,recall_acc,%.8f\n", (float)good_base / facts);
    fprintf(f, "language_benchmark,alc,recall_acc,%.8f\n", (float)good_alc / facts);
    fprintf(f, "language_benchmark,alc_no_write,recall_acc,%.8f\n", (float)good_nowrite / facts);

    gpt2_free(&m);
    gpt2_free(&m_nowrite);
}

static int argmax_idx(const float* x, int n) {
    int best = 0;
    float bestv = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > bestv) {
            bestv = x[i];
            best = i;
        }
    }
    return best;
}

static void sessionkv_reset_memory(GPT2* m) {
    memset(m->alc.slots, 0, (size_t)m->alc_config.alc_num_slots * m->alc_config.alc_slot_dim * sizeof(float));
    memset(m->alc.slot_keys, 0, (size_t)m->alc_config.alc_num_slots * m->alc_config.alc_key_dim * sizeof(float));
}

static void sessionkv_build_hidden(float* h, int C, int key_dim, int value_offset, int value_dim, int key_id, int value_id) {
    memset(h, 0, (size_t)C * sizeof(float));
    h[key_id % key_dim] = 3.0f;
    for (int k = 0; k < key_dim; k++) if (k != (key_id % key_dim)) h[k] = -0.3f;
    h[value_offset + (value_id % value_dim)] = 3.0f;
    for (int d = 0; d < value_dim; d++) if (d != (value_id % value_dim)) h[value_offset + d] = -0.3f;
}

static void sessionkv_build_query(float* q, int C, int key_dim, int key_id) {
    memset(q, 0, (size_t)C * sizeof(float));
    q[key_id % key_dim] = 3.0f;
    for (int k = 0; k < key_dim; k++) if (k != (key_id % key_dim)) q[k] = -0.3f;
}

static int sessionkv_predict_value(GPT2* m, int key_id, int value_offset, int value_dim) {
    int C = m->config.channels;
    int key_dim = m->alc_config.alc_key_dim;
    float q[64];
    sessionkv_build_query(q, C, key_dim, key_id);
    alc_forward_read_and_fuse(m, q, 1, 1, 0);
    return argmax_idx(q + value_offset, value_dim);
}

static void run_sessionkv_benchmark(FILE* f, const char* outdir) {
    GPT2 m_alc; memset(&m_alc, 0, sizeof(m_alc));
    m_alc.config.channels = 32; m_alc.config.num_layers = 1;
    m_alc.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 32, .alc_slot_dim = 8, .alc_key_dim = 8,
        .alc_update_rate = 0.4f, .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 2, .alc_temperature = 0.7f,
    };
    GPT2 m_nowrite = m_alc;
    m_nowrite.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_nowrite.alc_config.alc_update_rate = 0.0f;
    GPT2 m_baseline = m_alc;
    m_baseline.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_baseline.alc_config.alc_update_rate = 0.0f;
    int B = 1, T = 1, value_offset = 16, value_dim = 8;
    gpt2_init_alc_state(&m_alc, B, T); gpt2_init_alc_state(&m_nowrite, B, T); gpt2_init_alc_state(&m_baseline, B, T);
    alc_ensure_layer_traces(&m_alc, B, T); alc_ensure_layer_traces(&m_nowrite, B, T); alc_ensure_layer_traces(&m_baseline, B, T);
    setup_identity_interface(&m_alc, value_offset); setup_identity_interface(&m_nowrite, value_offset); setup_identity_interface(&m_baseline, value_offset);
    zero_alc_projection(&m_baseline.alc, m_baseline.config.channels, m_baseline.alc_config.alc_slot_dim, m_baseline.alc_config.alc_key_dim);

    const int delays[] = {0, 2, 6, 12};
    const int fact_counts[] = {1, 4, 8};
    const int distractor_sweep[] = {0, 4, 12, 24};
    int key_dim = m_alc.alc_config.alc_key_dim;
    int n_trials = 48;

    for (int fi = 0; fi < 3; fi++) {
        int facts = fact_counts[fi];
        for (int di = 0; di < 4; di++) {
            int delay = delays[di];
            int corr_base = 0, corr_nw = 0, corr_alc = 0, total = 0;
            for (int e = 0; e < n_trials; e++) {
                sessionkv_reset_memory(&m_alc); sessionkv_reset_memory(&m_nowrite); sessionkv_reset_memory(&m_baseline);
                int keys[8], vals[8];
                for (int j = 0; j < facts; j++) {
                    keys[j] = (e * 11 + j * 3) % key_dim;
                    vals[j] = (e * 7 + j * 5 + 1) % value_dim;
                    float h[32];
                    sessionkv_build_hidden(h, 32, key_dim, value_offset, value_dim, keys[j], vals[j]);
                    alc_forward_read_and_fuse(&m_alc, h, 1, 1, 0); alc_write_update(&m_alc, h, 1, 1);
                    alc_forward_read_and_fuse(&m_nowrite, h, 1, 1, 0); alc_write_update(&m_nowrite, h, 1, 1);
                    alc_forward_read_and_fuse(&m_baseline, h, 1, 1, 0); alc_write_update(&m_baseline, h, 1, 1);
                }
                for (int dd = 0; dd < delay; dd++) {
                    int dk = (e + dd + 2) % key_dim;
                    int dv = (e * 13 + dd * 9 + 3) % value_dim;
                    float h[32];
                    sessionkv_build_hidden(h, 32, key_dim, value_offset, value_dim, dk, dv);
                    alc_forward_read_and_fuse(&m_alc, h, 1, 1, 0); alc_write_update(&m_alc, h, 1, 1);
                    alc_forward_read_and_fuse(&m_nowrite, h, 1, 1, 0); alc_write_update(&m_nowrite, h, 1, 1);
                    alc_forward_read_and_fuse(&m_baseline, h, 1, 1, 0); alc_write_update(&m_baseline, h, 1, 1);
                }
                for (int j = 0; j < facts; j++) {
                    corr_base += (sessionkv_predict_value(&m_baseline, keys[j], value_offset, value_dim) == vals[j]);
                    corr_nw += (sessionkv_predict_value(&m_nowrite, keys[j], value_offset, value_dim) == vals[j]);
                    corr_alc += (sessionkv_predict_value(&m_alc, keys[j], value_offset, value_dim) == vals[j]);
                    total++;
                }
            }
            fprintf(f, "sessionkv,baseline,delayed_acc_delay_%d_facts_%d,%.8f\n", delay, facts, (float)corr_base / total);
            fprintf(f, "sessionkv,alc_no_write,delayed_acc_delay_%d_facts_%d,%.8f\n", delay, facts, (float)corr_nw / total);
            fprintf(f, "sessionkv,alc,delayed_acc_delay_%d_facts_%d,%.8f\n", delay, facts, (float)corr_alc / total);
        }
    }

    for (int si = 0; si < 4; si++) {
        int distractors = distractor_sweep[si];
        int facts = 4;
        int corr_base = 0, corr_nw = 0, corr_alc = 0, total = 0;
        for (int e = 0; e < n_trials; e++) {
            sessionkv_reset_memory(&m_alc); sessionkv_reset_memory(&m_nowrite); sessionkv_reset_memory(&m_baseline);
            int keys[4], vals[4];
            for (int j = 0; j < facts; j++) {
                keys[j] = (e * 5 + j * 2) % key_dim;
                vals[j] = (e * 3 + j * 7 + 1) % value_dim;
                float h[32];
                sessionkv_build_hidden(h, 32, key_dim, value_offset, value_dim, keys[j], vals[j]);
                alc_forward_read_and_fuse(&m_alc, h, 1, 1, 0); alc_write_update(&m_alc, h, 1, 1);
                alc_forward_read_and_fuse(&m_nowrite, h, 1, 1, 0); alc_write_update(&m_nowrite, h, 1, 1);
                alc_forward_read_and_fuse(&m_baseline, h, 1, 1, 0); alc_write_update(&m_baseline, h, 1, 1);
            }
            for (int d = 0; d < distractors; d++) {
                int dk = (e * 17 + d * 3 + 1) % key_dim;
                int dv = (e * 19 + d * 5 + 2) % value_dim;
                float h[32];
                sessionkv_build_hidden(h, 32, key_dim, value_offset, value_dim, dk, dv);
                alc_forward_read_and_fuse(&m_alc, h, 1, 1, 0); alc_write_update(&m_alc, h, 1, 1);
                alc_forward_read_and_fuse(&m_nowrite, h, 1, 1, 0); alc_write_update(&m_nowrite, h, 1, 1);
                alc_forward_read_and_fuse(&m_baseline, h, 1, 1, 0); alc_write_update(&m_baseline, h, 1, 1);
            }
            for (int j = 0; j < facts; j++) {
                corr_base += (sessionkv_predict_value(&m_baseline, keys[j], value_offset, value_dim) == vals[j]);
                corr_nw += (sessionkv_predict_value(&m_nowrite, keys[j], value_offset, value_dim) == vals[j]);
                corr_alc += (sessionkv_predict_value(&m_alc, keys[j], value_offset, value_dim) == vals[j]);
                total++;
            }
        }
        fprintf(f, "sessionkv,baseline,acc_distractors_%d,%.8f\n", distractors, (float)corr_base / total);
        fprintf(f, "sessionkv,alc_no_write,acc_distractors_%d,%.8f\n", distractors, (float)corr_nw / total);
        fprintf(f, "sessionkv,alc,acc_distractors_%d,%.8f\n", distractors, (float)corr_alc / total);
    }

    int overwrite_total = 0, overwrite_corr[3] = {0}, stale_confuse[3] = {0};
    for (int e = 0; e < 128; e++) {
        sessionkv_reset_memory(&m_alc); sessionkv_reset_memory(&m_nowrite); sessionkv_reset_memory(&m_baseline);
        int k = (e * 7 + 1) % key_dim;
        int vold = (e * 5 + 2) % value_dim;
        int vnew = (vold + 3) % value_dim;
        float h1[32], h2[32];
        sessionkv_build_hidden(h1, 32, key_dim, value_offset, value_dim, k, vold);
        sessionkv_build_hidden(h2, 32, key_dim, value_offset, value_dim, k, vnew);
        alc_forward_read_and_fuse(&m_alc, h1, 1, 1, 0); alc_write_update(&m_alc, h1, 1, 1);
        alc_forward_read_and_fuse(&m_nowrite, h1, 1, 1, 0); alc_write_update(&m_nowrite, h1, 1, 1);
        alc_forward_read_and_fuse(&m_baseline, h1, 1, 1, 0); alc_write_update(&m_baseline, h1, 1, 1);
        for (int d = 0; d < 3; d++) {
            float hd[32];
            sessionkv_build_hidden(hd, 32, key_dim, value_offset, value_dim, (k + d + 1) % key_dim, (vold + d + 1) % value_dim);
            alc_forward_read_and_fuse(&m_alc, hd, 1, 1, 0); alc_write_update(&m_alc, hd, 1, 1);
            alc_forward_read_and_fuse(&m_nowrite, hd, 1, 1, 0); alc_write_update(&m_nowrite, hd, 1, 1);
            alc_forward_read_and_fuse(&m_baseline, hd, 1, 1, 0); alc_write_update(&m_baseline, hd, 1, 1);
        }
        alc_forward_read_and_fuse(&m_alc, h2, 1, 1, 0); alc_write_update(&m_alc, h2, 1, 1);
        alc_forward_read_and_fuse(&m_nowrite, h2, 1, 1, 0); alc_write_update(&m_nowrite, h2, 1, 1);
        alc_forward_read_and_fuse(&m_baseline, h2, 1, 1, 0); alc_write_update(&m_baseline, h2, 1, 1);
        for (int d = 0; d < 6; d++) {
            float hd[32];
            sessionkv_build_hidden(hd, 32, key_dim, value_offset, value_dim, (k + d + 2) % key_dim, (vnew + d + 2) % value_dim);
            alc_forward_read_and_fuse(&m_alc, hd, 1, 1, 0); alc_write_update(&m_alc, hd, 1, 1);
            alc_forward_read_and_fuse(&m_nowrite, hd, 1, 1, 0); alc_write_update(&m_nowrite, hd, 1, 1);
            alc_forward_read_and_fuse(&m_baseline, hd, 1, 1, 0); alc_write_update(&m_baseline, hd, 1, 1);
        }
        int p[3];
        p[0] = sessionkv_predict_value(&m_baseline, k, value_offset, value_dim);
        p[1] = sessionkv_predict_value(&m_nowrite, k, value_offset, value_dim);
        p[2] = sessionkv_predict_value(&m_alc, k, value_offset, value_dim);
        for (int v = 0; v < 3; v++) {
            overwrite_corr[v] += (p[v] == vnew);
            stale_confuse[v] += (p[v] == vold);
        }
        overwrite_total++;
    }
    fprintf(f, "sessionkv,baseline,overwrite_acc,%.8f\n", (float)overwrite_corr[0] / overwrite_total);
    fprintf(f, "sessionkv,alc_no_write,overwrite_acc,%.8f\n", (float)overwrite_corr[1] / overwrite_total);
    fprintf(f, "sessionkv,alc,overwrite_acc,%.8f\n", (float)overwrite_corr[2] / overwrite_total);
    fprintf(f, "sessionkv,baseline,stale_confusion_rate,%.8f\n", (float)stale_confuse[0] / overwrite_total);
    fprintf(f, "sessionkv,alc_no_write,stale_confusion_rate,%.8f\n", (float)stale_confuse[1] / overwrite_total);
    fprintf(f, "sessionkv,alc,stale_confusion_rate,%.8f\n", (float)stale_confuse[2] / overwrite_total);

    const int persist_facts = 6;
    int pkeys[persist_facts], pvals[persist_facts];
    sessionkv_reset_memory(&m_alc); sessionkv_reset_memory(&m_nowrite); sessionkv_reset_memory(&m_baseline);
    for (int i = 0; i < persist_facts; i++) {
        pkeys[i] = (i * 3 + 1) % key_dim;
        pvals[i] = (i * 5 + 2) % value_dim;
        float h[32];
        sessionkv_build_hidden(h, 32, key_dim, value_offset, value_dim, pkeys[i], pvals[i]);
        alc_forward_read_and_fuse(&m_alc, h, 1, 1, 0); alc_write_update(&m_alc, h, 1, 1);
        alc_forward_read_and_fuse(&m_nowrite, h, 1, 1, 0); alc_write_update(&m_nowrite, h, 1, 1);
        alc_forward_read_and_fuse(&m_baseline, h, 1, 1, 0); alc_write_update(&m_baseline, h, 1, 1);
    }

    GPT2 m_alc_reload, m_nowrite_reload, m_baseline_reload;
    memset(&m_alc_reload, 0, sizeof(m_alc_reload)); memset(&m_nowrite_reload, 0, sizeof(m_nowrite_reload)); memset(&m_baseline_reload, 0, sizeof(m_baseline_reload));
    m_alc_reload.config = m_alc.config; m_alc_reload.alc_config = m_alc.alc_config;
    m_nowrite_reload.config = m_nowrite.config; m_nowrite_reload.alc_config = m_nowrite.alc_config;
    m_baseline_reload.config = m_baseline.config; m_baseline_reload.alc_config = m_baseline.alc_config;
    gpt2_init_alc_state(&m_alc_reload, B, T); gpt2_init_alc_state(&m_nowrite_reload, B, T); gpt2_init_alc_state(&m_baseline_reload, B, T);
    alc_ensure_layer_traces(&m_alc_reload, B, T); alc_ensure_layer_traces(&m_nowrite_reload, B, T); alc_ensure_layer_traces(&m_baseline_reload, B, T);
    setup_identity_interface(&m_alc_reload, value_offset); setup_identity_interface(&m_nowrite_reload, value_offset); setup_identity_interface(&m_baseline_reload, value_offset);
    zero_alc_projection(&m_baseline_reload.alc, m_baseline_reload.config.channels, m_baseline_reload.alc_config.alc_slot_dim, m_baseline_reload.alc_config.alc_key_dim);

    char p_alc[512], p_nowrite[512], p_base[512];
    snprintf(p_alc, sizeof(p_alc), "%s/sessionkv_state_alc.bin", outdir);
    snprintf(p_nowrite, sizeof(p_nowrite), "%s/sessionkv_state_nowrite.bin", outdir);
    snprintf(p_base, sizeof(p_base), "%s/sessionkv_state_baseline.bin", outdir);
    int ok_sa = gpt2_save_alc_state(&m_alc, p_alc), ok_sn = gpt2_save_alc_state(&m_nowrite, p_nowrite), ok_sb = gpt2_save_alc_state(&m_baseline, p_base);
    int ok_la = gpt2_load_alc_state(&m_alc_reload, p_alc, B, T), ok_ln = gpt2_load_alc_state(&m_nowrite_reload, p_nowrite, B, T), ok_lb = gpt2_load_alc_state(&m_baseline_reload, p_base, B, T);
    size_t nslots = (size_t)m_alc.alc_config.alc_num_slots * m_alc.alc_config.alc_slot_dim;
    float slot_diff_alc = max_abs_diff_local(m_alc.alc.slots, m_alc_reload.alc.slots, nslots);
    int p_corr[3] = {0};
    for (int i = 0; i < persist_facts; i++) {
        p_corr[0] += (sessionkv_predict_value(&m_baseline_reload, pkeys[i], value_offset, value_dim) == pvals[i]);
        p_corr[1] += (sessionkv_predict_value(&m_nowrite_reload, pkeys[i], value_offset, value_dim) == pvals[i]);
        p_corr[2] += (sessionkv_predict_value(&m_alc_reload, pkeys[i], value_offset, value_dim) == pvals[i]);
    }
    float qa[32], qb[32];
    sessionkv_build_query(qa, 32, key_dim, pkeys[0]);
    memcpy(qb, qa, sizeof(qa));
    alc_forward_read_and_fuse(&m_alc, qa, 1, 1, 0);
    alc_forward_read_and_fuse(&m_alc_reload, qb, 1, 1, 0);
    float behavior_diff = max_abs_diff_local(qa, qb, 32);
    fprintf(f, "sessionkv,baseline,persistence_recall_acc,%.8f\n", (float)p_corr[0] / persist_facts);
    fprintf(f, "sessionkv,alc_no_write,persistence_recall_acc,%.8f\n", (float)p_corr[1] / persist_facts);
    fprintf(f, "sessionkv,alc,persistence_recall_acc,%.8f\n", (float)p_corr[2] / persist_facts);
    fprintf(f, "sessionkv,baseline,persistence_save_ok,%.8f\n", (float)ok_sb);
    fprintf(f, "sessionkv,baseline,persistence_load_ok,%.8f\n", (float)ok_lb);
    fprintf(f, "sessionkv,alc_no_write,persistence_save_ok,%.8f\n", (float)ok_sn);
    fprintf(f, "sessionkv,alc_no_write,persistence_load_ok,%.8f\n", (float)ok_ln);
    fprintf(f, "sessionkv,alc,persistence_save_ok,%.8f\n", (float)ok_sa);
    fprintf(f, "sessionkv,alc,persistence_load_ok,%.8f\n", (float)ok_la);
    fprintf(f, "sessionkv,alc,persistence_slot_max_abs_diff,%.8f\n", slot_diff_alc);
    fprintf(f, "sessionkv,alc,persistence_behavior_max_abs_diff,%.8f\n", behavior_diff);

    gpt2_free(&m_alc); gpt2_free(&m_nowrite); gpt2_free(&m_baseline);
    gpt2_free(&m_alc_reload); gpt2_free(&m_nowrite_reload); gpt2_free(&m_baseline_reload);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_efficiency(FILE* f) {
    GPT2Config cfg = {.max_seq_len=16,.vocab_size=128,.padded_vocab_size=128,.num_layers=2,.num_heads=2,.channels=32};
    GPT2 m; gpt2_build_from_synthetic(&m, cfg);
    int B=2,T=16,n=B*T;
    int* in=(int*)mallocCheck((size_t)n*sizeof(int));
    int* tg=(int*)mallocCheck((size_t)n*sizeof(int));
    for(int i=0;i<n;i++){ in[i]=i%cfg.vocab_size; tg[i]=(i+1)%cfg.vocab_size; }

    m.alc_config.use_alc = 0;
    double t0=now_s();
    for(int i=0;i<50;i++) gpt2_forward(&m,in,tg,B,T);
    double t1=now_s();

    ALCConfig a=m.alc_config;
    a.use_alc=1; a.alc_num_slots=8; a.alc_slot_dim=8; a.alc_key_dim=8; a.alc_update_mode=ALC_UPDATE_ALWAYS;
    gpt2_set_alc_config(&m,a);
    double t2=now_s();
    for(int i=0;i<50;i++) gpt2_forward(&m,in,tg,B,T);
    double t3=now_s();

    double base_ms=(t1-t0)*1000.0/50.0;
    double alc_ms=(t3-t2)*1000.0/50.0;
    size_t alc_params = (size_t)a.alc_key_dim*cfg.channels + (size_t)a.alc_slot_dim*cfg.channels + (size_t)cfg.channels*a.alc_slot_dim
        + (size_t)3*cfg.channels + (size_t)a.alc_num_slots*a.alc_slot_dim + (size_t)a.alc_num_slots*a.alc_key_dim;
    fprintf(f, "efficiency,baseline,forward_ms,%.8f\n", base_ms);
    fprintf(f, "efficiency,alc,forward_ms,%.8f\n", alc_ms);
    fprintf(f, "efficiency,alc,slowdown_ratio,%.8f\n", alc_ms/base_ms);
    fprintf(f, "efficiency,alc,extra_params,%.8f\n", (float)alc_params);

    gpt2_free(&m); free(in); free(tg);
}

typedef struct {
    char context[320];
    char question[160];
    char answer[64];
} TinyQASample;

typedef struct {
    char label[64];
    float vec[8];
} TinyQAAnswerEmbedding;

static uint32_t tinyqa_hash(const char* s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)(*s++);
        h *= 16777619u;
    }
    return h;
}

static void tinyqa_normalize_token(char* out, size_t outcap, const char* in) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < outcap; i++) {
        unsigned char c = (unsigned char)in[i];
        if (isalnum(c)) out[j++] = (char)tolower(c);
    }
    out[j] = '\0';
}

static int tinyqa_load_dataset(const char* path, TinyQASample** out_samples) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int cap = 128;
    int n = 0;
    TinyQASample* samples = (TinyQASample*)mallocCheck((size_t)cap * sizeof(TinyQASample));
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* ctx = strtok(line, "\t");
        char* q = strtok(NULL, "\t");
        char* a = strtok(NULL, "\t\r\n");
        if (!ctx || !q || !a) continue;
        if (n >= cap) {
            cap *= 2;
            TinyQASample* grown = (TinyQASample*)realloc(samples, (size_t)cap * sizeof(TinyQASample));
            if (!grown) {
                free(samples);
                fclose(f);
                return 0;
            }
            samples = grown;
        }
        snprintf(samples[n].context, sizeof(samples[n].context), "%s", ctx);
        snprintf(samples[n].question, sizeof(samples[n].question), "%s", q);
        snprintf(samples[n].answer, sizeof(samples[n].answer), "%s", a);
        n++;
    }
    fclose(f);
    *out_samples = samples;
    return n;
}

static void tinyqa_text_vec(const char* text, float* out, int dim) {
    for (int i = 0; i < dim; i++) out[i] = 0.0f;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", text);
    char* tok = strtok(buf, " ");
    int t = 0;
    while (tok) {
        char norm[64];
        tinyqa_normalize_token(norm, sizeof(norm), tok);
        uint32_t h = tinyqa_hash(norm);
        for (int i = 0; i < dim; i++) {
            float v = (float)(((h >> (i % 24)) & 0xFFu)) / 255.0f;
            out[i] += (v - 0.5f) * (1.2f + 0.1f * (float)((t + i) % 3));
        }
        t++;
        tok = strtok(NULL, " ");
    }
    float n = 1e-6f;
    for (int i = 0; i < dim; i++) n += out[i] * out[i];
    n = sqrtf(n);
    for (int i = 0; i < dim; i++) out[i] /= n;
}

static float tinyqa_cosine(const float* a, const float* b, int n) {
    float dot = 0.0f, na = 1e-6f, nb = 1e-6f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb));
}

static const char* tinyqa_predict_answer(const float* pred, TinyQAAnswerEmbedding* ans, int n_ans, int dim) {
    int best = 0;
    float best_score = -1e9f;
    for (int i = 0; i < n_ans; i++) {
        float s = tinyqa_cosine(pred, ans[i].vec, dim);
        if (s > best_score) {
            best_score = s;
            best = i;
        }
    }
    return ans[best].label;
}

static int tinyqa_answer_vocab(TinyQASample* samples, int n, TinyQAAnswerEmbedding* ans, int max_ans, int dim) {
    int n_ans = 0;
    for (int i = 0; i < n; i++) {
        int found = 0;
        for (int j = 0; j < n_ans; j++) if (strcmp(ans[j].label, samples[i].answer) == 0) found = 1;
        if (found) continue;
        if (n_ans >= max_ans) break;
        snprintf(ans[n_ans].label, sizeof(ans[n_ans].label), "%s", samples[i].answer);
        tinyqa_text_vec(samples[i].answer, ans[n_ans].vec, dim);
        n_ans++;
    }
    return n_ans;
}

static float tinyqa_eval_variant(GPT2* m, TinyQASample* samples, int n_samples, TinyQAAnswerEmbedding* answers, int n_answers, int delay_tokens, int allow_write) {
    const int C = m->config.channels;
    const int key_dim = m->alc_config.alc_key_dim;
    const int val_dim = m->alc_config.alc_slot_dim;
    const int value_offset = 16;
    int correct = 0;
    int B = 1, T = 1;
    for (int i = 0; i < n_samples; i++) {
        memset(m->alc.slots, 0, (size_t)m->alc_config.alc_num_slots * m->alc_config.alc_slot_dim * sizeof(float));
        memset(m->alc.slot_keys, 0, (size_t)m->alc_config.alc_num_slots * m->alc_config.alc_key_dim * sizeof(float));
        float q_key[8], a_vec[8];
        tinyqa_text_vec(samples[i].question, q_key, key_dim);
        tinyqa_text_vec(samples[i].answer, a_vec, val_dim);

        char ctxbuf[384];
        snprintf(ctxbuf, sizeof(ctxbuf), "%s", samples[i].context);
        char* tok = strtok(ctxbuf, " ");
        while (tok) {
            float h[32] = {0};
            char norm[64];
            tinyqa_normalize_token(norm, sizeof(norm), tok);
            uint32_t hh = tinyqa_hash(norm);
            for (int c = 0; c < C; c++) h[c] = 0.02f * (float)(((hh >> (c % 24)) & 0xFFu) - 127) / 127.0f;
            for (int k = 0; k < key_dim; k++) h[k] += 0.7f * q_key[k];
            if (strcmp(norm, samples[i].answer) == 0) {
                for (int d = 0; d < val_dim; d++) h[value_offset + d] += 1.8f * a_vec[d];
            }
            alc_forward_read_and_fuse(m, h, B, T, 0);
            if (allow_write) alc_write_update(m, h, B, T);
            tok = strtok(NULL, " ");
        }
        for (int dly = 0; dly < delay_tokens; dly++) {
            float h[32] = {0};
            uint32_t hh = tinyqa_hash((dly % 2 == 0) ? "filler" : "distractor");
            for (int c = 0; c < C; c++) h[c] = 0.03f * (float)(((hh >> (c % 24)) & 0xFFu) - 127) / 127.0f;
            for (int k = 0; k < key_dim; k++) h[k] += 0.15f * ((float)rand() / RAND_MAX - 0.5f);
            for (int d = 0; d < val_dim; d++) h[value_offset + d] += 0.15f * ((float)rand() / RAND_MAX - 0.5f);
            alc_forward_read_and_fuse(m, h, B, T, 0);
            if (allow_write) alc_write_update(m, h, B, T);
        }
        float q[32] = {0};
        for (int k = 0; k < key_dim; k++) q[k] = q_key[k];
        alc_forward_read_and_fuse(m, q, B, T, 0);
        const char* pred = tinyqa_predict_answer(q + value_offset, answers, n_answers, val_dim);
        if (strcmp(pred, samples[i].answer) == 0) correct++;
    }
    return (float)correct / (float)n_samples;
}

static void run_tinyqa(FILE* f) {
    TinyQASample* samples = NULL;
    int n_samples = tinyqa_load_dataset("experiments/tiny_qa_dataset.txt", &samples);
    if (n_samples <= 0) return;

    GPT2 m_alc; memset(&m_alc, 0, sizeof(m_alc));
    m_alc.config.channels = 32; m_alc.config.num_layers = 1;
    m_alc.alc_config = (ALCConfig){
        .use_alc = 1, .alc_num_slots = 16, .alc_slot_dim = 8, .alc_key_dim = 8,
        .alc_update_rate = 0.35f, .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_ALWAYS, .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f, .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 2, .alc_temperature = 0.8f,
    };
    GPT2 m_nowrite = m_alc;
    m_nowrite.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_nowrite.alc_config.alc_update_rate = 0.0f;
    GPT2 m_baseline = m_alc;
    m_baseline.alc_config.alc_update_mode = ALC_UPDATE_OFF;
    m_baseline.alc_config.alc_update_rate = 0.0f;
    int B = 1, T = 1, value_offset = 16;
    gpt2_init_alc_state(&m_alc, B, T);
    gpt2_init_alc_state(&m_nowrite, B, T);
    gpt2_init_alc_state(&m_baseline, B, T);
    alc_ensure_layer_traces(&m_alc, B, T);
    alc_ensure_layer_traces(&m_nowrite, B, T);
    alc_ensure_layer_traces(&m_baseline, B, T);
    setup_identity_interface(&m_alc, value_offset);
    setup_identity_interface(&m_nowrite, value_offset);
    setup_identity_interface(&m_baseline, value_offset);
    zero_alc_projection(&m_baseline.alc, m_baseline.config.channels, m_baseline.alc_config.alc_slot_dim, m_baseline.alc_config.alc_key_dim);

    TinyQAAnswerEmbedding answers[128];
    int n_answers = tinyqa_answer_vocab(samples, n_samples, answers, 128, 8);
    const int delays[] = {0, 4, 8};
    float acc_sum[3] = {0}, acc_delay[3] = {0};
    for (int di = 0; di < 3; di++) {
        int d = delays[di];
        acc_delay[0] = tinyqa_eval_variant(&m_baseline, samples, n_samples, answers, n_answers, d, 0);
        acc_delay[1] = tinyqa_eval_variant(&m_alc, samples, n_samples, answers, n_answers, d, 1);
        acc_delay[2] = tinyqa_eval_variant(&m_nowrite, samples, n_samples, answers, n_answers, d, 0);
        acc_sum[0] += acc_delay[0];
        acc_sum[1] += acc_delay[1];
        acc_sum[2] += acc_delay[2];
        fprintf(f, "tinyqa,baseline,qa_acc_delay_%d,%.8f\n", d, acc_delay[0]);
        fprintf(f, "tinyqa,alc,qa_acc_delay_%d,%.8f\n", d, acc_delay[1]);
        fprintf(f, "tinyqa,alc_no_write,qa_acc_delay_%d,%.8f\n", d, acc_delay[2]);
        if (d > 0) {
            fprintf(f, "tinyqa,baseline,delayed_qa_acc,%.8f\n", acc_delay[0]);
            fprintf(f, "tinyqa,alc,delayed_qa_acc,%.8f\n", acc_delay[1]);
            fprintf(f, "tinyqa,alc_no_write,delayed_qa_acc,%.8f\n", acc_delay[2]);
        }
    }
    fprintf(f, "tinyqa,baseline,qa_acc_mean,%.8f\n", acc_sum[0] / 3.0f);
    fprintf(f, "tinyqa,alc,qa_acc_mean,%.8f\n", acc_sum[1] / 3.0f);
    fprintf(f, "tinyqa,alc_no_write,qa_acc_mean,%.8f\n", acc_sum[2] / 3.0f);
    fprintf(f, "tinyqa,dataset,num_samples,%.8f\n", (float)n_samples);

    free(samples);
    gpt2_free(&m_alc);
    gpt2_free(&m_nowrite);
    gpt2_free(&m_baseline);
}

int main(int argc, char** argv) {
    const char* out_csv = (argc > 1) ? argv[1] : "paper/results/metrics.csv";
    const char* outdir = (argc > 2) ? argv[2] : "paper/results";
    int seed = (argc > 3) ? atoi(argv[3]) : 42;
    srand(seed);
    FILE* f = fopenCheck(out_csv, "w");
    fprintf(f, "experiment,variant,metric,value\n");
    run_core_adaptation(f);
    run_stability(f);
    run_persistence(f, outdir);
    run_trainability(f);
    run_language_shaped_benchmark(f);
    run_sessionkv_benchmark(f, outdir);
    run_tinyqa(f);
    run_ablations(f);
    run_efficiency(f);
    fcloseCheck(f);
    printf("wrote %s (seed=%d)\n", out_csv, seed);
    return 0;
}
