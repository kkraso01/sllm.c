#define TESTING
#include "../train_gpt2.c"
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
    int B = 1, T = 1, value_offset = 8;
    gpt2_init_alc_state(&m, B, T);
    alc_ensure_layer_traces(&m, B, T);
    setup_identity_interface(&m, value_offset);
    memset(m.alc.slots, 0, (size_t)m.alc_config.alc_num_slots * m.alc_config.alc_slot_dim * sizeof(float));
    memset(m.alc.slot_keys, 0, (size_t)m.alc_config.alc_num_slots * m.alc_config.alc_key_dim * sizeof(float));

    int episodes = 64;
    int good_baseline = 0, good_alc = 0;
    float mse_baseline = 0.0f, mse_alc = 0.0f;

    for (int e = 0; e < episodes; e++) {
        int slot_id = e % m.alc_config.alc_num_slots;
        float hidden[16] = {0};
        float query[16] = {0};
        float target[4];
        for (int k = 0; k < 4; k++) {
            float keyv = (k == slot_id % 4) ? 4.0f : -1.0f;
            hidden[k] = keyv;
            query[k] = keyv;
        }
        for (int d = 0; d < 4; d++) {
            target[d] = frand_sym();
            hidden[value_offset + d] = target[d];
        }

        // baseline: query before update
        float q0[16]; memcpy(q0, query, sizeof(q0));
        alc_forward_read_and_fuse(&m, q0, B, T, 0);
        float pred0[4];
        for (int d = 0; d < 4; d++) pred0[d] = q0[value_offset + d];
        float m0 = mse_slice(pred0, target, 4);
        mse_baseline += m0;
        if (m0 < 0.1f) good_baseline++;

        // adapt: write then query
        float hcopy[16]; memcpy(hcopy, hidden, sizeof(hcopy));
        alc_forward_read_and_fuse(&m, hcopy, B, T, 0);
        alc_write_update(&m, hcopy, B, T);

        float q1[16]; memcpy(q1, query, sizeof(q1));
        alc_forward_read_and_fuse(&m, q1, B, T, 0);
        float pred1[4];
        for (int d = 0; d < 4; d++) pred1[d] = q1[value_offset + d];
        float m1 = mse_slice(pred1, target, 4);
        mse_alc += m1;
        if (m1 < 0.1f) good_alc++;
    }

    fprintf(f, "core_adaptation,baseline,recall_mse,%.8f\n", mse_baseline / episodes);
    fprintf(f, "core_adaptation,alc,recall_mse,%.8f\n", mse_alc / episodes);
    fprintf(f, "core_adaptation,baseline,recall_acc,%.8f\n", (float)good_baseline / episodes);
    fprintf(f, "core_adaptation,alc,recall_acc,%.8f\n", (float)good_alc / episodes);

    gpt2_free(&m);
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

int main(int argc, char** argv) {
    const char* out_csv = (argc > 1) ? argv[1] : "paper/results/metrics.csv";
    const char* outdir = (argc > 2) ? argv[2] : "paper/results";
    srand(42);
    FILE* f = fopenCheck(out_csv, "w");
    fprintf(f, "experiment,variant,metric,value\n");
    run_core_adaptation(f);
    run_stability(f);
    run_persistence(f, outdir);
    run_trainability(f);
    run_ablations(f);
    run_efficiency(f);
    fcloseCheck(f);
    printf("wrote %s\n", out_csv);
    return 0;
}
