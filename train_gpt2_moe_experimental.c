/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

typedef enum {
    ALC_FUSION_ADDITIVE = 0,
    ALC_FUSION_GATED = 1,
} ALCFusionMode;

typedef enum {
    ALC_UPDATE_OFF = 0,
    ALC_UPDATE_TRAIN_ONLY = 1,
    ALC_UPDATE_ALWAYS = 2,
} ALCUpdateMode;

typedef enum {
    ALC_ROUTING_HARD_TOP1 = 0,
    ALC_ROUTING_SOFTMAX = 1,
    ALC_ROUTING_TOPK_SOFTMAX = 2,
} ALCRoutingMode;

typedef enum {
    MOE_ROUTING_TOP1 = 0,
} MoERoutingMode;

typedef struct {
    int use_moe; // feature gate for experimental MoE path
    int moe_num_experts; // number of experts replacing FFN
    int moe_topk; // routing top-k (currently only top-1 scaffolded)
    int moe_apply_every_n_layers; // apply MoE on every nth layer
    int moe_router_mode; // see MoERoutingMode
    int moe_expert_memory_slots; // memory slots per expert
    int moe_expert_memory_dim; // local memory dimensionality per slot
    float moe_memory_update_rate; // EMA update rate for expert-local writes
    float moe_memory_fusion_scale; // additive fusion scale for expert memory readout
} MoEConfig;

typedef struct {
    float* router_w; // (E, C)
    float* router_b; // (E)
    float* expert_memory; // (E, S, D)
    float* expert_memory_keys; // (E, S, D)
    float* router_logits; // (B*T, E)
    float* router_probs; // (B*T, E)
    int* selected_expert; // (B*T)
    float* retrieved_buffer; // (B*T, C)
    int initialized;
    int scratch_bt;
    int logged_enable_message;
} MoEState;

typedef struct {
    int use_alc; // feature gate; keeps baseline path untouched when 0
    int alc_num_slots; // number of adaptive slots in memory table
    int alc_slot_dim; // dimensionality of stored adaptive slot vectors
    int alc_key_dim; // dimensionality used for lookup keys
    float alc_update_rate; // EMA update rate for write-back updates
    int alc_fusion_mode; // see ALCFusionMode
    int alc_update_mode; // see ALCUpdateMode
    int alc_apply_every_n_layers; // apply ALC on every nth transformer layer
    float alc_additive_scale; // scale for additive fusion
    int alc_routing_mode; // see ALCRoutingMode
    int alc_topk; // top-k slots for sparse soft routing
    float alc_temperature; // routing temperature (>0)
} ALCConfig;

typedef struct {
    // learned projections (initialized at runtime, currently not loaded from checkpoint)
    float* query_proj; // (K, C): maps hidden state -> ALC query/key space
    float* write_proj; // (D, C): maps hidden state -> slot update vector
    float* slot_to_hidden; // (C, D): maps retrieved slot back to hidden dim
    float* gate_h; // (C): gated fusion coefficient for hidden state path
    float* gate_a; // (C): gated fusion coefficient for retrieved slot path
    float* gate_b; // (C): gated fusion bias
    // adaptive state (continual-learning substrate)
    float* slot_keys; // (S, K): slot lookup keys
    float* slots; // (S, D): slot content vectors
    // per-forward scratch and traces
    float* query_buffer; // (B*T, K)
    float* retrieved_buffer; // (B*T, C)
    int* selected_slots; // (B*T)
    float* routing_probs; // (B*T, S)
    int initialized;
    int logged_enable_message;
    int debug_enabled;
    int scratch_bt;
    long forward_calls;
    long write_calls;
    long total_writes;
    long layers_applied;
    long layer_checks;
    int* slot_hit_counts;
    // per-layer forward traces for ALC backward (only populated when ALC active)
    float* hidden_pre_layers; // (L, B*T, C)
    float* retrieved_layers; // (L, B*T, C)
    int* selected_slots_layers; // (L, B*T)
    float* routing_probs_layers; // (L, B*T, S)
    int trace_bt;
    // gradients + optimizer state for trainable differentiable ALC tensors
    float* d_query_proj; // (K, C)
    float* d_slot_to_hidden; // (C, D)
    float* d_gate_h; // (C)
    float* d_gate_a; // (C)
    float* d_gate_b; // (C)
    float* m_slot_to_hidden; // Adam first moment
    float* v_slot_to_hidden; // Adam second moment
    float* m_query_proj;
    float* v_query_proj;
    float* m_gate_h;
    float* v_gate_h;
    float* m_gate_a;
    float* v_gate_a;
    float* m_gate_b;
    float* v_gate_b;
} ALCState;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    ALCConfig alc_config;
    MoEConfig moe_config;
    ALCState alc;
    MoEState moe;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

static int moe_layer_enabled(const GPT2* model, int layer) {
    if (!model->moe_config.use_moe) { return 0; }
    int every = model->moe_config.moe_apply_every_n_layers;
    if (every <= 0) { return 0; }
    return ((layer + 1) % every) == 0;
}

static void gpt2_validate_moe_config(const GPT2* model) {
    if (!model->moe_config.use_moe) { return; }
    const MoEConfig* cfg = &model->moe_config;
    if (cfg->moe_num_experts <= 0) { fprintf(stderr, "[MoE] invalid moe_num_experts=%d\n", cfg->moe_num_experts); exit(1); }
    if (cfg->moe_topk <= 0 || cfg->moe_topk > cfg->moe_num_experts) {
        fprintf(stderr, "[MoE] invalid moe_topk=%d\n", cfg->moe_topk);
        exit(1);
    }
    if (cfg->moe_expert_memory_slots <= 0) { fprintf(stderr, "[MoE] invalid moe_expert_memory_slots=%d\n", cfg->moe_expert_memory_slots); exit(1); }
    if (cfg->moe_expert_memory_dim <= 0 || cfg->moe_expert_memory_dim > model->config.channels) {
        fprintf(stderr, "[MoE] invalid moe_expert_memory_dim=%d (channels=%d)\n", cfg->moe_expert_memory_dim, model->config.channels);
        exit(1);
    }
    if (cfg->moe_memory_update_rate < 0.0f || cfg->moe_memory_update_rate > 1.0f) {
        fprintf(stderr, "[MoE] invalid moe_memory_update_rate=%.4f\n", cfg->moe_memory_update_rate);
        exit(1);
    }
}

static void gpt2_init_moe_state(GPT2* model, int B, int T) {
    if (!model->moe_config.use_moe) { return; }
    gpt2_validate_moe_config(model);
    int E = model->moe_config.moe_num_experts;
    int C = model->config.channels;
    int S = model->moe_config.moe_expert_memory_slots;
    int D = model->moe_config.moe_expert_memory_dim;
    int BT = B * T;
    if (model->moe.initialized) {
        if (model->moe.scratch_bt != BT) {
            free(model->moe.router_logits);
            free(model->moe.router_probs);
            free(model->moe.selected_expert);
            free(model->moe.retrieved_buffer);
            model->moe.router_logits = (float*)mallocCheck((size_t)BT * E * sizeof(float));
            model->moe.router_probs = (float*)mallocCheck((size_t)BT * E * sizeof(float));
            model->moe.selected_expert = (int*)mallocCheck((size_t)BT * sizeof(int));
            model->moe.retrieved_buffer = (float*)mallocCheck((size_t)BT * C * sizeof(float));
            model->moe.scratch_bt = BT;
        }
        return;
    }
    uint64_t seed = 0xC6BC279692B5C323ull;
    #define MOE_NEXT_RAND() (seed ^= seed << 7, seed ^= seed >> 9, ((seed & 0xFFFFFF) / (float)0x1000000))
    model->moe.router_w = (float*)mallocCheck((size_t)E * C * sizeof(float));
    model->moe.router_b = (float*)calloc((size_t)E, sizeof(float));
    model->moe.expert_memory = (float*)calloc((size_t)E * S * D, sizeof(float));
    model->moe.expert_memory_keys = (float*)mallocCheck((size_t)E * S * D * sizeof(float));
    model->moe.router_logits = (float*)mallocCheck((size_t)BT * E * sizeof(float));
    model->moe.router_probs = (float*)mallocCheck((size_t)BT * E * sizeof(float));
    model->moe.selected_expert = (int*)mallocCheck((size_t)BT * sizeof(int));
    model->moe.retrieved_buffer = (float*)mallocCheck((size_t)BT * C * sizeof(float));
    for (int i = 0; i < E * C; i++) { model->moe.router_w[i] = (MOE_NEXT_RAND() * 2.0f - 1.0f) * 0.02f; }
    for (int i = 0; i < E * S * D; i++) { model->moe.expert_memory_keys[i] = (MOE_NEXT_RAND() * 2.0f - 1.0f) * 0.02f; }
    model->moe.initialized = 1;
    model->moe.scratch_bt = BT;
    #undef MOE_NEXT_RAND
}

static void moe_forward_memory_scaffold(GPT2* model, float* l_ln2, float* l_fcproj, int B, int T, int C) {
    if (!model->moe_config.use_moe) { return; }
    int BT = B * T;
    int E = model->moe_config.moe_num_experts;
    int S = model->moe_config.moe_expert_memory_slots;
    int D = model->moe_config.moe_expert_memory_dim;
    float alpha = model->moe_config.moe_memory_update_rate;
    float scale = model->moe_config.moe_memory_fusion_scale;

    for (int bt = 0; bt < BT; bt++) {
        const float* h = l_ln2 + (size_t)bt * C;
        float* logits = model->moe.router_logits + (size_t)bt * E;
        float* probs = model->moe.router_probs + (size_t)bt * E;
        float max_logit = -FLT_MAX;
        int expert = 0;
        for (int e = 0; e < E; e++) {
            const float* rw = model->moe.router_w + (size_t)e * C;
            float logit = model->moe.router_b[e];
            for (int c = 0; c < C; c++) { logit += rw[c] * h[c]; }
            logits[e] = logit;
            if (logit > max_logit) { max_logit = logit; expert = e; }
        }
        float sum_exp = 0.0f;
        for (int e = 0; e < E; e++) {
            probs[e] = expf(logits[e] - max_logit);
            sum_exp += probs[e];
        }
        float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (int e = 0; e < E; e++) { probs[e] *= inv_sum; }
        model->moe.selected_expert[bt] = expert;

        const float* q = l_fcproj + (size_t)bt * C;
        float* retrieved = model->moe.retrieved_buffer + (size_t)bt * C;
        memset(retrieved, 0, (size_t)C * sizeof(float));

        int best_slot = 0;
        float best_score = -FLT_MAX;
        for (int s = 0; s < S; s++) {
            const float* key = model->moe.expert_memory_keys + ((size_t)expert * S + s) * D;
            float score = 0.0f;
            for (int d = 0; d < D; d++) { score += q[d] * key[d]; }
            if (score > best_score) { best_score = score; best_slot = s; }
        }

        float* mem = model->moe.expert_memory + ((size_t)expert * S + best_slot) * D;
        for (int d = 0; d < D; d++) {
            retrieved[d] = mem[d];
            l_fcproj[(size_t)bt * C + d] += scale * retrieved[d];
            mem[d] = (1.0f - alpha) * mem[d] + alpha * q[d];
        }
    }
}

static float alc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void alc_fail(const char* msg) {
    fprintf(stderr, "[ALC] %s\n", msg);
    exit(1);
}

static void alc_require(int cond, const char* msg) {
    if (!cond) { alc_fail(msg); }
}

static void alc_check_finite_scalar(float x, const char* name) {
    if (!isfinite(x)) {
        fprintf(stderr, "[ALC] non-finite value in %s\n", name);
        exit(1);
    }
}

static void alc_check_finite_buffer(const float* x, size_t n, const char* name) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            fprintf(stderr, "[ALC] non-finite value in %s at idx=%zu\n", name, i);
            exit(1);
        }
    }
}

static float alc_l2_norm(const float* x, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) { ss += x[i] * x[i]; }
    return sqrtf(ss);
}

static float alc_clamp(float x, float lo, float hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

static int alc_layer_enabled(const GPT2* model, int layer) {
    if (!model->alc_config.use_alc) { return 0; }
    int every = model->alc_config.alc_apply_every_n_layers;
    if (every <= 0) { return 0; }
    return ((layer + 1) % every) == 0;
}

static const char* alc_fusion_mode_name(int mode) {
    switch (mode) {
        case ALC_FUSION_ADDITIVE: return "additive";
        case ALC_FUSION_GATED: return "gated";
        default: return "unknown";
    }
}

static const char* alc_update_mode_name(int mode) {
    switch (mode) {
        case ALC_UPDATE_OFF: return "off";
        case ALC_UPDATE_TRAIN_ONLY: return "train_only";
        case ALC_UPDATE_ALWAYS: return "always";
        default: return "unknown";
    }
}

static const char* alc_routing_mode_name(int mode) {
    switch (mode) {
        case ALC_ROUTING_HARD_TOP1: return "hard_top1";
        case ALC_ROUTING_SOFTMAX: return "softmax";
        case ALC_ROUTING_TOPK_SOFTMAX: return "topk_softmax";
        default: return "unknown";
    }
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t endian_marker;
    uint32_t header_bytes;
    uint32_t channels;
    uint32_t num_slots;
    uint32_t slot_dim;
    uint32_t key_dim;
    uint32_t fusion_mode;
    uint32_t update_mode;
    uint32_t reserved[6];
} ALCCheckpointHeader;

void gpt2_set_alc_config(GPT2 *model, ALCConfig config) {
    model->alc_config = config;
}

static void gpt2_validate_alc_config(const GPT2* model) {
    if (!model->alc_config.use_alc) { return; }
    const ALCConfig* cfg = &model->alc_config;
    if (cfg->alc_num_slots <= 0) { fprintf(stderr, "[ALC] invalid config: alc_num_slots=%d (must be >0)\n", cfg->alc_num_slots); exit(1); }
    if (cfg->alc_slot_dim <= 0) { fprintf(stderr, "[ALC] invalid config: alc_slot_dim=%d (must be >0)\n", cfg->alc_slot_dim); exit(1); }
    if (cfg->alc_key_dim <= 0) { fprintf(stderr, "[ALC] invalid config: alc_key_dim=%d (must be >0)\n", cfg->alc_key_dim); exit(1); }
    if (cfg->alc_update_rate < 0.0f || cfg->alc_update_rate > 1.0f) {
        fprintf(stderr, "[ALC] invalid config: alc_update_rate=%.6f (must be in [0,1])\n", cfg->alc_update_rate);
        exit(1);
    }
    if (cfg->alc_apply_every_n_layers <= 0) {
        fprintf(stderr, "[ALC] invalid config: alc_apply_every_n_layers=%d (must be >0)\n", cfg->alc_apply_every_n_layers);
        exit(1);
    }
    if (cfg->alc_fusion_mode != ALC_FUSION_ADDITIVE && cfg->alc_fusion_mode != ALC_FUSION_GATED) {
        fprintf(stderr, "[ALC] invalid config: alc_fusion_mode=%d (valid: 0 additive, 1 gated)\n", cfg->alc_fusion_mode);
        exit(1);
    }
    if (cfg->alc_update_mode < ALC_UPDATE_OFF || cfg->alc_update_mode > ALC_UPDATE_ALWAYS) {
        fprintf(stderr, "[ALC] invalid config: alc_update_mode=%d (valid: 0 off, 1 train_only, 2 always)\n", cfg->alc_update_mode);
        exit(1);
    }
    if (cfg->alc_routing_mode < ALC_ROUTING_HARD_TOP1 || cfg->alc_routing_mode > ALC_ROUTING_TOPK_SOFTMAX) {
        fprintf(stderr, "[ALC] invalid config: alc_routing_mode=%d (valid: 0 hard_top1, 1 softmax, 2 topk_softmax)\n",
                cfg->alc_routing_mode);
        exit(1);
    }
    if (cfg->alc_temperature <= 0.0f) {
        fprintf(stderr, "[ALC] invalid config: alc_temperature=%.6f (must be >0)\n", cfg->alc_temperature);
        exit(1);
    }
    if (cfg->alc_topk <= 0 || cfg->alc_topk > cfg->alc_num_slots) {
        fprintf(stderr, "[ALC] invalid config: alc_topk=%d (must be in [1, alc_num_slots=%d])\n",
                cfg->alc_topk, cfg->alc_num_slots);
        exit(1);
    }
    if (model->config.channels <= 0) { fprintf(stderr, "[ALC] invalid model channels=%d\n", model->config.channels); exit(1); }
    alc_require(cfg->alc_slot_dim > 0 && cfg->alc_key_dim > 0, "invalid dimensions: slot_dim/key_dim must be > 0");
    alc_require(cfg->alc_num_slots > 0, "invalid dimensions: num_slots must be > 0");
    alc_require(cfg->alc_update_rate >= 0.0f && cfg->alc_update_rate <= 1.0f, "invalid update rate: must be in [0,1]");
    alc_require(cfg->alc_fusion_mode == ALC_FUSION_ADDITIVE || cfg->alc_fusion_mode == ALC_FUSION_GATED,
                "invalid fusion mode");
    alc_require(cfg->alc_update_mode >= ALC_UPDATE_OFF && cfg->alc_update_mode <= ALC_UPDATE_ALWAYS,
                "invalid update mode");
    alc_require(cfg->alc_routing_mode >= ALC_ROUTING_HARD_TOP1 && cfg->alc_routing_mode <= ALC_ROUTING_TOPK_SOFTMAX,
                "invalid routing mode");
    alc_require(cfg->alc_temperature > 0.0f, "invalid routing temperature: must be > 0");
    alc_require(cfg->alc_topk > 0 && cfg->alc_topk <= cfg->alc_num_slots, "invalid top-k routing value");
}

static void gpt2_init_alc_state(GPT2* model, int B, int T) {
    if (!model->alc_config.use_alc) { return; }
    gpt2_validate_alc_config(model);
    int S = model->alc_config.alc_num_slots;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    int C = model->config.channels;
    int BT = B * T;
    if (BT <= 0) { fprintf(stderr, "[ALC] invalid runtime B*T=%d\n", BT); exit(1); }
    if (model->alc.initialized) {
        if (model->alc.scratch_bt != BT) {
            free(model->alc.query_buffer);
            free(model->alc.retrieved_buffer);
            free(model->alc.selected_slots);
            free(model->alc.routing_probs);
            model->alc.query_buffer = (float*)mallocCheck((size_t)BT * K * sizeof(float));
            model->alc.retrieved_buffer = (float*)mallocCheck((size_t)BT * C * sizeof(float));
            model->alc.selected_slots = (int*)mallocCheck((size_t)BT * sizeof(int));
            model->alc.routing_probs = (float*)mallocCheck((size_t)BT * S * sizeof(float));
            model->alc.scratch_bt = BT;
        }
        return;
    }
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    #define ALC_NEXT_RAND() (seed ^= seed << 7, seed ^= seed >> 9, ((seed & 0xFFFFFF) / (float)0x1000000))
    model->alc.query_proj = (float*)mallocCheck((size_t)K * C * sizeof(float));
    model->alc.write_proj = (float*)mallocCheck((size_t)D * C * sizeof(float));
    model->alc.slot_to_hidden = (float*)mallocCheck((size_t)C * D * sizeof(float));
    model->alc.gate_h = (float*)mallocCheck((size_t)C * sizeof(float));
    model->alc.gate_a = (float*)mallocCheck((size_t)C * sizeof(float));
    model->alc.gate_b = (float*)mallocCheck((size_t)C * sizeof(float));
    model->alc.slot_keys = (float*)mallocCheck((size_t)S * K * sizeof(float));
    model->alc.slots = (float*)calloc((size_t)S * D, sizeof(float));
    model->alc.slot_hit_counts = (int*)calloc((size_t)S, sizeof(int));
    model->alc.query_buffer = (float*)mallocCheck((size_t)BT * K * sizeof(float));
    model->alc.retrieved_buffer = (float*)mallocCheck((size_t)BT * C * sizeof(float));
    model->alc.selected_slots = (int*)mallocCheck((size_t)BT * sizeof(int));
    model->alc.routing_probs = (float*)mallocCheck((size_t)BT * S * sizeof(float));
    for (int i = 0; i < K * C; i++) { model->alc.query_proj[i] = (ALC_NEXT_RAND() * 2.0f - 1.0f) * 0.02f; }
    for (int i = 0; i < D * C; i++) { model->alc.write_proj[i] = (ALC_NEXT_RAND() * 2.0f - 1.0f) * 0.01f; }
    for (int i = 0; i < C * D; i++) { model->alc.slot_to_hidden[i] = (ALC_NEXT_RAND() * 2.0f - 1.0f) * 0.01f; }
    for (int i = 0; i < C; i++) {
        model->alc.gate_h[i] = 0.2f;
        model->alc.gate_a[i] = 0.2f;
        model->alc.gate_b[i] = 0.0f;
    }
    for (int i = 0; i < S * K; i++) {
        model->alc.slot_keys[i] = (ALC_NEXT_RAND() * 2.0f - 1.0f) * 0.05f;
    }
    model->alc.initialized = 1;
    model->alc.scratch_bt = BT;
    #undef ALC_NEXT_RAND
}

static void alc_ensure_layer_traces(GPT2* model, int B, int T) {
    if (!model->alc_config.use_alc || !model->alc.initialized) { return; }
    int BT = B * T;
    int L = model->config.num_layers;
    int C = model->config.channels;
    if (model->alc.trace_bt == BT && model->alc.hidden_pre_layers != NULL &&
        model->alc.retrieved_layers != NULL && model->alc.selected_slots_layers != NULL &&
        model->alc.routing_probs_layers != NULL) {
        return;
    }
    free(model->alc.hidden_pre_layers);
    free(model->alc.retrieved_layers);
    free(model->alc.selected_slots_layers);
    free(model->alc.routing_probs_layers);
    model->alc.hidden_pre_layers = (float*)mallocCheck((size_t)L * BT * C * sizeof(float));
    model->alc.retrieved_layers = (float*)mallocCheck((size_t)L * BT * C * sizeof(float));
    model->alc.selected_slots_layers = (int*)mallocCheck((size_t)L * BT * sizeof(int));
    model->alc.routing_probs_layers = (float*)mallocCheck((size_t)L * BT * model->alc_config.alc_num_slots * sizeof(float));
    model->alc.trace_bt = BT;
}

static void alc_zero_param_grads(ALCState* alc, int C, int D, int K) {
    if (alc->d_query_proj != NULL) { memset(alc->d_query_proj, 0, (size_t)K * C * sizeof(float)); }
    if (alc->d_slot_to_hidden != NULL) { memset(alc->d_slot_to_hidden, 0, (size_t)C * D * sizeof(float)); }
    if (alc->d_gate_h != NULL) { memset(alc->d_gate_h, 0, (size_t)C * sizeof(float)); }
    if (alc->d_gate_a != NULL) { memset(alc->d_gate_a, 0, (size_t)C * sizeof(float)); }
    if (alc->d_gate_b != NULL) { memset(alc->d_gate_b, 0, (size_t)C * sizeof(float)); }
}

static void alc_ensure_grad_buffers(GPT2* model) {
    if (!model->alc_config.use_alc || !model->alc.initialized) { return; }
    int C = model->config.channels;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    if (model->alc.d_slot_to_hidden == NULL) {
        model->alc.d_query_proj = (float*)calloc((size_t)K * C, sizeof(float));
        model->alc.m_query_proj = (float*)calloc((size_t)K * C, sizeof(float));
        model->alc.v_query_proj = (float*)calloc((size_t)K * C, sizeof(float));
        model->alc.d_slot_to_hidden = (float*)calloc((size_t)C * D, sizeof(float));
        model->alc.m_slot_to_hidden = (float*)calloc((size_t)C * D, sizeof(float));
        model->alc.v_slot_to_hidden = (float*)calloc((size_t)C * D, sizeof(float));
        model->alc.d_gate_h = (float*)calloc((size_t)C, sizeof(float));
        model->alc.d_gate_a = (float*)calloc((size_t)C, sizeof(float));
        model->alc.d_gate_b = (float*)calloc((size_t)C, sizeof(float));
        model->alc.m_gate_h = (float*)calloc((size_t)C, sizeof(float));
        model->alc.v_gate_h = (float*)calloc((size_t)C, sizeof(float));
        model->alc.m_gate_a = (float*)calloc((size_t)C, sizeof(float));
        model->alc.v_gate_a = (float*)calloc((size_t)C, sizeof(float));
        model->alc.m_gate_b = (float*)calloc((size_t)C, sizeof(float));
        model->alc.v_gate_b = (float*)calloc((size_t)C, sizeof(float));
    }
}

static void alc_forward_read_and_fuse(GPT2* model, float* hidden, int B, int T, int layer) {
    int S = model->alc_config.alc_num_slots;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    int C = model->config.channels;
    int BT = B * T;
    const float routing_scale = 1.0f / sqrtf((float)K);
    const float gate_logit_abs_cap = 40.0f; // keeps sigmoid numerically stable in fp32
    const float additive_norm_ratio_cap = 100.0f; // very loose safety guard to prevent runaway fusion
    model->alc.forward_calls++;
    alc_require(model->alc.initialized, "forward called before ALC initialization");
    alc_require(layer >= 0 && layer < model->config.num_layers, "layer index out of bounds");
    alc_require(model->alc_config.alc_update_rate >= 0.0f && model->alc_config.alc_update_rate <= 1.0f,
                "alc_update_rate out of range in forward");
    alc_require(model->alc_config.alc_fusion_mode == ALC_FUSION_ADDITIVE || model->alc_config.alc_fusion_mode == ALC_FUSION_GATED,
                "invalid fusion mode in forward");
    alc_require(model->alc_config.alc_update_mode >= ALC_UPDATE_OFF && model->alc_config.alc_update_mode <= ALC_UPDATE_ALWAYS,
                "invalid update mode in forward");
    float* layer_hidden_pre = model->alc.hidden_pre_layers + (size_t)layer * BT * C;
    float* layer_retrieved = model->alc.retrieved_layers + (size_t)layer * BT * C;
    int* layer_slots = model->alc.selected_slots_layers + (size_t)layer * BT;
    float* layer_probs = model->alc.routing_probs_layers + (size_t)layer * BT * S;
    for (int bt = 0; bt < BT; bt++) {
        float* h = hidden + (size_t)bt * C;
        memcpy(layer_hidden_pre + (size_t)bt * C, h, (size_t)C * sizeof(float));
        float* q = model->alc.query_buffer + (size_t)bt * K;
        float* probs = model->alc.routing_probs + (size_t)bt * S;
        float scores[S];
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            const float* wrow = model->alc.query_proj + (size_t)k * C;
            for (int c = 0; c < C; c++) { sum += wrow[c] * h[c]; }
            q[k] = sum;
        }
        int best_slot = 0;
        float best_score = -INFINITY;
        for (int s = 0; s < S; s++) {
            const float* sk = model->alc.slot_keys + (size_t)s * K;
            float score = 0.0f;
            for (int k = 0; k < K; k++) { score += q[k] * sk[k]; }
            score *= routing_scale;
            alc_check_finite_scalar(score, "routing_scores");
            scores[s] = score;
            if (score > best_score) { best_score = score; best_slot = s; }
        }
        alc_require(best_slot >= 0 && best_slot < S, "selected slot index out of bounds");
        for (int s = 0; s < S; s++) { probs[s] = 0.0f; }
        if (model->alc_config.alc_routing_mode == ALC_ROUTING_HARD_TOP1) {
            probs[best_slot] = 1.0f;
        } else if (model->alc_config.alc_routing_mode == ALC_ROUTING_SOFTMAX) {
            float smax = -INFINITY;
            for (int s = 0; s < S; s++) { if (scores[s] > smax) { smax = scores[s]; } }
            float denom = 0.0f;
            float inv_tau = 1.0f / model->alc_config.alc_temperature;
            for (int s = 0; s < S; s++) {
                float z = (scores[s] - smax) * inv_tau;
                float e = expf(z);
                probs[s] = e;
                denom += e;
            }
            alc_require(denom > 0.0f && isfinite(denom), "invalid routing softmax denominator");
            for (int s = 0; s < S; s++) { probs[s] /= denom; }
        } else {
            int topk = model->alc_config.alc_topk;
            int topk_idx[topk];
            float topk_score[topk];
            for (int i = 0; i < topk; i++) { topk_idx[i] = -1; topk_score[i] = -INFINITY; }
            for (int s = 0; s < S; s++) {
                float sc = scores[s];
                int insert = -1;
                for (int i = 0; i < topk; i++) {
                    if (sc > topk_score[i]) { insert = i; break; }
                }
                if (insert >= 0) {
                    for (int i = topk - 1; i > insert; i--) {
                        topk_score[i] = topk_score[i - 1];
                        topk_idx[i] = topk_idx[i - 1];
                    }
                    topk_score[insert] = sc;
                    topk_idx[insert] = s;
                }
            }
            float smax = topk_score[0];
            float denom = 0.0f;
            float inv_tau = 1.0f / model->alc_config.alc_temperature;
            for (int i = 0; i < topk; i++) {
                alc_require(topk_idx[i] >= 0 && topk_idx[i] < S, "invalid top-k routing index");
                float z = (topk_score[i] - smax) * inv_tau;
                float e = expf(z);
                probs[topk_idx[i]] = e;
                denom += e;
            }
            alc_require(denom > 0.0f && isfinite(denom), "invalid top-k routing softmax denominator");
            for (int i = 0; i < topk; i++) {
                probs[topk_idx[i]] /= denom;
            }
        }
        float prob_sum = 0.0f;
        for (int s = 0; s < S; s++) { prob_sum += probs[s]; }
        alc_require(fabsf(prob_sum - 1.0f) < 1e-3f, "routing probabilities must sum to 1");
        model->alc.selected_slots[bt] = best_slot;
        layer_slots[bt] = best_slot;
        memcpy(layer_probs + (size_t)bt * S, probs, (size_t)S * sizeof(float));
        if (model->alc.slot_hit_counts != NULL) { model->alc.slot_hit_counts[best_slot]++; }
        float* retrieved = model->alc.retrieved_buffer + (size_t)bt * C;
        float adaptive[D];
        for (int d = 0; d < D; d++) { adaptive[d] = 0.0f; }
        for (int s = 0; s < S; s++) {
            const float p = probs[s];
            if (p == 0.0f) { continue; }
            const float* slot = model->alc.slots + (size_t)s * D;
            for (int d = 0; d < D; d++) {
                adaptive[d] += p * slot[d];
            }
        }
        for (int c = 0; c < C; c++) {
            const float* proj = model->alc.slot_to_hidden + (size_t)c * D;
            float mapped = 0.0f;
            for (int d = 0; d < D; d++) { mapped += proj[d] * adaptive[d]; }
            retrieved[c] = mapped;
        }
        alc_check_finite_buffer(retrieved, (size_t)C, "retrieved_buffer");
        memcpy(layer_retrieved + (size_t)bt * C, retrieved, (size_t)C * sizeof(float));
        if (model->alc_config.alc_fusion_mode == ALC_FUSION_ADDITIVE) {
            float h_norm = alc_l2_norm(h, C);
            float r_norm = alc_l2_norm(retrieved, C);
            float scale = model->alc_config.alc_additive_scale;
            if (r_norm > additive_norm_ratio_cap * (h_norm + 1e-6f)) {
                scale *= (additive_norm_ratio_cap * (h_norm + 1e-6f)) / (r_norm + 1e-12f);
            }
            for (int c = 0; c < C; c++) {
                h[c] += scale * retrieved[c];
            }
        } else {
            for (int c = 0; c < C; c++) {
                float gate_logit = model->alc.gate_h[c] * h[c] + model->alc.gate_a[c] * retrieved[c] + model->alc.gate_b[c];
                alc_check_finite_scalar(gate_logit, "gate_logits");
                float gate = alc_sigmoid(alc_clamp(gate_logit, -gate_logit_abs_cap, gate_logit_abs_cap));
                h[c] = gate * h[c] + (1.0f - gate) * retrieved[c];
            }
        }
        alc_check_finite_buffer(h, (size_t)C, "fused_hidden");
    }
}

static void alc_write_update(GPT2* model, float* hidden, int B, int T) {
    if (model->alc_config.alc_update_rate <= 0.0f) { return; }
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    int C = model->config.channels;
    int BT = B * T;
    float eta = model->alc_config.alc_update_rate;
    const float write_norm_cap = 1e6f;
    const float slot_norm_cap = 1e6f;
    const float key_norm_cap = 1e6f;
    model->alc.write_calls++;
    model->alc.total_writes += BT;
    alc_require(eta >= 0.0f && eta <= 1.0f, "alc_update_rate out of range in write_update");
    for (int bt = 0; bt < BT; bt++) {
        const float* probs = model->alc.routing_probs + (size_t)bt * model->alc_config.alc_num_slots;
        const float* q = model->alc.query_buffer + (size_t)bt * K;
        const float* h = hidden + (size_t)bt * C;
        float write_vec[D];
        float write_norm_ss = 0.0f;
        for (int d = 0; d < D; d++) {
            const float* wrow = model->alc.write_proj + (size_t)d * C;
            float projected = 0.0f;
            for (int j = 0; j < C; j++) { projected += wrow[j] * h[j]; }
            alc_check_finite_scalar(projected, "slot_write_projection");
            write_norm_ss += projected * projected;
            write_vec[d] = projected;
        }
        if (write_norm_ss > write_norm_cap * write_norm_cap) {
            float mul = write_norm_cap / (sqrtf(write_norm_ss) + 1e-12f);
            for (int d = 0; d < D; d++) { write_vec[d] *= mul; }
        }
        for (int s = 0; s < model->alc_config.alc_num_slots; s++) {
            float p = probs[s];
            if (p <= 0.0f) { continue; }
            float alpha = eta * p;
            float* slot = model->alc.slots + (size_t)s * D;
            float* key = model->alc.slot_keys + (size_t)s * K;
            for (int d = 0; d < D; d++) {
                slot[d] = (1.0f - alpha) * slot[d] + alpha * write_vec[d];
            }
            for (int k = 0; k < K; k++) {
                key[k] = (1.0f - alpha) * key[k] + alpha * q[k];
            }
            float slot_norm = alc_l2_norm(slot, D);
            float key_norm = alc_l2_norm(key, K);
            if (slot_norm > slot_norm_cap) {
                float mul = slot_norm_cap / (slot_norm + 1e-12f);
                for (int d = 0; d < D; d++) { slot[d] *= mul; }
            }
            if (key_norm > key_norm_cap) {
                float mul = key_norm_cap / (key_norm + 1e-12f);
                for (int k = 0; k < K; k++) { key[k] *= mul; }
            }
            alc_check_finite_buffer(slot, (size_t)D, "slot_updates");
            alc_check_finite_buffer(key, (size_t)K, "key_updates");
        }
    }
}

static void alc_backward_fuse_and_accumulate(GPT2* model, int layer, float* d_hidden_out, int B, int T) {
    if (!model->alc_config.use_alc || !alc_layer_enabled(model, layer)) { return; }
    int BT = B * T;
    int C = model->config.channels;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    int S = model->alc_config.alc_num_slots;
    const float gate_logit_abs_cap = 40.0f;
    const float routing_scale = 1.0f / sqrtf((float)K);
    const float* layer_hidden_pre = model->alc.hidden_pre_layers + (size_t)layer * BT * C;
    const float* layer_retrieved = model->alc.retrieved_layers + (size_t)layer * BT * C;
    const int* layer_slots = model->alc.selected_slots_layers + (size_t)layer * BT;
    const float* layer_probs = model->alc.routing_probs_layers + (size_t)layer * BT * S;
    alc_require(model->alc.d_slot_to_hidden != NULL, "missing grad buffer d_slot_to_hidden");
    alc_require(model->alc.d_query_proj != NULL, "missing grad buffer d_query_proj");
    for (int bt = 0; bt < BT; bt++) {
        float* d_h = d_hidden_out + (size_t)bt * C;
        const float* h_pre = layer_hidden_pre + (size_t)bt * C;
        const float* retrieved = layer_retrieved + (size_t)bt * C;
        int slot_idx = layer_slots[bt];
        const float* probs = layer_probs + (size_t)bt * S;
        alc_require(slot_idx >= 0 && slot_idx < model->alc_config.alc_num_slots, "selected slot index out of bounds in backward");
        float adaptive[D];
        for (int d = 0; d < D; d++) { adaptive[d] = 0.0f; }
        for (int s = 0; s < S; s++) {
            if (probs[s] == 0.0f) { continue; }
            const float* slot = model->alc.slots + (size_t)s * D;
            for (int d = 0; d < D; d++) { adaptive[d] += probs[s] * slot[d]; }
        }
        float d_adaptive[D];
        for (int d = 0; d < D; d++) { d_adaptive[d] = 0.0f; }
        for (int c = 0; c < C; c++) {
            float d_out = d_h[c];
            float d_retrieved = model->alc_config.alc_additive_scale * d_out;
            if (model->alc_config.alc_fusion_mode == ALC_FUSION_GATED) {
                float gh = model->alc.gate_h[c];
                float ga = model->alc.gate_a[c];
                float z_raw = gh * h_pre[c] + ga * retrieved[c] + model->alc.gate_b[c];
                float z = alc_clamp(z_raw, -gate_logit_abs_cap, gate_logit_abs_cap);
                float g = alc_sigmoid(z);
                float d_g = d_out * (h_pre[c] - retrieved[c]);
                float d_z = d_g * g * (1.0f - g);
                if (z != z_raw) { d_z = 0.0f; }
                alc_check_finite_scalar(d_z, "gate_grad_dz");
                model->alc.d_gate_h[c] += d_z * h_pre[c];
                model->alc.d_gate_a[c] += d_z * retrieved[c];
                model->alc.d_gate_b[c] += d_z;
                d_retrieved = d_out * (1.0f - g) + d_z * ga;
                d_h[c] = d_out * g + d_z * gh;
            }
            float* d_proj_row = model->alc.d_slot_to_hidden + (size_t)c * D;
            for (int d = 0; d < D; d++) {
                d_proj_row[d] += d_retrieved * adaptive[d];
                d_adaptive[d] += d_retrieved * model->alc.slot_to_hidden[(size_t)c * D + d];
            }
        }
        if (model->alc_config.alc_routing_mode != ALC_ROUTING_HARD_TOP1) {
            float d_prob[S];
            float pdotp = 0.0f;
            for (int s = 0; s < S; s++) {
                const float* slot = model->alc.slots + (size_t)s * D;
                float dp = 0.0f;
                for (int d = 0; d < D; d++) { dp += d_adaptive[d] * slot[d]; }
                d_prob[s] = dp;
                pdotp += probs[s] * dp;
            }
            float d_q[K];
            for (int k = 0; k < K; k++) { d_q[k] = 0.0f; }
            float inv_tau = 1.0f / model->alc_config.alc_temperature;
            for (int s = 0; s < S; s++) {
                if (probs[s] == 0.0f) { continue; }
                float d_score = probs[s] * (d_prob[s] - pdotp) * inv_tau;
                const float* sk = model->alc.slot_keys + (size_t)s * K;
                for (int k = 0; k < K; k++) {
                    d_q[k] += d_score * routing_scale * sk[k];
                }
            }
            for (int k = 0; k < K; k++) {
                float* d_qrow = model->alc.d_query_proj + (size_t)k * C;
                for (int c = 0; c < C; c++) {
                    d_qrow[c] += d_q[k] * h_pre[c];
                }
            }
        }
        alc_check_finite_buffer(d_h, (size_t)C, "d_hidden_post_alc");
    }
}

static void alc_adamw_update(float* p, float* g, float* m, float* v, size_t n,
                             float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    for (size_t i = 0; i < n; i++) {
        float grad = g[i];
        float param = p[i];
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        float m_hat = m_new / beta1_correction;
        float v_hat = v_new / beta2_correction;
        m[i] = m_new;
        v[i] = v_new;
        p[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

int gpt2_save_alc_state(const GPT2* model, const char* path) {
    if (!model->alc_config.use_alc) {
        fprintf(stderr, "[ALC] refusing to save state while ALC is disabled.\n");
        return 0;
    }
    if (!model->alc.initialized) {
        fprintf(stderr, "[ALC] refusing to save state before ALC is initialized.\n");
        return 0;
    }
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "[ALC] failed to open save path '%s': %s\n", path, strerror(errno));
        return 0;
    }
    ALCCheckpointHeader h = {
        .magic = 0x414C4331u, // ASCII "ALC1"
        .version = 1,
        .endian_marker = 0x01020304u,
        .header_bytes = (uint32_t)sizeof(ALCCheckpointHeader),
        .channels = (uint32_t)model->config.channels,
        .num_slots = (uint32_t)model->alc_config.alc_num_slots,
        .slot_dim = (uint32_t)model->alc_config.alc_slot_dim,
        .key_dim = (uint32_t)model->alc_config.alc_key_dim,
        .fusion_mode = (uint32_t)model->alc_config.alc_fusion_mode,
        .update_mode = (uint32_t)model->alc_config.alc_update_mode,
    };
    fwriteCheck(&h, sizeof(ALCCheckpointHeader), 1, f);
    int C = model->config.channels;
    int S = model->alc_config.alc_num_slots;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    fwriteCheck(model->alc.query_proj, sizeof(float), (size_t)K * C, f);
    fwriteCheck(model->alc.write_proj, sizeof(float), (size_t)D * C, f);
    fwriteCheck(model->alc.slot_to_hidden, sizeof(float), (size_t)C * D, f);
    fwriteCheck(model->alc.gate_h, sizeof(float), (size_t)C, f);
    fwriteCheck(model->alc.gate_a, sizeof(float), (size_t)C, f);
    fwriteCheck(model->alc.gate_b, sizeof(float), (size_t)C, f);
    fwriteCheck(model->alc.slot_keys, sizeof(float), (size_t)S * K, f);
    fwriteCheck(model->alc.slots, sizeof(float), (size_t)S * D, f);
    fcloseCheck(f);
    return 1;
}

int gpt2_load_alc_state(GPT2* model, const char* path, int B, int T) {
    if (!model->alc_config.use_alc) {
        fprintf(stderr, "[ALC] refusing to load state while ALC is disabled.\n");
        return 0;
    }
    gpt2_init_alc_state(model, B, T);
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "[ALC] failed to open state path '%s': %s\n", path, strerror(errno));
        return 0;
    }
    ALCCheckpointHeader h;
    freadCheck(&h, sizeof(ALCCheckpointHeader), 1, f);
    if (h.magic != 0x414C4331u || h.version != 1u) {
        fprintf(stderr, "[ALC] incompatible state file '%s': bad magic/version (%u/%u)\n", path, h.magic, h.version);
        fcloseCheck(f);
        exit(1);
    }
    if (h.endian_marker != 0x01020304u) {
        fprintf(stderr, "[ALC] incompatible state file '%s': unsupported endianness marker 0x%08x\n", path, h.endian_marker);
        fcloseCheck(f);
        exit(1);
    }
    if (h.header_bytes != sizeof(ALCCheckpointHeader)) {
        fprintf(stderr, "[ALC] incompatible state file '%s': header size mismatch (%u vs %zu)\n",
                path, h.header_bytes, sizeof(ALCCheckpointHeader));
        fcloseCheck(f);
        exit(1);
    }
    if (h.channels != model->config.channels || h.num_slots != model->alc_config.alc_num_slots ||
        h.slot_dim != model->alc_config.alc_slot_dim || h.key_dim != model->alc_config.alc_key_dim ||
        h.fusion_mode != model->alc_config.alc_fusion_mode || h.update_mode != model->alc_config.alc_update_mode) {
        fprintf(stderr,
                "[ALC] state mismatch for '%s': file(C=%d,S=%d,D=%d,K=%d,fusion=%d,update=%d) vs runtime(C=%d,S=%d,D=%d,K=%d,fusion=%d,update=%d)\n",
                path, h.channels, h.num_slots, h.slot_dim, h.key_dim, h.fusion_mode, h.update_mode,
                model->config.channels, model->alc_config.alc_num_slots, model->alc_config.alc_slot_dim,
                model->alc_config.alc_key_dim, model->alc_config.alc_fusion_mode, model->alc_config.alc_update_mode);
        fcloseCheck(f);
        exit(1);
    }
    int C = model->config.channels;
    int S = model->alc_config.alc_num_slots;
    int D = model->alc_config.alc_slot_dim;
    int K = model->alc_config.alc_key_dim;
    freadCheck(model->alc.query_proj, sizeof(float), (size_t)K * C, f);
    freadCheck(model->alc.write_proj, sizeof(float), (size_t)D * C, f);
    freadCheck(model->alc.slot_to_hidden, sizeof(float), (size_t)C * D, f);
    freadCheck(model->alc.gate_h, sizeof(float), (size_t)C, f);
    freadCheck(model->alc.gate_a, sizeof(float), (size_t)C, f);
    freadCheck(model->alc.gate_b, sizeof(float), (size_t)C, f);
    freadCheck(model->alc.slot_keys, sizeof(float), (size_t)S * K, f);
    freadCheck(model->alc.slots, sizeof(float), (size_t)S * D, f);
    alc_check_finite_buffer(model->alc.query_proj, (size_t)K * C, "loaded.query_proj");
    alc_check_finite_buffer(model->alc.write_proj, (size_t)D * C, "loaded.write_proj");
    alc_check_finite_buffer(model->alc.slot_to_hidden, (size_t)C * D, "loaded.slot_to_hidden");
    alc_check_finite_buffer(model->alc.gate_h, (size_t)C, "loaded.gate_h");
    alc_check_finite_buffer(model->alc.gate_a, (size_t)C, "loaded.gate_a");
    alc_check_finite_buffer(model->alc.gate_b, (size_t)C, "loaded.gate_b");
    alc_check_finite_buffer(model->alc.slot_keys, (size_t)S * K, "loaded.slot_keys");
    alc_check_finite_buffer(model->alc.slots, (size_t)S * D, "loaded.slots");
    int next = fgetc(f);
    if (next != EOF) {
        fprintf(stderr, "[ALC] incompatible state file '%s': trailing bytes detected\n", path);
        fcloseCheck(f);
        exit(1);
    }
    fcloseCheck(f);
    return 1;
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
    model->alc_config = (ALCConfig){
        .use_alc = 0,
        .alc_num_slots = 64,
        .alc_slot_dim = C,
        .alc_key_dim = 64,
        .alc_update_rate = 0.05f,
        .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_TRAIN_ONLY,
        .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f,
        .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 4,
        .alc_temperature = 1.0f,
    };
    memset(&model->alc, 0, sizeof(ALCState));
    model->moe_config = (MoEConfig){
        .use_moe = 0,
        .moe_num_experts = 4,
        .moe_topk = 1,
        .moe_apply_every_n_layers = 1,
        .moe_router_mode = MOE_ROUTING_TOP1,
        .moe_expert_memory_slots = 8,
        .moe_expert_memory_dim = C < 64 ? C : 64,
        .moe_memory_update_rate = 0.05f,
        .moe_memory_fusion_scale = 0.1f,
    };
    memset(&model->moe, 0, sizeof(MoEState));
}

void gpt2_build_from_synthetic(GPT2 *model, GPT2Config config) {
    model->config = config;
    fill_in_parameter_sizes(model->param_sizes, model->config);
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) { num_parameters += model->param_sizes[i]; }
    model->num_parameters = num_parameters;
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    uint64_t seed = 0xD1B54A32D192ED03ull;
    for (size_t i = 0; i < num_parameters; i++) {
        seed ^= seed << 7;
        seed ^= seed >> 9;
        float u = (seed & 0xFFFFFF) / (float)0x1000000;
        model->params_memory[i] = (u * 2.0f - 1.0f) * 0.02f;
    }
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    int C = model->config.channels;
    model->alc_config = (ALCConfig){
        .use_alc = 0,
        .alc_num_slots = 32,
        .alc_slot_dim = C,
        .alc_key_dim = C < 32 ? C : 32,
        .alc_update_rate = 0.05f,
        .alc_fusion_mode = ALC_FUSION_ADDITIVE,
        .alc_update_mode = ALC_UPDATE_TRAIN_ONLY,
        .alc_apply_every_n_layers = 1,
        .alc_additive_scale = 1.0f,
        .alc_routing_mode = ALC_ROUTING_TOPK_SOFTMAX,
        .alc_topk = 4,
        .alc_temperature = 1.0f,
    };
    memset(&model->alc, 0, sizeof(ALCState));
    model->moe_config = (MoEConfig){
        .use_moe = 0,
        .moe_num_experts = 4,
        .moe_topk = 1,
        .moe_apply_every_n_layers = 1,
        .moe_router_mode = MOE_ROUTING_TOP1,
        .moe_expert_memory_slots = 8,
        .moe_expert_memory_dim = C < 32 ? C : 32,
        .moe_memory_update_rate = 0.05f,
        .moe_memory_fusion_scale = 0.1f,
    };
    memset(&model->moe, 0, sizeof(MoEState));
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    if (model->moe_config.use_moe) {
        gpt2_init_moe_state(model, B, T);
        if (!model->moe.logged_enable_message) {
            printf("[MoE-experimental] enabled: experts=%d topk=%d memory_slots=%d memory_dim=%d update_rate=%.4f fusion_scale=%.4f apply_every_n_layers=%d\n",
                   model->moe_config.moe_num_experts,
                   model->moe_config.moe_topk,
                   model->moe_config.moe_expert_memory_slots,
                   model->moe_config.moe_expert_memory_dim,
                   model->moe_config.moe_memory_update_rate,
                   model->moe_config.moe_memory_fusion_scale,
                   model->moe_config.moe_apply_every_n_layers);
            model->moe.logged_enable_message = 1;
        }
    }

    if (model->alc_config.use_alc) {
        gpt2_init_alc_state(model, B, T);
        alc_ensure_layer_traces(model, B, T);
        if (!model->alc.logged_enable_message) {
            printf("[ALC] enabled: slots=%d slot_dim=%d key_dim=%d fusion_mode=%s update_mode=%s routing_mode=%s topk=%d temperature=%.4f apply_every_n_layers=%d update_rate=%.4f\n",
                   model->alc_config.alc_num_slots,
                   model->alc_config.alc_slot_dim,
                   model->alc_config.alc_key_dim,
                   alc_fusion_mode_name(model->alc_config.alc_fusion_mode),
                   alc_update_mode_name(model->alc_config.alc_update_mode),
                   alc_routing_mode_name(model->alc_config.alc_routing_mode),
                   model->alc_config.alc_topk,
                   model->alc_config.alc_temperature,
                   model->alc_config.alc_apply_every_n_layers,
                   model->alc_config.alc_update_rate);
            printf("[ALC] training semantics: gradients={slot_to_hidden%s%s} state_updates={slots,slot_keys} nondiff={write_proj}\n",
                   model->alc_config.alc_fusion_mode == ALC_FUSION_GATED ? ",gate_h,gate_a,gate_b" : "",
                   model->alc_config.alc_routing_mode == ALC_ROUTING_HARD_TOP1 ? "" : ",query_proj");
            model->alc.logged_enable_message = 1;
        }
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        if (moe_layer_enabled(model, l)) {
            moe_forward_memory_scaffold(model, l_ln2, l_fcproj, B, T, C);
        }
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        // post-FFN extension region ordering (future intent):
        // attention -> FFN/MoE -> Engram(optional) -> ALC(optional) -> residual continuation
        int alc_apply = alc_layer_enabled(model, l);
        if (model->alc_config.use_alc) { model->alc.layer_checks++; }
        if (alc_apply) {
            model->alc.layers_applied++;
            alc_forward_read_and_fuse(model, l_residual3, B, T, l);
            int do_update = (model->alc_config.alc_update_mode == ALC_UPDATE_ALWAYS)
                || (model->alc_config.alc_update_mode == ALC_UPDATE_TRAIN_ONLY && targets != NULL);
            if (do_update) {
                alc_write_update(model, l_residual3, B, T);
            }
        }
    }
    if (model->alc_config.use_alc && model->alc.debug_enabled && model->alc.forward_calls % 10 == 0) {
        int S = model->alc_config.alc_num_slots;
        int top_slot = 0;
        int top_hits = 0;
        for (int s = 0; s < S; s++) {
            int hits = model->alc.slot_hit_counts == NULL ? 0 : model->alc.slot_hit_counts[s];
            if (hits > top_hits) { top_hits = hits; top_slot = s; }
        }
        printf("[ALC][debug] fwd=%ld layers_applied=%ld/%ld writes=%ld tokens_written=%ld fusion=%s update=%s top_slot=%d hits=%d\n",
               model->alc.forward_calls, model->alc.layers_applied, model->alc.layer_checks,
               model->alc.write_calls, model->alc.total_writes,
               alc_fusion_mode_name(model->alc_config.alc_fusion_mode),
               alc_update_mode_name(model->alc_config.alc_update_mode),
               top_slot, top_hits);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
    if (model->alc_config.use_alc && model->alc.initialized) {
        alc_zero_param_grads(&model->alc, model->config.channels, model->alc_config.alc_slot_dim, model->alc_config.alc_key_dim);
    }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;
    if (model->alc_config.use_alc && model->alc.initialized) {
        alc_ensure_grad_buffers(model);
    }

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        if (model->alc_config.use_alc && alc_layer_enabled(model, l)) {
            alc_backward_fuse_and_accumulate(model, l, dl_residual3, B, T);
        }
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
    if (model->alc_config.use_alc && model->alc.initialized && model->alc.d_slot_to_hidden != NULL) {
        size_t C = model->config.channels;
        size_t D = model->alc_config.alc_slot_dim;
        size_t K = model->alc_config.alc_key_dim;
        if (model->alc_config.alc_routing_mode != ALC_ROUTING_HARD_TOP1) {
            alc_adamw_update(model->alc.query_proj, model->alc.d_query_proj,
                             model->alc.m_query_proj, model->alc.v_query_proj,
                             K * C, learning_rate, beta1, beta2, eps, weight_decay, t);
        }
        alc_adamw_update(model->alc.slot_to_hidden, model->alc.d_slot_to_hidden,
                         model->alc.m_slot_to_hidden, model->alc.v_slot_to_hidden,
                         C * D, learning_rate, beta1, beta2, eps, weight_decay, t);
        if (model->alc_config.alc_fusion_mode == ALC_FUSION_GATED) {
            alc_adamw_update(model->alc.gate_h, model->alc.d_gate_h, model->alc.m_gate_h, model->alc.v_gate_h,
                             C, learning_rate, beta1, beta2, eps, weight_decay, t);
            alc_adamw_update(model->alc.gate_a, model->alc.d_gate_a, model->alc.m_gate_a, model->alc.v_gate_a,
                             C, learning_rate, beta1, beta2, eps, weight_decay, t);
            alc_adamw_update(model->alc.gate_b, model->alc.d_gate_b, model->alc.m_gate_b, model->alc.v_gate_b,
                             C, learning_rate, beta1, beta2, eps, weight_decay, t);
        }
    }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
    free(model->alc.query_proj);
    free(model->alc.write_proj);
    free(model->alc.slot_to_hidden);
    free(model->alc.gate_h);
    free(model->alc.gate_a);
    free(model->alc.gate_b);
    free(model->alc.slot_keys);
    free(model->alc.slots);
    free(model->alc.query_buffer);
    free(model->alc.retrieved_buffer);
    free(model->alc.selected_slots);
    free(model->alc.routing_probs);
    free(model->alc.slot_hit_counts);
    free(model->alc.hidden_pre_layers);
    free(model->alc.retrieved_layers);
    free(model->alc.selected_slots_layers);
    free(model->alc.routing_probs_layers);
    free(model->alc.d_query_proj);
    free(model->alc.d_slot_to_hidden);
    free(model->alc.d_gate_h);
    free(model->alc.d_gate_a);
    free(model->alc.d_gate_b);
    free(model->alc.m_slot_to_hidden);
    free(model->alc.v_slot_to_hidden);
    free(model->alc.m_query_proj);
    free(model->alc.v_query_proj);
    free(model->alc.m_gate_h);
    free(model->alc.v_gate_h);
    free(model->alc.m_gate_a);
    free(model->alc.v_gate_a);
    free(model->alc.m_gate_b);
    free(model->alc.v_gate_b);
    free(model->moe.router_w);
    free(model->moe.router_b);
    free(model->moe.expert_memory);
    free(model->moe.expert_memory_keys);
    free(model->moe.router_logits);
    free(model->moe.router_probs);
    free(model->moe.selected_expert);
    free(model->moe.retrieved_buffer);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

static int env_get_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value == NULL ? fallback : atoi(value);
}

static float env_get_float(const char* name, float fallback) {
    const char* value = getenv(name);
    return value == NULL ? fallback : atof(value);
}

static const char* env_get_str(const char* name, const char* fallback) {
    const char* value = getenv(name);
    return value == NULL ? fallback : value;
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

    // build the GPT-2 model (checkpoint by default, synthetic via env opt-in)
    GPT2 model;
    int use_synth = env_get_int("LLMC_USE_SYNTHETIC_MODEL", 0);
    if (use_synth) {
        GPT2Config synth_cfg = {
            .max_seq_len = env_get_int("LLMC_SYNTH_MAX_SEQ_LEN", 16),
            .vocab_size = env_get_int("LLMC_SYNTH_VOCAB_SIZE", 128),
            .padded_vocab_size = env_get_int("LLMC_SYNTH_PADDED_VOCAB_SIZE", 128),
            .num_layers = env_get_int("LLMC_SYNTH_NUM_LAYERS", 2),
            .num_heads = env_get_int("LLMC_SYNTH_NUM_HEADS", 2),
            .channels = env_get_int("LLMC_SYNTH_CHANNELS", 32),
        };
        if (synth_cfg.padded_vocab_size < synth_cfg.vocab_size) {
            fprintf(stderr, "LLMC_SYNTH_PADDED_VOCAB_SIZE (%d) must be >= LLMC_SYNTH_VOCAB_SIZE (%d)\n",
                    synth_cfg.padded_vocab_size, synth_cfg.vocab_size);
            exit(1);
        }
        gpt2_build_from_synthetic(&model, synth_cfg);
        printf("[GPT-2 synthetic] tiny validation model initialized\n");
    } else {
        const char* checkpoint_path = env_get_str("LLMC_CHECKPOINT_PATH", "gpt2_124M.bin");
        gpt2_build_from_checkpoint(&model, checkpoint_path);
    }
    ALCConfig alc_cfg = model.alc_config;
    alc_cfg.use_alc = env_get_int("LLMC_USE_ALC", alc_cfg.use_alc);
    alc_cfg.alc_num_slots = env_get_int("LLMC_ALC_NUM_SLOTS", alc_cfg.alc_num_slots);
    alc_cfg.alc_slot_dim = env_get_int("LLMC_ALC_SLOT_DIM", alc_cfg.alc_slot_dim);
    alc_cfg.alc_key_dim = env_get_int("LLMC_ALC_KEY_DIM", alc_cfg.alc_key_dim);
    alc_cfg.alc_update_rate = env_get_float("LLMC_ALC_UPDATE_RATE", alc_cfg.alc_update_rate);
    alc_cfg.alc_fusion_mode = env_get_int("LLMC_ALC_FUSION_MODE", alc_cfg.alc_fusion_mode);
    alc_cfg.alc_update_mode = env_get_int("LLMC_ALC_UPDATE_MODE", alc_cfg.alc_update_mode);
    alc_cfg.alc_apply_every_n_layers = env_get_int("LLMC_ALC_APPLY_EVERY_N_LAYERS", alc_cfg.alc_apply_every_n_layers);
    alc_cfg.alc_additive_scale = env_get_float("LLMC_ALC_ADDITIVE_SCALE", alc_cfg.alc_additive_scale);
    alc_cfg.alc_routing_mode = env_get_int("LLMC_ALC_ROUTING_MODE", alc_cfg.alc_routing_mode);
    alc_cfg.alc_topk = env_get_int("LLMC_ALC_TOPK", alc_cfg.alc_topk);
    alc_cfg.alc_temperature = env_get_float("LLMC_ALC_TEMPERATURE", alc_cfg.alc_temperature);
    gpt2_set_alc_config(&model, alc_cfg);
    model.alc.debug_enabled = env_get_int("LLMC_ALC_DEBUG", 0);
    model.moe_config.use_moe = env_get_int("LLMC_USE_MOE", model.moe_config.use_moe);
    model.moe_config.moe_num_experts = env_get_int("LLMC_MOE_NUM_EXPERTS", model.moe_config.moe_num_experts);
    model.moe_config.moe_topk = env_get_int("LLMC_MOE_TOPK", model.moe_config.moe_topk);
    model.moe_config.moe_apply_every_n_layers = env_get_int("LLMC_MOE_APPLY_EVERY_N_LAYERS", model.moe_config.moe_apply_every_n_layers);
    model.moe_config.moe_expert_memory_slots = env_get_int("LLMC_MOE_EXPERT_MEMORY_SLOTS", model.moe_config.moe_expert_memory_slots);
    model.moe_config.moe_expert_memory_dim = env_get_int("LLMC_MOE_EXPERT_MEMORY_DIM", model.moe_config.moe_expert_memory_dim);
    model.moe_config.moe_memory_update_rate = env_get_float("LLMC_MOE_MEMORY_UPDATE_RATE", model.moe_config.moe_memory_update_rate);
    model.moe_config.moe_memory_fusion_scale = env_get_float("LLMC_MOE_MEMORY_FUSION_SCALE", model.moe_config.moe_memory_fusion_scale);

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    if (use_synth) {
        B = env_get_int("LLMC_SYNTH_BATCH_SIZE", 2);
        T = env_get_int("LLMC_SYNTH_SEQ_LEN", 8);
        if (T > model.config.max_seq_len) {
            fprintf(stderr, "LLMC_SYNTH_SEQ_LEN (%d) must be <= max_seq_len (%d)\n", T, model.config.max_seq_len);
            exit(1);
        }
    }
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;
    if (use_synth) { val_num_batches = 1; }

    const char* alc_state_in = env_get_str("LLMC_ALC_STATE_IN", "");
    const char* alc_state_out = env_get_str("LLMC_ALC_STATE_OUT", "");
    if (model.alc_config.use_alc && strlen(alc_state_in) > 0) {
        int loaded = gpt2_load_alc_state(&model, alc_state_in, B, T);
        if (loaded) { printf("[ALC] loaded state: %s\n", alc_state_in); }
    }

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    if (model.alc_config.use_alc && strlen(alc_state_out) > 0) {
        if (gpt2_save_alc_state(&model, alc_state_out)) {
            printf("[ALC] saved state: %s\n", alc_state_out);
        } else {
            fprintf(stderr, "[ALC] failed to save state: %s\n", alc_state_out);
        }
    }
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif
