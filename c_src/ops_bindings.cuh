#pragma once

#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define THREADS_PER_BLOCK 256

void copy_embedding(Weights* weights, RunState* run_state, size_t token);

void rmsnorm(__nv_bfloat16_raw* o, const __nv_bfloat16_raw* x, const __nv_bfloat16_raw* weight, size_t dim);

void matmul_cublas(
    cublasHandle_t handle, 
    __nv_bfloat16_raw* y, 
    const __nv_bfloat16_raw* W, 
    const __nv_bfloat16_raw* x, 
    size_t m, 
    size_t n,
    float alpha,
    float beta
);

void qk_norm_fused(
    __nv_bfloat16_raw* q,
    __nv_bfloat16_raw* k,
    const __nv_bfloat16_raw* q_norm_weight,
    const __nv_bfloat16_raw* k_norm_weight
);

void rope(
    __nv_bfloat16_raw* q,
    __nv_bfloat16_raw* k,
    size_t pos
);

void mha(
    RunState* s,
    size_t l, 
    size_t pos
);

void swiglu(
    __nv_bfloat16_raw* hb, 
    const __nv_bfloat16_raw* hb2, 
    size_t size
);

typedef struct Logits {
    float _inner[VOCAB_SIZE];
} Logits;

Logits get_logits(
    RunState* s
);

#ifdef __cplusplus
}
#endif