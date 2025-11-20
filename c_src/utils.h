#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define SEQ_LEN 8192
#define HEAD_DIM 128
#define HIDDEN_SIZE 1024
#define NUM_ATTN_HEADS 16
#define NUM_HIDDEN_LAYERS 28
#define NUM_KV_HEADS 8
#define INTERMEDIATE_SIZE 3072
#define VOCAB_SIZE 151936
#define Q_DIM (NUM_ATTN_HEADS * HEAD_DIM)
#define KV_DIM (NUM_KV_HEADS * HEAD_DIM)

#define INV_HEAD_DIM (1.0f / HEAD_DIM)
#define INV_DIM (1.0f / HIDDEN_SIZE)

#define ROPE_THETA 1000000.0f
#define EPS 1e-6f

typedef struct Attention {
    __nv_bfloat16_raw* q_proj;
    __nv_bfloat16_raw* k_proj;
    __nv_bfloat16_raw* v_proj;
    __nv_bfloat16_raw* o_proj;
    __nv_bfloat16_raw* q_norm;
    __nv_bfloat16_raw* k_norm;
} Attention;

typedef struct FFN {
    __nv_bfloat16_raw* gate_proj;
    __nv_bfloat16_raw* up_proj;
    __nv_bfloat16_raw* down_proj;
} FFN;

typedef struct TransformerBlocks {
    __nv_bfloat16_raw* input_layernorm;
    __nv_bfloat16_raw* post_attention_layernorm;
    Attention attention;
    FFN ffn;
} TransformerBlock;

typedef struct Weights {
    __nv_bfloat16_raw* embedding; // model.embed_tokens.weight
    TransformerBlock transformer_blocks[NUM_HIDDEN_LAYERS];
    __nv_bfloat16_raw* norm;     // model.norm.weight
    __nv_bfloat16_raw* lm_head;    // lm_head.weight
    void* gpu_mem_block;
} Weights;

typedef struct
{
    __nv_bfloat16_raw* x;       // activation at current time stamp (HIDDEN_SIZE,)
    __nv_bfloat16_raw* xb;      // buffer for residual branch (HIDDEN_SIZE,)
    __nv_bfloat16_raw* xb2;     // an additional buffer (HIDDEN_SIZE,)
    __nv_bfloat16_raw* hb;      // buffer for hidden dimension in ffn (INTERMEDIATE_SIZE,)
    __nv_bfloat16_raw* hb2;     // buffer for hidden dimension in ffn (INTERMEDIATE_SIZE,)
    __nv_bfloat16_raw* q;       // query buffer (Q_DIM,) - NOTE: This is larger than HIDDEN_SIZE now

    float* att;    // buffer for scores/attention values (NUM_ATTN_HEADS, SEQ_LEN)
    __nv_bfloat16_raw* logits;  // output logits on the GPU (VOCAB_SIZE,)

    // kv cache
    __nv_bfloat16_raw* key_cache;   // (NUM_HIDDEN_LAYERS, SEQ_LEN, KV_DIM)
    __nv_bfloat16_raw* value_cache; // (NUM_HIDDEN_LAYERS, SEQ_LEN, KV_DIM)
    
    // buffer for final logits converted to fp32 on the GPU
    float* d_logits_fp32;
} RunState;

RunState init_run_state() {
    RunState s;

    cudaMalloc((void**)&s.x, HIDDEN_SIZE * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.xb, HIDDEN_SIZE * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.xb2, HIDDEN_SIZE * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.hb, INTERMEDIATE_SIZE * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.hb2, INTERMEDIATE_SIZE * sizeof(__nv_bfloat16_raw));
    // query buffer must be Q_DIM, which is NUM_ATTN_HEADS * HEAD_DIM = 2048 for this model.
    cudaMalloc((void**)&s.q, Q_DIM * sizeof(__nv_bfloat16_raw));
    
    cudaMalloc((void**)&s.att, (size_t)NUM_ATTN_HEADS * SEQ_LEN * sizeof(float));
    cudaMalloc((void**)&s.logits, VOCAB_SIZE * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.key_cache, (size_t)NUM_HIDDEN_LAYERS * SEQ_LEN * KV_DIM * sizeof(__nv_bfloat16_raw));
    cudaMalloc((void**)&s.value_cache, (size_t)NUM_HIDDEN_LAYERS * SEQ_LEN * KV_DIM * sizeof(__nv_bfloat16_raw));

    cudaMalloc((void**)&s.d_logits_fp32, VOCAB_SIZE * sizeof(float));

    return s;
}

cublasHandle_t init_cublas_handle() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

