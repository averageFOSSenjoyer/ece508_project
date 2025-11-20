#pragma once

#include "utils.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>

template<int TPB>
__global__ void __launch_bounds__(TPB)
rms_norm_kernel(
    __nv_bfloat16_raw* __restrict__ Y,
    const __nv_bfloat16_raw* __restrict__ X,
    const __nv_bfloat16_raw* __restrict__ weight,
    size_t D
) {
    const int t_idx = threadIdx.x;
    const int vec_iters = D / 2;

    const __nv_bfloat162* row_in = reinterpret_cast<const __nv_bfloat162*>(X);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out = reinterpret_cast<__nv_bfloat162*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += TPB) {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        // convert to fp32 for math
        float2 v_fp32 = __bfloat1622float2(v_bf16);

        // lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    __shared__ float mul_val;
    if (t_idx == 0) {
        float val = __fmaf_rn(block_sum, INV_DIM, EPS);
        float approx = __frsqrt_rn(val);
        mul_val = rsqrtf(val);
    }
    __syncthreads();

    for (int idx = t_idx; idx < vec_iters; idx += TPB)
    {
        __nv_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        // convert back to BF16 for storing
        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

template<int TPB, int HD>
__global__ void __launch_bounds__(TPB)
fused_multi_rmsnorm_kernel(
    __nv_bfloat16_raw* __restrict__ vecs,
    const __nv_bfloat16_raw* __restrict__ weight,
    size_t num_vecs
) {
    // each block processes one vector/head
    const int vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const int t_idx = threadIdx.x;
    const int vec_iters = HD / 2;

    __nv_bfloat16_raw* vec_start = vecs + vec_idx * HD;

    const __nv_bfloat162* row_in = reinterpret_cast<const __nv_bfloat162*>(vec_start);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out = reinterpret_cast<__nv_bfloat162*>(vec_start);

    // 1. calculate sum of squares
    float lsum = 0.0f;
    for (int idx = t_idx; idx < vec_iters; idx += TPB)
    {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        float2 v_fp32 = __bfloat1622float2(v_bf16);
        lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
    }

    // 2. reduce sum within the block
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) { mul_val = rsqrtf(block_sum * INV_HEAD_DIM + EPS); }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += TPB)
    {
        __nv_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

__global__ void qwen_naive_rope_kernel(
    __nv_bfloat16_raw* q,
    __nv_bfloat16_raw* k_cache_pos,
    size_t pos
) {
    // `blockIdx.x` will correspond to the head index 'h'
    int h = blockIdx.x;
    // `threadIdx.x` will correspond to the inner loop index 'j'
    int j = threadIdx.x;

    if (h < NUM_ATTN_HEADS && j < HEAD_DIM / 2)
    {
        __nv_bfloat16_raw* q_head = q + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float q_real = __bfloat162float(q_head[j]);
        float q_imag = __bfloat162float(q_head[j + HEAD_DIM / 2]);

        float q_rotated_real = q_real * fcr - q_imag * fci;
        float q_rotated_imag = q_real * fci + q_imag * fcr;

        q_head[j]              = __float2bfloat16_rn(q_rotated_real);
        q_head[j + HEAD_DIM/2] = __float2bfloat16_rn(q_rotated_imag);
    }

    if (h < NUM_KV_HEADS && j < HEAD_DIM / 2)
    {
        __nv_bfloat16_raw* k_head = k_cache_pos + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float k_real = __bfloat162float(k_head[j]);
        float k_imag = __bfloat162float(k_head[j + HEAD_DIM / 2]);

        // perform rotation in fp32
        float k_rotated_real = k_real * fcr - k_imag * fci;
        float k_rotated_imag = k_real * fci + k_imag * fcr;

        k_head[j]              = __float2bfloat16_rn(k_rotated_real);
        k_head[j + HEAD_DIM/2] = __float2bfloat16_rn(k_rotated_imag);
    }
}

__global__ void
attention_qk_kernel(
    float* att,
    const __nv_bfloat16_raw* q,
    const __nv_bfloat16_raw* k_cache,
    int pos)
{
    // grid: NUM_ATTN_HEADS, block: pos + 1 (up to 1024)
    int h = blockIdx.x; 
    int t = threadIdx.x;

    constexpr int kv_mul = NUM_ATTN_HEADS / NUM_KV_HEADS;

    if (t <= pos)
    {
        const __nv_bfloat16_raw* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        const __nv_bfloat16_raw* k_vec = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++)
        {
            __nv_bfloat162 q_pair = reinterpret_cast<const __nv_bfloat162*>(q_head)[i];
            __nv_bfloat162 k_pair = reinterpret_cast<const __nv_bfloat162*>(k_vec)[i];

            float2 q_vals = __bfloat1622float2(q_pair);
            float2 k_vals = __bfloat1622float2(k_pair);

            // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
    }
}

__global__ void
softmax_kernel(
    float* att, 
    size_t pos
) {
    // grid: NUM_ATTN_HEADS, block: 1
    int h = blockIdx.x;

    float* scores = att + (size_t)h * SEQ_LEN;
    int len = pos + 1;

    // find max value for numerical stability
    // float max_val = -HUGE_VALF;
    float max_val = -1e9f;
    for (int i = 0; i < len; i++)
    {
        if (scores[i] > max_val) { max_val = scores[i]; }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) { scores[i] *= inv_sum; }
}

__global__ void attention_qk_kernel(
    float* att,
    const __nv_bfloat16_raw* q,
    const __nv_bfloat16_raw* k_cache,
    size_t pos
) {
    // grid: NUM_ATTN_HEADS, block: pos + 1 (up to 1024)
    int h = blockIdx.x; 
    int t = threadIdx.x;

    constexpr int kv_mul = NUM_ATTN_HEADS / NUM_KV_HEADS;

    if (t <= pos)
    {
        const __nv_bfloat16_raw* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        const __nv_bfloat16_raw* k_vec = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++)
        {
            __nv_bfloat162 q_pair = reinterpret_cast<const __nv_bfloat162*>(q_head)[i];
            __nv_bfloat162 k_pair = reinterpret_cast<const __nv_bfloat162*>(k_vec)[i];

            float2 q_vals = __bfloat1622float2(q_pair);
            float2 k_vals = __bfloat1622float2(k_pair);

            // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
    }
}

__global__ void attention_v_kernel(
    __nv_bfloat16_raw* out,
    const float* att,
    const __nv_bfloat16_raw* v_cache,
    size_t pos
) {
    // grid: NUM_ATTN_HEADS, block: HEAD_DIM
    int h = blockIdx.x;
    int i = threadIdx.x; // idx within the head dimension
    constexpr int kv_mul = NUM_ATTN_HEADS / NUM_KV_HEADS;

    __nv_bfloat16_raw* out_head = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * SEQ_LEN;
    int kv_head_idx = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const __nv_bfloat16_raw* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        // weighted_sum += att_head[t] * __bfloat162float(v_vec[i]);   
        weighted_sum = __fmaf_rn(att_head[t], __bfloat162float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2bfloat16_rn(weighted_sum);
}

__global__ void swiglu_kernel(
    __nv_bfloat16_raw* hb, 
    const __nv_bfloat16_raw* hb2, 
    size_t size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float val_fp32 = __bfloat162float(hb[i]);
        float hb2_fp32 = __bfloat162float(hb2[i]);
        
        float silu_val = val_fp32 * (1.0f / (1.0f + expf(-val_fp32)));
        float result_fp32 = silu_val * hb2_fp32;
        hb[i] = __float2bfloat16_rn(result_fp32);
    }
}

__global__ void convert_bf16_to_fp32_kernel(
    __nv_bfloat16_raw* bf16_in, 
    float* fp32_out, 
    size_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){ fp32_out[i] = __bfloat162float(bf16_in[i]); }
}