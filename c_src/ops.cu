#include "utils.h"
#include "ops_bindings.cuh"
#include "ops_impl.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>

extern "C"  void copy_embedding(Weights* weights, RunState* run_state, size_t token) {
    __nv_bfloat16_raw* token_embedding_ptr = weights->embedding + token * HIDDEN_SIZE;
    cudaMemcpy(run_state->x, token_embedding_ptr, (size_t)HIDDEN_SIZE * sizeof(__nv_bfloat16_raw), cudaMemcpyDeviceToDevice);
}

extern "C" void rmsnorm(
    __nv_bfloat16_raw* o,
    const __nv_bfloat16_raw* x,
    const __nv_bfloat16_raw* weight,
    size_t dim
) {
    if (dim % 2 != 0)
    {
        printf("FATAL: rmsnorm dim %lu is not divisible by 2. Vectorized kernel cannot run.\n", dim);
        exit(EXIT_FAILURE);
    }
    // if dim > (THREADS_PER_BLOCK * some_threshold), a multi-block reduction might be needed,
    // but for typical dimensions up to 8192, a single block is sufficient and simpler.
    const int num_blocks = 1;

    rms_norm_kernel<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(o, x, weight, dim);
}

// performs a matrix-vector multiplication y = Wx using cuBLAS.
// W is a matrix (m rows, n cols), x is a vector (len n), y is a vector (len m).
extern "C" void matmul_cublas(
    cublasHandle_t handle, 
    __nv_bfloat16_raw* y, 
    const __nv_bfloat16_raw* W, 
    const __nv_bfloat16_raw* x, 
    size_t m, 
    size_t n,
    float alpha,
    float beta
) {
    // in cuBLAS, matrices are column-major by default. 
    // weights are row-major.
    // W matrix is (m, n) in row-major layout, which is (n, m) in column-major.
    // we want to compute y = Wx.
    // by telling cublasSgemv to use the transpose of W (CUBLAS_OP_T),
    // it correctly treats our row-major matrix as a row-major matrix.

    // C = alpha * (A @ B) + beta * C

    // cublasSgemv: y = alpha * op(A) * x + beta * y
    // op(A) is our W matrix. handle is the cuBLAS context.
    // CUBLAS_OP_T means "use the transpose of A".
    // n, m are the dimensions of the matrix as seen by cuBLAS (column-major).
    // So for our (m, n) row-major matrix, it's seen as (n, m) column-major.
    // W is the pointer to the matrix. n is the leading dimension (width of the row-major matrix).
    // x is the input vector.  1 is its stride.
    // y is the output vector. 1 is its stride.
    cublasGemmEx(handle,
                 CUBLAS_OP_T,        // Transpose W (since it's row-major)
                 CUBLAS_OP_N,        // Don't transpose x
                 m,                  // rows of C (output y)
                 1,                  // columns of C (output y is a vector)
                 n,                  // common dimension (k)
                 
                 &alpha,             // host pointer
                 W,                  // A matrix (W)
                 CUDA_R_16BF,        // A datatype
                 n,                  // leading dimension of A

                 x,                  // B matrix (x)
                 CUDA_R_16BF,        // B datatype
                 n,                  // leading dimension of B
                 
                 &beta,              // host pointer
                 y,                  // C matrix (y)
                 CUDA_R_16BF,        // C datatype
                 m,                  // leading dimension of C
                 
                 CUDA_R_32F,         // compute type: use fp32 for precision
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

extern "C" void qk_norm_fused(
    __nv_bfloat16_raw* q,
    __nv_bfloat16_raw* k,
    const __nv_bfloat16_raw* q_norm_weight,
    const __nv_bfloat16_raw* k_norm_weight
) {
    constexpr int QK_NORM_THREADS_PER_BLOCK = 64;

    // launching ONE kernel for all query heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<NUM_ATTN_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (q, q_norm_weight, NUM_ATTN_HEADS);

    // launching ONE kernel for all key heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<NUM_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (k, k_norm_weight, NUM_KV_HEADS);
}

extern "C" void rope(
    __nv_bfloat16_raw* q,
    __nv_bfloat16_raw* k,
    size_t pos
) {
    dim3 grid(NUM_ATTN_HEADS, 1, 1);
    dim3 block(HEAD_DIM / 2, 1, 1);

    qwen_naive_rope_kernel<<<grid, block>>>(q, k, pos);
}

extern "C" void mha(
    RunState* s,
    size_t l, 
    size_t pos
) {
    __nv_bfloat16_raw* layer_key_cache = s->key_cache     + (size_t)l * SEQ_LEN * KV_DIM;
    __nv_bfloat16_raw* layer_value_cache = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM;

    // kernel 1: calculate QK scores
    int qk_threads_per_block = std::min((size_t)1024, pos + 1);
    attention_qk_kernel<<<NUM_ATTN_HEADS, qk_threads_per_block>>>(
        s->att, s->q, layer_key_cache, pos
    );

    // kernel 2: softmax
    softmax_kernel<<<NUM_ATTN_HEADS, 1>>>(s->att, pos);

    // kernel 3: aggregate V values
    attention_v_kernel<<<NUM_ATTN_HEADS, HEAD_DIM>>>(
        s->q, s->att, layer_value_cache, pos
    );
}

extern "C" void swiglu(
    __nv_bfloat16_raw* hb, 
    const __nv_bfloat16_raw* hb2, 
    size_t size
) {
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    swiglu_kernel<<<grid_size, THREADS_PER_BLOCK>>>(hb, hb2, size);
}

extern "C" Logits get_logits(
    RunState* s
) {
    Logits logits;

    int grid_size = (VOCAB_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    convert_bf16_to_fp32_kernel<<<grid_size, THREADS_PER_BLOCK>>>(s->logits, s->d_logits_fp32, VOCAB_SIZE);

    // 13. copy the fp32 logits from GPU device to pinned host memory for the CPU to access
    cudaMemcpy(&logits._inner, s->d_logits_fp32, (size_t)VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return logits;
}