#include "linear_cuda_kernels.cuh"
#include <vector>

#ifdef TENSORN_CUDA_AVAILABLE

constexpr int LBLOCK_SIZE = 256;

namespace linear_cuda {

__global__ void causal_linear_attn_fwd_kernel(
    const float* __restrict__ Q_phi,
    const float* __restrict__ K_phi,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ den_cache,
    float* __restrict__ z_cache,
    float* __restrict__ S_buf,
    float* __restrict__ z_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    if (b >= B) return;

    size_t DxE = D * E;
    float* state = S_buf + b * DxE;
    float* z = z_buf + b * D;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < DxE; i += blockDim.x)
        state[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x)
        z[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++) {
        size_t idx = base + t;

        for (size_t de = threadIdx.x; de < DxE; de += blockDim.x) {
            size_t d = de / E;
            size_t e = de % E;
            state[de] += K_phi[idx * D + d] * V[idx * E + e];
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            z[d] += K_phi[idx * D + d];
        }
        __syncthreads();

        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            z_cache[idx * D + d] = z[d];
        }
        __syncthreads();

        float norm = 0.0f;
        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            norm += Q_phi[idx * D + d] * z[d];
        }

        extern __shared__ float shared_norm[];
        shared_norm[threadIdx.x] = norm;
        __syncthreads();
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                shared_norm[threadIdx.x] += shared_norm[threadIdx.x + s];
            __syncthreads();
        }
        float total_norm = shared_norm[0] + 1e-8f;
        den_cache[idx] = total_norm;
        __syncthreads();

        for (size_t e = threadIdx.x; e < E; e += blockDim.x) {
            float num = 0.0f;
            for (size_t d = 0; d < D; d++) {
                num += Q_phi[idx * D + d] * state[d * E + e];
            }
            O[idx * E + e] = num / total_norm;
        }
        __syncthreads();
    }
}

void causal_linear_attention_fwd(
    const CudaTensor<float>& Q_phi, const CudaTensor<float>& K_phi,
    const CudaTensor<float>& V,
    CudaTensor<float>& O, CudaTensor<float>& den_cache,
    CudaTensor<float>& z_cache,
    CudaTensor<float>& S_buf, CudaTensor<float>& z_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t block_threads = (size_t)min(max(D, E), (size_t)LBLOCK_SIZE);
    dim3 grid((unsigned int)B);
    dim3 block((unsigned int)block_threads);
    size_t shared_mem = block_threads * sizeof(float);
    causal_linear_attn_fwd_kernel<<<grid, block, shared_mem>>>(
        Q_phi.device_ptr(), K_phi.device_ptr(), V.device_ptr(),
        O.device_ptr(), den_cache.device_ptr(), z_cache.device_ptr(),
        S_buf.device_ptr(), z_buf.device_ptr(),
        B, S, D, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void causal_linear_attn_bwd_step1_kernel(
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ den_cache,
    float* __restrict__ dnum,
    float* __restrict__ dden,
    size_t N, size_t E)
{
    size_t i = blockIdx.x;
    if (i >= N) return;

    const float* o_row = O + i * E;
    const float* dout_row = dO + i * E;
    float* dnum_row = dnum + i * E;
    float inv_den = 1.0f / den_cache[i];

    float sum = 0.0f;
    for (size_t e = threadIdx.x; e < E; e += blockDim.x) {
        float val = dout_row[e] * inv_den;
        dnum_row[e] = val;
        sum += dout_row[e] * o_row[e];
    }

    extern __shared__ float shared_sum[];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dden[i] = -shared_sum[0] * inv_den;
}

void causal_linear_attention_bwd_step1(
    const CudaTensor<float>& O, const CudaTensor<float>& dO,
    const CudaTensor<float>& den_cache,
    CudaTensor<float>& dnum, CudaTensor<float>& dden,
    size_t N, size_t E)
{
    size_t block_threads = (size_t)min(E, (size_t)LBLOCK_SIZE);
    dim3 grid((unsigned int)N);
    dim3 block((unsigned int)block_threads);
    size_t shared_mem = block_threads * sizeof(float);
    causal_linear_attn_bwd_step1_kernel<<<grid, block, shared_mem>>>(
        O.device_ptr(), dO.device_ptr(), den_cache.device_ptr(),
        dnum.device_ptr(), dden.device_ptr(), N, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void causal_linear_attn_bwd_dQ_kernel(
    const float* __restrict__ K_phi,
    const float* __restrict__ V,
    const float* __restrict__ dnum,
    const float* __restrict__ dden,
    const float* __restrict__ z_cache,
    float* __restrict__ dQ_phi,
    float* __restrict__ S_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    if (b >= B) return;

    size_t DxE = D * E;
    float* state = S_buf + b * DxE;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < DxE; i += blockDim.x)
        state[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++) {
        size_t idx = base + t;

        for (size_t de = threadIdx.x; de < DxE; de += blockDim.x) {
            size_t d = de / E;
            size_t e = de % E;
            state[de] += K_phi[idx * D + d] * V[idx * E + e];
        }
        __syncthreads();

        const float* z_row = z_cache + idx * D;
        float* dq_row = dQ_phi + idx * D;
        const float* dnum_row = dnum + idx * E;
        float dd = dden[idx];

        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            float val = z_row[d] * dd;
            for (size_t e = 0; e < E; e++) {
                val += state[d * E + e] * dnum_row[e];
            }
            dq_row[d] = val;
        }
        __syncthreads();
    }
}

void causal_linear_attention_bwd_dQ(
    const CudaTensor<float>& K_phi, const CudaTensor<float>& V,
    const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
    const CudaTensor<float>& z_cache,
    CudaTensor<float>& dQ_phi, CudaTensor<float>& S_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t block_threads = (size_t)min(max(D, E), (size_t)LBLOCK_SIZE);
    dim3 grid((unsigned int)B);
    dim3 block((unsigned int)block_threads);
    causal_linear_attn_bwd_dQ_kernel<<<grid, block>>>(
        K_phi.device_ptr(), V.device_ptr(),
        dnum.device_ptr(), dden.device_ptr(), z_cache.device_ptr(),
        dQ_phi.device_ptr(), S_buf.device_ptr(),
        B, S, D, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void causal_linear_attn_bwd_dKdV_kernel(
    const float* __restrict__ Q_phi,
    const float* __restrict__ K_phi,
    const float* __restrict__ V,
    const float* __restrict__ dnum,
    const float* __restrict__ dden,
    float* __restrict__ dK_phi,
    float* __restrict__ dV,
    float* __restrict__ S_star_buf,
    float* __restrict__ z_star_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    if (b >= B) return;

    size_t DxE = D * E;
    float* S_star = S_star_buf + b * DxE;
    float* z_star = z_star_buf + b * D;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < DxE; i += blockDim.x)
        S_star[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x)
        z_star[i] = 0.0f;
    __syncthreads();

    for (int t_int = (int)S - 1; t_int >= 0; t_int--) {
        size_t t = (size_t)t_int;
        size_t idx = base + t;

        for (size_t de = threadIdx.x; de < DxE; de += blockDim.x) {
            size_t d = de / E;
            size_t e = de % E;
            S_star[de] += Q_phi[idx * D + d] * dnum[idx * E + e];
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            z_star[d] += Q_phi[idx * D + d] * dden[idx];
        }
        __syncthreads();

        for (size_t e = threadIdx.x; e < E; e += blockDim.x) {
            float val = 0.0f;
            for (size_t d = 0; d < D; d++) {
                val += S_star[d * E + e] * K_phi[idx * D + d];
            }
            dV[idx * E + e] = val;
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
            float val = z_star[d];
            for (size_t e = 0; e < E; e++) {
                val += S_star[d * E + e] * V[idx * E + e];
            }
            dK_phi[idx * D + d] = val;
        }
        __syncthreads();
    }
}

void causal_linear_attention_bwd_dKdV(
    const CudaTensor<float>& Q_phi, const CudaTensor<float>& K_phi,
    const CudaTensor<float>& V,
    const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
    CudaTensor<float>& dK_phi, CudaTensor<float>& dV,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t block_threads = (size_t)min(max(D, E), (size_t)LBLOCK_SIZE);
    dim3 grid((unsigned int)B);
    dim3 block((unsigned int)block_threads);

    CudaTensor<float> S_star_buf({B * D, E});
    CudaTensor<float> z_star_buf({B, D});
    S_star_buf.memset_zero();
    z_star_buf.memset_zero();

    causal_linear_attn_bwd_dKdV_kernel<<<grid, block>>>(
        Q_phi.device_ptr(), K_phi.device_ptr(), V.device_ptr(),
        dnum.device_ptr(), dden.device_ptr(),
        dK_phi.device_ptr(), dV.device_ptr(),
        S_star_buf.device_ptr(), z_star_buf.device_ptr(),
        B, S, D, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace linear_cuda

#endif
