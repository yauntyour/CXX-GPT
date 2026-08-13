#include "TwoStage/two_stage_cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef TENSORN_CUDA_AVAILABLE

namespace two_stage_cuda {

// ------------------------------------------------------------
// shared tuning constants
// ------------------------------------------------------------
constexpr int TS_WARP = 32;      // lanes per warp
constexpr int TS_WARPS = 32;     // warps per block (each warp owns one query row)
constexpr int TS_FB_CPT = 8;     // max columns per lane -> head_dim <= 256
constexpr int TS_NTHREADS = TS_WARP * TS_WARPS;

// ------------------------------------------------------------
// dynamic shared memory selection:
// primary kernels are used while their smem fits the device,
// otherwise generic warp-per-row fallbacks take over.
// ------------------------------------------------------------
static int device_max_smem_optin()
{
    static int max_optin = -1;
    if (max_optin < 0)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
            max_optin = (int)prop.sharedMemPerBlockOptin;
        else
            max_optin = 0;
    }
    return max_optin;
}

template <typename K, typename... Args>
static bool launch_with_smem(K kernel, size_t smem_bytes, dim3 grid, dim3 block, Args... args)
{
    if (smem_bytes > (size_t)device_max_smem_optin())
        return false;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    kernel<<<grid, block, smem_bytes>>>(args...);
    return true;
}

// ============================================================
// Stage 1 forward: fused causal softmax MHA.
// One block per (b, h); K/V cached in smem; each warp owns a
// query row and runs an online (flash-style) softmax with only
// warp-shuffle reductions -- zero __syncthreads in the hot loop.
// ============================================================
__global__ void fused_mha_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t bh = blockIdx.x;
    size_t b = bh / H, h = bh % H;
    size_t warp = threadIdx.x / TS_WARP;
    size_t lane = threadIdx.x % TS_WARP;
    size_t s = (size_t)blockIdx.y * TS_WARPS + warp;

    const size_t stride = H * Dh;
    const size_t base = b * S;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + S * Dh;

    for (size_t i = threadIdx.x; i < S * Dh; i += blockDim.x)
    {
        size_t t = i / Dh, c = i % Dh;
        size_t row = (base + t) * stride + h * Dh;
        Ks[i] = K[row + c];
        Vs[i] = V[row + c];
    }
    __syncthreads();
    if (s >= S)
        return;

    const size_t qrow = (base + s) * stride + h * Dh;
    float qv[TS_FB_CPT];
    float ov[TS_FB_CPT];
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        qv[j] = (c < Dh) ? Q[qrow + c] * scale : 0.0f;
        ov[j] = 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    for (size_t t = 0; t <= s; t++)
    {
        float score = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                score += qv[j] * Ks[t * Dh + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            score += __shfl_xor_sync(0xffffffffu, score, off);

        float m_new = fmaxf(m, score);
        float p = __expf(score - m_new);
        float alpha = __expf(m - m_new);
        l = l * alpha + p;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                ov[j] = ov[j] * alpha + p * Vs[t * Dh + c];
        }
        m = m_new;
    }
    float inv = 1.0f / l;
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        if (c < Dh)
            O[qrow + c] = ov[j] * inv;
    }
}

// ============================================================
// Stage 1 forward, long-sequence variant: K/V streamed through
// smem in tiles of TS_TILE_ROWS rows, so smem stays at
// 2*TS_TILE_ROWS*Dh regardless of S. Online softmax continues
// across tiles. Block = 8 warps, each warp owns one query row.
// ============================================================
constexpr int TS_TILE_ROWS = 64;
constexpr int TS_BROWS = 8;

__global__ void fused_mha_fwd_tiled_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t bh = blockIdx.x;
    size_t b = bh / H, h = bh % H;
    size_t warp = threadIdx.x / TS_WARP;
    size_t lane = threadIdx.x % TS_WARP;
    size_t s = (size_t)blockIdx.y * TS_BROWS + warp;
    bool active = (s < S);
    if (s >= S)
        s = 0; // clamp: keeps every load address in-bounds even if
               // the compiler speculates past the `active` guard

    const size_t stride = H * Dh;
    const size_t base = b * S;
    const size_t qrow = (base + s) * stride + h * Dh;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + TS_TILE_ROWS * Dh;

    float qv[TS_FB_CPT];
    float ov[TS_FB_CPT];
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        qv[j] = (c < Dh) ? Q[qrow + c] * scale : 0.0f;
        ov[j] = 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    const size_t ntile = (S + TS_TILE_ROWS - 1) / TS_TILE_ROWS;
    for (size_t tb = 0; tb < ntile; tb++)
    {
        size_t t0 = tb * TS_TILE_ROWS;
        size_t t1 = (t0 + TS_TILE_ROWS < S) ? t0 + TS_TILE_ROWS : S;
        for (size_t i = threadIdx.x; i < (t1 - t0) * Dh; i += blockDim.x)
        {
            size_t t = t0 + i / Dh, c = i % Dh;
            size_t row = (base + t) * stride + h * Dh;
            Ks[i] = K[row + c];
            Vs[i] = V[row + c];
        }
        __syncthreads();
        if (active)
        {
            size_t colcnt = (t1 <= s) ? TS_TILE_ROWS
                                      : ((s >= t0) ? (s + 1 - t0) : 0);
            for (size_t i = 0; i < colcnt; i++)
            {
                float score = 0.0f;
                for (int j = 0; j < TS_FB_CPT; j++)
                {
                    size_t c = lane * TS_FB_CPT + j;
                    if (c < Dh)
                        score += qv[j] * Ks[i * Dh + c];
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    score += __shfl_xor_sync(0xffffffffu, score, off);

                float m_new = fmaxf(m, score);
                float p = __expf(score - m_new);
                float alpha = __expf(m - m_new);
                l = l * alpha + p;
                for (int j = 0; j < TS_FB_CPT; j++)
                {
                    size_t c = lane * TS_FB_CPT + j;
                    if (c < Dh)
                        ov[j] = ov[j] * alpha + p * Vs[i * Dh + c];
                }
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (active)
    {
        float inv = 1.0f / l;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                O[qrow + c] = ov[j] * inv;
        }
    }
}

// generic fallback: one warp computes one output row (K/V from L2)
__global__ void fused_mha_fwd_warp_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t id = blockIdx.x;
    size_t s = id % S;
    size_t h = (id / S) % H;
    size_t b = id / (S * H);
    size_t lane = threadIdx.x;

    const size_t stride = H * Dh;
    const size_t base = b * S;
    const size_t qrow = (base + s) * stride + h * Dh;

    float qv[TS_FB_CPT];
    float ov[TS_FB_CPT];
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        qv[j] = (c < Dh) ? Q[qrow + c] * scale : 0.0f;
        ov[j] = 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    for (size_t t = 0; t <= s; t++)
    {
        const size_t krow = (base + t) * stride + h * Dh;
        float score = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                score += qv[j] * K[krow + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            score += __shfl_xor_sync(0xffffffffu, score, off);

        float m_new = fmaxf(m, score);
        float p = __expf(score - m_new);
        float alpha = __expf(m - m_new);
        l = l * alpha + p;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                ov[j] = ov[j] * alpha + p * V[krow + c];
        }
        m = m_new;
    }
    float inv = 1.0f / l;
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        if (c < Dh)
            O[qrow + c] = ov[j] * inv;
    }
}

void fused_mha_fwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                   const CudaTensor<float>& V, CudaTensor<float>& O,
                   size_t B, size_t S, size_t H, size_t Dh)
{
    float scale = 1.0f / sqrtf((float)Dh);
    size_t smem_bytes = 2 * S * Dh * sizeof(float);
    if (Dh <= 32 * TS_FB_CPT)
    {
        dim3 grid((unsigned)(B * H), (unsigned)((S + TS_WARPS - 1) / TS_WARPS));
        dim3 block(TS_NTHREADS);
        if (launch_with_smem(fused_mha_fwd_kernel, smem_bytes, grid, block,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             O.device_ptr(), scale, B, S, H, Dh))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
        // Whole K/V does not fit in smem: stream it through a bounded tile.
        size_t smem_tiled = 2 * (size_t)TS_TILE_ROWS * Dh * sizeof(float);
        dim3 grid_t((unsigned)(B * H), (unsigned)((S + TS_BROWS - 1) / TS_BROWS));
        dim3 block_t(TS_WARP * TS_BROWS);
        if (launch_with_smem(fused_mha_fwd_tiled_kernel, smem_tiled, grid_t, block_t,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             O.device_ptr(), scale, B, S, H, Dh))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
    }
    dim3 grid((unsigned)(B * H * S));
    dim3 block(TS_WARP);
    if (Dh > 32 * TS_FB_CPT)
    {
        fprintf(stderr, "two_stage_cuda::fused_mha_fwd: head_dim %zu > %d not supported\n",
                Dh, 32 * TS_FB_CPT);
        std::abort();
    }
    fused_mha_fwd_warp_kernel<<<grid, block>>>(
        Q.device_ptr(), K.device_ptr(), V.device_ptr(),
        O.device_ptr(), scale, B, S, H, Dh);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Stage 1 backward: fused causal softmax MHA.
// Phase A (warp = query row s): recompute scores, store P row and
//   dP = P * (dO . v_t - dO . O_s) in smem, emit dQ row.
// Phase B (warp = key/value row t): accumulate dK, dV from P/dP.
// ============================================================
__global__ void fused_mha_bwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ O, const float* __restrict__ dO,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t bh = blockIdx.x;
    size_t b = bh / H, h = bh % H;
    size_t warp = threadIdx.x / TS_WARP;
    size_t lane = threadIdx.x % TS_WARP;

    const size_t stride = H * Dh;
    const size_t base = b * S;

    extern __shared__ float smem[];
    float* P = smem;
    float* dP = smem + S * S;

    // ---- Phase A ----
    for (size_t s = warp; s < S; s += TS_WARPS)
    {
        const size_t qrow = (base + s) * stride + h * Dh;
        float qv[TS_FB_CPT];
        float dov[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            qv[j] = (c < Dh) ? Q[qrow + c] * scale : 0.0f;
            dov[j] = (c < Dh) ? dO[qrow + c] : 0.0f;
        }

        // raw scores
        for (size_t t = 0; t <= s; t++)
        {
            const size_t krow = (base + t) * stride + h * Dh;
            float score = 0.0f;
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    score += qv[j] * K[krow + c];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_xor_sync(0xffffffffu, score, off);
            P[s * S + t] = score;
        }

        // row softmax statistics (lane-cyclic reduce)
        float m = -1e30f;
        for (size_t t = lane; t <= s; t += TS_WARP)
            m = fmaxf(m, P[s * S + t]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

        float lsum = 0.0f;
        for (size_t t = lane; t <= s; t += TS_WARP)
            lsum += __expf(P[s * S + t] - m);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lsum += __shfl_xor_sync(0xffffffffu, lsum, off);

        float gs = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                gs += dov[j] * O[qrow + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gs += __shfl_xor_sync(0xffffffffu, gs, off);

        float inv_l = 1.0f / lsum;
        for (size_t t = lane; t <= s; t += TS_WARP)
            P[s * S + t] = __expf(P[s * S + t] - m) * inv_l;
        __syncwarp();

        // dP[s][t] = P[s][t] * (dO_s . v_t - gs)
        for (size_t t = 0; t <= s; t++)
        {
            const size_t vrow = (base + t) * stride + h * Dh;
            float ds = 0.0f;
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    ds += dov[j] * V[vrow + c];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ds += __shfl_xor_sync(0xffffffffu, ds, off);
            dP[s * S + t] = P[s * S + t] * (ds - gs);
        }
        __syncwarp();

        // dQ_s = scale * sum_t dP[s][t] * k_t
        float dq_acc[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
            dq_acc[j] = 0.0f;
        for (size_t t = 0; t <= s; t++)
        {
            const size_t krow = (base + t) * stride + h * Dh;
            float dp = dP[s * S + t];
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    dq_acc[j] += dp * K[krow + c];
            }
        }
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                dQ[qrow + c] = scale * dq_acc[j];
        }
    }
    __syncthreads();

    // ---- Phase B: dK_t = scale * sum_s dP[s][t] * q_s ; dV_t = sum_s P[s][t] * dO_s ----
    for (size_t t = warp; t < S; t += TS_WARPS)
    {
        const size_t trow = (base + t) * stride + h * Dh;
        float dk_acc[TS_FB_CPT];
        float dv_acc[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            dk_acc[j] = 0.0f;
            dv_acc[j] = 0.0f;
        }
        for (size_t s = t; s < S; s++)
        {
            const size_t srow = (base + s) * stride + h * Dh;
            float dp = dP[s * S + t];
            float p = P[s * S + t];
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                {
                    dk_acc[j] += dp * Q[srow + c] * scale;
                    dv_acc[j] += p * dO[srow + c];
                }
            }
        }
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
            {
                dK[trow + c] = dk_acc[j];
                dV[trow + c] = dv_acc[j];
            }
        }
    }
}

// Long-sequence variant: same two-phase algorithm but P/dP live in
// global scratch (per (b,h): S*S) instead of smem, so any S works.
// Phase A: warp = query row (32 rows in flight per block, K/V rows
// broadcast across warps -> L1 reuse). Phase B: warp = key row.
__global__ void fused_mha_bwd_scratch_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ O, const float* __restrict__ dO,
    float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
    float* __restrict__ Pscr, float* __restrict__ dPscr,
    float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t bh = blockIdx.x;
    size_t b = bh / H, h = bh % H;
    size_t warp = threadIdx.x / TS_WARP;
    size_t lane = threadIdx.x % TS_WARP;

    const size_t stride = H * Dh;
    const size_t base = b * S;
    float* P = Pscr + bh * S * S;
    float* dP = dPscr + bh * S * S;

    // ---- Phase A ----
    for (size_t s = warp; s < S; s += TS_WARPS)
    {
        const size_t qrow = (base + s) * stride + h * Dh;
        float qv[TS_FB_CPT];
        float dov[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            qv[j] = (c < Dh) ? Q[qrow + c] * scale : 0.0f;
            dov[j] = (c < Dh) ? dO[qrow + c] : 0.0f;
        }

        for (size_t t = 0; t <= s; t++)
        {
            const size_t krow = (base + t) * stride + h * Dh;
            float score = 0.0f;
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    score += qv[j] * K[krow + c];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                score += __shfl_xor_sync(0xffffffffu, score, off);
            P[s * S + t] = score;
        }

        float m = -1e30f;
        for (size_t t = lane; t <= s; t += TS_WARP)
            m = fmaxf(m, P[s * S + t]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

        float lsum = 0.0f;
        for (size_t t = lane; t <= s; t += TS_WARP)
            lsum += __expf(P[s * S + t] - m);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lsum += __shfl_xor_sync(0xffffffffu, lsum, off);

        float gs = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                gs += dov[j] * O[qrow + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gs += __shfl_xor_sync(0xffffffffu, gs, off);

        float inv_l = 1.0f / lsum;
        for (size_t t = lane; t <= s; t += TS_WARP)
            P[s * S + t] = __expf(P[s * S + t] - m) * inv_l;

        for (size_t t = 0; t <= s; t++)
        {
            const size_t vrow = (base + t) * stride + h * Dh;
            float ds = 0.0f;
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    ds += dov[j] * V[vrow + c];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ds += __shfl_xor_sync(0xffffffffu, ds, off);
            dP[s * S + t] = P[s * S + t] * (ds - gs);
        }

        float dq_acc[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
            dq_acc[j] = 0.0f;
        for (size_t t = 0; t <= s; t++)
        {
            const size_t krow = (base + t) * stride + h * Dh;
            float dp = dP[s * S + t];
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    dq_acc[j] += dp * K[krow + c];
            }
        }
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                dQ[qrow + c] = scale * dq_acc[j];
        }
    }
    __syncthreads();

    // ---- Phase B: dK_t = scale * sum_s dP[s][t] * q_s ; dV_t = sum_s P[s][t] * dO_s ----
    for (size_t t = warp; t < S; t += TS_WARPS)
    {
        const size_t trow = (base + t) * stride + h * Dh;
        float dk_acc[TS_FB_CPT];
        float dv_acc[TS_FB_CPT];
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            dk_acc[j] = 0.0f;
            dv_acc[j] = 0.0f;
        }
        for (size_t s = t; s < S; s++)
        {
            const size_t srow = (base + s) * stride + h * Dh;
            float dp = dP[s * S + t];
            float p = P[s * S + t];
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                {
                    dk_acc[j] += dp * Q[srow + c] * scale;
                    dv_acc[j] += p * dO[srow + c];
                }
            }
        }
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
            {
                dK[trow + c] = dk_acc[j];
                dV[trow + c] = dv_acc[j];
            }
        }
    }
}

// fallback backward, part 1: one warp computes one dQ row
__global__ void mha_bwd_fallback_dq_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ O, const float* __restrict__ dO,
    float* __restrict__ dQ, float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t id = blockIdx.x;
    size_t s = id % S;
    size_t h = (id / S) % H;
    size_t b = id / (S * H);
    size_t lane = threadIdx.x;

    const size_t stride = H * Dh;
    const size_t base = b * S;
    const size_t qrow = (base + s) * stride + h * Dh;

    extern __shared__ float scores[];

    float m = -1e30f, lsum = 0.0f;
    for (size_t u = 0; u <= s; u++)
    {
        const size_t urow = (base + u) * stride + h * Dh;
        float sc = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                sc += Q[qrow + c] * scale * K[urow + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sc += __shfl_xor_sync(0xffffffffu, sc, off);
        scores[u] = sc;
        m = fmaxf(m, sc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    // NOTE: every lane already accumulated the FULL row sum above,
    // so no cross-lane reduction must be applied to lsum.
    for (size_t u = 0; u <= s; u++)
        lsum += __expf(scores[u] - m);

    float gs = 0.0f;
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        if (c < Dh)
            gs += dO[qrow + c] * O[qrow + c];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        gs += __shfl_xor_sync(0xffffffffu, gs, off);

    float dq_acc[TS_FB_CPT];
    for (int j = 0; j < TS_FB_CPT; j++)
        dq_acc[j] = 0.0f;
    for (size_t u = 0; u <= s; u++)
    {
        const size_t urow = (base + u) * stride + h * Dh;
        float p = __expf(scores[u] - m) / lsum;
        float ds = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                ds += dO[qrow + c] * V[urow + c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ds += __shfl_xor_sync(0xffffffffu, ds, off);
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
                dq_acc[j] += p * (ds - gs) * K[urow + c];
        }
    }
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        if (c < Dh)
            dQ[qrow + c] = scale * dq_acc[j];
    }
}

// fallback backward, part 2: one warp computes one dK/dV row pair
__global__ void mha_bwd_fallback_dkdv_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ O, const float* __restrict__ dO,
    float* __restrict__ dK, float* __restrict__ dV,
    float scale, size_t B, size_t S, size_t H, size_t Dh)
{
    size_t id = blockIdx.x;
    size_t t = id % S;
    size_t h = (id / S) % H;
    size_t b = id / (S * H);
    size_t lane = threadIdx.x;

    const size_t stride = H * Dh;
    const size_t base = b * S;
    const size_t trow = (base + t) * stride + h * Dh;

    extern __shared__ float scores[];

    float dk_acc[TS_FB_CPT];
    float dv_acc[TS_FB_CPT];
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        dk_acc[j] = 0.0f;
        dv_acc[j] = 0.0f;
    }

    for (size_t s = t; s < S; s++)
    {
        const size_t srow = (base + s) * stride + h * Dh;
        float m = -1e30f, lsum = 0.0f;
        for (size_t u = 0; u <= s; u++)
        {
            const size_t urow = (base + u) * stride + h * Dh;
            float sc = 0.0f;
            for (int j = 0; j < TS_FB_CPT; j++)
            {
                size_t c = lane * TS_FB_CPT + j;
                if (c < Dh)
                    sc += Q[srow + c] * scale * K[urow + c];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sc += __shfl_xor_sync(0xffffffffu, sc, off);
            scores[u] = sc;
            m = fmaxf(m, sc);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        // NOTE: every lane already accumulated the FULL row sum above,
        // so no cross-lane reduction must be applied to lsum.
        for (size_t u = 0; u <= s; u++)
            lsum += __expf(scores[u] - m);
        float p = __expf(scores[t] - m) / lsum;

        float gs = 0.0f, ds = 0.0f;
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
            {
                gs += dO[srow + c] * O[srow + c];
                ds += dO[srow + c] * V[trow + c];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gs += __shfl_xor_sync(0xffffffffu, gs, off);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ds += __shfl_xor_sync(0xffffffffu, ds, off);

        float dscore = p * (ds - gs);
        for (int j = 0; j < TS_FB_CPT; j++)
        {
            size_t c = lane * TS_FB_CPT + j;
            if (c < Dh)
            {
                dk_acc[j] += dscore * Q[srow + c] * scale;
                dv_acc[j] += p * dO[srow + c];
            }
        }
    }
    for (int j = 0; j < TS_FB_CPT; j++)
    {
        size_t c = lane * TS_FB_CPT + j;
        if (c < Dh)
        {
            dK[trow + c] = dk_acc[j];
            dV[trow + c] = dv_acc[j];
        }
    }
}

void fused_mha_bwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                   const CudaTensor<float>& V, const CudaTensor<float>& O,
                   const CudaTensor<float>& dO,
                   CudaTensor<float>& dQ, CudaTensor<float>& dK,
                   CudaTensor<float>& dV,
                   size_t B, size_t S, size_t H, size_t Dh)
{
    float scale = 1.0f / sqrtf((float)Dh);
    size_t smem_bytes = (2 * S * S + S) * sizeof(float);
    if (Dh <= 32 * TS_FB_CPT)
    {
        dim3 grid((unsigned)(B * H));
        dim3 block(TS_NTHREADS);
        if (launch_with_smem(fused_mha_bwd_kernel, smem_bytes, grid, block,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             O.device_ptr(), dO.device_ptr(),
                             dQ.device_ptr(), dK.device_ptr(), dV.device_ptr(),
                             scale, B, S, H, Dh))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
        // P/dP (S*S each) do not fit in smem: spill them to global scratch.
        CudaTensor<float> Pscr({B * H * S * S});
        CudaTensor<float> dPscr({B * H * S * S});
        dim3 grid_s((unsigned)(B * H));
        dim3 block_s(TS_NTHREADS);
        fused_mha_bwd_scratch_kernel<<<grid_s, block_s>>>(
            Q.device_ptr(), K.device_ptr(), V.device_ptr(),
            O.device_ptr(), dO.device_ptr(),
            dQ.device_ptr(), dK.device_ptr(), dV.device_ptr(),
            Pscr.device_ptr(), dPscr.device_ptr(),
            scale, B, S, H, Dh);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return;
    }
    // generic fallback path
    if (Dh > 32 * TS_FB_CPT)
    {
        fprintf(stderr, "two_stage_cuda::fused_mha_bwd: head_dim %zu > %d not supported\n",
                Dh, 32 * TS_FB_CPT);
        std::abort();
    }
    dim3 grid((unsigned)(B * H * S));
    dim3 block(TS_WARP);
    mha_bwd_fallback_dq_kernel<<<grid, block, S * sizeof(float)>>>(
        Q.device_ptr(), K.device_ptr(), V.device_ptr(), O.device_ptr(), dO.device_ptr(),
        dQ.device_ptr(), scale, B, S, H, Dh);
    mha_bwd_fallback_dkdv_kernel<<<grid, block, S * sizeof(float)>>>(
        Q.device_ptr(), K.device_ptr(), V.device_ptr(), O.device_ptr(), dO.device_ptr(),
        dK.device_ptr(), dV.device_ptr(), scale, B, S, H, Dh);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Stage 2 forward: ELU+1 kernel attention, causal prefix scan.
// State layout S_buf[b][e][d] (E x D), one block per batch item.
// phi(Q), phi(K) and the denominator are fused into the scan.
// ============================================================
__global__ void kernel_attn_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float* __restrict__ den, float* __restrict__ z_cache,
    float* __restrict__ S_buf, float* __restrict__ z_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    const size_t NCH = NT / E; // threads per e-column
    const size_t chunk_sz = D / NCH;

    size_t b = blockIdx.x;
    float* state = S_buf + (size_t)b * E * D;
    float* z = z_buf + (size_t)b * D;

    size_t e = threadIdx.x % E;
    size_t chunk = threadIdx.x / E;

    extern __shared__ float smem[];
    float* qsm = smem;                     // phi(Q) row: D
    float* numsm = smem + D;               // partial numerators: NT
    float* densm = smem + D + NT;          // partial denominators: NT

    for (size_t i = threadIdx.x; i < E * D; i += NT)
        state[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += NT)
        z[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = b * S + t;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;
        const float* qr = Q + idx * D;

        // state[e][d] += phi(k[d]) * v[e]
        for (size_t d = chunk * chunk_sz; d < (chunk + 1) * chunk_sz; d++)
        {
            float kv = kr[d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            state[(size_t)e * D + d] += kp * vr[e];
        }
        // z[d] += phi(k[d]), and cache it for the backward pass
        if (threadIdx.x < D)
        {
            float kv = kr[threadIdx.x];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            z[threadIdx.x] += kp;
            z_cache[idx * D + threadIdx.x] = z[threadIdx.x];
        }
        // phi(Q) row in smem (fused, no extra kernel launch)
        if (threadIdx.x < D)
        {
            float qv = qr[threadIdx.x];
            qsm[threadIdx.x] = qv > 0.0f ? qv + 1.0f : __expf(qv);
        }
        __syncthreads();

        // partial numerators / denominators
        float num = 0.0f, dpart = 0.0f;
        for (size_t d = chunk * chunk_sz; d < (chunk + 1) * chunk_sz; d++)
        {
            float qp = qsm[d];
            num += qp * state[(size_t)e * D + d];
            dpart += qp * z[d];
        }
        numsm[threadIdx.x] = num;
        densm[threadIdx.x] = dpart;
        __syncthreads();

        // reduce over the per-e chunks
        for (size_t k = NCH / 2; k > 0; k >>= 1)
        {
            if (chunk < k)
            {
                numsm[chunk * E + e] += numsm[(chunk + k) * E + e];
                densm[chunk * E + e] += densm[(chunk + k) * E + e];
            }
            __syncthreads();
        }
        if (threadIdx.x < E)
        {
            den[idx] = densm[e];
            O[idx * E + e] = numsm[e] / (densm[e] + 1e-8f);
        }
        __syncthreads();
    }
}

// ============================================================
// Stage 2, high-occupancy variant: each batch item is split into
// TS_S2_CHUNK blocks along the E axis (fwd/dV) or the D axis
// (dQ/dK). z/z_cache/den are recomputed redundantly by every
// chunk block of a batch item with identical values, so no
// cross-block synchronization is required.
// ============================================================
constexpr int TS_S2_CHUNK = 4;
constexpr int TS_S2_MAXCS = 32; // max d-elements per thread in the chunked scan

__global__ void kernel_attn_fwd_chunk_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float* __restrict__ den, float* __restrict__ z_cache,
    float* __restrict__ S_buf, float* __restrict__ z_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    size_t b = blockIdx.x / TS_S2_CHUNK;
    size_t ce = blockIdx.x % TS_S2_CHUNK;
    const size_t Ew = E / TS_S2_CHUNK;
    const size_t NCH = NT / Ew;
    const size_t chunk_sz = D / NCH;
    const size_t e0 = ce * Ew;

    const size_t e = e0 + threadIdx.x % Ew;
    const size_t el = threadIdx.x % Ew;
    const size_t chunk = threadIdx.x / Ew;
    const size_t d0 = chunk * chunk_sz;

    // The scan state (E x D per batch item) is only consumed inside this
    // kernel (the backward passes rescan from K/Q), so each thread keeps
    // its own (e, d-chunk) slice in registers -- zero global RMW traffic.
    // phi(K), phi(Q), z and the reduction buffers live in smem; the
    // per-batch z is per-block here, which also removes the cross-block
    // RMW race that a global z would have.
    extern __shared__ float smem[];
    float* phik = smem;
    float* phiq = smem + D;
    float* zsm = smem + 2 * D;
    float* numsm = smem + 3 * D;
    float* densm = smem + 3 * D + 512;

    if (threadIdx.x < D)
        zsm[threadIdx.x] = 0.0f;
    __syncthreads();

    float acc[TS_S2_MAXCS];
    for (size_t i = 0; i < chunk_sz; i++)
        acc[i] = 0.0f;

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = b * S + t;
        const float* kr = K + idx * D;
        const float* qr = Q + idx * D;
        const float* vr = V + idx * E;

        if (threadIdx.x < D)
        {
            float kv = kr[threadIdx.x];
            phik[threadIdx.x] = kv > 0.0f ? kv + 1.0f : __expf(kv);
            float qv = qr[threadIdx.x];
            phiq[threadIdx.x] = qv > 0.0f ? qv + 1.0f : __expf(qv);
            zsm[threadIdx.x] += phik[threadIdx.x];
            z_cache[idx * D + threadIdx.x] = zsm[threadIdx.x];
        }
        __syncthreads();

        float ve = vr[e];
        for (size_t i = 0; i < chunk_sz; i++)
            acc[i] += phik[d0 + i] * ve;

        float num = 0.0f, dpart = 0.0f;
        for (size_t i = 0; i < chunk_sz; i++)
        {
            float qp = phiq[d0 + i];
            num += qp * acc[i];
            dpart += qp * zsm[d0 + i];
        }
        numsm[threadIdx.x] = num;
        densm[threadIdx.x] = dpart;
        __syncthreads();

        for (size_t k = NCH / 2; k > 0; k >>= 1)
        {
            if (chunk < k)
            {
                numsm[chunk * Ew + el] += numsm[(chunk + k) * Ew + el];
                densm[chunk * Ew + el] += densm[(chunk + k) * Ew + el];
            }
            __syncthreads();
        }
        if (threadIdx.x < Ew)
        {
            den[idx] = densm[el];
            O[idx * E + e] = numsm[el] / (densm[el] + 1e-8f);
        }
        __syncthreads();
    }
}

// generic fallback (any D/E): one block per batch item
__global__ void kernel_attn_fwd_flat_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float* __restrict__ den, float* __restrict__ z_cache,
    float* __restrict__ S_buf, float* __restrict__ z_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    float* state = S_buf + (size_t)b * E * D;
    float* z = z_buf + (size_t)b * D;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < E * D; i += blockDim.x)
        state[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x)
        z[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = base + t;
        for (size_t de = threadIdx.x; de < E * D; de += blockDim.x)
        {
            size_t e = de / D, d = de % D;
            float kv = K[idx * D + d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            state[de] += kp * V[idx * E + e];
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x)
        {
            float kv = K[idx * D + d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            z[d] += kp;
            z_cache[idx * D + d] = z[d];
        }
        __syncthreads();
        if (threadIdx.x < E)
        {
            const float* qr = Q + idx * D;
            float num = 0.0f, dpart = 0.0f;
            for (size_t d = 0; d < D; d++)
            {
                float qv = qr[d];
                float qp = qv > 0.0f ? qv + 1.0f : __expf(qv);
                num += qp * state[(size_t)threadIdx.x * D + d];
                dpart += qp * z[d];
            }
            den[idx] = dpart;
            O[idx * E + threadIdx.x] = num / (dpart + 1e-8f);
        }
        __syncthreads();
    }
}

void kernel_attn_fwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                     const CudaTensor<float>& V,
                     CudaTensor<float>& O, CudaTensor<float>& den_cache,
                     CudaTensor<float>& z_cache,
                     CudaTensor<float>& S_buf, CudaTensor<float>& z_buf,
                     size_t B, size_t S, size_t D, size_t E)
{
    const size_t Ew = E / TS_S2_CHUNK;
    const size_t NCH = 512 / Ew;
    const size_t chunk_sz = D / NCH;
    bool chunk_ok = (E >= TS_S2_CHUNK) && (E % TS_S2_CHUNK == 0) &&
                    (Ew > 0) && (512 % Ew == 0) && (NCH <= 512) &&
                    (D % NCH == 0) && (chunk_sz <= TS_S2_MAXCS) && (chunk_sz >= 1);
    if (chunk_ok)
    {
        size_t smem_bytes = (3 * D + 2 * 512) * sizeof(float);
        dim3 grid((unsigned)(B * TS_S2_CHUNK));
        dim3 block(512);
        if (launch_with_smem(kernel_attn_fwd_chunk_kernel, smem_bytes, grid, block,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             O.device_ptr(), den_cache.device_ptr(), z_cache.device_ptr(),
                             S_buf.device_ptr(), z_buf.device_ptr(),
                             B, S, D, E))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
    }
    bool fused_ok = (E <= 512) && (512 % E == 0) && (D % (512 / E) == 0);
    if (fused_ok)
    {
        size_t smem_bytes = (D + 2 * 512) * sizeof(float);
        dim3 grid((unsigned)B);
        dim3 block(512);
        if (launch_with_smem(kernel_attn_fwd_kernel, smem_bytes, grid, block,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             O.device_ptr(), den_cache.device_ptr(), z_cache.device_ptr(),
                             S_buf.device_ptr(), z_buf.device_ptr(),
                             B, S, D, E))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
    }
    dim3 grid((unsigned)B);
    dim3 block(256);
    kernel_attn_fwd_flat_kernel<<<grid, block>>>(
        Q.device_ptr(), K.device_ptr(), V.device_ptr(),
        O.device_ptr(), den_cache.device_ptr(), z_cache.device_ptr(),
        S_buf.device_ptr(), z_buf.device_ptr(),
        B, S, D, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Stage 2 backward, step 1: dnum = dO / den ; dden = -sum(dO*O)/den
// ============================================================
__global__ void kernel_attn_bwd_step1_kernel(
    const float* __restrict__ O, const float* __restrict__ dO,
    const float* __restrict__ den,
    float* __restrict__ dnum, float* __restrict__ dden,
    size_t N, size_t E)
{
    size_t i = blockIdx.x;
    const float* o_row = O + i * E;
    const float* d_row = dO + i * E;
    float* dn_row = dnum + i * E;
    float inv_den = 1.0f / den[i];

    float sum = 0.0f;
    for (size_t e = threadIdx.x; e < E; e += blockDim.x)
    {
        dn_row[e] = d_row[e] * inv_den;
        sum += d_row[e] * o_row[e];
    }
    extern __shared__ float sh[];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dden[i] = -sh[0] * inv_den;
}

static size_t next_pow2(size_t n)
{
    if (n == 0)
        return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

void kernel_attn_bwd_step1(const CudaTensor<float>& O, const CudaTensor<float>& dO,
                           const CudaTensor<float>& den_cache,
                           CudaTensor<float>& dnum, CudaTensor<float>& dden,
                           size_t N, size_t E)
{
    // the block reduction below requires a power-of-2 block size
    size_t bt = next_pow2(std::min(E, (size_t)256));
    dim3 grid((unsigned)N);
    dim3 block((unsigned)bt);
    kernel_attn_bwd_step1_kernel<<<grid, block, bt * sizeof(float)>>>(
        O.device_ptr(), dO.device_ptr(), den_cache.device_ptr(),
        dnum.device_ptr(), dden.device_ptr(), N, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Stage 2 backward, dQ: re-scan state, dQ = (z*dden + S.dnum) * phi'(Q)
// ============================================================
__global__ void kernel_attn_bwd_dQ_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    const float* __restrict__ z_cache, const float* __restrict__ Q,
    float* __restrict__ dQ, float* __restrict__ S_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    const size_t NCH = NT / E;
    const size_t chunk_sz = D / NCH;

    size_t b = blockIdx.x;
    float* state = S_buf + (size_t)b * E * D;
    size_t e = threadIdx.x % E;
    size_t chunk = threadIdx.x / E;

    for (size_t i = threadIdx.x; i < E * D; i += NT)
        state[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = b * S + t;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;
        for (size_t d = chunk * chunk_sz; d < (chunk + 1) * chunk_sz; d++)
        {
            float kv = kr[d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            state[(size_t)e * D + d] += kp * vr[e];
        }
        __syncthreads();

        if (threadIdx.x < D)
        {
            const float* zc = z_cache + idx * D;
            const float* dn = dnum + idx * E;
            float acc = zc[threadIdx.x] * dden[idx];
            const float* sr = state + threadIdx.x;
            for (size_t e2 = 0; e2 < E; e2++)
                acc += sr[e2 * D] * dn[e2];
            float qv = Q[idx * D + threadIdx.x];
            float der = qv > 0.0f ? 1.0f : __expf(qv);
            dQ[idx * D + threadIdx.x] = acc * der;
        }
        __syncthreads();
    }
}

// dQ, chunked along D: each block owns Dw = D/4 columns. The scan
// state lives in per-thread registers (thread = (d, e-subset) owns
// its slice), and the per-d sum over E is finished with smem
// atomics, so there is no global state traffic at all.
__global__ void kernel_attn_bwd_dQ_chunk_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    const float* __restrict__ z_cache, const float* __restrict__ Q,
    float* __restrict__ dQ, float* __restrict__ S_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    size_t b = blockIdx.x / TS_S2_CHUNK;
    size_t cd = blockIdx.x % TS_S2_CHUNK;
    const size_t Dw = D / TS_S2_CHUNK;
    const size_t Es = E / (NT / Dw); // e-elements per thread
    const size_t d0 = cd * Dw;

    const size_t d = d0 + threadIdx.x / (NT / Dw);
    const size_t e0 = (threadIdx.x % (NT / Dw)) * Es;

    extern __shared__ float smem[];
    float* phik = smem;      // own d-chunk
    float* par = smem + Dw;  // per-d partial sums

    float acc[TS_S2_MAXCS];
    for (size_t i = 0; i < Es; i++)
        acc[i] = 0.0f;

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = b * S + t;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;

        if (threadIdx.x < Dw)
        {
            float kv = kr[d0 + threadIdx.x];
            phik[threadIdx.x] = kv > 0.0f ? kv + 1.0f : __expf(kv);
            par[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        float kp = phik[d - d0];
        for (size_t i = 0; i < Es; i++)
            acc[i] += kp * vr[e0 + i];

        float partial = 0.0f;
        for (size_t i = 0; i < Es; i++)
            partial += acc[i] * dnum[idx * E + e0 + i];
        atomicAdd(&par[d - d0], partial);
        __syncthreads();

        if (threadIdx.x < Dw)
        {
            size_t dd = d0 + threadIdx.x;
            float tot = z_cache[idx * D + dd] * dden[idx] + par[threadIdx.x];
            float qv = Q[idx * D + dd];
            float der = qv > 0.0f ? 1.0f : __expf(qv);
            dQ[idx * D + dd] = tot * der;
        }
        __syncthreads();
    }
}

__global__ void kernel_attn_bwd_dQ_flat_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    const float* __restrict__ z_cache, const float* __restrict__ Q,
    float* __restrict__ dQ, float* __restrict__ S_buf,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    float* state = S_buf + (size_t)b * E * D;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < E * D; i += blockDim.x)
        state[i] = 0.0f;
    __syncthreads();

    for (size_t t = 0; t < S; t++)
    {
        size_t idx = base + t;
        for (size_t de = threadIdx.x; de < E * D; de += blockDim.x)
        {
            size_t e = de / D, d = de % D;
            float kv = K[idx * D + d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            state[de] += kp * V[idx * E + e];
        }
        __syncthreads();
        for (size_t d = threadIdx.x; d < D; d += blockDim.x)
        {
            const float* zc = z_cache + idx * D;
            const float* dn = dnum + idx * E;
            float acc = zc[d] * dden[idx];
            for (size_t e2 = 0; e2 < E; e2++)
                acc += state[e2 * D + d] * dn[e2];
            float qv = Q[idx * D + d];
            float der = qv > 0.0f ? 1.0f : __expf(qv);
            dQ[idx * D + d] = acc * der;
        }
        __syncthreads();
    }
}

void kernel_attn_bwd_dQ(const CudaTensor<float>& K, const CudaTensor<float>& V,
                        const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
                        const CudaTensor<float>& z_cache, const CudaTensor<float>& Q,
                        CudaTensor<float>& dQ, CudaTensor<float>& S_buf,
                        size_t B, size_t S, size_t D, size_t E)
{
    const size_t Dw = D / TS_S2_CHUNK;
    const size_t NPE = 512 / Dw;    // threads per d-column
    const size_t Es = E / NPE;      // e-elements per thread
    bool chunk_ok = (D >= TS_S2_CHUNK) && (D % TS_S2_CHUNK == 0) &&
                    (Dw > 0) && (512 % Dw == 0) && (NPE >= 1) &&
                    (E % NPE == 0) && (Es >= 1) && (Es <= TS_S2_MAXCS);
    if (chunk_ok)
    {
        size_t smem_bytes = (Dw + Dw) * sizeof(float);
        dim3 grid((unsigned)(B * TS_S2_CHUNK));
        dim3 block(512);
        kernel_attn_bwd_dQ_chunk_kernel<<<grid, block, smem_bytes>>>(
            K.device_ptr(), V.device_ptr(), dnum.device_ptr(), dden.device_ptr(),
            z_cache.device_ptr(), Q.device_ptr(),
            dQ.device_ptr(), S_buf.device_ptr(),
            B, S, D, E);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return;
    }
    dim3 grid((unsigned)B);
    if (E <= 512 && 512 % E == 0 && D % (512 / E) == 0)
    {
        dim3 block(512);
        kernel_attn_bwd_dQ_kernel<<<grid, block>>>(
            K.device_ptr(), V.device_ptr(), dnum.device_ptr(), dden.device_ptr(),
            z_cache.device_ptr(), Q.device_ptr(),
            dQ.device_ptr(), S_buf.device_ptr(),
            B, S, D, E);
    }
    else
    {
        dim3 block(256);
        kernel_attn_bwd_dQ_flat_kernel<<<grid, block>>>(
            K.device_ptr(), V.device_ptr(), dnum.device_ptr(), dden.device_ptr(),
            z_cache.device_ptr(), Q.device_ptr(),
            dQ.device_ptr(), S_buf.device_ptr(),
            B, S, D, E);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Stage 2 backward, dK/dV: reverse scan
//   S*[e][d] = sum_{s>=t} phi(Q_s)[d] * dnum_s[e]
//   z*[d]    = sum_{s>=t} phi(Q_s)[d] * dden_s
//   dV_t[e]  = sum_d S*[e][d] * phi(K_t)[d]
//   dK_t[d]  = (z*[d] + sum_e S*[e][d] * V_t[e]) * phi'(K_t)[d]
// dV needs the full D range per e-row, dK the full E range per
// d-column, so the scan is split into two chunked kernels (one
// chunked along E, one along D) for higher occupancy.
// ============================================================
// dK, chunked along D. Thread = (d, e-subset) owns its state slice in
// registers; the per-d sum over E is finished with smem atomics.
__global__ void kernel_attn_bwd_dK_chunk_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    float* __restrict__ dK,
    float* __restrict__ S_star, float* __restrict__ z_star,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    size_t b = blockIdx.x / TS_S2_CHUNK;
    size_t cd = blockIdx.x % TS_S2_CHUNK;
    const size_t Dw = D / TS_S2_CHUNK;
    const size_t NPE = NT / Dw;
    const size_t Es = E / NPE;
    const size_t d0 = cd * Dw;

    const size_t d = d0 + threadIdx.x % Dw;
    const size_t e0 = (threadIdx.x / Dw) * Es;

    extern __shared__ float smem[];
    float* phiq = smem;       // own d-chunk
    float* par = smem + Dw;   // per-d partial sums

    float acc[TS_S2_MAXCS];
    for (size_t i = 0; i < Es; i++)
        acc[i] = 0.0f;
    float zs = 0.0f;

    for (int it = (int)S - 1; it >= 0; it--)
    {
        size_t t = (size_t)it;
        size_t idx = b * S + t;
        const float* qr = Q + idx * D;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;

        if (threadIdx.x < Dw)
        {
            float qv = qr[d0 + threadIdx.x];
            phiq[threadIdx.x] = qv > 0.0f ? qv + 1.0f : __expf(qv);
            par[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        float qp = phiq[d - d0];
        for (size_t i = 0; i < Es; i++)
            acc[i] += qp * dnum[idx * E + e0 + i];
        if (threadIdx.x < Dw)
            zs += qp * dden[idx];

        float partial = 0.0f;
        for (size_t i = 0; i < Es; i++)
            partial += acc[i] * vr[e0 + i];
        atomicAdd(&par[d - d0], partial);
        __syncthreads();

        if (threadIdx.x < Dw)
        {
            size_t dd = d0 + threadIdx.x;
            float kv = kr[dd];
            float der = kv > 0.0f ? 1.0f : __expf(kv);
            dK[idx * D + dd] = (zs + par[threadIdx.x]) * der;
        }
        __syncthreads();
    }
}

// dV, chunked along E. Thread = (e, d-subset) keeps its state slice in
// registers; the per-e sum over D is finished with smem atomics.
__global__ void kernel_attn_bwd_dV_chunk_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    float* __restrict__ dV,
    float* __restrict__ S_star,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    size_t b = blockIdx.x / TS_S2_CHUNK;
    size_t ce = blockIdx.x % TS_S2_CHUNK;
    const size_t Ew = E / TS_S2_CHUNK;
    const size_t NPD = NT / Ew;
    const size_t Ds = D / NPD;
    const size_t e0 = ce * Ew;

    const size_t e = e0 + threadIdx.x % Ew;
    const size_t d0 = (threadIdx.x / Ew) * Ds;

    extern __shared__ float smem[];
    float* phiq = smem;           // D
    float* phik = smem + D;       // D
    float* par = smem + 2 * D;    // Ew

    float acc[TS_S2_MAXCS];
    for (size_t i = 0; i < Ds; i++)
        acc[i] = 0.0f;

    for (int it = (int)S - 1; it >= 0; it--)
    {
        size_t t = (size_t)it;
        size_t idx = b * S + t;
        const float* qr = Q + idx * D;
        const float* kr = K + idx * D;

        if (threadIdx.x < D)
        {
            float qv = qr[threadIdx.x];
            phiq[threadIdx.x] = qv > 0.0f ? qv + 1.0f : __expf(qv);
            float kv = kr[threadIdx.x];
            phik[threadIdx.x] = kv > 0.0f ? kv + 1.0f : __expf(kv);
        }
        if (threadIdx.x < Ew)
            par[threadIdx.x] = 0.0f;
        __syncthreads();

        float dnum_e = dnum[idx * E + e];
        for (size_t i = 0; i < Ds; i++)
            acc[i] += phiq[d0 + i] * dnum_e;

        float partial = 0.0f;
        for (size_t i = 0; i < Ds; i++)
            partial += acc[i] * phik[d0 + i];
        atomicAdd(&par[e - e0], partial);
        __syncthreads();

        if (threadIdx.x < Ew)
            dV[idx * E + e0 + threadIdx.x] = par[threadIdx.x];
        __syncthreads();
    }
}

__global__ void kernel_attn_bwd_dKdV_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    float* __restrict__ dK, float* __restrict__ dV,
    float* __restrict__ S_star, float* __restrict__ z_star,
    size_t B, size_t S, size_t D, size_t E)
{
    const size_t NT = 512;
    const size_t NCH = NT / E;
    const size_t chunk_sz = D / NCH;

    size_t b = blockIdx.x;
    float* S_st = S_star + (size_t)b * E * D;
    float* zs = z_star + (size_t)b * D;
    size_t e = threadIdx.x % E;
    size_t chunk = threadIdx.x / E;

    extern __shared__ float smem[];
    float* psm = smem; // dV partials: NT

    for (size_t i = threadIdx.x; i < E * D; i += NT)
        S_st[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += NT)
        zs[i] = 0.0f;
    __syncthreads();

    for (int it = (int)S - 1; it >= 0; it--)
    {
        size_t t = (size_t)it;
        size_t idx = b * S + t;
        const float* qr = Q + idx * D;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;

        for (size_t d = chunk * chunk_sz; d < (chunk + 1) * chunk_sz; d++)
        {
            float qv = qr[d];
            float qp = qv > 0.0f ? qv + 1.0f : __expf(qv);
            S_st[(size_t)e * D + d] += qp * dnum[idx * E + e];
        }
        if (threadIdx.x < D)
        {
            float qv = qr[threadIdx.x];
            float qp = qv > 0.0f ? qv + 1.0f : __expf(qv);
            zs[threadIdx.x] += qp * dden[idx];
        }
        __syncthreads();

        // dV_t[e] = sum_d S*[e][d] * phi(k[d])
        float dpart = 0.0f;
        for (size_t d = chunk * chunk_sz; d < (chunk + 1) * chunk_sz; d++)
        {
            float kv = kr[d];
            float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
            dpart += S_st[(size_t)e * D + d] * kp;
        }
        psm[threadIdx.x] = dpart;
        __syncthreads();
        for (size_t k = NCH / 2; k > 0; k >>= 1)
        {
            if (chunk < k)
                psm[chunk * E + e] += psm[(chunk + k) * E + e];
            __syncthreads();
        }
        if (threadIdx.x < E)
            dV[idx * E + e] = psm[e];

        // dK_t[d] = (z*[d] + sum_e S*[e][d] * v[e]) * phi'(k[d])
        if (threadIdx.x < D)
        {
            float acc = zs[threadIdx.x];
            const float* sr = S_st + threadIdx.x;
            for (size_t e2 = 0; e2 < E; e2++)
                acc += sr[e2 * D] * vr[e2];
            float kv = kr[threadIdx.x];
            float der = kv > 0.0f ? 1.0f : __expf(kv);
            dK[idx * D + threadIdx.x] = acc * der;
        }
        __syncthreads();
    }
}

__global__ void kernel_attn_bwd_dKdV_flat_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ dnum, const float* __restrict__ dden,
    float* __restrict__ dK, float* __restrict__ dV,
    float* __restrict__ S_star, float* __restrict__ z_star,
    size_t B, size_t S, size_t D, size_t E)
{
    size_t b = blockIdx.x;
    float* S_st = S_star + (size_t)b * E * D;
    float* zs = z_star + (size_t)b * D;
    size_t base = b * S;

    for (size_t i = threadIdx.x; i < E * D; i += blockDim.x)
        S_st[i] = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x)
        zs[i] = 0.0f;
    __syncthreads();

    for (int it = (int)S - 1; it >= 0; it--)
    {
        size_t t = (size_t)it;
        size_t idx = base + t;
        const float* qr = Q + idx * D;
        const float* kr = K + idx * D;
        const float* vr = V + idx * E;

        for (size_t de = threadIdx.x; de < E * D; de += blockDim.x)
        {
            size_t e = de / D, d = de % D;
            float qv = qr[d];
            float qp = qv > 0.0f ? qv + 1.0f : __expf(qv);
            S_st[de] += qp * dnum[idx * E + e];
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x)
        {
            float qv = qr[d];
            float qp = qv > 0.0f ? qv + 1.0f : __expf(qv);
            zs[d] += qp * dden[idx];
        }
        __syncthreads();

        for (size_t e2 = threadIdx.x; e2 < E; e2 += blockDim.x)
        {
            float acc = 0.0f;
            for (size_t d = 0; d < D; d++)
            {
                float kv = kr[d];
                float kp = kv > 0.0f ? kv + 1.0f : __expf(kv);
                acc += S_st[e2 * D + d] * kp;
            }
            dV[idx * E + e2] = acc;
        }
        for (size_t d = threadIdx.x; d < D; d += blockDim.x)
        {
            float acc = zs[d];
            for (size_t e2 = 0; e2 < E; e2++)
                acc += S_st[e2 * D + d] * vr[e2];
            float kv = kr[d];
            float der = kv > 0.0f ? 1.0f : __expf(kv);
            dK[idx * D + d] = acc * der;
        }
        __syncthreads();
    }
}

void kernel_attn_bwd_dKdV(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                          const CudaTensor<float>& V,
                          const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
                          CudaTensor<float>& dK, CudaTensor<float>& dV,
                          size_t B, size_t S, size_t D, size_t E)
{
    CudaTensor<float> S_star_buf({B * E, D});
    CudaTensor<float> z_star_buf({B, D});
    S_star_buf.memset_zero();
    z_star_buf.memset_zero();

    const size_t Ew = E / TS_S2_CHUNK;
    const size_t Dw = D / TS_S2_CHUNK;
    const size_t NPE_d = 512 / Dw; // threads per d-column (dK)
    const size_t NPD_e = 512 / Ew; // threads per e-row (dV)
    const size_t Es_d = E / NPE_d;
    const size_t Ds_e = D / NPD_e;
    bool ok_dK = (D >= TS_S2_CHUNK) && (D % TS_S2_CHUNK == 0) &&
                 (Dw > 0) && (512 % Dw == 0) && (NPE_d >= 1) &&
                 (E % NPE_d == 0) && (Es_d >= 1) && (Es_d <= TS_S2_MAXCS);
    bool ok_dV = (E >= TS_S2_CHUNK) && (E % TS_S2_CHUNK == 0) &&
                 (Ew > 0) && (512 % Ew == 0) && (NPD_e >= 1) &&
                 (D % NPD_e == 0) && (Ds_e >= 1) && (Ds_e <= TS_S2_MAXCS);
    if (ok_dK && ok_dV)
    {
        dim3 grid((unsigned)(B * TS_S2_CHUNK));
        dim3 block(512);
        kernel_attn_bwd_dK_chunk_kernel<<<grid, block, 2 * Dw * sizeof(float)>>>(
            Q.device_ptr(), K.device_ptr(), V.device_ptr(),
            dnum.device_ptr(), dden.device_ptr(),
            dK.device_ptr(),
            S_star_buf.device_ptr(), z_star_buf.device_ptr(),
            B, S, D, E);
        kernel_attn_bwd_dV_chunk_kernel<<<grid, block, (2 * D + Ew) * sizeof(float)>>>(
            Q.device_ptr(), K.device_ptr(), V.device_ptr(),
            dnum.device_ptr(), dden.device_ptr(),
            dV.device_ptr(),
            S_star_buf.device_ptr(),
            B, S, D, E);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return;
    }
    dim3 grid((unsigned)B);
    if (E <= 512 && 512 % E == 0 && D % (512 / E) == 0)
    {
        size_t smem_bytes = 512 * sizeof(float);
        dim3 block(512);
        if (launch_with_smem(kernel_attn_bwd_dKdV_kernel, smem_bytes, grid, block,
                             Q.device_ptr(), K.device_ptr(), V.device_ptr(),
                             dnum.device_ptr(), dden.device_ptr(),
                             dK.device_ptr(), dV.device_ptr(),
                             S_star_buf.device_ptr(), z_star_buf.device_ptr(),
                             B, S, D, E))
        {
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
    }
    dim3 block(256);
    kernel_attn_bwd_dKdV_flat_kernel<<<grid, block>>>(
        Q.device_ptr(), K.device_ptr(), V.device_ptr(),
        dnum.device_ptr(), dden.device_ptr(),
        dK.device_ptr(), dV.device_ptr(),
        S_star_buf.device_ptr(), z_star_buf.device_ptr(),
        B, S, D, E);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// RoPE: one thread per (row, pair-of-dims). Pairs are grouped in
// consecutive heads of width head_dim:
//   x'_{2i}   = x_{2i} * cos(p*w_i) - x_{2i+1} * sin(p*w_i)
//   x'_{2i+1} = x_{2i} * sin(p*w_i) + x_{2i+1} * cos(p*w_i)
//   w_i = base^(-2i / head_dim), base = 10000, p = row % S
// sign = +1 for the forward rotation, -1 for the (orthogonal)
// inverse used in the backward pass.
// ============================================================
__global__ void rope_kernel(
    float* __restrict__ Q, float* __restrict__ K,
    size_t B, size_t S, size_t D, size_t head_dim, float sign)
{
    const size_t npairs = D / 2;
    const size_t pairs_per_head = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * S * npairs;
    if (idx >= total)
        return;

    size_t row = idx / npairs;
    size_t pair = idx % npairs;
    float p = (float)(row % S);

    size_t pair_in_head = pair % pairs_per_head;
    size_t off = row * D + (pair / pairs_per_head) * head_dim + 2 * pair_in_head;

    float ang = sign * p * __expf(-2.0f * (float)pair_in_head / (float)head_dim *
                                  9.210340371976184f); // ln(10000)
    float c = __cosf(ang);
    float s = __sinf(ang);

    if (Q)
    {
        float x0 = Q[off];
        float x1 = Q[off + 1];
        Q[off] = x0 * c - x1 * s;
        Q[off + 1] = x0 * s + x1 * c;
    }
    if (K)
    {
        float x0 = K[off];
        float x1 = K[off + 1];
        K[off] = x0 * c - x1 * s;
        K[off + 1] = x0 * s + x1 * c;
    }
}

static void rope_launch(CudaTensor<float>& Q, CudaTensor<float>& K,
                        size_t B, size_t S, size_t D, size_t head_dim, float sign)
{
    size_t total = B * S * (D / 2);
    dim3 grid((unsigned)((total + 255) / 256));
    dim3 block(256);
    rope_kernel<<<grid, block>>>(Q.device_ptr(), K.device_ptr(),
                                 B, S, D, head_dim, sign);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void rope_fwd(CudaTensor<float>& Q, CudaTensor<float>& K,
              size_t B, size_t S, size_t D, size_t head_dim)
{
    rope_launch(Q, K, B, S, D, head_dim, 1.0f);
}

void rope_bwd(CudaTensor<float>& dQ, CudaTensor<float>& dK,
              size_t B, size_t S, size_t D, size_t head_dim)
{
    rope_launch(dQ, dK, B, S, D, head_dim, -1.0f);
}

// ============================================================
// Exact GELU matching torch.nn.GELU() (erf form):
//   gelu(x)      = 0.5 * x * (1 + erf(x / sqrt(2)))
//   gelu'(x)     = 0.5 * (1 + erf(x / sqrt(2)))
//                  + x * exp(-x^2 / 2) / sqrt(2*pi)
// ============================================================
__global__ void gelu_exact_kernel(const float* __restrict__ A, float* __restrict__ C,
                                  size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = A[idx];
    C[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

void gelu_exact(const CudaTensor<float>& A, CudaTensor<float>& C)
{
    size_t n = A.size();
    dim3 grid((n + 255) / 256);
    dim3 block(256);
    gelu_exact_kernel<<<grid, block>>>(A.device_ptr(), C.device_ptr(), n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void gelu_exact_deriv_kernel(const float* __restrict__ A, float* __restrict__ C,
                                        size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = A[idx];
    float a = x * 0.7071067811865476f;
    C[idx] = 0.5f * (1.0f + erff(a)) +
             x * __expf(-0.5f * x * x) * 0.3989422804014327f;
}

void gelu_exact_deriv(const CudaTensor<float>& A, CudaTensor<float>& C)
{
    size_t n = A.size();
    dim3 grid((n + 255) / 256);
    dim3 block(256);
    gelu_exact_deriv_kernel<<<grid, block>>>(A.device_ptr(), C.device_ptr(), n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace two_stage_cuda

#endif




