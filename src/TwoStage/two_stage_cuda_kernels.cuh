#pragma once
#include "TensorN.hpp"
#include <cuda_runtime.h>

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

namespace two_stage_cuda {

// ============================================================
// Stage 1: fused causal softmax MHA.
// Q/K/V/O are (B*S, H*Dh) row-major (head h of row r lives at
// [(r*H + h)*Dh, (r*H + h)*Dh + Dh)). The scale 1/sqrt(Dh),
// the causal mask and the softmax are all fused into one kernel.
// ============================================================

void fused_mha_fwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                   const CudaTensor<float>& V, CudaTensor<float>& O,
                   size_t B, size_t S, size_t H, size_t Dh);

void fused_mha_bwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                   const CudaTensor<float>& V, const CudaTensor<float>& O,
                   const CudaTensor<float>& dO,
                   CudaTensor<float>& dQ, CudaTensor<float>& dK,
                   CudaTensor<float>& dV,
                   size_t B, size_t S, size_t H, size_t Dh);

// ============================================================
// Stage 2: fused linear "kernel" attention with feature map
// phi(x) = ELU(x) + 1, computed with causal prefix sums:
//   C_t = sum_{s<=t} phi(K_s)          (z_cache)
//   S_t = sum_{s<=t} phi(K_s) x V_s    (scan state, E x D)
//   Y_t = (phi(Q_t) . S_t) / (phi(Q_t) . C_t)
// phi() and its derivative are fused into the scan kernels.
// Q/K are (B*S, D), V/O (B*S, E).
// ============================================================

void kernel_attn_fwd(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                     const CudaTensor<float>& V,
                     CudaTensor<float>& O, CudaTensor<float>& den_cache,
                     CudaTensor<float>& z_cache,
                     CudaTensor<float>& S_buf, CudaTensor<float>& z_buf,
                     size_t B, size_t S, size_t D, size_t E);

void kernel_attn_bwd_step1(const CudaTensor<float>& O, const CudaTensor<float>& dO,
                           const CudaTensor<float>& den_cache,
                           CudaTensor<float>& dnum, CudaTensor<float>& dden,
                           size_t N, size_t E);

void kernel_attn_bwd_dQ(const CudaTensor<float>& K, const CudaTensor<float>& V,
                        const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
                        const CudaTensor<float>& z_cache, const CudaTensor<float>& Q,
                        CudaTensor<float>& dQ, CudaTensor<float>& S_buf,
                        size_t B, size_t S, size_t D, size_t E);

void kernel_attn_bwd_dKdV(const CudaTensor<float>& Q, const CudaTensor<float>& K,
                          const CudaTensor<float>& V,
                          const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
                          CudaTensor<float>& dK, CudaTensor<float>& dV,
                          size_t B, size_t S, size_t D, size_t E);

// ============================================================
// RoPE (rotary position embedding), applied in-place to Q and K.
// Rows are (B*S, D) row-major; the position of row r is r % S.
// The feature dim D is viewed as consecutive heads of width
// head_dim (head_dim == D applies RoPE over the whole vector).
// Backward applies the inverse rotation (negated angle), since
// RoPE is an orthogonal map: dX = R(-p)^T dX_rot.
// ============================================================

void rope_fwd(CudaTensor<float>& Q, CudaTensor<float>& K,
              size_t B, size_t S, size_t D, size_t head_dim);

void rope_bwd(CudaTensor<float>& dQ, CudaTensor<float>& dK,
              size_t B, size_t S, size_t D, size_t head_dim);

// ============================================================
// Exact GELU matching torch.nn.GELU() (erf form) and its
// derivative. Used by the q/k/v expander FFNs and the final FFN
// so the C++ port matches the PyTorch reference numerically.
// ============================================================

void gelu_exact(const CudaTensor<float>& A, CudaTensor<float>& C);
void gelu_exact_deriv(const CudaTensor<float>& A, CudaTensor<float>& C);

} // namespace two_stage_cuda

#endif
