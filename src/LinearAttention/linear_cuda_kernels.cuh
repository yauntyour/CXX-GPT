#pragma once
#include "TensorN.hpp"
#include <cuda_runtime.h>

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

namespace linear_cuda {

void causal_linear_attention_fwd(
    const CudaTensor<float>& Q_phi, const CudaTensor<float>& K_phi,
    const CudaTensor<float>& V,
    CudaTensor<float>& O, CudaTensor<float>& den_cache,
    CudaTensor<float>& z_cache,
    CudaTensor<float>& S_buf, CudaTensor<float>& z_buf,
    size_t B, size_t S, size_t D, size_t E);

void causal_linear_attention_bwd_step1(
    const CudaTensor<float>& O, const CudaTensor<float>& dO,
    const CudaTensor<float>& den_cache,
    CudaTensor<float>& dnum, CudaTensor<float>& dden,
    size_t N, size_t E);

void causal_linear_attention_bwd_dQ(
    const CudaTensor<float>& K_phi, const CudaTensor<float>& V,
    const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
    const CudaTensor<float>& z_cache,
    CudaTensor<float>& dQ_phi, CudaTensor<float>& S_buf,
    size_t B, size_t S, size_t D, size_t E);

void causal_linear_attention_bwd_dKdV(
    const CudaTensor<float>& Q_phi, const CudaTensor<float>& K_phi,
    const CudaTensor<float>& V,
    const CudaTensor<float>& dnum, const CudaTensor<float>& dden,
    CudaTensor<float>& dK_phi, CudaTensor<float>& dV,
    size_t B, size_t S, size_t D, size_t E);

} // namespace linear_cuda

#endif
