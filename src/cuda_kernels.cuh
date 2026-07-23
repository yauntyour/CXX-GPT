#pragma once
#include "TensorN.hpp"
#include <cuda_runtime.h>

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

namespace gpt_cuda {

void gelu_deriv(const CudaTensor<float>& x, CudaTensor<float>& out);
void add_row_bias(const CudaTensor<float>& A, const CudaTensor<float>& bias, CudaTensor<float>& C);
void embedding_forward(const CudaTensor<float>& weight, const int* indices, CudaTensor<float>& out, size_t N);
void embedding_backward(const CudaTensor<float>& dout, const int* indices, CudaTensor<float>& grad_weight, size_t N);
void softmax_backward(const CudaTensor<float>& s, const CudaTensor<float>& ds, CudaTensor<float>& dz);

void layernorm_forward(const CudaTensor<float>& x, const CudaTensor<float>& gamma,
    const CudaTensor<float>& beta, CudaTensor<float>& out,
    CudaTensor<float>& mean, CudaTensor<float>& rstd, float eps);

void layernorm_backward(const CudaTensor<float>& dout, const CudaTensor<float>& x,
    const CudaTensor<float>& gamma, const CudaTensor<float>& mean,
    const CudaTensor<float>& rstd, CudaTensor<float>& dx,
    CudaTensor<float>& dgamma, CudaTensor<float>& dbeta);

float cross_entropy_loss_forward(const CudaTensor<float>& logits, const int* targets,
    CudaTensor<float>& probs, size_t N, size_t V);
void cross_entropy_loss_backward(const CudaTensor<float>& probs, const int* targets,
    CudaTensor<float>& dlogits, size_t N, size_t V);

void attention_mask(CudaTensor<float>& scores, size_t B, size_t S);

void adamw_step(CudaTensor<float>& data, const CudaTensor<float>& grad,
    CudaTensor<float>& m, CudaTensor<float>& v,
    float lr, float beta1, float beta2, float wd, float eps,
    float corr1, float corr2);

float grad_norm_squared(const CudaTensor<float>& grad);

void fill_row(CudaTensor<float>& out, const CudaTensor<float>& row_vec);

} // namespace gpt_cuda

#endif
