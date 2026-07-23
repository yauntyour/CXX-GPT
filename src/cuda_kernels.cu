#include "cuda_kernels.cuh"

#ifdef TENSORN_CUDA_AVAILABLE

constexpr int BLOCK_SIZE = 256;

namespace gpt_cuda {

// ============================================================
// GELU derivative: dx = gelu'(x)
// ============================================================
__global__ void gelu_deriv_kernel(const float* x, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    const float c = 0.7978845608028654f;
    float x2 = v * v;
    float x3 = x2 * v;
    float inner = c * (v + 0.044715f * x3);
    float t = tanhf(inner);
    float d = 1.0f - t * t;
    out[idx] = 0.5f * (1.0f + t) + 0.5f * v * c * (1.0f + 3.0f * 0.044715f * x2) * d;
}

void gelu_deriv(const CudaTensor<float>& x, CudaTensor<float>& out) {
    size_t n = x.size();
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    gelu_deriv_kernel<<<grid, block>>>(x.device_ptr(), out.device_ptr(), n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Row-wise bias add: C[i][j] = A[i][j] + bias[j]
// A shape: [N, D], bias shape: [D], C shape: [N, D]
// ============================================================
__global__ void add_row_bias_kernel(const float* A, const float* bias, float* C, size_t N, size_t D) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * D;
    if (idx >= total) return;
    size_t col = idx % D;
    C[idx] = A[idx] + bias[col];
}

void add_row_bias(const CudaTensor<float>& A, const CudaTensor<float>& bias, CudaTensor<float>& C) {
    auto sh = A.shape();
    size_t N = sh[0], D = sh[1];
    size_t total = N * D;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    add_row_bias_kernel<<<grid, block>>>(A.device_ptr(), bias.device_ptr(), C.device_ptr(), N, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Fill row: out[i][j] = row[j] for all i
// out shape: [N, D], row shape: [D]
// ============================================================
__global__ void fill_row_kernel(float* out, const float* row, size_t N, size_t D) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * D;
    if (idx >= total) return;
    size_t col = idx % D;
    out[idx] = row[col];
}

void fill_row(CudaTensor<float>& out, const CudaTensor<float>& row) {
    auto sh = out.shape();
    size_t N = sh[0], D = sh[1];
    size_t total = N * D;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    fill_row_kernel<<<grid, block>>>(out.device_ptr(), row.device_ptr(), N, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Embedding forward: out[i * D + j] = weight[indices[i] * D + j]
// ============================================================
__global__ void embedding_fwd_kernel(const float* weight, const int* indices, float* out, size_t N, size_t D, size_t vocab_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * D;
    if (idx >= total) return;
    size_t row = idx / D;
    size_t col = idx % D;
    int token = indices[row];
    if (token >= 0 && token < (int)vocab_size)
        out[idx] = weight[(size_t)token * D + col];
    else
        out[idx] = 0.0f;
}

void embedding_forward(const CudaTensor<float>& weight, const int* indices, CudaTensor<float>& out, size_t N) {
    auto sh = out.shape();
    size_t D = sh[1];
    size_t total = N * D;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    size_t vocab_size = weight.shape()[0];
    embedding_fwd_kernel<<<grid, block>>>(weight.device_ptr(), indices, out.device_ptr(), N, D, vocab_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Embedding backward: grad_weight[token * D + j] += dout[i * D + j]
// ============================================================
__global__ void embedding_bwd_kernel(const float* dout, const int* indices, float* grad, size_t N, size_t D) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * D;
    if (idx >= total) return;
    size_t row = idx / D;
    size_t col = idx % D;
    int token = indices[row];
    atomicAdd(&grad[(size_t)token * D + col], dout[idx]);
}

void embedding_backward(const CudaTensor<float>& dout, const int* indices, CudaTensor<float>& grad_weight, size_t N) {
    auto sh = grad_weight.shape();
    size_t D = sh[1];
    size_t total = N * D;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    embedding_bwd_kernel<<<grid, block>>>(dout.device_ptr(), indices, grad_weight.device_ptr(), N, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Softmax backward: dz[i][j] = s[i][j] * (ds[i][j] - sum_k(s[i][k] * ds[i][k]))
// s, ds, dz shape: [N, D]
// ============================================================
__global__ void softmax_bwd_kernel(const float* s, const float* ds, float* dz, size_t N, size_t D) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x;
    if (i >= N) return;

    const float* s_row = s + i * D;
    const float* ds_row = ds + i * D;
    float* dz_row = dz + i * D;

    // Compute sum_sds = sum(s[k] * ds[k]) for this row
    float sum_sds = 0.0f;
    for (size_t k = tid; k < D; k += blockDim.x) {
        sum_sds += s_row[k] * ds_row[k];
    }
    shared[tid] = sum_sds;
    __syncthreads();

    // Reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // Compute dz
    for (size_t k = tid; k < D; k += blockDim.x) {
        dz_row[k] = s_row[k] * (ds_row[k] - row_sum);
    }
}

void softmax_backward(const CudaTensor<float>& s, const CudaTensor<float>& ds, CudaTensor<float>& dz) {
    auto sh = s.shape();
    size_t N = sh[0], D = sh[1];
    int block = (D < 1024) ? 256 : 512;
    if (D < 256) block = (int)D;
    dim3 grid((unsigned int)N);
    dim3 threads((unsigned int)block);
    size_t shared_mem = block * sizeof(float);
    softmax_bwd_kernel<<<grid, threads, shared_mem>>>(s.device_ptr(), ds.device_ptr(), dz.device_ptr(), N, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// LayerNorm forward (fused):
// mean[i] = sum(x[i,:]) / D
// var[i] = sum((x[i,:] - mean[i])^2) / D
// rstd[i] = 1/sqrt(var[i] + eps)
// out[i,j] = gamma[j] * (x[i,j] - mean[i]) * rstd[i] + beta[j]
// ============================================================
__global__ void layernorm_fwd_kernel(const float* x, const float* gamma, const float* beta,
                                      float* out, float* mean, float* rstd,
                                      size_t N, size_t D, float eps) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x;
    if (i >= N) return;

    const float* x_row = x + i * D;
    float* out_row = out + i * D;

    // Pass 1: compute mean
    float sum_val = 0.0f;
    for (size_t k = tid; k < D; k += blockDim.x) {
        sum_val += x_row[k];
    }
    shared[tid] = sum_val;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float m = shared[0] / D;
    mean[i] = m;
    __syncthreads();

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (size_t k = tid; k < D; k += blockDim.x) {
        float diff = x_row[k] - m;
        var_sum += diff * diff;
    }
    shared[tid] = var_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float var = shared[0] / D;
    float r = rsqrtf(var + eps);
    rstd[i] = r;
    __syncthreads();

    // Pass 3: compute output
    for (size_t k = tid; k < D; k += blockDim.x) {
        float xhat = (x_row[k] - m) * r;
        out_row[k] = xhat * gamma[k] + beta[k];
    }
}

void layernorm_forward(const CudaTensor<float>& x, const CudaTensor<float>& gamma,
    const CudaTensor<float>& beta, CudaTensor<float>& out,
    CudaTensor<float>& mean, CudaTensor<float>& rstd, float eps)
{
    auto sh = x.shape();
    size_t N = sh[0], D = sh[1];
    int block = (int)min(D, (size_t)512);
    dim3 grid((unsigned int)N);
    dim3 threads((unsigned int)block);
    size_t shared_mem = block * sizeof(float);
    layernorm_fwd_kernel<<<grid, threads, shared_mem>>>(
        x.device_ptr(), gamma.device_ptr(), beta.device_ptr(),
        out.device_ptr(), mean.device_ptr(), rstd.device_ptr(),
        N, D, eps);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// LayerNorm backward (fused):
// ============================================================
__global__ void layernorm_bwd_kernel(const float* dout, const float* x, const float* gamma,
                                      const float* mean, const float* rstd,
                                      float* dx, float* dgamma, float* dbeta,
                                      size_t N, size_t D) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x;
    if (i >= N) return;

    const float* dout_row = dout + i * D;
    const float* x_row = x + i * D;
    float* dx_row = dx + i * D;
    float m = mean[i];
    float r = rstd[i];

    // Compute dxhat_sum and dxhat_xhat_sum
    float ds_sum = 0.0f, ds_xs_sum = 0.0f;
    for (size_t k = tid; k < D; k += blockDim.x) {
        float xhat = (x_row[k] - m) * r;
        float dxhat = dout_row[k] * gamma[k];
        ds_sum += dxhat;
        ds_xs_sum += dxhat * xhat;
    }
    shared[tid] = ds_sum;
    shared[tid + blockDim.x] = ds_xs_sum;
    __syncthreads();

    // Reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
            shared[tid + blockDim.x] += shared[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    float dxhat_sum = shared[0];
    float dxhat_xhat_sum = shared[blockDim.x];
    __syncthreads();

    // Compute dx, dgamma, dbeta
    float invD = 1.0f / D;
    for (size_t k = tid; k < D; k += blockDim.x) {
        float xhat = (x_row[k] - m) * r;
        float dxhat = dout_row[k] * gamma[k];
        dx_row[k] = r * (dxhat - (dxhat_xhat_sum * xhat + dxhat_sum) * invD);
        // Atomic add for dgamma, dbeta
        atomicAdd(&dgamma[k], dout_row[k] * xhat);
        atomicAdd(&dbeta[k], dout_row[k]);
    }
}

void layernorm_backward(const CudaTensor<float>& dout, const CudaTensor<float>& x,
    const CudaTensor<float>& gamma, const CudaTensor<float>& mean,
    const CudaTensor<float>& rstd, CudaTensor<float>& dx,
    CudaTensor<float>& dgamma, CudaTensor<float>& dbeta)
{
    auto sh = dout.shape();
    size_t N = sh[0], D = sh[1];
    int block = (int)min(D, (size_t)512);
    dim3 grid((unsigned int)N);
    dim3 threads((unsigned int)block);
    size_t shared_mem = block * 2 * sizeof(float);
    layernorm_bwd_kernel<<<grid, threads, shared_mem>>>(
        dout.device_ptr(), x.device_ptr(), gamma.device_ptr(),
        mean.device_ptr(), rstd.device_ptr(),
        dx.device_ptr(), dgamma.device_ptr(), dbeta.device_ptr(),
        N, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Cross-entropy loss forward
// probs = softmax(logits), return mean(-log(probs[i, targets[i]]))
// ============================================================
__global__ void cross_entropy_loss_fwd_kernel(const float* logits, const float* probs,
                                               const int* targets, float* loss_out,
                                               size_t N, size_t V) {
    size_t i = blockIdx.x;
    if (i >= N) return;
    if (threadIdx.x != 0) return;

    float p = fmaxf(probs[i * V + targets[i]], 1e-12f);
    atomicAdd(loss_out, -logf(p));
}

float cross_entropy_loss_forward(const CudaTensor<float>& logits, const int* targets,
    CudaTensor<float>& probs, size_t N, size_t V)
{
    TensorN::cuda::softmax(logits, probs, -1);

    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    dim3 grid((unsigned int)N);
    dim3 threads(1);
    cross_entropy_loss_fwd_kernel<<<grid, threads>>>(
        logits.device_ptr(), probs.device_ptr(), targets, d_loss, N, V);
    CHECK_CUDA_ERROR(cudaGetLastError());

    float loss = 0.0f;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return loss / N;
}

// ============================================================
// Cross-entropy backward: dlogits = (probs - one_hot(targets)) / N
// ============================================================
__global__ void cross_entropy_bwd_kernel(float* dlogits, const float* probs,
                                          const int* targets, size_t N, size_t V) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * V;
    if (idx >= total) return;
    size_t i = idx / V;
    size_t j = idx % V;
    dlogits[idx] = (probs[idx] - (j == (size_t)targets[i] ? 1.0f : 0.0f)) / N;
}

void cross_entropy_loss_backward(const CudaTensor<float>& probs, const int* targets,
    CudaTensor<float>& dlogits, size_t N, size_t V)
{
    size_t total = N * V;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    cross_entropy_bwd_kernel<<<grid, block>>>(dlogits.device_ptr(), probs.device_ptr(), targets, N, V);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// Attention causal + block-diagonal mask
// scores shape: [B*S, B*S]
// Set cross-batch and upper-triangular entries to -inf
// ============================================================
__global__ void attention_mask_kernel(float* scores, size_t B, size_t S) {
    size_t dim = B * S;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim * dim;
    if (idx >= total) return;
    size_t i = idx / dim;
    size_t j = idx % dim;
    size_t batch_i = i / S;
    size_t batch_j = j / S;
    size_t pos_i = i % S;
    size_t pos_j = j % S;
    if (batch_i != batch_j || pos_i < pos_j) {
        scores[idx] = -1e10f;
    }
}

void attention_mask(CudaTensor<float>& scores, size_t B, size_t S) {
    size_t total = B * S * B * S;
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    attention_mask_kernel<<<grid, block>>>(scores.device_ptr(), B, S);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================
// AdamW fused step
// m = beta1*m + (1-beta1)*g
// v = beta2*v + (1-beta2)*g*g
// data -= alpha * m / (sqrt(v) + eps_hat) + lr * wd * data
// ============================================================
__global__ void adamw_kernel(float* data, const float* grad, float* m, float* v,
                              float lr, float beta1, float beta2, float wd, float eps,
                              float corr1, float corr2, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float new_m = beta1 * m[idx] + (1.0f - beta1) * g;
    float new_v = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = new_m;
    v[idx] = new_v;
    float alpha = lr * sqrtf(corr2) / corr1;
    float eps_hat = eps * sqrtf(corr2);
    data[idx] -= alpha * new_m / (sqrtf(new_v) + eps_hat) + lr * wd * data[idx];
}

void adamw_step(CudaTensor<float>& data, const CudaTensor<float>& grad,
    CudaTensor<float>& m, CudaTensor<float>& v,
    float lr, float beta1, float beta2, float wd, float eps,
    float corr1, float corr2)
{
    size_t n = data.size();
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    adamw_kernel<<<grid, block>>>(data.device_ptr(), grad.device_ptr(),
        m.device_ptr(), v.device_ptr(),
        lr, beta1, beta2, wd, eps, corr1, corr2, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void grad_norm_sq_kernel(const float* grad, float* out, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + tid;
    float val = 0.0f;
    if (i < n) val += grad[i] * grad[i];
    if (i + blockDim.x < n) val += grad[i + blockDim.x] * grad[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

float grad_norm_squared(const CudaTensor<float>& grad) {
    size_t n = grad.size();
    if (n == 0) return 0.0f;
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    size_t grid = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    grad_norm_sq_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        grad.device_ptr(), d_out, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    float result = 0.0f;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}

} // namespace gpt_cuda

#endif
