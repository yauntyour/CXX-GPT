#pragma once

#include "TensorN.hpp"
#include "cuda_kernels.cuh"
#include <vector>
#include <string>
#include <array>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <memory>
#include <chrono>
#include <functional>
#include <nlohmann/json.hpp>

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

struct Param {
    CudaTensor<float> data;
    CudaTensor<float> grad;

    Param() {}
    Param(const std::vector<size_t>& shape)
        : data(shape), grad(shape)
    {
        grad.memset_zero();
    }

    void zero_grad() { grad.memset_zero(); }
    size_t numel() const { return data.size(); }
};

class RNG {
    std::mt19937 gen;
public:
    RNG(unsigned seed = 42) : gen(seed) {}

    float normal(float mean = 0.0f, float std = 1.0f) {
        std::normal_distribution<float> dist(mean, std);
        return dist(gen);
    }

    int randint(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    CudaTensor<float> normal_init_tensor(const std::vector<size_t>& shape, float stddev) {
        Tensor<float> cpu(shape);
        for (size_t i = 0; i < cpu.size(); i++)
            cpu[i] = normal(0.0f, stddev);
        return CudaTensor<float>::fromTensor(cpu);
    }

    void normal_init(CudaTensor<float>& t, float stddev) {
        Tensor<float> cpu = t.toTensor();
        for (size_t i = 0; i < cpu.size(); i++)
            cpu[i] = normal(0.0f, stddev);
        t.copyFromHost(cpu.data->data(), cpu.size());
    }
};

struct Linear {
    Param W, b;
    size_t in_feat, out_feat;
    CudaTensor<float> cached_input;

    Linear(size_t in_features, size_t out_features, RNG& rng)
        : W({out_features, in_features}), b({out_features}),
          in_feat(in_features), out_feat(out_features)
    {
        float kaiming = std::sqrt(2.0f / in_features);
        auto cpu_W = rng.normal_init_tensor({out_features, in_features}, kaiming);
        W.data = std::move(cpu_W);
    }

    CudaTensor<float> forward(const CudaTensor<float>& x) {
        cached_input = x;
        CudaTensor<float> Wt({in_feat, out_feat});
        TensorN::cuda::transpose(W.data, Wt);
        CudaTensor<float> prod({x.shape()[0], out_feat});
        TensorN::cuda::matmul(x, Wt, prod);
        CudaTensor<float> out(prod.shape());
        gpt_cuda::add_row_bias(prod, b.data, out);
        return out;
    }

    CudaTensor<float> backward(const CudaTensor<float>& dout) {
        size_t N = dout.shape()[0];
        CudaTensor<float> dout_T({out_feat, N});
        TensorN::cuda::transpose(dout, dout_T);
        CudaTensor<float> dW_mat({out_feat, in_feat});
        TensorN::cuda::matmul(dout_T, cached_input, dW_mat);
        CudaTensor<float> new_grad(W.grad.shape());
        TensorN::cuda::add(W.grad, dW_mat, new_grad);
        W.grad = std::move(new_grad);
        auto db_vec = TensorN::cuda::sum_axis(dout, 0);
        CudaTensor<float> new_b_grad(b.grad.shape());
        TensorN::cuda::add(b.grad, db_vec, new_b_grad);
        b.grad = std::move(new_b_grad);
        CudaTensor<float> dx({N, in_feat});
        TensorN::cuda::matmul(dout, W.data, dx);
        return dx;
    }

    void zero_grad() { W.zero_grad(); b.zero_grad(); }
    std::vector<Param*> parameters() { return {&W, &b}; }
};

struct Embedding {
    Param weight;
    size_t vocab_size, embd_dim;

    Embedding(size_t vocab_sz, size_t embd, RNG& rng)
        : weight({vocab_sz, embd}), vocab_size(vocab_sz), embd_dim(embd)
    {
        auto cpu_W = rng.normal_init_tensor({vocab_sz, embd}, 0.02f);
        weight.data = std::move(cpu_W);
    }

    CudaTensor<float> forward(const std::vector<int>& idxs) {
        size_t N = idxs.size();
        CudaTensor<float> out({N, embd_dim});
        int* d_indices = nullptr;
        cudaMalloc(&d_indices, N * sizeof(int));
        cudaMemcpy(d_indices, idxs.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        gpt_cuda::embedding_forward(weight.data, d_indices, out, N);
        cudaFree(d_indices);
        return out;
    }

    void backward(const CudaTensor<float>& dout, const std::vector<int>& idxs) {
        size_t N = idxs.size();
        int* d_indices = nullptr;
        cudaMalloc(&d_indices, N * sizeof(int));
        cudaMemcpy(d_indices, idxs.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        gpt_cuda::embedding_backward(dout, d_indices, weight.grad, N);
        cudaFree(d_indices);
    }

    void zero_grad() { weight.zero_grad(); }
    std::vector<Param*> parameters() { return {&weight}; }
};

struct LayerNorm {
    Param gamma, beta;
    float eps;
    CudaTensor<float> x_cache, mean_cache, rstd_cache;

    LayerNorm(size_t dim, float eps, RNG& rng)
        : gamma({dim}), beta({dim}), eps(eps)
    {
        Tensor<float> cpu_gamma({dim});
        Tensor<float> cpu_beta({dim});
        for (size_t i = 0; i < dim; i++) {
            cpu_gamma[i] = 1.0f;
            cpu_beta[i] = 0.0f;
        }
        gamma.data = CudaTensor<float>::fromTensor(cpu_gamma);
        beta.data = CudaTensor<float>::fromTensor(cpu_beta);
    }

    CudaTensor<float> forward(const CudaTensor<float>& x) {
        auto sh = x.shape();
        size_t N = sh[0], D = sh[1];
        x_cache = x;
        mean_cache = CudaTensor<float>({N});
        rstd_cache = CudaTensor<float>({N});
        CudaTensor<float> out(sh);
        gpt_cuda::layernorm_forward(x, gamma.data, beta.data, out, mean_cache, rstd_cache, eps);
        return out;
    }

    CudaTensor<float> backward(const CudaTensor<float>& dout) {
        auto sh = dout.shape();
        size_t N = sh[0], D = sh[1];
        CudaTensor<float> dx(sh);
        CudaTensor<float> dgamma({D});
        dgamma.memset_zero();
        CudaTensor<float> dbeta({D});
        dbeta.memset_zero();
        gpt_cuda::layernorm_backward(dout, x_cache, gamma.data, mean_cache, rstd_cache,
                                      dx, dgamma, dbeta);
        CudaTensor<float> new_gamma_grad(gamma.grad.shape());
        TensorN::cuda::add(gamma.grad, dgamma, new_gamma_grad);
        gamma.grad = std::move(new_gamma_grad);
        CudaTensor<float> new_beta_grad(beta.grad.shape());
        TensorN::cuda::add(beta.grad, dbeta, new_beta_grad);
        beta.grad = std::move(new_beta_grad);
        return dx;
    }

    void zero_grad() { gamma.zero_grad(); beta.zero_grad(); }
    std::vector<Param*> parameters() { return {&gamma, &beta}; }
};

inline std::pair<float, CudaTensor<float>> cross_entropy_loss(
    const CudaTensor<float>& logits, const std::vector<int>& targets)
{
    auto sh = logits.shape();
    size_t N = sh[0], V = sh[1];

    int* d_targets = nullptr;
    cudaMalloc(&d_targets, N * sizeof(int));
    cudaMemcpy(d_targets, targets.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    CudaTensor<float> probs(sh);
    float loss = gpt_cuda::cross_entropy_loss_forward(logits, d_targets, probs, N, V);

    CudaTensor<float> dlogits(sh);
    gpt_cuda::cross_entropy_loss_backward(probs, d_targets, dlogits, N, V);

    cudaFree(d_targets);
    return {loss, dlogits};
}

class AdamW {
    std::vector<Param*> params;
    float lr, beta1, beta2, wd, eps;
    std::vector<CudaTensor<float>> m, v;
    int t;

public:
    AdamW(std::vector<Param*> p, float lr = 3e-4f, float b1 = 0.9f,
          float b2 = 0.999f, float wd = 0.01f, float eps = 1e-8f)
        : params(p), lr(lr), beta1(b1), beta2(b2), wd(wd), eps(eps), t(0)
    {
        for (size_t i = 0; i < params.size(); i++) {
            m.emplace_back(params[i]->data.shape());
            m.back().memset_zero();
            v.emplace_back(params[i]->data.shape());
            v.back().memset_zero();
        }
    }

    void step() {
        t++;
        float corr1 = 1.0f - std::pow(beta1, (float)t);
        float corr2 = 1.0f - std::pow(beta2, (float)t);
        float alpha = lr * std::sqrt(corr2) / corr1;
        float eps_hat = eps * std::sqrt(corr2);

        for (size_t i = 0; i < params.size(); i++) {
            gpt_cuda::adamw_step(params[i]->data, params[i]->grad,
                                  m[i], v[i],
                                  lr, beta1, beta2, wd, eps,
                                  corr1, corr2);
        }
    }

    int step_count() const { return t; }

    void save(std::ofstream& f) const {
        f.write(reinterpret_cast<const char*>(&t), sizeof(int));
        f.write(reinterpret_cast<const char*>(&lr), sizeof(float));
        f.write(reinterpret_cast<const char*>(&beta1), sizeof(float));
        f.write(reinterpret_cast<const char*>(&beta2), sizeof(float));
        f.write(reinterpret_cast<const char*>(&wd), sizeof(float));

        uint32_t np = static_cast<uint32_t>(params.size());
        f.write(reinterpret_cast<const char*>(&np), sizeof(uint32_t));

        for (size_t i = 0; i < params.size(); i++) {
            auto m_cpu = m[i].toTensor();
            auto sh = m_cpu.shape();
            uint32_t ndim = static_cast<uint32_t>(sh.size());
            f.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
            for (size_t d = 0; d < sh.size(); d++) {
                size_t dim = sh[d];
                f.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
            }
            for (size_t j = 0; j < m_cpu.size(); j++) {
                float val = m_cpu[j];
                f.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }

            auto v_cpu = v[i].toTensor();
            for (size_t j = 0; j < v_cpu.size(); j++) {
                float val = v_cpu[j];
                f.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
    }

    void load(std::ifstream& f) {
        f.read(reinterpret_cast<char*>(&t), sizeof(int));
        f.read(reinterpret_cast<char*>(&lr), sizeof(float));
        f.read(reinterpret_cast<char*>(&beta1), sizeof(float));
        f.read(reinterpret_cast<char*>(&beta2), sizeof(float));
        f.read(reinterpret_cast<char*>(&wd), sizeof(float));

        uint32_t np = 0;
        f.read(reinterpret_cast<char*>(&np), sizeof(uint32_t));

        for (uint32_t i = 0; i < np && i < (uint32_t)params.size(); i++) {
            uint32_t ndim = 0;
            f.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
            std::vector<size_t> shape(ndim);
            size_t numel = 1;
            for (uint32_t d = 0; d < ndim; d++) {
                f.read(reinterpret_cast<char*>(&shape[d]), sizeof(size_t));
                numel *= shape[d];
            }

            Tensor<float> m_cpu(shape);
            for (size_t j = 0; j < numel; j++) {
                float val = 0;
                f.read(reinterpret_cast<char*>(&val), sizeof(float));
                m_cpu[j] = val;
            }
            m[i] = CudaTensor<float>::fromTensor(m_cpu);

            Tensor<float> v_cpu(shape);
            for (size_t j = 0; j < numel; j++) {
                float val = 0;
                f.read(reinterpret_cast<char*>(&val), sizeof(float));
                v_cpu[j] = val;
            }
            v[i] = CudaTensor<float>::fromTensor(v_cpu);
        }
    }
};

struct GPTConfig {
    size_t vocab_size = 4096;
    size_t block_size = 64;
    size_t n_embd = 128;
    size_t n_layer = 4;
    float ln_eps = 1e-5f;
};

class GPT {
public:
    GPTConfig cfg;

    Embedding wte, wpe;

    struct Block {
        LayerNorm ln_1, ln_2;
        Linear attn_q, attn_k, attn_v, attn_proj;
        Linear mlp_fc, mlp_proj;

        CudaTensor<float> resid1_cache;
        CudaTensor<float> ln1_out_cache;
        CudaTensor<float> Q_cache, K_cache, V_cache;
        CudaTensor<float> attn_w_cache;
        CudaTensor<float> resid2_cache;
        CudaTensor<float> ln2_out_cache;
        CudaTensor<float> fc_out_cache;

        size_t B_cur, S_cur, E_cur;

        Block(size_t n_embd, size_t block_size, RNG& rng)
            : ln_1(n_embd, 1e-5f, rng), ln_2(n_embd, 1e-5f, rng),
              attn_q(n_embd, n_embd, rng), attn_k(n_embd, n_embd, rng),
              attn_v(n_embd, n_embd, rng), attn_proj(n_embd, n_embd, rng),
              mlp_fc(n_embd, 4 * n_embd, rng), mlp_proj(4 * n_embd, n_embd, rng) {}

        CudaTensor<float> forward(const CudaTensor<float>& x, size_t B, size_t S, size_t E) {
            B_cur = B; S_cur = S; E_cur = E;
            resid1_cache = x;

            auto xn = ln_1.forward(x);
            ln1_out_cache = xn;

            Q_cache = attn_q.forward(xn);
            K_cache = attn_k.forward(xn);
            V_cache = attn_v.forward(xn);

            size_t dim = B * S;
            CudaTensor<float> Kt({E, dim});
            TensorN::cuda::transpose(K_cache, Kt);
            CudaTensor<float> scores({dim, dim});
            TensorN::cuda::matmul(Q_cache, Kt, scores);

            float scale = 1.0f / std::sqrt((float)E);
            TensorN::cuda::multiply_scalar(scores, scale, scores);

            gpt_cuda::attention_mask(scores, B, S);

            attn_w_cache = CudaTensor<float>({dim, dim});
            TensorN::cuda::softmax(scores, attn_w_cache, -1);

            CudaTensor<float> attn_out({dim, E});
            TensorN::cuda::matmul(attn_w_cache, V_cache, attn_out);

            auto attn_proj_out = attn_proj.forward(attn_out);
            CudaTensor<float> x1({dim, E});
            TensorN::cuda::add(resid1_cache, attn_proj_out, x1);
            resid2_cache = x1;

            auto xn2 = ln_2.forward(x1);
            ln2_out_cache = xn2;

            auto fc_out = mlp_fc.forward(xn2);
            fc_out_cache = fc_out;
            CudaTensor<float> act_out(fc_out.shape());
            TensorN::cuda::gelu(fc_out, act_out);
            auto mlp_out = mlp_proj.forward(act_out);

            CudaTensor<float> x2({dim, E});
            TensorN::cuda::add(resid2_cache, mlp_out, x2);
            return x2;
        }

        CudaTensor<float> backward(const CudaTensor<float>& dout) {
            size_t B = B_cur, S = S_cur, E = E_cur;
            size_t dim = B * S;

            auto d_res2 = dout;
            auto d_mlp_out = dout;
            auto d_act = mlp_proj.backward(d_mlp_out);
            CudaTensor<float> gelu_deriv_out(d_act.shape());
            gpt_cuda::gelu_deriv(fc_out_cache, gelu_deriv_out);
            CudaTensor<float> d_act_gated(d_act.shape());
            TensorN::cuda::multiply(d_act, gelu_deriv_out, d_act_gated);
            auto d_xn2 = mlp_fc.backward(d_act_gated);
            auto d_x1_from_ln2 = ln_2.backward(d_xn2);
            CudaTensor<float> d_x1({dim, E});
            TensorN::cuda::add(d_x1_from_ln2, d_res2, d_x1);

            auto d_attn_proj_out = d_x1;
            auto d_attn_out = attn_proj.backward(d_attn_proj_out);

            CudaTensor<float> d_attn_w({dim, dim});
            CudaTensor<float> V_t({E, dim});
            TensorN::cuda::transpose(V_cache, V_t);
            TensorN::cuda::matmul(d_attn_out, V_t, d_attn_w);

            CudaTensor<float> d_scores({dim, dim});
            gpt_cuda::softmax_backward(attn_w_cache, d_attn_w, d_scores);

            float scale = 1.0f / std::sqrt((float)E);
            TensorN::cuda::multiply_scalar(d_scores, scale, d_scores);

            CudaTensor<float> dQ({dim, E}), dK({dim, E}), dV({dim, E});
            TensorN::cuda::matmul(d_scores, K_cache, dQ);
            CudaTensor<float> d_scores_T({dim, dim});
            TensorN::cuda::transpose(d_scores, d_scores_T);
            TensorN::cuda::matmul(d_scores_T, Q_cache, dK);
            CudaTensor<float> attn_w_T({dim, dim});
            TensorN::cuda::transpose(attn_w_cache, attn_w_T);
            TensorN::cuda::matmul(attn_w_T, d_attn_out, dV);

            auto dx_q = attn_q.backward(dQ);
            auto dx_k = attn_k.backward(dK);
            auto dx_v = attn_v.backward(dV);
            CudaTensor<float> d_ln1({dim, E});
            TensorN::cuda::add(dx_q, dx_k, d_ln1);
            TensorN::cuda::add(d_ln1, dx_v, d_ln1);
            auto d_ln1_out = ln_1.backward(d_ln1);
            CudaTensor<float> d_x({dim, E});
            TensorN::cuda::add(d_ln1_out, d_x1, d_x);
            return d_x;
        }

        void zero_grad() {
            ln_1.zero_grad(); ln_2.zero_grad();
            attn_q.zero_grad(); attn_k.zero_grad(); attn_v.zero_grad();
            attn_proj.zero_grad();
            mlp_fc.zero_grad(); mlp_proj.zero_grad();
        }

        std::vector<Param*> parameters() {
            std::vector<Param*> ps;
            for (auto* p : ln_1.parameters())  ps.push_back(p);
            for (auto* p : ln_2.parameters())  ps.push_back(p);
            for (auto* p : attn_q.parameters())  ps.push_back(p);
            for (auto* p : attn_k.parameters())  ps.push_back(p);
            for (auto* p : attn_v.parameters())  ps.push_back(p);
            for (auto* p : attn_proj.parameters())  ps.push_back(p);
            for (auto* p : mlp_fc.parameters())  ps.push_back(p);
            for (auto* p : mlp_proj.parameters())  ps.push_back(p);
            return ps;
        }
    };

    std::vector<Block> blocks;
    LayerNorm ln_f;
    Linear lm_head;
    CudaTensor<float> final_x_cache;
    std::vector<int> final_idx_cache;
    size_t B_fwd, S_fwd;

    GPT(GPTConfig config, RNG& rng)
        : cfg(config),
          wte(config.vocab_size, config.n_embd, rng),
          wpe(config.block_size, config.n_embd, rng),
          ln_f(config.n_embd, config.ln_eps, rng),
          lm_head(config.n_embd, config.vocab_size, rng)
    {
        for (size_t i = 0; i < cfg.n_layer; i++)
            blocks.emplace_back(cfg.n_embd, cfg.block_size, rng);
    }

    CudaTensor<float> forward(const std::vector<int>& idx, size_t B, size_t S) {
        B_fwd = B; S_fwd = S;
        final_idx_cache = idx;

        auto tok_emb = wte.forward(idx);
        std::vector<int> pos_idx(B * S);
        for (size_t b = 0; b < B; b++)
            for (size_t s = 0; s < S; s++)
                pos_idx[b * S + s] = (int)s;
        auto pos_emb = wpe.forward(pos_idx);

        size_t dim = B * S;
        CudaTensor<float> x({dim, cfg.n_embd});
        TensorN::cuda::add(tok_emb, pos_emb, x);
        for (auto& blk : blocks)
            x = blk.forward(x, B, S, cfg.n_embd);

        final_x_cache = x;
        auto xn = ln_f.forward(x);
        return lm_head.forward(xn);
    }

    void backward(const CudaTensor<float>& dlogits) {
        auto d_xn = lm_head.backward(dlogits);
        auto d_x = ln_f.backward(d_xn);

        for (int i = (int)blocks.size() - 1; i >= 0; i--)
            d_x = blocks[i].backward(d_x);

        std::vector<int> pos_idx(B_fwd * S_fwd);
        for (size_t b = 0; b < B_fwd; b++)
            for (size_t s = 0; s < S_fwd; s++)
                pos_idx[b * S_fwd + s] = (int)s;
        wpe.backward(d_x, pos_idx);
        wte.backward(d_x, final_idx_cache);
    }

    void zero_grad() {
        wte.zero_grad(); wpe.zero_grad(); ln_f.zero_grad();
        lm_head.zero_grad();
        for (auto& blk : blocks) blk.zero_grad();
    }

    std::vector<Param*> parameters() {
        std::vector<Param*> ps;
        for (auto* p : wte.parameters()) ps.push_back(p);
        for (auto* p : wpe.parameters()) ps.push_back(p);
        for (auto* p : ln_f.parameters()) ps.push_back(p);
        for (auto* p : lm_head.parameters()) ps.push_back(p);
        for (auto& blk : blocks)
            for (auto* p : blk.parameters()) ps.push_back(p);
        return ps;
    }

    size_t total_params() const {
        size_t n = 0;
        n += wte.weight.numel();
        n += wpe.weight.numel();
        n += ln_f.gamma.numel() + ln_f.beta.numel();
        n += lm_head.W.numel() + lm_head.b.numel();
        for (auto& blk : blocks) {
            n += blk.ln_1.gamma.numel() + blk.ln_1.beta.numel();
            n += blk.ln_2.gamma.numel() + blk.ln_2.beta.numel();
            n += blk.attn_q.W.numel() + blk.attn_q.b.numel();
            n += blk.attn_k.W.numel() + blk.attn_k.b.numel();
            n += blk.attn_v.W.numel() + blk.attn_v.b.numel();
            n += blk.attn_proj.W.numel() + blk.attn_proj.b.numel();
            n += blk.mlp_fc.W.numel() + blk.mlp_fc.b.numel();
            n += blk.mlp_proj.W.numel() + blk.mlp_proj.b.numel();
        }
        return n;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Cannot open for writing: " << path << std::endl;
            return;
        }

        constexpr uint32_t magic = 0x4754504D;
        constexpr uint32_t version = 1;

        f.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

        size_t vs = cfg.vocab_size, bs = cfg.block_size;
        size_t ne = cfg.n_embd, nl = cfg.n_layer;
        float eps = cfg.ln_eps;
        f.write(reinterpret_cast<const char*>(&vs), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&bs), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&ne), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&nl), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&eps), sizeof(float));

        auto params = const_cast<GPT*>(this)->parameters();
        uint32_t np = static_cast<uint32_t>(params.size());
        f.write(reinterpret_cast<const char*>(&np), sizeof(uint32_t));

        for (auto* p : params) {
            auto cpu = p->data.toTensor();
            auto sh = cpu.shape();
            uint32_t ndim = static_cast<uint32_t>(sh.size());
            f.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
            for (size_t d = 0; d < sh.size(); d++) {
                size_t dim = sh[d];
                f.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
            }
            for (size_t i = 0; i < cpu.size(); i++) {
                float val = cpu[i];
                f.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }

        std::cout << "Model saved to " << path << " (" << np << " params)" << std::endl;
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Cannot open for reading: " << path << std::endl;
            return;
        }

        constexpr uint32_t magic = 0x4754504D;

        uint32_t fmagic = 0, fversion = 0;
        f.read(reinterpret_cast<char*>(&fmagic), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&fversion), sizeof(uint32_t));

        if (fmagic != magic) {
            std::cerr << "Invalid model file: bad magic" << std::endl;
            return;
        }

        size_t vs = 0, bs = 0, ne = 0, nl = 0;
        float eps = 0;
        f.read(reinterpret_cast<char*>(&vs), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&bs), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&ne), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&nl), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&eps), sizeof(float));

        if (vs != cfg.vocab_size || bs != cfg.block_size ||
            ne != cfg.n_embd || nl != cfg.n_layer) {
            std::cerr << "Warning: saved model config differs from current. "
                      << "Loading anyway..." << std::endl;
        }

        uint32_t np = 0;
        f.read(reinterpret_cast<char*>(&np), sizeof(uint32_t));

        auto params = parameters();

        for (uint32_t pi = 0; pi < np && pi < (uint32_t)params.size(); pi++) {
            uint32_t ndim = 0;
            f.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
            std::vector<size_t> shape(ndim);
            size_t numel = 1;
            for (uint32_t d = 0; d < ndim; d++) {
                f.read(reinterpret_cast<char*>(&shape[d]), sizeof(size_t));
                numel *= shape[d];
            }

            Tensor<float> cpu(shape);
            for (size_t i = 0; i < numel; i++) {
                float val = 0;
                f.read(reinterpret_cast<char*>(&val), sizeof(float));
                cpu[i] = val;
            }

            params[pi]->data = CudaTensor<float>::fromTensor(cpu);
        }

        if (!f) {
            std::cerr << "Warning: partial or corrupt model file" << std::endl;
        }

        std::cout << "Model loaded from " << path << " (" << np << " params)" << std::endl;
    }

    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens,
                               int eos_id, float temperature, int top_k, RNG& rng) {
        std::vector<int> ids = prompt;

        for (size_t step = 0; step < max_new_tokens; step++) {
            size_t context_len = std::min(ids.size(), cfg.block_size);
            size_t offset = ids.size() - context_len;

            std::vector<int> x(context_len);
            std::vector<int> pos(context_len);
            for (size_t i = 0; i < context_len; i++) {
                x[i] = ids[offset + i];
                pos[i] = (int)i;
            }

            auto tok_emb = wte.forward(x);
            auto pos_emb = wpe.forward(pos);
            CudaTensor<float> hid({context_len, cfg.n_embd});
            TensorN::cuda::add(tok_emb, pos_emb, hid);

            for (auto& blk : blocks)
                hid = blk.forward(hid, 1, context_len, cfg.n_embd);

            auto xn = ln_f.forward(hid);
            auto logits = lm_head.forward(xn);

            auto logits_cpu = logits.toTensor();
            size_t last_row = (context_len - 1) * cfg.vocab_size;

            std::vector<std::pair<float, int>> top_v;
            top_v.reserve(cfg.vocab_size);
            float max_logit = -1e30f;
            for (size_t v = 0; v < cfg.vocab_size; v++) {
                float val = logits_cpu[last_row + v];
                if (temperature > 0.01f) val /= temperature;
                if (val > max_logit) max_logit = val;
                top_v.push_back({val, (int)v});
            }

            if (top_k > 0 && top_k < (int)top_v.size()) {
                std::partial_sort(top_v.begin(), top_v.begin() + top_k, top_v.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                top_v.resize(top_k);
            }

            int next_token;
            if (temperature < 0.01f) {
                float best = -1e30f;
                int best_id = 0;
                for (auto& p : top_v) {
                    if (p.first > best) { best = p.first; best_id = p.second; }
                }
                next_token = best_id;
            } else {
                float sum_exp = 0.0f;
                for (auto& p : top_v) {
                    float exp_val = std::exp(p.first - max_logit);
                    p.first = exp_val;
                    sum_exp += exp_val;
                }

                float r = (float)rng.randint(0, 1000000) / 1000000.0f;
                float cum = 0.0f;
                int chosen = top_v[0].second;
                for (auto& p : top_v) {
                    cum += p.first / sum_exp;
                    if (r <= cum) { chosen = p.second; break; }
                }
                next_token = chosen;
            }

            if (next_token == eos_id) break;
            ids.push_back(next_token);
        }

        return ids;
    }
};

inline void numerical_grad_check(GPT& model, const std::vector<int>& x,
    const std::vector<int>& y, size_t batch_size, size_t block_size)
{
    auto params = model.parameters();
    auto logits = model.forward(x, batch_size, block_size);
    auto [loss0, dlogits] = cross_entropy_loss(logits, y);
    model.zero_grad();
    model.backward(dlogits);

    std::cout << "\n=== Numerical Gradient Check ===" << std::endl;
    std::cout << "loss = " << loss0 << std::endl;

    float eps = 1e-3f;
    int check_count = std::min((int)params.size(), 5);

    for (int idx = 0; idx < check_count; idx++) {
        auto* p = params[idx];
        size_t n = p->data.size();
        int check_elems = std::min((int)n, 3);

        Tensor<float> cpu_data = p->data.toTensor();
        Tensor<float> cpu_grad = p->grad.toTensor();

        for (int e = 0; e < check_elems; e++) {
            float old_val = cpu_data[e];

            cpu_data[e] = old_val + eps;
            p->data.copyFromHost(cpu_data.data->data(), n);
            auto logits_p = model.forward(x, batch_size, block_size);
            auto [loss_p, _] = cross_entropy_loss(logits_p, y);

            cpu_data[e] = old_val - eps;
            p->data.copyFromHost(cpu_data.data->data(), n);
            auto logits_m = model.forward(x, batch_size, block_size);
            auto [loss_m, __] = cross_entropy_loss(logits_m, y);

            cpu_data[e] = old_val;
            p->data.copyFromHost(cpu_data.data->data(), n);

            float numerical = (loss_p - loss_m) / (2.0f * eps);
            float analytical = cpu_grad[e];
            float rel_err = std::abs(numerical - analytical) /
                            (std::abs(numerical) + std::abs(analytical) + 1e-8f);

            std::cout << "param[" << idx << "][" << e << "]"
                      << "  analytical=" << analytical
                      << "  numerical=" << numerical
                      << "  rel_err=" << rel_err
                      << (rel_err < 1e-3f ? " OK" : " MISMATCH") << std::endl;
        }
    }
    std::cout << "================================\n" << std::endl;
}

// ============================================================
// BPE Tokenizer (unchanged from CPU version)
// ============================================================
class BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_ranks;
    int bos_id, eos_id;

    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const {
            return std::hash<std::string>()(p.first) ^
                   (std::hash<std::string>()(p.second) << 1);
        }
    };

    static int utf8_char_len(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;
        if ((first_byte & 0xE0) == 0xC0) return 2;
        if ((first_byte & 0xF0) == 0xE0) return 3;
        if ((first_byte & 0xF8) == 0xF0) return 4;
        return 1;
    }

    static int utf8_decode(const std::string& s, size_t& i) {
        if (i >= s.size()) return -1;
        unsigned char c = (unsigned char)s[i];
        int cp = 0, len = utf8_char_len(c);
        if (len == 1) { cp = c; }
        else if (len == 2) { cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F); }
        else if (len == 3) { cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F); }
        else if (len == 4) { cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F); }
        i += len;
        return cp;
    }

    bool is_cjk_char(const std::string& utf8_char) const {
        if (utf8_char.empty()) return false;
        size_t pos = 0;
        int cp = utf8_decode(utf8_char, pos);
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x3040 && cp <= 0x309F) ||
               (cp >= 0x30A0 && cp <= 0x30FF) ||
               (cp >= 0xAC00 && cp <= 0xD7AF);
    }

    static std::string utf8_encode(int cp) {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    static constexpr int BL_UNMAPPED = 256;

    static const std::array<uint16_t, 324>& byte_table() {
        static std::array<uint16_t, 324> tbl;
        static bool ready = false;
        if (!ready) {
            tbl.fill((uint16_t)BL_UNMAPPED);
            for (int b = 33; b <= 126; b++) tbl[b] = (uint16_t)b;
            for (int b = 161; b <= 172; b++) tbl[b] = (uint16_t)b;
            for (int b = 174; b <= 255; b++) tbl[b] = (uint16_t)b;
            int n = 0;
            for (int b = 0; b < 256; b++)
                if (tbl[b] == BL_UNMAPPED)
                    tbl[256 + n++] = (uint16_t)b;
            ready = true;
        }
        return tbl;
    }

    static int byte_to_cp(uint8_t b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            return b;
        int n = 0;
        for (int x = 0; x < b; x++)
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255)))
                n++;
        return 256 + n;
    }

    static bool is_byte_level_cp(int cp) {
        return cp > 0 && cp <= 323 && byte_table()[cp] != BL_UNMAPPED;
    }

    static uint8_t cp_to_byte(int cp) {
        return (uint8_t)byte_table()[cp];
    }

    static std::string byte_to_char(uint8_t b) {
        return utf8_encode(byte_to_cp(b));
    }

    int get_merge_rank(const std::string& a, const std::string& b) const {
        auto it = merge_ranks.find(a + " " + b);
        if (it != merge_ranks.end()) return it->second;
        return -1;
    }

    std::vector<std::string> bpe_encode_word(const std::string& word) const {
        std::string bytes_str;
        for (size_t i = 0; i < word.size(); i++) {
            uint8_t b = (uint8_t)word[i];
            bytes_str += byte_to_char(b);
        }

        std::vector<std::string> symbols;
        for (size_t i = 0; i < bytes_str.size(); ) {
            int len = utf8_char_len((unsigned char)bytes_str[i]);
            symbols.push_back(bytes_str.substr(i, len));
            i += len;
        }

        while (symbols.size() > 1) {
            int best_rank = 1 << 30;
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                int r = get_merge_rank(symbols[i], symbols[i + 1]);
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                }
            }
            if (best_rank == (1 << 30)) break;

            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }
        return symbols;
    }

public:
    BPETokenizer() : bos_id(-1), eos_id(-1) {}

    bool load(const std::string& json_path) {
        std::ifstream f(json_path);
        if (!f.is_open()) return false;

        nlohmann::json j;
        f >> j;

        auto& model = j["model"];
        auto& v = model["vocab"];
        id_to_token.resize(v.size());
        for (auto it = v.begin(); it != v.end(); ++it) {
            std::string token = it.key();
            int id = it.value();
            vocab[token] = id;
            if (id < (int)id_to_token.size())
                id_to_token[id] = token;
        }

        auto& merges_arr = model["merges"];
        for (size_t i = 0; i < merges_arr.size(); i++) {
            std::string merge_str = merges_arr[i][0].get<std::string>() + " " +
                                    merges_arr[i][1].get<std::string>();
            merge_ranks[merge_str] = (int)i;
        }

        auto bos_it = vocab.find("<|im_start|>");
        if (bos_it != vocab.end()) bos_id = bos_it->second;
        auto eos_it = vocab.find("<|im_end|>");
        if (eos_it != vocab.end()) eos_id = eos_it->second;

        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        if (bos_id >= 0) ids.push_back(bos_id);

        std::string current;
        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = (unsigned char)text[i];
            int len = 1;
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            if (i + len > text.size()) len = (int)(text.size() - i);

            std::string ch = text.substr(i, len);
            if (ch == " " || ch == "\n" || ch == "\t" || ch == "\r") {
                if (!current.empty()) {
                    auto syms = bpe_encode_word(current);
                    for (auto& s : syms) {
                        auto it = vocab.find(s);
                        if (it != vocab.end()) ids.push_back(it->second);
                    }
                    current.clear();
                }
                auto syms = bpe_encode_word(ch);
                for (auto& s : syms) {
                    auto it = vocab.find(s);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            } else if (is_cjk_char(ch)) {
                if (!current.empty()) {
                    auto syms = bpe_encode_word(current);
                    for (auto& s : syms) {
                        auto it = vocab.find(s);
                        if (it != vocab.end()) ids.push_back(it->second);
                    }
                    current.clear();
                }
                auto syms = bpe_encode_word(ch);
                for (auto& s : syms) {
                    auto it = vocab.find(s);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            } else {
                current += ch;
            }
            i += len;
        }
        if (!current.empty()) {
            auto syms = bpe_encode_word(current);
            for (auto& s : syms) {
                auto it = vocab.find(s);
                if (it != vocab.end()) ids.push_back(it->second);
            }
        }

        if (eos_id >= 0) ids.push_back(eos_id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size()) continue;
            const auto& token = id_to_token[id];
            for (size_t i = 0; i < token.size(); ) {
                int cp = utf8_decode(token, i);
                if (is_byte_level_cp(cp)) {
                    result += (char)cp_to_byte(cp);
                }
            }
        }
        return result;
    }

    size_t vocab_size() const { return id_to_token.size(); }
    int get_bos_id() const { return bos_id; }
    int get_eos_id() const { return eos_id; }
};

#endif // TENSORN_CUDA_AVAILABLE
