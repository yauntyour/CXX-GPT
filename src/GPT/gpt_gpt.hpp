#pragma once

#include "TensorN.hpp"
#include "gpt_cuda_kernels.cuh"
#include "tokenizer.hpp"
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
    float learning_rate() const { return lr; }
    float get_beta1() const { return beta1; }
    float get_beta2() const { return beta2; }
    float get_weight_decay() const { return wd; }

    std::vector<std::pair<std::string, Tensor<float>>> named_state() const {
        std::vector<std::pair<std::string, Tensor<float>>> state;
        for (size_t i = 0; i < params.size(); i++) {
            std::string prefix = "param." + std::to_string(i) + ".";
            state.push_back({prefix + "m", m[i].toTensor()});
            state.push_back({prefix + "v", v[i].toTensor()});
        }
        return state;
    }

    void load_state(const std::unordered_map<std::string, Tensor<float>>& state, int saved_step) {
        t = saved_step;
        for (size_t i = 0; i < params.size(); i++) {
            std::string m_key = "param." + std::to_string(i) + ".m";
            std::string v_key = "param." + std::to_string(i) + ".v";
            if (state.count(m_key)) {
                m[i] = CudaTensor<float>::fromTensor(state.at(m_key));
            }
            if (state.count(v_key)) {
                v[i] = CudaTensor<float>::fromTensor(state.at(v_key));
            }
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

    std::vector<std::pair<std::string, Param*>> named_parameters() {
        std::vector<std::pair<std::string, Param*>> np;
        np.push_back({"wte.weight", &wte.weight});
        np.push_back({"wpe.weight", &wpe.weight});
        np.push_back({"ln_f.gamma", &ln_f.gamma});
        np.push_back({"ln_f.beta", &ln_f.beta});
        np.push_back({"lm_head.W", &lm_head.W});
        np.push_back({"lm_head.b", &lm_head.b});
        for (size_t i = 0; i < blocks.size(); i++) {
            std::string prefix = "blocks." + std::to_string(i) + ".";
            np.push_back({prefix + "ln_1.gamma", &blocks[i].ln_1.gamma});
            np.push_back({prefix + "ln_1.beta", &blocks[i].ln_1.beta});
            np.push_back({prefix + "ln_2.gamma", &blocks[i].ln_2.gamma});
            np.push_back({prefix + "ln_2.beta", &blocks[i].ln_2.beta});
            np.push_back({prefix + "attn_q.W", &blocks[i].attn_q.W});
            np.push_back({prefix + "attn_q.b", &blocks[i].attn_q.b});
            np.push_back({prefix + "attn_k.W", &blocks[i].attn_k.W});
            np.push_back({prefix + "attn_k.b", &blocks[i].attn_k.b});
            np.push_back({prefix + "attn_v.W", &blocks[i].attn_v.W});
            np.push_back({prefix + "attn_v.b", &blocks[i].attn_v.b});
            np.push_back({prefix + "attn_proj.W", &blocks[i].attn_proj.W});
            np.push_back({prefix + "attn_proj.b", &blocks[i].attn_proj.b});
            np.push_back({prefix + "mlp_fc.W", &blocks[i].mlp_fc.W});
            np.push_back({prefix + "mlp_fc.b", &blocks[i].mlp_fc.b});
            np.push_back({prefix + "mlp_proj.W", &blocks[i].mlp_proj.W});
            np.push_back({prefix + "mlp_proj.b", &blocks[i].mlp_proj.b});
        }
        return np;
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
        std::string gguf_path = path;
        if (gguf_path.size() < 5 || gguf_path.substr(gguf_path.size() - 5) != ".gguf") {
            gguf_path += ".gguf";
        }

        std::vector<std::pair<std::string, Tensor<float>>> tensors;
        auto named = const_cast<GPT*>(this)->named_parameters();
        for (auto& [name, param] : named) {
            tensors.push_back({name, param->data.toTensor()});
        }

        std::unordered_map<std::string, GGUFMetadataValue> meta;
        meta["general.architecture"] = std::string("cxxgpt");
        meta["general.name"] = std::string("CXX-GPT");
        meta["cxxgpt.vocab_size"] = uint64_t(cfg.vocab_size);
        meta["cxxgpt.block_size"] = uint64_t(cfg.block_size);
        meta["cxxgpt.n_embd"] = uint64_t(cfg.n_embd);
        meta["cxxgpt.n_layer"] = uint64_t(cfg.n_layer);
        meta["cxxgpt.layer_norm_epsilon"] = cfg.ln_eps;

        save_gguf_multi(tensors, gguf_path, meta);
        std::cout << "Model saved to " << gguf_path << " (" << tensors.size() << " tensors)" << std::endl;
    }

    static GPTConfig load_config(const std::string& path) {
        auto meta = gguf_read_metadata(path);
        GPTConfig cfg;

        auto get_u64 = [&](const std::string& key, size_t& target) {
            auto it = meta.find(key);
            if (it != meta.end()) {
                if (std::holds_alternative<uint64_t>(it->second))
                    target = (size_t)std::get<uint64_t>(it->second);
                else if (std::holds_alternative<int64_t>(it->second))
                    target = (size_t)std::get<int64_t>(it->second);
            }
        };

        get_u64("cxxgpt.vocab_size", cfg.vocab_size);
        get_u64("cxxgpt.block_size", cfg.block_size);
        get_u64("cxxgpt.n_embd", cfg.n_embd);
        get_u64("cxxgpt.n_layer", cfg.n_layer);

        auto it_eps = meta.find("cxxgpt.layer_norm_epsilon");
        if (it_eps != meta.end() && std::holds_alternative<float>(it_eps->second))
            cfg.ln_eps = std::get<float>(it_eps->second);

        return cfg;
    }

    void load(const std::string& path) {
        auto meta = gguf_read_metadata(path);

        auto it_arch = meta.find("general.architecture");
        if (it_arch != meta.end() && std::holds_alternative<std::string>(it_arch->second)) {
            std::string arch = std::get<std::string>(it_arch->second);
            if (arch != "cxxgpt") {
                std::cerr << "Warning: model architecture is '" << arch << "', expected 'cxxgpt'" << std::endl;
            }
        }

        auto tensors = load_gguf_multi<float>(path);
        auto named = named_parameters();

        size_t loaded = 0;
        for (auto& [name, param] : named) {
            if (tensors.count(name)) {
                param->data = CudaTensor<float>::fromTensor(tensors[name]);
                loaded++;
            } else {
                std::cerr << "Warning: missing tensor '" << name << "'" << std::endl;
            }
        }

        std::cout << "Model loaded from " << path << " (" << loaded << " tensors)" << std::endl;
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
#endif // TENSORN_CUDA_AVAILABLE
