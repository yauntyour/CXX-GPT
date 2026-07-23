#pragma once

#include "GPT/gpt.hpp"
#include "LinearAttention/linear_cuda_kernels.cuh"

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

static size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

static Tensor<float> generate_hadamard_submatrix(size_t D, size_t E) {
    std::vector<float> full(D * D, 0.0f);
    full[0] = 1.0f;
    for (size_t k = 1; k < D; k *= 2) {
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < k; j++) {
                full[i * D + j + k]       =  full[i * D + j];
                full[(i + k) * D + j]     =  full[i * D + j];
                full[(i + k) * D + j + k] = -full[i * D + j];
            }
        }
    }
    Tensor<float> H_sub({D, E});
    for (size_t d = 0; d < D; d++)
        for (size_t e = 0; e < E; e++)
            H_sub[d * E + e] = full[d * D + e];
    return H_sub;
}

struct LinearGPTConfig {
    size_t vocab_size = 4096;
    size_t block_size = 64;
    size_t n_embd = 256;
    size_t n_layer = 4;
    float ln_eps = 1e-5f;
};

inline CudaTensor<float> compute_phi(const CudaTensor<float>& x,
    const CudaTensor<float>& H, size_t D, size_t E)
{
    size_t N = x.shape()[0];
    float inv_sqrt_E = 1.0f / std::sqrt((float)E);
    CudaTensor<float> Ht({E, D});
    TensorN::cuda::transpose(H, Ht);
    CudaTensor<float> raw({N, D});
    TensorN::cuda::matmul(x, Ht, raw);
    TensorN::cuda::multiply_scalar(raw, inv_sqrt_E, raw);
    CudaTensor<float> phi({N, D});
    TensorN::cuda::exp(raw, phi);
    return phi;
}

inline void backward_phi(const CudaTensor<float>& dphi,
    const CudaTensor<float>& phi, const CudaTensor<float>& H,
    size_t D, size_t E, CudaTensor<float>& dx)
{
    size_t N = dphi.shape()[0];
    float inv_sqrt_E = 1.0f / std::sqrt((float)E);
    CudaTensor<float> d_raw({N, D});
    TensorN::cuda::multiply(dphi, phi, d_raw);
    TensorN::cuda::matmul(d_raw, H, dx);
    TensorN::cuda::multiply_scalar(dx, inv_sqrt_E, dx);
}

class LinearGPT {
public:
    LinearGPTConfig cfg;
    size_t D;

    CudaTensor<float> H;

    Embedding wte, wpe;
    CudaTensor<float> wpe_all;

    struct LinearBlock {
        LayerNorm ln_1, ln_2;
        Linear attn_q, attn_k, attn_v, attn_proj;
        Linear mlp_fc, mlp_proj;

        CudaTensor<float> resid1_cache;
        CudaTensor<float> ln1_out_cache;
        CudaTensor<float> Q_cache, K_cache, V_cache;
        CudaTensor<float> Q_phi_cache, K_phi_cache;
        CudaTensor<float> O_cache;
        CudaTensor<float> den_cache, z_cache;
        CudaTensor<float> resid2_cache;
        CudaTensor<float> ln2_out_cache;
        CudaTensor<float> fc_out_cache;
        CudaTensor<float> S_buf, z_buf;

        size_t B_cur, S_cur, E_cur, D_cur;
        const CudaTensor<float>* H_ptr;

        LinearBlock(size_t n_embd, size_t block_size, size_t D,
                    const CudaTensor<float>& hadamard, RNG& rng)
            : ln_1(n_embd, 1e-5f, rng), ln_2(n_embd, 1e-5f, rng),
              attn_q(n_embd, n_embd, rng), attn_k(n_embd, n_embd, rng),
              attn_v(n_embd, n_embd, rng), attn_proj(n_embd, n_embd, rng),
              mlp_fc(n_embd, 4 * n_embd, rng), mlp_proj(4 * n_embd, n_embd, rng),
              H_ptr(&hadamard) {}

        CudaTensor<float> forward(const CudaTensor<float>& x, size_t B, size_t S,
                                   size_t E, size_t D) {
            B_cur = B; S_cur = S; E_cur = E; D_cur = D;
            size_t dim = B * S;
            resid1_cache = x;

            auto xn = ln_1.forward(x);
            ln1_out_cache = xn;

            Q_cache = attn_q.forward(xn);
            K_cache = attn_k.forward(xn);
            V_cache = attn_v.forward(xn);

            Q_phi_cache = compute_phi(Q_cache, *H_ptr, D, E);
            K_phi_cache = compute_phi(K_cache, *H_ptr, D, E);

            S_buf = CudaTensor<float>({B * D, E});
            z_buf = CudaTensor<float>({B, D});
            den_cache = CudaTensor<float>({dim});
            z_cache = CudaTensor<float>({dim, D});
            O_cache = CudaTensor<float>({dim, E});

            linear_cuda::causal_linear_attention_fwd(
                Q_phi_cache, K_phi_cache, V_cache,
                O_cache, den_cache, z_cache,
                S_buf, z_buf, B, S, D, E);

            auto attn_proj_out = attn_proj.forward(O_cache);
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
            size_t B = B_cur, S = S_cur, E = E_cur, D = D_cur;
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

            CudaTensor<float> dnum({dim, E});
            CudaTensor<float> dden({dim});
            linear_cuda::causal_linear_attention_bwd_step1(
                O_cache, d_attn_out, den_cache, dnum, dden, dim, E);

            CudaTensor<float> dQ_phi({dim, D});
            linear_cuda::causal_linear_attention_bwd_dQ(
                K_phi_cache, V_cache, dnum, dden, z_cache,
                dQ_phi, S_buf, B, S, D, E);

            CudaTensor<float> dK_phi({dim, D});
            CudaTensor<float> dV({dim, E});
            linear_cuda::causal_linear_attention_bwd_dKdV(
                Q_phi_cache, K_phi_cache, V_cache,
                dnum, dden, dK_phi, dV,
                B, S, D, E);

            CudaTensor<float> dQ({dim, E});
            backward_phi(dQ_phi, Q_phi_cache, *H_ptr, D, E, dQ);
            CudaTensor<float> dK({dim, E});
            backward_phi(dK_phi, K_phi_cache, *H_ptr, D, E, dK);

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

    std::vector<LinearBlock> blocks;
    LayerNorm ln_f;
    Linear lm_head;
    CudaTensor<float> final_x_cache;
    std::vector<int> final_idx_cache;
    size_t B_fwd, S_fwd;

    LinearGPT(LinearGPTConfig config, RNG& rng)
        : cfg(config),
          D(next_power_of_2(config.n_embd)),
          wte(config.vocab_size, config.n_embd, rng),
          wpe(config.block_size, config.n_embd, rng),
          ln_f(config.n_embd, config.ln_eps, rng),
          lm_head(config.n_embd, config.vocab_size, rng)
    {
        auto H_cpu = generate_hadamard_submatrix(D, cfg.n_embd);
        H = CudaTensor<float>::fromTensor(H_cpu);

        for (size_t i = 0; i < cfg.n_layer; i++)
            blocks.emplace_back(cfg.n_embd, cfg.block_size, D, H, rng);
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
            x = blk.forward(x, B, S, cfg.n_embd, D);

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

        constexpr uint32_t magic = 0x4C474D44;
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

        auto params = const_cast<LinearGPT*>(this)->parameters();
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

        constexpr uint32_t magic = 0x4C474D44;

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
            for (size_t i = 0; i < context_len; i++)
                x[i] = ids[offset + i];

            auto tok_emb = wte.forward(x);
            std::vector<int> pos(context_len);
            for (size_t i = 0; i < context_len; i++)
                pos[i] = (int)i;
            auto pos_emb = wpe.forward(pos);

            CudaTensor<float> hid({context_len, cfg.n_embd});
            TensorN::cuda::add(tok_emb, pos_emb, hid);

            for (auto& blk : blocks)
                hid = blk.forward(hid, 1, context_len, cfg.n_embd, D);

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

inline void linear_numerical_grad_check(LinearGPT& model,
    const std::vector<int>& x, const std::vector<int>& y,
    size_t batch_size, size_t block_size)
{
    auto params = model.parameters();
    auto logits = model.forward(x, batch_size, block_size);
    auto [loss0, dlogits] = cross_entropy_loss(logits, y);
    model.zero_grad();
    model.backward(dlogits);

    std::cout << "\n=== Numerical Gradient Check (LinearGPT) ===" << std::endl;
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

#endif
