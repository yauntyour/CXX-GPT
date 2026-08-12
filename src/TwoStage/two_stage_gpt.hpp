#pragma once

#include "GPT/gpt_gpt.hpp"
#include "TwoStage/two_stage_cuda_kernels.cuh"
#include <cassert>

#ifdef TENSORN_CUDA_AVAILABLE

using namespace TensorN;

// Two-stage attention model:
//   1) expanded-feature softmax MHA  (q/k/v through per-head FFN expanders)
//   2) ELU+1 linear "kernel" attention with causal prefix sums
// then a final FFN and a weight-tied lm head. All projections are
// bias-free, matching the reference PyTorch spec.
struct TwoStageConfig
{
    size_t vocab_size = 4096;
    size_t block_size = 64;
    size_t n_embd = 256;
    size_t n_head = 8;
    size_t N = 2; // feature expansion factor: head_dim = N * (n_embd / n_head)
};

class TwoStageGPT
{
public:
    TwoStageConfig cfg;
    size_t head_dim;

    Embedding wte;

    struct StageBlock
    {
        Linear attn_q, attn_k, attn_v; // n_embd -> n_embd
        Linear q_ffn_fc, q_ffn_proj;   // n_embd -> 4N*n_embd -> N*n_embd
        Linear k_ffn_fc, k_ffn_proj;
        Linear v_ffn_fc, v_ffn_proj;
        Linear lq_proj, lk_proj, lv_proj; // N*n_embd -> n_embd
        Linear ffn_fc, ffn_proj;          // n_embd -> 4N*n_embd -> n_embd

        CudaTensor<float> Qc, Kc, Vc;       // after q/k/v proj
        CudaTensor<float> qact, kact, vact; // GELU outputs of the expander fc layers
        CudaTensor<float> Qx, Kx, Vx;       // after expander FFNs (dim, H*Dh)
        CudaTensor<float> ra;               // stage-1 attention output
        CudaTensor<float> lqc, lkc, lvc;    // after l*_proj
        CudaTensor<float> Ac;               // stage-2 attention output
        CudaTensor<float> ffc_out;          // final FFN fc output (for GELU deriv)
        CudaTensor<float> den, z_cache;     // stage-2 backward caches
        CudaTensor<float> S_buf, z_buf;     // stage-2 scan buffers

        size_t B_cur, S_cur, E_cur, H_cur, Dh_cur;

        StageBlock(size_t n_embd, size_t N, size_t n_head, RNG &rng)
            : attn_q(n_embd, n_embd, rng, false),
              attn_k(n_embd, n_embd, rng, false),
              attn_v(n_embd, n_embd, rng, false),
              q_ffn_fc(n_embd, 4 * N * n_embd, rng, false),
              q_ffn_proj(4 * N * n_embd, N * n_embd, rng, false),
              k_ffn_fc(n_embd, 4 * N * n_embd, rng, false),
              k_ffn_proj(4 * N * n_embd, N * n_embd, rng, false),
              v_ffn_fc(n_embd, 4 * N * n_embd, rng, false),
              v_ffn_proj(4 * N * n_embd, N * n_embd, rng, false),
              lq_proj(N * n_embd, n_embd, rng, false),
              lk_proj(N * n_embd, n_embd, rng, false),
              lv_proj(N * n_embd, n_embd, rng, false),
              ffn_fc(n_embd, 4 * N * n_embd, rng, false),
              ffn_proj(4 * N * n_embd, n_embd, rng, false) {}

        CudaTensor<float> forward(const CudaTensor<float> &x, size_t B, size_t S,
                                  size_t E, size_t H, size_t Dh)
        {
            B_cur = B;
            S_cur = S;
            E_cur = E;
            H_cur = H;
            Dh_cur = Dh;
            size_t dim = B * S;
            size_t big = H * Dh;

            Qc = attn_q.forward(x);
            Kc = attn_k.forward(x);
            Vc = attn_v.forward(x);

            auto qf = q_ffn_fc.forward(Qc);
            qact = CudaTensor<float>(qf.shape());
            TensorN::cuda::gelu(qf, qact);
            Qx = q_ffn_proj.forward(qact);

            auto kf = k_ffn_fc.forward(Kc);
            kact = CudaTensor<float>(kf.shape());
            TensorN::cuda::gelu(kf, kact);
            Kx = k_ffn_proj.forward(kact);

            auto vf = v_ffn_fc.forward(Vc);
            vact = CudaTensor<float>(vf.shape());
            TensorN::cuda::gelu(vf, vact);
            Vx = v_ffn_proj.forward(vact);

            // stage 1: fused causal softmax MHA over the expanded heads
            ra = CudaTensor<float>({dim, big});
            two_stage_cuda::fused_mha_fwd(Qx, Kx, Vx, ra, B, S, H, Dh);

            lqc = lq_proj.forward(ra);
            lkc = lk_proj.forward(ra);
            lvc = lv_proj.forward(ra);

            // stage 2: fused ELU+1 kernel attention (D == E == n_embd)
            den = CudaTensor<float>({dim});
            z_cache = CudaTensor<float>({dim, E});
            S_buf = CudaTensor<float>({B * E, E});
            z_buf = CudaTensor<float>({B, E});
            Ac = CudaTensor<float>({dim, E});
            two_stage_cuda::kernel_attn_fwd(lqc, lkc, lvc, Ac, den, z_cache,
                                            S_buf, z_buf, B, S, E, E);

            // final FFN (reuses the standard fc -> GELU -> proj pattern)
            auto ff = ffn_fc.forward(Ac);
            ffc_out = ff;
            CudaTensor<float> f_act(ff.shape());
            TensorN::cuda::gelu(ff, f_act);
            return ffn_proj.forward(f_act);
        }

        CudaTensor<float> backward(const CudaTensor<float> &dout)
        {
            size_t B = B_cur, S = S_cur, E = E_cur, H = H_cur, Dh = Dh_cur;
            size_t dim = B * S;
            size_t big = H * Dh;

            // final FFN
            auto d_ffn_proj = ffn_proj.backward(dout);
            CudaTensor<float> gd(ffc_out.shape());
            gpt_cuda::gelu_deriv(ffc_out, gd);
            CudaTensor<float> d_act(d_ffn_proj.shape());
            TensorN::cuda::multiply(d_ffn_proj, gd, d_act);
            auto dA = ffn_fc.backward(d_act);

            // stage 2 backward (ELU+1 derivative fused into the kernels)
            CudaTensor<float> dnum({dim, E});
            CudaTensor<float> dden({dim});
            two_stage_cuda::kernel_attn_bwd_step1(Ac, dA, den, dnum, dden, dim, E);

            CudaTensor<float> dlq({dim, E});
            two_stage_cuda::kernel_attn_bwd_dQ(lkc, lvc, dnum, dden, z_cache, lqc,
                                               dlq, S_buf, B, S, E, E);
            CudaTensor<float> dlk({dim, E});
            CudaTensor<float> dlv({dim, E});
            two_stage_cuda::kernel_attn_bwd_dKdV(lqc, lkc, lvc, dnum, dden,
                                                 dlk, dlv, B, S, E, E);

            auto d_ra_lq = lq_proj.backward(dlq);
            auto d_ra_lk = lk_proj.backward(dlk);
            auto d_ra_lv = lv_proj.backward(dlv);
            CudaTensor<float> dra({dim, big});
            TensorN::cuda::add(d_ra_lq, d_ra_lk, dra);
            TensorN::cuda::add(dra, d_ra_lv, dra);

            // stage 1 backward (softmax recomputed inside the fused kernel)
            CudaTensor<float> dQx({dim, big}), dKx({dim, big}), dVx({dim, big});
            two_stage_cuda::fused_mha_bwd(Qx, Kx, Vx, ra, dra,
                                          dQx, dKx, dVx, B, S, H, Dh);

            // q/k/v expander FFNs
            auto d_qproj = q_ffn_proj.backward(dQx);
            CudaTensor<float> gq(qact.shape());
            gpt_cuda::gelu_deriv(qact, gq);
            CudaTensor<float> dqact(d_qproj.shape());
            TensorN::cuda::multiply(d_qproj, gq, dqact);
            auto d_qfc = q_ffn_fc.backward(dqact);
            auto dx_q = attn_q.backward(d_qfc);

            auto d_kproj = k_ffn_proj.backward(dKx);
            CudaTensor<float> gk(kact.shape());
            gpt_cuda::gelu_deriv(kact, gk);
            CudaTensor<float> dkact(d_kproj.shape());
            TensorN::cuda::multiply(d_kproj, gk, dkact);
            auto d_kfc = k_ffn_fc.backward(dkact);
            auto dx_k = attn_k.backward(d_kfc);

            auto d_vproj = v_ffn_proj.backward(dVx);
            CudaTensor<float> gv(vact.shape());
            gpt_cuda::gelu_deriv(vact, gv);
            CudaTensor<float> dvact(d_vproj.shape());
            TensorN::cuda::multiply(d_vproj, gv, dvact);
            auto d_vfc = v_ffn_fc.backward(dvact);
            auto dx_v = attn_v.backward(d_vfc);

            CudaTensor<float> dx({dim, E});
            TensorN::cuda::add(dx_q, dx_k, dx);
            TensorN::cuda::add(dx, dx_v, dx);
            return dx;
        }

        void zero_grad()
        {
            attn_q.zero_grad();
            attn_k.zero_grad();
            attn_v.zero_grad();
            q_ffn_fc.zero_grad();
            q_ffn_proj.zero_grad();
            k_ffn_fc.zero_grad();
            k_ffn_proj.zero_grad();
            v_ffn_fc.zero_grad();
            v_ffn_proj.zero_grad();
            lq_proj.zero_grad();
            lk_proj.zero_grad();
            lv_proj.zero_grad();
            ffn_fc.zero_grad();
            ffn_proj.zero_grad();
        }

        std::vector<Param *> parameters()
        {
            std::vector<Param *> ps;
            for (auto *p : attn_q.parameters()) ps.push_back(p);
            for (auto *p : attn_k.parameters()) ps.push_back(p);
            for (auto *p : attn_v.parameters()) ps.push_back(p);
            for (auto *p : q_ffn_fc.parameters()) ps.push_back(p);
            for (auto *p : q_ffn_proj.parameters()) ps.push_back(p);
            for (auto *p : k_ffn_fc.parameters()) ps.push_back(p);
            for (auto *p : k_ffn_proj.parameters()) ps.push_back(p);
            for (auto *p : v_ffn_fc.parameters()) ps.push_back(p);
            for (auto *p : v_ffn_proj.parameters()) ps.push_back(p);
            for (auto *p : lq_proj.parameters()) ps.push_back(p);
            for (auto *p : lk_proj.parameters()) ps.push_back(p);
            for (auto *p : lv_proj.parameters()) ps.push_back(p);
            for (auto *p : ffn_fc.parameters()) ps.push_back(p);
            for (auto *p : ffn_proj.parameters()) ps.push_back(p);
            return ps;
        }
    };

    StageBlock blk;
    CudaTensor<float> final_x_cache;
    std::vector<int> final_idx_cache;
    const int *d_idx_cache = nullptr;
    size_t B_fwd, S_fwd;

    TwoStageGPT(TwoStageConfig config, RNG &rng)
        : cfg(config),
          head_dim(config.N * (config.n_embd / config.n_head)),
          wte(config.vocab_size, config.n_embd, rng),
          blk(config.n_embd, config.N, config.n_head, rng)
    {
        assert(config.n_embd % config.n_head == 0);
    }

    CudaTensor<float> forward(const std::vector<int> &idx, size_t B, size_t S)
    {
        B_fwd = B;
        S_fwd = S;
        final_idx_cache = idx;
        d_idx_cache = nullptr;
        size_t dim = B * S;

        auto x = wte.forward(idx);
        final_x_cache = blk.forward(x, B, S, cfg.n_embd, cfg.n_head, head_dim);

        // weight-tied lm head: logits = O @ wte^T
        CudaTensor<float> Wt({cfg.n_embd, cfg.vocab_size});
        TensorN::cuda::transpose(wte.weight.data, Wt);
        CudaTensor<float> logits({dim, cfg.vocab_size});
        TensorN::cuda::matmul(final_x_cache, Wt, logits);
        return logits;
    }

    // Device-side path: indices already on GPU (pipelined batch feeders).
    CudaTensor<float> forward(const int *d_idx, size_t B, size_t S)
    {
        B_fwd = B;
        S_fwd = S;
        final_idx_cache.clear();
        d_idx_cache = d_idx;
        size_t dim = B * S;

        auto x = wte.forward(d_idx, dim);
        final_x_cache = blk.forward(x, B, S, cfg.n_embd, cfg.n_head, head_dim);

        // weight-tied lm head: logits = O @ wte^T
        CudaTensor<float> Wt({cfg.n_embd, cfg.vocab_size});
        TensorN::cuda::transpose(wte.weight.data, Wt);
        CudaTensor<float> logits({dim, cfg.vocab_size});
        TensorN::cuda::matmul(final_x_cache, Wt, logits);
        return logits;
    }

    void backward(const CudaTensor<float> &dlogits)
    {
        size_t dim = B_fwd * S_fwd;

        // tied lm head backward: dO = dlogits @ wte ; grad_wte += dlogits^T @ O
        CudaTensor<float> dO({dim, cfg.n_embd});
        TensorN::cuda::matmul(dlogits, wte.weight.data, dO);
        CudaTensor<float> dlogits_T({cfg.vocab_size, dim});
        TensorN::cuda::transpose(dlogits, dlogits_T);
        CudaTensor<float> dW({cfg.vocab_size, cfg.n_embd});
        TensorN::cuda::matmul(dlogits_T, final_x_cache, dW);
        CudaTensor<float> new_grad(wte.weight.grad.shape());
        TensorN::cuda::add(wte.weight.grad, dW, new_grad);
        wte.weight.grad = std::move(new_grad);

        auto dx = blk.backward(dO);
        if (d_idx_cache)
            wte.backward(dx, d_idx_cache, B_fwd * S_fwd);
        else
            wte.backward(dx, final_idx_cache);
    }

    void zero_grad()
    {
        wte.zero_grad();
        blk.zero_grad();
    }

    std::vector<std::pair<std::string, Param *>> named_parameters()
    {
        std::vector<std::pair<std::string, Param *>> np;
        np.push_back({"embedding.weight", &wte.weight});
        auto add_linear = [&](const std::string &name, Linear &lin)
        {
            np.push_back({name + ".W", &lin.W});
            if (lin.has_bias())
                np.push_back({name + ".b", &lin.b});
        };
        add_linear("attn_q", blk.attn_q);
        add_linear("attn_k", blk.attn_k);
        add_linear("attn_v", blk.attn_v);
        add_linear("q_ffn_fc", blk.q_ffn_fc);
        add_linear("q_ffn_proj", blk.q_ffn_proj);
        add_linear("k_ffn_fc", blk.k_ffn_fc);
        add_linear("k_ffn_proj", blk.k_ffn_proj);
        add_linear("v_ffn_fc", blk.v_ffn_fc);
        add_linear("v_ffn_proj", blk.v_ffn_proj);
        add_linear("lq_proj", blk.lq_proj);
        add_linear("lk_proj", blk.lk_proj);
        add_linear("lv_proj", blk.lv_proj);
        add_linear("ffn_fc", blk.ffn_fc);
        add_linear("ffn_proj", blk.ffn_proj);
        return np;
    }

    std::vector<Param *> parameters()
    {
        std::vector<Param *> ps;
        for (auto *p : wte.parameters())
            ps.push_back(p);
        for (auto *p : blk.parameters())
            ps.push_back(p);
        return ps;
    }

    size_t total_params() const
    {
        size_t n = 0;
        for (auto *p : const_cast<TwoStageGPT *>(this)->parameters())
            n += p->numel();
        return n;
    }

    void save(const std::string &path) const
    {
        std::string gguf_path = path;
        if (gguf_path.size() < 5 || gguf_path.substr(gguf_path.size() - 5) != ".gguf")
            gguf_path += ".gguf";

        std::vector<std::pair<std::string, Tensor<float>>> tensors;
        auto named = const_cast<TwoStageGPT *>(this)->named_parameters();
        for (auto &[name, param] : named)
            tensors.push_back({name, param->data.toTensor()});

        std::unordered_map<std::string, GGUFMetadataValue> meta;
        meta["general.architecture"] = std::string("cxxstage");
        meta["general.name"] = std::string("CXX-TwoStageGPT");
        meta["cxxstage.vocab_size"] = uint64_t(cfg.vocab_size);
        meta["cxxstage.block_size"] = uint64_t(cfg.block_size);
        meta["cxxstage.n_embd"] = uint64_t(cfg.n_embd);
        meta["cxxstage.n_head"] = uint64_t(cfg.n_head);
        meta["cxxstage.N"] = uint64_t(cfg.N);

        save_gguf_multi(tensors, gguf_path, meta);
        std::cout << "Model saved to " << gguf_path << " (" << tensors.size() << " tensors)" << std::endl;
    }

    static TwoStageConfig load_config(const std::string &path)
    {
        auto meta = gguf_read_metadata(path);
        TwoStageConfig cfg;

        auto get_u64 = [&](const std::string &key, size_t &target)
        {
            auto it = meta.find(key);
            if (it != meta.end())
            {
                if (std::holds_alternative<uint64_t>(it->second))
                    target = (size_t)std::get<uint64_t>(it->second);
                else if (std::holds_alternative<int64_t>(it->second))
                    target = (size_t)std::get<int64_t>(it->second);
            }
        };

        get_u64("cxxstage.vocab_size", cfg.vocab_size);
        get_u64("cxxstage.block_size", cfg.block_size);
        get_u64("cxxstage.n_embd", cfg.n_embd);
        get_u64("cxxstage.n_head", cfg.n_head);
        get_u64("cxxstage.N", cfg.N);
        return cfg;
    }

    void load(const std::string &path)
    {
        auto meta = gguf_read_metadata(path);

        auto it_arch = meta.find("general.architecture");
        if (it_arch != meta.end() && std::holds_alternative<std::string>(it_arch->second))
        {
            std::string arch = std::get<std::string>(it_arch->second);
            if (arch != "cxxstage")
                std::cerr << "Warning: model architecture is '" << arch << "', expected 'cxxstage'" << std::endl;
        }

        auto tensors = load_gguf_multi<float>(path);
        auto named = named_parameters();

        size_t loaded = 0;
        for (auto &[name, param] : named)
        {
            if (tensors.count(name))
            {
                param->data = CudaTensor<float>::fromTensor(tensors[name]);
                loaded++;
            }
            else
            {
                std::cerr << "Warning: missing tensor '" << name << "'" << std::endl;
            }
        }
        std::cout << "Model loaded from " << path << " (" << loaded << " tensors)" << std::endl;
    }

    std::vector<int> generate(const std::vector<int> &prompt, size_t max_new_tokens,
                              int eos_id, float temperature, int top_k, RNG &rng)
    {
        std::vector<int> ids = prompt;

        for (size_t step = 0; step < max_new_tokens; step++)
        {
            size_t context_len = std::min(ids.size(), cfg.block_size);
            size_t offset = ids.size() - context_len;

            std::vector<int> x(context_len);
            for (size_t i = 0; i < context_len; i++)
                x[i] = ids[offset + i];

            auto tok_emb = wte.forward(x);
            CudaTensor<float> hid({context_len, cfg.n_embd});
            hid = blk.forward(tok_emb, 1, context_len, cfg.n_embd, cfg.n_head, head_dim);

            CudaTensor<float> Wt({cfg.n_embd, cfg.vocab_size});
            TensorN::cuda::transpose(wte.weight.data, Wt);
            CudaTensor<float> logits({context_len, cfg.vocab_size});
            TensorN::cuda::matmul(hid, Wt, logits);

            auto logits_cpu = logits.toTensor();
            size_t last_row = (context_len - 1) * cfg.vocab_size;

            std::vector<std::pair<float, int>> top_v;
            top_v.reserve(cfg.vocab_size);
            float max_logit = -1e30f;
            for (size_t v = 0; v < cfg.vocab_size; v++)
            {
                float val = logits_cpu[last_row + v];
                if (temperature > 0.01f)
                    val /= temperature;
                if (val > max_logit)
                    max_logit = val;
                top_v.push_back({val, (int)v});
            }

            if (top_k > 0 && top_k < (int)top_v.size())
            {
                std::partial_sort(top_v.begin(), top_v.begin() + top_k, top_v.end(),
                                  [](const auto &a, const auto &b)
                                  { return a.first > b.first; });
                top_v.resize(top_k);
            }

            int next_token;
            if (temperature < 0.01f)
            {
                float best = -1e30f;
                int best_id = 0;
                for (auto &p : top_v)
                {
                    if (p.first > best)
                    {
                        best = p.first;
                        best_id = p.second;
                    }
                }
                next_token = best_id;
            }
            else
            {
                float sum_exp = 0.0f;
                for (auto &p : top_v)
                {
                    float exp_val = std::exp(p.first - max_logit);
                    p.first = exp_val;
                    sum_exp += exp_val;
                }
                float r = (float)rng.randint(0, 1000000) / 1000000.0f;
                float cum = 0.0f;
                int chosen = top_v[0].second;
                for (auto &p : top_v)
                {
                    cum += p.first / sum_exp;
                    if (r <= cum)
                    {
                        chosen = p.second;
                        break;
                    }
                }
                next_token = chosen;
            }

            if (next_token == eos_id)
                break;
            ids.push_back(next_token);
        }
        return ids;
    }
};

inline void two_stage_numerical_grad_check(TwoStageGPT &model,
                                           const std::vector<int> &x, const std::vector<int> &y,
                                           size_t batch_size, size_t block_size)
{
    auto params = model.parameters();
    auto logits = model.forward(x, batch_size, block_size);
    auto [loss0, dlogits] = cross_entropy_loss(logits, y);
    model.zero_grad();
    model.backward(dlogits);

    std::cout << "\n=== Numerical Gradient Check (TwoStageGPT) ===" << std::endl;
    std::cout << "loss = " << loss0 << std::endl;

    float eps = 1e-3f;
    int check_count = std::min((int)params.size(), 6);

    for (int idx = 0; idx < check_count; idx++)
    {
        auto *p = params[idx];
        size_t n = p->data.size();
        int check_elems = std::min((int)n, 3);

        Tensor<float> cpu_data = p->data.toTensor();
        Tensor<float> cpu_grad = p->grad.toTensor();

        for (int e = 0; e < check_elems; e++)
        {
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
    std::cout << "================================\n"
              << std::endl;
}

#endif
