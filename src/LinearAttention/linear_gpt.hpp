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

inline CudaTensor<float> compute_phi_polynomial(const CudaTensor<float>& x, size_t D, size_t E)
{
    size_t N = x.shape()[0];
    // 计算 x 的平方和：norm_sq = sum(x^2, axis=1)
    CudaTensor<float> x_sq({N, E});
    TensorN::cuda::multiply(x, x, x_sq);
    CudaTensor<float> norm_sq = TensorN::cuda::sum_axis(x_sq, 1); // 形状 [N]
    // 计算 x^2 + 1
    CudaTensor<float> norm_sq_plus1({N});
    TensorN::cuda::add_scalar(norm_sq, 1.0f, norm_sq_plus1);
    // 构造 phi，形状 [N, D]
    CudaTensor<float> phi({N, D});
    // 将 phi 初始化为零
    TensorN::cuda::multiply_scalar(phi, 0.0f, phi);
    // 设置第一个分量：phi[:, 0] = norm_sq_plus1
    // 我们将 norm_sq_plus1 视为列向量 [N, 1]，与全 1 行向量 [1, D] 相乘，但只取第一列
    // 更高效的方法是直接赋值，但我们没有切片操作。
    // 作为临时解决方案，我们使用循环填充第一列。
    // 注意：这会在 GPU 上产生同步开销，但暂时可以接受。
    // 将 norm_sq_plus1 复制到主机，然后逐元素设置。
    auto norm_sq_plus1_cpu = norm_sq_plus1.toTensor();
    auto phi_cpu = phi.toTensor();
    for (size_t i = 0; i < N; ++i) {
        phi_cpu[i * D + 0] = norm_sq_plus1_cpu[i];
    }
    // 对于 i = 1..E-1，设置 phi[:, i] = norm_sq_plus1 * x[:, i-1]
    auto x_cpu = x.toTensor();
    for (size_t i = 0; i < N; ++i) {
        float scale = norm_sq_plus1_cpu[i];
        for (size_t j = 1; j < E; ++j) {
            phi_cpu[i * D + j] = scale * x_cpu[i * E + (j-1)];
        }
    }
    // 将结果复制回 GPU
    phi.copyFromHost(phi_cpu.data->data(), N * D);
    return phi;
}

inline void backward_phi_polynomial(const CudaTensor<float>& dphi,
    const CudaTensor<float>& x, const CudaTensor<float>& norm_sq_plus1,
    size_t D, size_t E, CudaTensor<float>& dx)
{
    size_t N = dphi.shape()[0];
    // 将数据复制到主机
    auto dphi_cpu = dphi.toTensor();
    auto x_cpu = x.toTensor();
    auto norm_sq_plus1_cpu = norm_sq_plus1.toTensor();
    auto dx_cpu = dx.toTensor();
    // 初始化 dx 为零
    for (size_t i = 0; i < N * E; ++i) dx_cpu[i] = 0.0f;
    // 计算梯度
    for (size_t i = 0; i < N; ++i) {
        float B = norm_sq_plus1_cpu[i];
        // 来自 φ0 的梯度
        float dphi0 = dphi_cpu[i * D + 0];
        for (size_t j = 0; j < E; ++j) {
            // ∂φ0/∂x_j = 2 * x_j
            dx_cpu[i * E + j] += dphi0 * 2.0f * x_cpu[i * E + j];
        }
        // 来自 φk (k=1..E) 的梯度
        for (size_t k = 1; k <= E; ++k) {
            float dphik = dphi_cpu[i * D + k];
            // φk = B * x_{k-1}
            // ∂φk/∂x_j = 2 * x_j * x_{k-1} + B * δ_{j, k-1}
            float x_km1 = x_cpu[i * E + (k-1)];
            for (size_t j = 0; j < E; ++j) {
                float grad = 2.0f * x_cpu[i * E + j] * x_km1;
                if (j == k-1) grad += B;
                dx_cpu[i * E + j] += dphik * grad;
            }
        }
    }
    // 将结果复制回 GPU
    dx.copyFromHost(dx_cpu.data->data(), N * E);
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
        CudaTensor<float> Q_norm_sq_plus1_cache, K_norm_sq_plus1_cache;
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

            Q_phi_cache = compute_phi_polynomial(Q_cache, D, E);
            K_phi_cache = compute_phi_polynomial(K_cache, D, E);
            // 我们需要计算 norm_sq_plus1 用于反向传播，但 compute_phi_polynomial 已经计算了它，但没有返回。
            // 我们需要修改 compute_phi_polynomial 以返回 norm_sq_plus1。
            // 暂时，我们重新计算它，尽管效率低下。
            // 在下一步中，我们将修改函数以返回 norm_sq_plus1。
            // 为了快速实现，我们在这里重新计算。
            CudaTensor<float> Q_sq({Q_cache.shape()[0], E});
            TensorN::cuda::multiply(Q_cache, Q_cache, Q_sq);
            Q_norm_sq_plus1_cache = TensorN::cuda::sum_axis(Q_sq, 1);
            TensorN::cuda::add_scalar(Q_norm_sq_plus1_cache, 1.0f, Q_norm_sq_plus1_cache);
            CudaTensor<float> K_sq({K_cache.shape()[0], E});
            TensorN::cuda::multiply(K_cache, K_cache, K_sq);
            K_norm_sq_plus1_cache = TensorN::cuda::sum_axis(K_sq, 1);
            TensorN::cuda::add_scalar(K_norm_sq_plus1_cache, 1.0f, K_norm_sq_plus1_cache);

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
            backward_phi_polynomial(dQ_phi, Q_cache, Q_norm_sq_plus1_cache, D, E, dQ);
            CudaTensor<float> dK({dim, E});
            backward_phi_polynomial(dK_phi, K_cache, K_norm_sq_plus1_cache, D, E, dK);

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
        auto named = const_cast<LinearGPT*>(this)->named_parameters();
        for (auto& [name, param] : named) {
            tensors.push_back({name, param->data.toTensor()});
        }

        std::unordered_map<std::string, GGUFMetadataValue> meta;
        meta["general.architecture"] = std::string("cxxlinear");
        meta["general.name"] = std::string("CXX-LinearGPT");
        meta["cxxlinear.vocab_size"] = uint64_t(cfg.vocab_size);
        meta["cxxlinear.block_size"] = uint64_t(cfg.block_size);
        meta["cxxlinear.n_embd"] = uint64_t(cfg.n_embd);
        meta["cxxlinear.n_layer"] = uint64_t(cfg.n_layer);
        meta["cxxlinear.layer_norm_epsilon"] = cfg.ln_eps;

        save_gguf_multi(tensors, gguf_path, meta);
        std::cout << "Model saved to " << gguf_path << " (" << tensors.size() << " tensors)" << std::endl;
    }

    void load(const std::string& path) {
        auto meta = gguf_read_metadata(path);

        auto it_arch = meta.find("general.architecture");
        if (it_arch != meta.end() && std::holds_alternative<std::string>(it_arch->second)) {
            std::string arch = std::get<std::string>(it_arch->second);
            if (arch != "cxxlinear") {
                std::cerr << "Warning: model architecture is '" << arch << "', expected 'cxxlinear'" << std::endl;
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
