// Kernel-level unit test for TwoStage fused kernels.
// Compares CUDA kernels against straightforward CPU reference math.
#include "TwoStage/two_stage_gpt.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>

using namespace TensorN;

static std::mt19937 rgen(1234);

static float frand() { return (float)rgen() / (float)rgen.max() * 2.0f - 1.0f; }

static float phi_elu(float x) { return x > 0.0f ? x + 1.0f : std::exp(x); }
static float phi_elu_der(float x) { return x > 0.0f ? 1.0f : std::exp(x); }

static size_t idx3(size_t b, size_t s, size_t h, size_t c, size_t S, size_t H, size_t Dh)
{
    return ((b * S + s) * H + h) * Dh + c;
}

static float ref_mha_fwd(const std::vector<float> &Q, const std::vector<float> &K,
                         const std::vector<float> &V, size_t b, size_t s, size_t h,
                         size_t c, size_t B, size_t S, size_t H, size_t Dh)
{
    float scale = 1.0f / std::sqrt((float)Dh);
    std::vector<float> scores(s + 1);
    float m = -1e30f;
    for (size_t t = 0; t <= s; t++)
    {
        float acc = 0.0f;
        for (size_t d = 0; d < Dh; d++)
            acc += Q[idx3(b, s, h, d, S, H, Dh)] * K[idx3(b, t, h, d, S, H, Dh)];
        scores[t] = acc * scale;
        m = std::max(m, scores[t]);
    }
    float l = 0.0f;
    for (size_t t = 0; t <= s; t++)
        l += std::exp(scores[t] - m);
    float out = 0.0f;
    for (size_t t = 0; t <= s; t++)
        out += std::exp(scores[t] - m) / l * V[idx3(b, t, h, c, S, H, Dh)];
    return out;
}

static void ref_mha_bwd(const std::vector<float> &Q, const std::vector<float> &K,
                        const std::vector<float> &V, const std::vector<float> &O,
                        const std::vector<float> &dO,
                        std::vector<float> &dQ, std::vector<float> &dK, std::vector<float> &dV,
                        size_t B, size_t S, size_t H, size_t Dh)
{
    float scale = 1.0f / std::sqrt((float)Dh);
    for (size_t b = 0; b < B; b++)
        for (size_t s = 0; s < S; s++)
            for (size_t h = 0; h < H; h++)
            {
                std::vector<float> p(s + 1);
                std::vector<float> score(s + 1);
                float m = -1e30f;
                for (size_t t = 0; t <= s; t++)
                {
                    float acc = 0.0f;
                    for (size_t d = 0; d < Dh; d++)
                        acc += Q[idx3(b, s, h, d, S, H, Dh)] * K[idx3(b, t, h, d, S, H, Dh)];
                    score[t] = acc * scale;
                    m = std::max(m, score[t]);
                }
                float l = 0.0f;
                for (size_t t = 0; t <= s; t++)
                {
                    p[t] = std::exp(score[t] - m);
                    l += p[t];
                }
                float gs = 0.0f; // dO_s . O_s
                for (size_t c = 0; c < Dh; c++)
                    gs += dO[idx3(b, s, h, c, S, H, Dh)] * O[idx3(b, s, h, c, S, H, Dh)];
                for (size_t t = 0; t <= s; t++)
                {
                    float pv = p[t] / l;
                    float ds = 0.0f; // dO_s . v_t
                    for (size_t c = 0; c < Dh; c++)
                        ds += dO[idx3(b, s, h, c, S, H, Dh)] * V[idx3(b, t, h, c, S, H, Dh)];
                    float dscore = pv * (ds - gs);
                    for (size_t c = 0; c < Dh; c++)
                    {
                        dQ[idx3(b, s, h, c, S, H, Dh)] += scale * dscore * K[idx3(b, t, h, c, S, H, Dh)];
                        dK[idx3(b, t, h, c, S, H, Dh)] += scale * dscore * Q[idx3(b, s, h, c, S, H, Dh)];
                        dV[idx3(b, t, h, c, S, H, Dh)] += pv * dO[idx3(b, s, h, c, S, H, Dh)];
                    }
                }
            }
}

static void ref_kernel_attn_fwd(const std::vector<float> &Q, const std::vector<float> &K,
                                const std::vector<float> &V, std::vector<float> &O,
                                std::vector<float> &den, size_t B, size_t S, size_t D, size_t E)
{
    for (size_t b = 0; b < B; b++)
    {
        std::vector<float> z(D, 0.0f);
        std::vector<float> state(E * D, 0.0f);
        for (size_t t = 0; t < S; t++)
        {
            size_t idx = b * S + t;
            for (size_t d = 0; d < D; d++)
            {
                z[d] += phi_elu(K[idx * D + d]);
                for (size_t e = 0; e < E; e++)
                    state[e * D + d] += phi_elu(K[idx * D + d]) * V[idx * E + e];
            }
            float dpart = 0.0f;
            for (size_t d = 0; d < D; d++)
                dpart += phi_elu(Q[idx * D + d]) * z[d];
            den[idx] = dpart;
            for (size_t e = 0; e < E; e++)
            {
                float num = 0.0f;
                for (size_t d = 0; d < D; d++)
                    num += phi_elu(Q[idx * D + d]) * state[e * D + d];
                O[idx * E + e] = num / (dpart + 1e-8f);
            }
        }
    }
}

static void ref_kernel_attn_bwd(const std::vector<float> &Q, const std::vector<float> &K,
                                const std::vector<float> &V, const std::vector<float> &O,
                                const std::vector<float> &dO, const std::vector<float> &den,
                                std::vector<float> &dQ, std::vector<float> &dK, std::vector<float> &dV,
                                size_t B, size_t S, size_t D, size_t E)
{
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < S; t++)
        {
            size_t idx = b * S + t;
            // prefix state (forward scan)
            std::vector<float> z(D, 0.0f);
            std::vector<float> state(E * D, 0.0f);
            for (size_t s = 0; s <= t; s++)
            {
                size_t j = b * S + s;
                for (size_t d = 0; d < D; d++)
                {
                    z[d] += phi_elu(K[j * D + d]);
                    for (size_t e = 0; e < E; e++)
                        state[e * D + d] += phi_elu(K[j * D + d]) * V[j * E + e];
                }
            }
            float inv_den = 1.0f / den[idx];
            float dden = 0.0f;
            for (size_t e = 0; e < E; e++)
                dden -= dO[idx * E + e] * O[idx * E + e] * inv_den;
            for (size_t d = 0; d < D; d++)
            {
                float dnum = z[d] * dden;
                for (size_t e = 0; e < E; e++)
                    dnum += state[e * D + d] * (dO[idx * E + e] * inv_den);
                dQ[idx * D + d] = dnum * phi_elu_der(Q[idx * D + d]);
            }
            // suffix state (reverse scan)
            std::vector<float> zs(D, 0.0f);
            std::vector<float> Sst(E * D, 0.0f);
            for (size_t s = S; s-- > t;)
            {
                size_t j = b * S + s;
                float dd = 0.0f;
                for (size_t e = 0; e < E; e++)
                    dd -= dO[j * E + e] * O[j * E + e] / den[j];
                for (size_t d = 0; d < D; d++)
                {
                    zs[d] += phi_elu(Q[j * D + d]) * dd;
                    for (size_t e = 0; e < E; e++)
                        Sst[e * D + d] += phi_elu(Q[j * D + d]) * (dO[j * E + e] / den[j]);
                }
            }
            for (size_t e = 0; e < E; e++)
            {
                float acc = 0.0f;
                for (size_t d = 0; d < D; d++)
                    acc += Sst[e * D + d] * phi_elu(K[idx * D + d]);
                dV[idx * E + e] = acc;
            }
            for (size_t d = 0; d < D; d++)
            {
                float acc = zs[d];
                for (size_t e = 0; e < E; e++)
                    acc += Sst[e * D + d] * V[idx * E + e];
                dK[idx * D + d] = acc * phi_elu_der(K[idx * D + d]);
            }
        }
}

template <typename T>
static std::vector<T> t2v(const Tensor<T> &t)
{
    return *t.data;
}

template <typename T>
static float max_abs_diff(const std::vector<T> &a, const std::vector<T> &b)
{
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

int main()
{
    std::cout << std::unitbuf;
    std::cout << std::scientific << std::setprecision(2);

    // ---------------- Stage 1 ----------------
    {
        size_t B = 2, S = 6, H = 2, Dh = 8, dim = B * S, big = H * Dh;
        size_t n = dim * big;
        std::vector<float> Q(n), K(n), V(n), dO(n);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, V));
        CudaTensor<float> cO({dim, big});
        two_stage_cuda::fused_mha_fwd(cQ, cK, cV, cO, B, S, H, Dh);

        std::vector<float> O = t2v(cO.toTensor());
        std::vector<float> rO(n);
        for (size_t b = 0; b < B; b++)
            for (size_t s = 0; s < S; s++)
                for (size_t h = 0; h < H; h++)
                    for (size_t c = 0; c < Dh; c++)
                        rO[idx3(b, s, h, c, S, H, Dh)] =
                            ref_mha_fwd(Q, K, V, b, s, h, c, B, S, H, Dh);
        std::cout << "stage1 fwd  maxdiff = " << max_abs_diff(O, rO) << std::endl;

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, dO));
        CudaTensor<float> cdQ({dim, big}), cdK({dim, big}), cdV({dim, big});
        two_stage_cuda::fused_mha_bwd(cQ, cK, cV, cO, cdO, cdQ, cdK, cdV, B, S, H, Dh);

        std::vector<float> rQ(n, 0.0f), rK(n, 0.0f), rV(n, 0.0f);
        ref_mha_bwd(Q, K, V, O, dO, rQ, rK, rV, B, S, H, Dh);
        std::cout << "stage1 dQ  maxdiff = " << max_abs_diff(t2v(cdQ.toTensor()), rQ) << std::endl;
        std::cout << "stage1 dK  maxdiff = " << max_abs_diff(t2v(cdK.toTensor()), rK) << std::endl;
        std::cout << "stage1 dV  maxdiff = " << max_abs_diff(t2v(cdV.toTensor()), rV) << std::endl;
    }

    // ---------------- Stage 2 ----------------
    {
        size_t B = 2, S = 6, D = 8, E = 8, dim = B * S;
        std::vector<float> Q(dim * D), K(dim * D), V(dim * E), dO(dim * E);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, V));
        CudaTensor<float> cO({dim, E}), den({dim}), zc({dim, D});
        CudaTensor<float> Sb({B * E, D}), zb({B, D});
        two_stage_cuda::kernel_attn_fwd(cQ, cK, cV, cO, den, zc, Sb, zb, B, S, D, E);

        std::vector<float> O = t2v(cO.toTensor());
        std::vector<float> rO(dim * E), rden(dim);
        ref_kernel_attn_fwd(Q, K, V, rO, rden, B, S, D, E);
        std::cout << "stage2 fwd  maxdiff = " << max_abs_diff(O, rO) << std::endl;
        std::cout << "stage2 den  maxdiff = " << max_abs_diff(t2v(den.toTensor()), rden) << std::endl;

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, dO));
        CudaTensor<float> dnum({dim, E}), dden({dim});
        two_stage_cuda::kernel_attn_bwd_step1(cO, cdO, den, dnum, dden, dim, E);
        std::vector<float> rdnum(dim * E), rdden(dim);
        for (size_t i = 0; i < dim; i++)
        {
            rdden[i] = 0.0f;
            for (size_t e = 0; e < E; e++)
            {
                rdnum[i * E + e] = dO[i * E + e] / rden[i];
                rdden[i] -= dO[i * E + e] * rO[i * E + e] / rden[i];
            }
        }
        std::cout << "stage2 dnum maxdiff = " << max_abs_diff(t2v(dnum.toTensor()), rdnum) << std::endl;
        std::cout << "stage2 dden maxdiff = " << max_abs_diff(t2v(dden.toTensor()), rdden) << std::endl;

        CudaTensor<float> dlq({dim, D}), dlk({dim, D}), dlv({dim, E});
        two_stage_cuda::kernel_attn_bwd_dQ(cK, cV, dnum, dden, zc, cQ, dlq, Sb, B, S, D, E);
        two_stage_cuda::kernel_attn_bwd_dKdV(cQ, cK, cV, dnum, dden, dlk, dlv, B, S, D, E);
        std::vector<float> rQ2(dim * D), rK2(dim * D), rV2(dim * E);
        ref_kernel_attn_bwd(Q, K, V, O, dO, rden, rQ2, rK2, rV2, B, S, D, E);
        std::cout << "stage2 dlq maxdiff = " << max_abs_diff(t2v(dlq.toTensor()), rQ2) << std::endl;
        std::cout << "stage2 dlk maxdiff = " << max_abs_diff(t2v(dlk.toTensor()), rK2) << std::endl;
        std::cout << "stage2 dlv maxdiff = " << max_abs_diff(t2v(dlv.toTensor()), rV2) << std::endl;
    }

    // ---------------- Fallback paths (forced) ----------------
    {
        // S*Dh so large that K/V do not fit in smem -> tiled / warp-per-row paths
        size_t B = 2, S = 300, H = 1, Dh = 128, dim = B * S, big = H * Dh;
        size_t n = dim * big;
        std::vector<float> Q(n), K(n), V(n), dO(n);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, V));
        CudaTensor<float> cO({dim, big});
        two_stage_cuda::fused_mha_fwd(cQ, cK, cV, cO, B, S, H, Dh);
        std::vector<float> O = t2v(cO.toTensor());
        std::vector<float> rO(n);
        for (size_t b = 0; b < B; b++)
            for (size_t s = 0; s < S; s++)
                for (size_t h = 0; h < H; h++)
                    for (size_t c = 0; c < Dh; c++)
                        rO[idx3(b, s, h, c, S, H, Dh)] =
                            ref_mha_fwd(Q, K, V, b, s, h, c, B, S, H, Dh);
        std::cout << "fallback fwd maxdiff = " << max_abs_diff(O, rO) << std::endl;

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, dO));
        CudaTensor<float> cdQ({dim, big}), cdK({dim, big}), cdV({dim, big});
        two_stage_cuda::fused_mha_bwd(cQ, cK, cV, cO, cdO, cdQ, cdK, cdV, B, S, H, Dh);
        std::vector<float> rQ(n, 0.0f), rK(n, 0.0f), rV(n, 0.0f);
        ref_mha_bwd(Q, K, V, O, dO, rQ, rK, rV, B, S, H, Dh);
        std::cout << "fallback dQ  maxdiff = " << max_abs_diff(t2v(cdQ.toTensor()), rQ) << std::endl;
        std::cout << "fallback dK  maxdiff = " << max_abs_diff(t2v(cdK.toTensor()), rK) << std::endl;
        std::cout << "fallback dV  maxdiff = " << max_abs_diff(t2v(cdV.toTensor()), rV) << std::endl;
    }
    {
        // E not dividing 512 forces the flat stage-2 kernels
        size_t B = 2, S = 5, D = 7, E = 6, dim = B * S;
        std::vector<float> Q(dim * D), K(dim * D), V(dim * E), dO(dim * E);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, V));
        CudaTensor<float> cO({dim, E}), den({dim}), zc({dim, D});
        CudaTensor<float> Sb({B * E, D}), zb({B, D});
        two_stage_cuda::kernel_attn_fwd(cQ, cK, cV, cO, den, zc, Sb, zb, B, S, D, E);
        std::vector<float> O = t2v(cO.toTensor());
        std::vector<float> rO(dim * E), rden(dim);
        ref_kernel_attn_fwd(Q, K, V, rO, rden, B, S, D, E);
        std::cout << "flat fwd     maxdiff = " << max_abs_diff(O, rO) << std::endl;
        std::vector<float> rdden(dim);
        for (size_t i = 0; i < dim; i++)
        {
            rdden[i] = 0.0f;
            for (size_t e = 0; e < E; e++)
                rdden[i] -= dO[i * E + e] * rO[i * E + e] / rden[i];
        }

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, dO));
        CudaTensor<float> dnum({dim, E}), dden({dim});
        two_stage_cuda::kernel_attn_bwd_step1(cO, cdO, den, dnum, dden, dim, E);
        CudaTensor<float> dlq({dim, D}), dlk({dim, D}), dlv({dim, E});
        two_stage_cuda::kernel_attn_bwd_dQ(cK, cV, dnum, dden, zc, cQ, dlq, Sb, B, S, D, E);
        two_stage_cuda::kernel_attn_bwd_dKdV(cQ, cK, cV, dnum, dden, dlk, dlv, B, S, D, E);
        std::vector<float> rQ2(dim * D), rK2(dim * D), rV2(dim * E);
        ref_kernel_attn_bwd(Q, K, V, O, dO, rden, rQ2, rK2, rV2, B, S, D, E);
        std::cout << "flat dlq     maxdiff = " << max_abs_diff(t2v(dlq.toTensor()), rQ2) << std::endl;
        std::cout << "flat dlk     maxdiff = " << max_abs_diff(t2v(dlk.toTensor()), rK2) << std::endl;
        std::cout << "flat dlv     maxdiff = " << max_abs_diff(t2v(dlv.toTensor()), rV2) << std::endl;
        std::cout << "flat dden    maxdiff = " << max_abs_diff(t2v(dden.toTensor()), rdden) << std::endl;
    }

    // ---------------- Large-S stage 1 (tiled fwd / scratch bwd) ----------------
    {
        size_t B = 2, S = 256, H = 2, Dh = 64, dim = B * S, big = H * Dh;
        size_t n = dim * big;
        std::vector<float> Q(n), K(n), V(n), dO(n);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, V));
        CudaTensor<float> cO({dim, big});
        two_stage_cuda::fused_mha_fwd(cQ, cK, cV, cO, B, S, H, Dh);

        std::vector<float> rO(n);
        for (size_t b = 0; b < B; b++)
            for (size_t s = 0; s < S; s++)
                for (size_t h = 0; h < H; h++)
                    for (size_t c = 0; c < Dh; c++)
                        rO[idx3(b, s, h, c, S, H, Dh)] =
                            ref_mha_fwd(Q, K, V, b, s, h, c, B, S, H, Dh);
        std::cout << "big S=256 fwd maxdiff = " << max_abs_diff(t2v(cO.toTensor()), rO) << std::endl;

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, big}, dO));
        CudaTensor<float> cdQ({dim, big}), cdK({dim, big}), cdV({dim, big});
        two_stage_cuda::fused_mha_bwd(cQ, cK, cV, cO, cdO, cdQ, cdK, cdV, B, S, H, Dh);
        std::vector<float> rQ(n, 0.0f), rK(n, 0.0f), rV(n, 0.0f);
        ref_mha_bwd(Q, K, V, t2v(cO.toTensor()), dO, rQ, rK, rV, B, S, H, Dh);
        std::cout << "big S=256 dQ  maxdiff = " << max_abs_diff(t2v(cdQ.toTensor()), rQ) << std::endl;
        std::cout << "big S=256 dK  maxdiff = " << max_abs_diff(t2v(cdK.toTensor()), rK) << std::endl;
        std::cout << "big S=256 dV  maxdiff = " << max_abs_diff(t2v(cdV.toTensor()), rV) << std::endl;
    }
    {
        // Chunked stage-2 path (E=D=256, TS_S2_CHUNK=4)
        size_t B = 2, S = 32, D = 256, E = 256, dim = B * S;
        std::vector<float> Q(dim * D), K(dim * D), V(dim * E), dO(dim * E);
        for (auto &x : Q) x = frand();
        for (auto &x : K) x = frand();
        for (auto &x : V) x = frand();
        for (auto &x : dO) x = frand();

        CudaTensor<float> cQ = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, Q));
        CudaTensor<float> cK = CudaTensor<float>::fromTensor(Tensor<float>({dim, D}, K));
        CudaTensor<float> cV = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, V));
        CudaTensor<float> cO({dim, E}), den({dim}), zc({dim, D});
        CudaTensor<float> Sb({B * E, D}), zb({B, D});
        two_stage_cuda::kernel_attn_fwd(cQ, cK, cV, cO, den, zc, Sb, zb, B, S, D, E);
        std::vector<float> rO(dim * E), rden(dim);
        ref_kernel_attn_fwd(Q, K, V, rO, rden, B, S, D, E);
        std::cout << "chunk fwd    maxdiff = " << max_abs_diff(t2v(cO.toTensor()), rO) << std::endl;
        std::cout << "chunk den    maxdiff = " << max_abs_diff(t2v(den.toTensor()), rden) << std::endl;

        CudaTensor<float> cdO = CudaTensor<float>::fromTensor(Tensor<float>({dim, E}, dO));
        CudaTensor<float> dnum({dim, E}), dden({dim});
        two_stage_cuda::kernel_attn_bwd_step1(cO, cdO, den, dnum, dden, dim, E);
        CudaTensor<float> dlq({dim, D}), dlk({dim, D}), dlv({dim, E});
        two_stage_cuda::kernel_attn_bwd_dQ(cK, cV, dnum, dden, zc, cQ, dlq, Sb, B, S, D, E);
        two_stage_cuda::kernel_attn_bwd_dKdV(cQ, cK, cV, dnum, dden, dlk, dlv, B, S, D, E);
        std::vector<float> rQ2(dim * D), rK2(dim * D), rV2(dim * E);
        ref_kernel_attn_bwd(Q, K, V, rO, dO, rden, rQ2, rK2, rV2, B, S, D, E);
        std::cout << "chunk dlq    maxdiff = " << max_abs_diff(t2v(dlq.toTensor()), rQ2) << std::endl;
        std::cout << "chunk dlk    maxdiff = " << max_abs_diff(t2v(dlk.toTensor()), rK2) << std::endl;
        std::cout << "chunk dlv    maxdiff = " << max_abs_diff(t2v(dlv.toTensor()), rV2) << std::endl;
    }

    std::cout << "done" << std::endl;
    return 0;
}
