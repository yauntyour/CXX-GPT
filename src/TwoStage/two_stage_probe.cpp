// Forward-pass probe: loads the TwoStageGPT checkpoint and dumps the
// logits on a fixed token span from the dataset bin, so the C++ model
// can be cross-checked against the PyTorch reference (model.py) with
// identical weights and inputs.
#include "TwoStage/two_stage_gpt.hpp"
#include "dataset.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <clocale>
#ifdef _WIN32
#include <windows.h>
#endif

static constexpr const char *CHECKPOINT_FILE = "models/two_stage_checkpoint.gguf";

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::setlocale(LC_ALL, "en_US.UTF-8");

    size_t B = 2, S = 256;
    if (argc >= 2) B = (size_t)std::atoll(argv[1]);
    if (argc >= 3) S = (size_t)std::atoll(argv[2]);
    bool use_tf32 = (argc >= 4) ? (std::atoi(argv[3]) != 0) : false;

    try
    {
        std::cout << "Loading tokenizer..." << std::endl;
        BPETokenizer tokenizer;
        if (!tokenizer.load("tokenizer/tokenizer.json"))
        {
            std::cerr << "Failed to load tokenizer!" << std::endl;
            return 1;
        }
        size_t vocab = tokenizer.vocab_size();
        std::cout << "Vocabulary size: " << vocab << std::endl;

        if (use_tf32)
            TensorN::cuda::set_tf32(true);
        std::cout << "TF32 GEMMs: " << (use_tf32 ? "ON" : "OFF") << std::endl;

        TwoStageConfig cfg;
        cfg.vocab_size = vocab;
        cfg.block_size = 256;
        cfg.n_embd = 256;
        cfg.n_head = 8;
        cfg.N = 6;

        RNG rng(42);
        TwoStageGPT model(cfg, rng);

        auto tensors = load_gguf_multi<float>(CHECKPOINT_FILE);
        auto named = model.named_parameters();
        size_t loaded = 0;
        for (auto &[name, param] : named)
        {
            std::string key = "model." + name;
            if (tensors.count(key))
            {
                param->data = CudaTensor<float>::fromTensor(tensors[key]);
                loaded++;
            }
            else
            {
                std::cerr << "Missing tensor: " << key << std::endl;
            }
        }
        std::cout << "Loaded " << loaded << " tensors from " << CHECKPOINT_FILE << std::endl;

        // Read the first B*S+1 tokens straight from the dataset bin.
        std::ifstream f("dataset/pretrain_t2t_mini.bin", std::ios::binary);
        if (!f.is_open())
        {
            std::cerr << "Cannot open dataset bin" << std::endl;
            return 1;
        }
        f.seekg(sizeof(DataBlockHeader));
        std::vector<uint16_t> raw(B * S + 1);
        f.read(reinterpret_cast<char *>(raw.data()), raw.size() * sizeof(uint16_t));
        if ((size_t)f.gcount() != raw.size() * sizeof(uint16_t))
        {
            std::cerr << "Short read from dataset bin" << std::endl;
            return 1;
        }

        std::vector<int> x(B * S), y(B * S);
        for (size_t i = 0; i < B * S; i++)
        {
            x[i] = raw[i];
            y[i] = raw[i + 1];
        }

        auto logits = model.forward(x, B, S);
        auto [loss, dlogits] = cross_entropy_loss(logits, y);
        std::cout << "CE loss on probe batch = " << loss << std::endl;

        auto lg = logits.toTensor();
        std::ofstream out("probe_logits.bin", std::ios::binary);
        out.write(reinterpret_cast<const char *>(lg.data->data()),
                  lg.data->size() * sizeof(float));
        std::ofstream tout("probe_tokens.bin", std::ios::binary);
        tout.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(int));
        std::ofstream tout2("probe_targets.bin", std::ios::binary);
        tout2.write(reinterpret_cast<const char *>(y.data()), y.size() * sizeof(int));
        std::cout << "Dumped " << lg.data->size() << " logits ("
                  << B * S << " x " << cfg.vocab_size << ")" << std::endl;

        float mx = 0.0f;
        for (size_t i = 0; i < lg.data->size(); i++)
            mx = (std::max)(mx, std::abs((*lg.data)[i]));
        std::cout << "max |logit| = " << mx << std::endl;

        // Per-group gradient norms on the probe batch (diagnostic).
        model.zero_grad();
        model.backward(dlogits);
        auto groups = model.named_parameters();
        std::vector<std::pair<std::string, float>> norms;
        for (auto &[name, param] : groups)
        {
            float gn = std::sqrt(gpt_cuda::grad_norm_squared(param->grad));
            norms.push_back({name, gn});
        }
        std::sort(norms.begin(), norms.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });
        std::cout << "\nPer-group grad norms (probe batch):" << std::endl;
        for (auto &[name, gn] : norms)
            std::cout << std::setw(24) << name << "  " << gn << std::endl;

        float total_sq = 0.0f;
        for (auto &[name, gn] : norms)
            total_sq += gn * gn;
        std::cout << "total grad_norm = " << std::sqrt(total_sq) << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
