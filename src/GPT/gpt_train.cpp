#include "gpt_gpt.hpp"
#include "dataset.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdint>
#include <clocale>
#include <csignal>
#include <cstdlib>
#include <atomic>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

static std::atomic<bool> g_interrupted{false};

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        g_interrupted = true;
        return TRUE;
    }
    return FALSE;
}
#else
extern "C" void sigint_handler(int) {
    g_interrupted = true;
}
#endif

static constexpr const char* CHECKPOINT_FILE = "models/gpt_checkpoint.gguf";
static constexpr const char* MODEL_FILE = "models/gpt_model.gguf";

void save_checkpoint(const GPT& model, const AdamW& optim, int step) {
    std::cout << "\nSaving checkpoint at step " << step << "..." << std::endl;

    std::vector<std::pair<std::string, Tensor<float>>> tensors;

    auto named = const_cast<GPT&>(model).named_parameters();
    for (auto& [name, param] : named) {
        tensors.push_back({"model." + name, param->data.toTensor()});
    }

    auto opt_state = optim.named_state();
    for (auto& [name, tensor] : opt_state) {
        tensors.push_back({"optimizer." + name, tensor});
    }

    std::unordered_map<std::string, GGUFMetadataValue> meta;
    meta["general.architecture"] = std::string("cxxgpt_checkpoint");
    meta["checkpoint.step"] = int32_t(step);
    meta["optimizer.learning_rate"] = optim.learning_rate();
    meta["optimizer.beta1"] = optim.get_beta1();
    meta["optimizer.beta2"] = optim.get_beta2();
    meta["optimizer.weight_decay"] = optim.get_weight_decay();

    save_gguf_multi(tensors, CHECKPOINT_FILE, meta);
    std::cout << "Checkpoint saved to " << CHECKPOINT_FILE << std::endl;
}

bool load_checkpoint(GPT& model, AdamW& optim, int& step) {
    if (!std::filesystem::exists(CHECKPOINT_FILE)) {
        return false;
    }

    try {
        auto meta = gguf_read_metadata(CHECKPOINT_FILE);

        auto it_step = meta.find("checkpoint.step");
        if (it_step != meta.end() && std::holds_alternative<int32_t>(it_step->second)) {
            step = std::get<int32_t>(it_step->second);
        }

        auto tensors = load_gguf_multi<float>(CHECKPOINT_FILE);

        auto named = model.named_parameters();
        for (auto& [name, param] : named) {
            std::string key = "model." + name;
            if (tensors.count(key)) {
                param->data = CudaTensor<float>::fromTensor(tensors[key]);
            }
        }

        std::unordered_map<std::string, Tensor<float>> opt_state;
        for (auto& [key, tensor] : tensors) {
            if (key.substr(0, 10) == "optimizer.") {
                opt_state[key.substr(10)] = tensor;
            }
        }
        optim.load_state(opt_state, step);

        std::cout << "Loaded checkpoint from step " << step << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load checkpoint: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::setlocale(LC_ALL, "en_US.UTF-8");
    try
    {
    std::cout << "Loading tokenizer..." << std::endl;
    BPETokenizer tokenizer;
    if (!tokenizer.load("tokenizer/tokenizer.json"))
    {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;

    Dataset train_ds;

    std::cout << "Loading dataset..." << std::endl;
    // Prefer the preprocessed binary "big block" (BOS/EOS wrapped, flat).
    if (!train_ds.load_bin("dataset/pretrain_t2t_mini.bin"))
    {
        if (!train_ds.load_jsonl("dataset/pretrain_t2t_mini.jsonl", tokenizer))
        {
            std::cerr << "Failed to load training data!" << std::endl;
            return 1;
        }
    }

    std::cout << "Train: " << train_ds.num_docs() << " docs, "
              << train_ds.num_tokens() << " tokens" << std::endl;

    GPTConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size();
    cfg.block_size = 256;
    cfg.n_embd = 512;
    cfg.n_layer = 8;

    RNG rng(42);
    GPT model(cfg, rng);
    std::cout << "Model parameters: " << model.total_params() << std::endl;

    auto params = model.parameters();
    AdamW optim(params, 3e-4f, 0.9f, 0.999f, 0.01f);

    auto compute_grad_norm = [&params]() {
        float total = 0.0f;
        for (auto* p : params)
            total += gpt_cuda::grad_norm_squared(p->grad);
        return std::sqrt(total);
    };

    size_t batch_size = 8;
    int num_epochs = 3;
    int eval_interval = 500;
    int eval_iters = 10;

    size_t tokens_per_step = batch_size * cfg.block_size;
    size_t steps_per_epoch = train_ds.num_tokens() / tokens_per_step;
    int max_steps = num_epochs * (int)steps_per_epoch;

    // Optional CLI overrides: gpt_train.exe [epochs] [batch_size]
    if (argc >= 2) num_epochs = std::atoi(argv[1]);
    if (argc >= 3) batch_size = (size_t)(std::max)(1, std::atoi(argv[2]));
    if (argc >= 2) max_steps = num_epochs * (int)(train_ds.num_tokens() / (batch_size * cfg.block_size));
    std::cout << "Epochs=" << num_epochs << " steps=" << max_steps
              << " batch_size=" << batch_size << std::endl;

    int start_step = 0;

#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, sigint_handler);
#endif

    std::cout << "\nStarting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = start_step; step < max_steps; step++)
    {
        if (g_interrupted) {
            std::cout << "\n\nUser interrupted! Saving checkpoint..." << std::endl;
            save_checkpoint(model, optim, step);
            std::cout << "Checkpoint saved. You can resume later with the same command." << std::endl;
            return 0;
        }

        auto [x, y] = train_ds.next_batch(batch_size, cfg.block_size);

        auto logits = model.forward(x, batch_size, cfg.block_size);
        auto [loss, dlogits] = cross_entropy_loss(logits, y);

        if (step == 0) {
            std::cout << "Initial forward loss (raw): " << loss << std::endl;
            numerical_grad_check(model, x, y, batch_size, cfg.block_size);
        }

        model.zero_grad();
        model.backward(dlogits);

        if (step % eval_interval == 0 || step == max_steps - 1)
        {
            float grad_norm = compute_grad_norm();
            float train_loss = estimate_loss(model, train_ds,
                                             batch_size, cfg.block_size, rng, eval_iters);

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            std::cout << "step " << std::setw(4) << step
                      << " | train loss " << train_loss
                      << " | grad_norm " << grad_norm
                      << " | time " << elapsed << "s" << std::endl;

            if (step > 0) {
                save_checkpoint(model, optim, step);
            }
        }

        optim.step();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\nTraining completed in " << total_sec << "s" << std::endl;

    model.save(MODEL_FILE);

    if (std::filesystem::exists(CHECKPOINT_FILE)) {
        std::filesystem::remove(CHECKPOINT_FILE);
    }

    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        std::cerr << "Try reducing batch_size, block_size, n_embd, or n_layer." << std::endl;
        return 1;
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
