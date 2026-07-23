#include "LinearAttention/linear_gpt.hpp"
#include "dataset.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdint>
#include <clocale>
#include <csignal>
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

static constexpr const char* CHECKPOINT_FILE = "linear_checkpoint.bin";

void save_checkpoint(const LinearGPT& model, const AdamW& optim, int step) {
    std::cout << "\nSaving checkpoint at step " << step << "..." << std::endl;

    std::ofstream f(CHECKPOINT_FILE, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open checkpoint for writing!" << std::endl;
        return;
    }

    constexpr uint32_t magic = 0x4C434B50;
    constexpr uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    int32_t s = static_cast<int32_t>(step);
    f.write(reinterpret_cast<const char*>(&s), sizeof(int32_t));

    auto params = const_cast<LinearGPT&>(model).parameters();
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

    optim.save(f);

    std::cout << "Checkpoint saved to " << CHECKPOINT_FILE << std::endl;
}

bool load_checkpoint(LinearGPT& model, AdamW& optim, int& step) {
    if (!std::filesystem::exists(CHECKPOINT_FILE)) {
        return false;
    }

    std::ifstream f(CHECKPOINT_FILE, std::ios::binary);
    if (!f.is_open()) return false;

    uint32_t magic = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    if (magic != 0x4C434B50 || version != 1) {
        std::cerr << "Invalid checkpoint file!" << std::endl;
        return false;
    }

    int32_t s = 0;
    f.read(reinterpret_cast<char*>(&s), sizeof(int32_t));
    step = static_cast<int>(s);

    uint32_t np = 0;
    f.read(reinterpret_cast<char*>(&np), sizeof(uint32_t));

    auto params = model.parameters();
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

    optim.load(f);

    std::cout << "Loaded checkpoint from step " << step << std::endl;
    return true;
}

float estimate_loss_linear(LinearGPT& model, Dataset& ds,
    size_t batch_size, size_t block_size, RNG& rng, int num_batches)
{
    float total_loss = 0.0f;
    for (int i = 0; i < num_batches; i++) {
        auto [x, y] = ds.next_batch(batch_size, block_size);
        auto logits = model.forward(x, batch_size, block_size);
        auto [loss, dl] = cross_entropy_loss(logits, y);
        total_loss += loss;
    }
    return total_loss / num_batches;
}

int main()
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

    Dataset train_ds, val_ds;

    std::cout << "Loading dataset..." << std::endl;
    if (!train_ds.load_jsonl("dataset/train.jsonl", tokenizer))
    {
        std::cerr << "Failed to load training data!" << std::endl;
        return 1;
    }
    val_ds.load_jsonl("dataset/val.jsonl", tokenizer);

    std::cout << "Train: " << train_ds.num_docs() << " docs, "
              << train_ds.num_tokens() << " tokens" << std::endl;

    LinearGPTConfig cfg;
    cfg.vocab_size = tokenizer.vocab_size();
    cfg.block_size = 64;
    cfg.n_embd = 128;
    cfg.n_layer = 6;

    size_t D = next_power_of_2(cfg.n_embd);
    std::cout << "Embedding dim: " << cfg.n_embd
              << ", Feature dim (D): " << D << std::endl;

    RNG rng(42);
    LinearGPT model(cfg, rng);
    std::cout << "Model parameters: " << model.total_params() << std::endl;

    auto params = model.parameters();
    AdamW optim(params, 1e-3f, 0.9f, 0.999f, 0.01f);

    auto compute_grad_norm = [&params]() {
        float total = 0.0f;
        for (auto* p : params)
            total += gpt_cuda::grad_norm_squared(p->grad);
        return std::sqrt(total);
    };

    size_t batch_size = 8;
    int max_steps = 2769;
    int eval_interval = 200;
    int eval_iters = 10;

    int start_step = 0;

#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, sigint_handler);
#endif

    std::cout << "\n=== Training LinearGPT (Hadamard+Exp Kernel) ===" << std::endl;
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
            linear_numerical_grad_check(model, x, y, batch_size, cfg.block_size);
        }

        model.zero_grad();
        model.backward(dlogits);

        if (step % eval_interval == 0 || step == max_steps - 1)
        {
            float grad_norm = compute_grad_norm();
            float train_loss = estimate_loss_linear(model, train_ds,
                batch_size, cfg.block_size, rng, eval_iters);
            float val_loss = estimate_loss_linear(model, val_ds,
                batch_size, cfg.block_size, rng, eval_iters);

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            std::cout << "step " << std::setw(4) << step
                      << " | train loss " << train_loss
                      << " | val loss " << val_loss
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

    model.save("linear_model.bin");

    if (std::filesystem::exists(CHECKPOINT_FILE)) {
        std::filesystem::remove(CHECKPOINT_FILE);
    }

    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
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
