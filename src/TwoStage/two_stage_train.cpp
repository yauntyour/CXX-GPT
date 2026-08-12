#include "TwoStage/two_stage_gpt.hpp"
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
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT)
    {
        g_interrupted = true;
        return TRUE;
    }
    return FALSE;
}
#else
extern "C" void sigint_handler(int)
{
    g_interrupted = true;
}
#endif

static constexpr const char *CHECKPOINT_FILE = "models/two_stage_checkpoint.gguf";
static constexpr const char *MODEL_FILE = "models/two_stage_model.gguf";

void save_checkpoint(const TwoStageGPT &model, const AdamW &optim, int step)
{
    std::cout << "\nSaving checkpoint at step " << step << "..." << std::endl;

    std::vector<std::pair<std::string, Tensor<float>>> tensors;

    auto named = const_cast<TwoStageGPT &>(model).named_parameters();
    for (auto &[name, param] : named)
    {
        tensors.push_back({"model." + name, param->data.toTensor()});
    }

    auto opt_state = optim.named_state();
    for (auto &[name, tensor] : opt_state)
    {
        tensors.push_back({"optimizer." + name, tensor});
    }

    std::unordered_map<std::string, GGUFMetadataValue> meta;
    meta["general.architecture"] = std::string("cxxstage_checkpoint");
    meta["checkpoint.step"] = int32_t(step);
    meta["optimizer.learning_rate"] = optim.learning_rate();
    meta["optimizer.beta1"] = optim.get_beta1();
    meta["optimizer.beta2"] = optim.get_beta2();
    meta["optimizer.weight_decay"] = optim.get_weight_decay();

    save_gguf_multi(tensors, CHECKPOINT_FILE, meta);
    std::cout << "Checkpoint saved to " << CHECKPOINT_FILE << std::endl;
}

bool load_checkpoint(TwoStageGPT &model, AdamW &optim, int &step)
{
    if (!std::filesystem::exists(CHECKPOINT_FILE))
    {
        return false;
    }

    try
    {
        auto meta = gguf_read_metadata(CHECKPOINT_FILE);

        auto it_step = meta.find("checkpoint.step");
        if (it_step != meta.end() && std::holds_alternative<int32_t>(it_step->second))
        {
            step = std::get<int32_t>(it_step->second);
        }

        auto tensors = load_gguf_multi<float>(CHECKPOINT_FILE);

        auto named = model.named_parameters();
        for (auto &[name, param] : named)
        {
            std::string key = "model." + name;
            if (tensors.count(key))
            {
                param->data = CudaTensor<float>::fromTensor(tensors[key]);
            }
        }

        std::unordered_map<std::string, Tensor<float>> opt_state;
        for (auto &[key, tensor] : tensors)
        {
            if (key.substr(0, 10) == "optimizer.")
            {
                opt_state[key.substr(10)] = tensor;
            }
        }
        optim.load_state(opt_state, step);

        std::cout << "Loaded checkpoint from step " << step << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load checkpoint: " << e.what() << std::endl;
        return false;
    }
}

float estimate_loss_two_stage(TwoStageGPT &model, Dataset &ds,
                              size_t batch_size, size_t block_size, RNG &rng, int num_batches)
{
    float total_loss = 0.0f;
    for (int i = 0; i < num_batches; i++)
    {
        auto [x, y] = ds.next_batch(batch_size, block_size);
        auto logits = model.forward(x, batch_size, block_size);
        auto [loss, dl] = cross_entropy_loss(logits, y);
        total_loss += loss;
    }
    return total_loss / num_batches;
}

int main(int argc, char **argv)
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
        if (!train_ds.load_bin("dataset/pretrain_t2t_mini.bin"))
        {
            if (!train_ds.load_jsonl("dataset/pretrain_t2t_mini.jsonl", tokenizer))
            {
                std::cerr << "Failed to load training data!" << std::endl;
                return 1;
            }
        }

        std::cout << "Train: " << train_ds.num_docs() << " docs, "
                  << train_ds.num_tokens() << " tokens"
                  << (train_ds.is_bin() ? " (preprocessed bin)" : " (streaming jsonl)") << std::endl;

        TwoStageConfig cfg;
        cfg.vocab_size = tokenizer.vocab_size();
        cfg.block_size = 256;
        cfg.n_embd = 256;
        cfg.n_head = 8;
        cfg.N = 2;

        std::cout << "Embedding dim: " << cfg.n_embd
                  << ", heads: " << cfg.n_head
                  << ", head_dim: " << cfg.N * (cfg.n_embd / cfg.n_head)
                  << " (N=" << cfg.N << ")" << std::endl;

        RNG rng(42);
        TwoStageGPT model(cfg, rng);
        std::cout << "Model parameters: " << model.total_params() << std::endl;

        auto params = model.parameters();
        AdamW optim(params, 3e-4f, 0.9f, 0.999f, 0.01f);

        auto compute_grad_norm = [&params]()
        {
            float total = 0.0f;
            for (auto *p : params)
                total += gpt_cuda::grad_norm_squared(p->grad);
            return std::sqrt(total);
        };

        size_t batch_size = 32;
        int grad_accum = 4; // micro-batches per optimizer step
        int num_epochs = 3;
        int eval_interval = 500;
        int eval_iters = 10;

        size_t tokens_per_step = batch_size * (size_t)grad_accum * cfg.block_size;
        size_t steps_per_epoch = train_ds.num_tokens() / tokens_per_step;
        int max_steps = num_epochs * (int)steps_per_epoch;

        if (argc >= 2)
            num_epochs = std::atoi(argv[1]);
        if (argc >= 3)
            batch_size = (size_t)(std::max)(1, std::atoi(argv[2]));
        if (argc >= 4)
            grad_accum = (std::max)(1, std::atoi(argv[3]));
        if (argc >= 2)
            max_steps = num_epochs * (int)(train_ds.num_tokens() /
                                           (batch_size * (size_t)grad_accum * cfg.block_size));
        std::cout << "Epochs=" << num_epochs << " steps=" << max_steps
                  << " batch_size=" << batch_size << " (micro) x " << grad_accum
                  << " (accum) = " << batch_size * (size_t)grad_accum
                  << " tokens/step" << std::endl;

        int start_step = 0;
        if (load_checkpoint(model, optim, start_step))
        {
            std::cout << "Resuming from step " << start_step << std::endl;
        }

#ifdef _WIN32
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
        signal(SIGINT, sigint_handler);
#endif

        std::cout << "\n=== Training TwoStageGPT (Softmax + ELU Kernel Attention) ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);

        // ------------------------------------------------------------------
        // Pipelined batch feeding: pinned host buffers + double-buffered
        // async H2D copies on a copy stream, overlapped with GPU compute.
        // ------------------------------------------------------------------
        size_t total_tokens = batch_size * cfg.block_size;
        size_t bytes = total_tokens * sizeof(int);

        std::array<int*, 2> hx{}, hy{}; // pinned host (dataset slices)
        std::array<int*, 2> dx{}, dy{}; // device (kernel inputs)
        for (int s = 0; s < 2; s++) {
            hx[s] = (int*)TensorN::PinnedMemoryPool::instance().acquire(bytes);
            hy[s] = (int*)TensorN::PinnedMemoryPool::instance().acquire(bytes);
            dx[s] = (int*)TensorN::CudaMemoryPool::instance().acquire(bytes);
            dy[s] = (int*)TensorN::CudaMemoryPool::instance().acquire(bytes);
        }
        // Non-blocking copy stream: uploads overlap with null-stream compute
        // instead of being implicitly serialized by legacy default-stream rules.
        cudaStream_t copy_stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        std::array<CudaEvent, 2> copy_done, compute_done;
        // Pre-signal so the first wait on a slot returns immediately.
        for (int s = 0; s < 2; s++)
        {
            compute_done[s].record(nullptr);
        }

        auto upload_batch = [&](int slot)
        {
            // Don't overwrite a slot whose backward may still be reading it.
            cudaStreamWaitEvent(copy_stream, compute_done[slot].get(), 0);
            train_ds.next_batch(hx[slot], hy[slot], batch_size, cfg.block_size);
            TensorN::async_copy_h2d(dx[slot], hx[slot], bytes, copy_stream);
            TensorN::async_copy_h2d(dy[slot], hy[slot], bytes, copy_stream);
            copy_done[slot].record(copy_stream);
        };

        // Numerical gradient check on one fresh batch (caches are overwritten,
        // so run it before the pipelined loop starts; params are restored).
        // Only on a fresh model - matches the original step-0 behavior.
        if (start_step == 0)
        {
            auto [gx, gy] = train_ds.next_batch(batch_size, cfg.block_size);
            two_stage_numerical_grad_check(model, gx, gy, batch_size, cfg.block_size);
            model.zero_grad();
        }

        // FP32 GEMMs keep the numerical check honest; after it passes,
        // switch the GEMMs to TF32 tensor cores (2x throughput).
        TensorN::cuda::set_tf32(true);
        std::cout << "TF32 tensor-core math enabled for GEMMs" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_print_time = start_time;

        size_t slot = 0;
        upload_batch(slot);

        float window_loss = 0.0f;
        size_t micro_batches = 0;

        for (int step = start_step; step < max_steps; step++)
        {
            if (g_interrupted)
            {
                std::cout << "\n\nUser interrupted! Saving checkpoint..." << std::endl;
                save_checkpoint(model, optim, step);
                std::cout << "Checkpoint saved. You can resume later with the same command." << std::endl;
                return 0;
            }

            // Default stream waits until this slot's async upload is done.
            cudaStreamWaitEvent(nullptr, copy_done[slot].get(), 0);
            int cur = slot;
            slot = 1 - slot;
            // Prefetch next batch while the GPU computes the current one.
            upload_batch(slot);

            auto logits = model.forward(dx[cur], batch_size, cfg.block_size);
            auto [loss, dlogits] = cross_entropy_loss(logits, dy[cur]);
            window_loss += loss;
            micro_batches++;

            if (micro_batches == 1)
                model.zero_grad();
            model.backward(dlogits);
            // Signal that this slot's buffers are free to be overwritten.
            compute_done[cur].record(nullptr);

            if (micro_batches == (size_t)grad_accum)
            {
                float avg_loss = window_loss / (float)grad_accum;
                window_loss = 0.0f;
                micro_batches = 0;

                auto now = std::chrono::high_resolution_clock::now();
                double step_sec = std::chrono::duration<double>(now - last_print_time).count();
                last_print_time = now;
                double tok_per_s = (step_sec > 0.0) ? tokens_per_step / step_sec : 0.0;

                // Per-step loss + throughput.
                std::cout << "step " << std::setw(4) << step
                          << " | loss " << avg_loss
                          << " | " << (unsigned long long)tok_per_s << " tok/s"
                          << std::endl;

                if (step % eval_interval == 0 || step == max_steps - 1)
                {
                    float grad_norm = compute_grad_norm();
                    float train_loss = estimate_loss_two_stage(model, train_ds,
                                                               batch_size, cfg.block_size, rng, eval_iters);

                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                    std::cout << "step " << std::setw(4) << step
                              << " | train loss " << train_loss
                              << " | accum loss " << avg_loss
                              << " | grad_norm " << grad_norm
                              << " | time " << elapsed << "s" << std::endl;

                    if (step > 0)
                    {
                        save_checkpoint(model, optim, step);
                    }
                }

                optim.step();
            }
        }

        for (int s = 0; s < 2; s++)
        {
            TensorN::PinnedMemoryPool::instance().release(hx[s]);
            TensorN::PinnedMemoryPool::instance().release(hy[s]);
            TensorN::CudaMemoryPool::instance().release(dx[s]);
            TensorN::CudaMemoryPool::instance().release(dy[s]);
        }
        cudaStreamSynchronize(copy_stream);
        CHECK_CUDA_ERROR(cudaStreamDestroy(copy_stream));
        cudaStreamSynchronize(nullptr);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "\nTraining completed in " << total_sec << "s" << std::endl;

        model.save(MODEL_FILE);

        if (std::filesystem::exists(CHECKPOINT_FILE))
        {
            std::filesystem::remove(CHECKPOINT_FILE);
        }
    }
    catch (const std::bad_alloc &e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
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
