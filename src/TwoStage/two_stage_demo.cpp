#include "TwoStage/two_stage_gpt.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <clocale>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif

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

        TwoStageConfig cfg = TwoStageGPT::load_config("models/two_stage_model.gguf");
        std::cout << "Model config: n_embd=" << cfg.n_embd
                  << " n_head=" << cfg.n_head
                  << " N=" << cfg.N
                  << " block_size=" << cfg.block_size << std::endl;

        RNG rng(42);
        TwoStageGPT model(cfg, rng);
        std::cout << "Model parameters: " << model.total_params() << std::endl;

        model.load("models/two_stage_model.gguf");

        int eos_id = tokenizer.get_eos_id();

        std::cout << "\n=== Interactive Chat (TwoStageGPT) ===" << std::endl;
        std::cout << "Type 'quit' or 'exit' to stop.\n"
                  << std::endl;

        std::string input;
        while (true)
        {
            std::cout << "You: ";
            std::getline(std::cin, input);
            if (input == "quit" || input == "exit" || input.empty())
            {
                std::cout << "Goodbye!" << std::endl;
                break;
            }

            std::string prompt_text = "<|im_start|>user\n" + input + "<|im_end|>";
            std::vector<int> prompt_ids = tokenizer.encode(prompt_text);
            if (!prompt_ids.empty() && prompt_ids.back() == eos_id)
                prompt_ids.pop_back();

            size_t prompt_len = prompt_ids.size();

            auto start = std::chrono::high_resolution_clock::now();
            auto ids = model.generate(prompt_ids, cfg.block_size * 2,
                                      eos_id, 0.8f, 40, rng);
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::string response;
            for (size_t i = prompt_len; i < ids.size(); i++)
            {
                int id = ids[i];
                if (id == eos_id)
                    break;
                if (id == tokenizer.get_bos_id())
                    continue;
                response += tokenizer.decode({id});
            }

            std::cout << "Bot: " << response << "\r\n[" << ms << "ms]" << std::endl;
            std::cout << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
