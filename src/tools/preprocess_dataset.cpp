// ============================================================
// preprocess_dataset: JSONL -> binary "big block" dataset
//
// Reads a JSONL dataset ({"text": ...} or {"messages": [...]}),
// tokenizes every document with the BPE tokenizer, explicitly
// prepends BOS and appends EOS, then merges everything into a
// single flat token block saved as a binary file.
//
// Usage:
//   preprocess_dataset <input.jsonl> [output.bin] [tokenizer.json]
//
// Output format: DataBlockHeader + uint16_t tokens[num_tokens]
// ============================================================

#include "tokenizer.hpp"
#include "data_bin.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr size_t FLUSH_CHUNK = 1u << 20; // flush every 1M tokens

// Mirror Dataset::parse_line: extract text from a single JSONL line.
std::string extract_text(const nlohmann::json& data) {
    std::string text;
    if (data.contains("messages")) {
        for (auto& msg : data["messages"]) {
            std::string role = msg.value("role", "");
            std::string content = msg.value("content", "");
            if (role == "system")
                text += "<|im_start|>system\n" + content + "<|im_end|>\n";
            else if (role == "user")
                text += "<|im_start|>user\n" + content + "<|im_end|>\n";
            else if (role == "assistant")
                text += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        }
    } else {
        text = data.value("text", "");
    }
    return text;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 2) {
        std::cerr << "Usage: preprocess_dataset <input.jsonl> [output.bin] [tokenizer.json]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = (argc >= 3) ? argv[2] : (input_path + ".bin");
    std::string tokenizer_path = (argc >= 4) ? argv[3] : "tokenizer/tokenizer.json";

    std::cout << "Tokenizer : " << tokenizer_path << std::endl;
    std::cout << "Input     : " << input_path << std::endl;
    std::cout << "Output    : " << output_path << std::endl;

    BPETokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path)) {
        std::cerr << "Failed to load tokenizer: " << tokenizer_path << std::endl;
        return 1;
    }
    if (tokenizer.vocab_size() > 65535) {
        std::cerr << "Vocab size " << tokenizer.vocab_size()
                  << " exceeds uint16_t limit (65535)" << std::endl;
        return 1;
    }
    int bos_id = tokenizer.get_bos_id();
    int eos_id = tokenizer.get_eos_id();
    std::cout << "Vocab     : " << tokenizer.vocab_size()
              << "  bos=" << bos_id << "  eos=" << eos_id << std::endl;

    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open input: " << input_path << std::endl;
        return 1;
    }

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open output: " << output_path << std::endl;
        return 1;
    }

    // Write placeholder header; patch real counts at the end.
    DataBlockHeader header{};
    for (size_t i = 0; i < 8; i++) header.magic[i] = DATA_BLOCK_MAGIC[i];
    header.version = DATA_BLOCK_VERSION;
    header.vocab_size = (uint32_t)tokenizer.vocab_size();
    header.bos_id = (uint32_t)bos_id;
    header.eos_id = (uint32_t)eos_id;
    header.num_tokens = 0;
    header.num_docs = 0;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    std::vector<uint16_t> chunk;
    chunk.reserve(FLUSH_CHUNK + 4096);

    uint64_t num_tokens = 0, num_docs = 0, bad_lines = 0;
    auto t_start = std::chrono::steady_clock::now();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        try {
            auto data = nlohmann::json::parse(line);
            std::string text = extract_text(data);
            if (text.empty()) {
                bad_lines++;
                continue;
            }

            // Encode WITHOUT specials, then explicitly wrap BOS/EOS.
            auto ids = tokenizer.encode(text, false);
            chunk.push_back((uint16_t)bos_id);
            for (int id : ids) chunk.push_back((uint16_t)id);
            chunk.push_back((uint16_t)eos_id);

            num_docs++;
            num_tokens += ids.size() + 2;

            if (chunk.size() >= FLUSH_CHUNK) {
                out.write(reinterpret_cast<const char*>(chunk.data()),
                          chunk.size() * sizeof(uint16_t));
                chunk.clear();
            }
        } catch (const std::exception&) {
            bad_lines++;
        }

        if (num_docs % 100000 == 0) {
            auto t_now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t_now - t_start).count();
            double rate = sec > 0 ? (double)num_docs / sec : 0;
            std::cout << "\rdocs=" << num_docs
                      << " tokens=" << num_tokens
                      << " (" << rate << " docs/s)   ";
            std::cout.flush();
        }
    }

    if (!chunk.empty()) {
        out.write(reinterpret_cast<const char*>(chunk.data()),
                  chunk.size() * sizeof(uint16_t));
        chunk.clear();
    }

    // Patch header with final counts.
    header.num_tokens = num_tokens;
    header.num_docs = num_docs;
    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.close();

    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nDone." << std::endl;
    std::cout << "Documents : " << num_docs << std::endl;
    std::cout << "Tokens    : " << num_tokens << std::endl;
    std::cout << "Bad lines : " << bad_lines << std::endl;
    std::cout << "Avg len   : "
              << (num_docs ? (double)(num_tokens - 2 * (double)num_docs) / num_docs : 0.0)
              << " tokens/doc (excl. BOS/EOS)" << std::endl;
    std::cout << "Bin size  : " << (double)(sizeof(header) + num_tokens * 2) / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "Elapsed   : " << total_sec << "s" << std::endl;
    return 0;
}
