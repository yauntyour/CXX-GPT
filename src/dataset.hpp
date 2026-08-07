#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "data_bin.hpp"

class BPETokenizer;
class RNG;
class GPT;

// ============================================================
// Dataset: two loading modes
//
// 1) Binary "big block" mode (load_bin): all documents are
//    tokenized once, wrapped with BOS/EOS and concatenated into
//    a single flat block. next_batch just slices contiguous
//    spans from the block — no JSON parsing, no tokenization,
//    max batch throughput.
//
// 2) Streaming JSONL mode (load_jsonl): sequential line-by-line
//    reading with on-the-fly tokenization (legacy).
// ============================================================
class Dataset {
public:
    size_t num_docs() const { return num_docs_; }
    size_t num_tokens() const { return num_tokens_; }
    bool empty() const { return num_docs_ == 0; }
    bool is_bin() const { return !block_.empty(); }

    bool load_jsonl(const std::string& path, BPETokenizer& tok);
    bool load_bin(const std::string& path);

    // Fill a batch from sequential reads (wraps at EOF)
    std::pair<std::vector<int>, std::vector<int>> next_batch(
        size_t batch_size, size_t block_size);

private:
    std::string filepath_;
    BPETokenizer* tok_ = nullptr;
    size_t num_docs_ = 0;
    size_t num_tokens_ = 0;
    mutable size_t error_count_ = 0;

    // Binary big-block mode
    std::vector<uint16_t> block_;
    mutable size_t block_pos_ = 0;

    // Open file handle and line buffer for sequential reading
    mutable std::ifstream file_;
    mutable std::vector<std::vector<int>> buffer_;
    mutable size_t buffer_pos_ = 0;
    static constexpr size_t PREFETCH_SIZE = 500;

    void refill_buffer() const;
    std::vector<int> parse_line(const std::string& line) const;
};

// ============================================================
// Implementation
// ============================================================

inline bool Dataset::load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return false;
    }

    DataBlockHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (f.gcount() != sizeof(header) ||
        std::string(header.magic, 8) != DATA_BLOCK_MAGIC) {
        std::cerr << "Bad dataset header: " << path << std::endl;
        return false;
    }

    block_.resize(header.num_tokens);
    f.read(reinterpret_cast<char*>(block_.data()),
           block_.size() * sizeof(uint16_t));
    if ((size_t)f.gcount() != block_.size() * sizeof(uint16_t)) {
        std::cerr << "Truncated token block: " << path << std::endl;
        block_.clear();
        return false;
    }

    num_docs_ = (size_t)header.num_docs;
    num_tokens_ = (size_t)header.num_tokens;
    block_pos_ = 0;
    filepath_ = path;

    std::cout << "Loaded " << num_docs_ << " docs, "
              << num_tokens_ << " tokens from " << path
              << " (big block, " << (double)(sizeof(header) + block_.size() * 2) / (1024 * 1024)
              << " MB)" << std::endl;
    return num_tokens_ > 1;
}

inline bool Dataset::load_jsonl(const std::string& path, BPETokenizer& tok) {
    block_.clear();
    block_pos_ = 0;
    num_tokens_ = 0;

    // Count lines
    std::ifstream count_file(path);
    if (!count_file.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return false;
    }
    num_docs_ = 0;
    std::string line;
    while (std::getline(count_file, line)) {
        if (!line.empty()) num_docs_++;
    }
    count_file.close();

    filepath_ = path;
    tok_ = &tok;
    file_.open(path);
    buffer_.clear();
    buffer_pos_ = 0;
    error_count_ = 0;

    std::cout << "Indexed " << num_docs_ << " docs from " << path
              << " (streaming)" << std::endl;
    return num_docs_ > 0;
}

inline std::vector<int> Dataset::parse_line(const std::string& line) const {
    try {
        auto data = nlohmann::json::parse(line);
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
        if (!text.empty()) {
            return tok_->encode(text);
        }
    } catch (const std::exception& e) {
        error_count_++;
    }
    return {};
}

inline void Dataset::refill_buffer() const {
    buffer_.clear();
    buffer_pos_ = 0;
    std::string line;
    while (buffer_.size() < PREFETCH_SIZE) {
        if (!std::getline(file_, line)) {
            // Wrap around: reopen and continue
            file_.clear();
            file_.seekg(0);
            if (buffer_.empty() && !std::getline(file_, line)) {
                break; // truly empty file
            }
            continue;
        }
        if (line.empty()) continue;
        auto ids = parse_line(line);
        if (!ids.empty()) {
            buffer_.push_back(std::move(ids));
        }
    }
}

inline std::pair<std::vector<int>, std::vector<int>> Dataset::next_batch(
    size_t batch_size, size_t block_size)
{
    // Binary big-block mode: slice a contiguous span, zero tokenization cost.
    if (!block_.empty()) {
        size_t total = batch_size * block_size;
        std::vector<int> x(total);
        std::vector<int> y(total);

        if (block_.size() <= total + 1) {
            // Degenerate case: whole block in one batch, wrap within block.
            for (size_t i = 0; i < total; i++) {
                size_t p = (i + block_pos_) % block_.size();
                size_t q = (p + 1) % block_.size();
                x[i] = block_[p];
                y[i] = block_[q];
            }
        } else {
            if (block_pos_ + total + 1 > block_.size()) {
                block_pos_ = 0; // wrap at EOF
            }
            for (size_t i = 0; i < total; i++) {
                x[i] = block_[block_pos_ + i];
                y[i] = block_[block_pos_ + i + 1];
            }
            block_pos_ += total;
        }
        return {x, y};
    }

    std::vector<int> x(batch_size * block_size, 0);
    std::vector<int> y(batch_size * block_size, 0);

    size_t filled = 0;
    while (filled < batch_size) {
        if (buffer_pos_ >= buffer_.size()) {
            refill_buffer();
            if (buffer_.empty()) break;
        }

        const auto& doc = buffer_[buffer_pos_++];
        if (doc.size() < block_size + 1) {
            size_t copy_len = std::min(doc.size(), block_size);
            for (size_t s = 0; s < copy_len; s++) {
                x[filled * block_size + s] = doc[s];
                y[filled * block_size + s] = (s + 1 < doc.size()) ? doc[s + 1] : 0;
            }
        } else {
            size_t start = 0; // always use offset 0 for simplicity in streaming
            for (size_t s = 0; s < block_size; s++) {
                x[filled * block_size + s] = doc[start + s];
                y[filled * block_size + s] = doc[start + s + 1];
            }
        }
        filled++;
    }

    if (error_count_ > 0) {
        std::cerr << "\r[Skipped " << error_count_ << " bad lines so far]   ";
    }

    return {x, y};
}

inline float estimate_loss(GPT& model, Dataset& ds,
    size_t batch_size, size_t block_size, RNG& rng, int num_batches)
{
    float total_loss = 0.0f;
    for (int i = 0; i < num_batches; i++) {
        auto [x, y] = ds.next_batch(batch_size, block_size);
        auto logits = model.forward(x, batch_size, block_size);
        auto [loss, _] = cross_entropy_loss(logits, y);
        total_loss += loss;
    }
    return total_loss / num_batches;
}
