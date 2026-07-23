#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

class BPETokenizer;
class RNG;
class GPT;

// ============================================================
// Streaming Dataset: sequential line-by-line JSONL reading
//
// Scans the file once to count lines (for num_docs()), then
// reads + tokenizes on demand during training. Maintains a
// prefetch buffer for efficiency.
// ============================================================
class Dataset {
public:
    size_t num_docs() const { return num_docs_; }
    size_t num_tokens() const { return 0; } // unknown in streaming mode
    bool empty() const { return num_docs_ == 0; }

    bool load_jsonl(const std::string& path, BPETokenizer& tok);

    // Fill a batch from sequential reads (wraps at EOF)
    std::pair<std::vector<int>, std::vector<int>> next_batch(
        size_t batch_size, size_t block_size);

private:
    std::string filepath_;
    BPETokenizer* tok_ = nullptr;
    size_t num_docs_ = 0;
    mutable size_t error_count_ = 0;

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

inline bool Dataset::load_jsonl(const std::string& path, BPETokenizer& tok) {
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
