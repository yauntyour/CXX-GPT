#pragma once

#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// ============================================================
// BPE Tokenizer (CPU-only, standalone)
//
// Loads a HuggingFace-style tokenizer.json and provides
// encode/decode with BOS/EOS special-token control.
// ============================================================
class BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_ranks;
    int bos_id, eos_id;

    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const {
            return std::hash<std::string>()(p.first) ^
                   (std::hash<std::string>()(p.second) << 1);
        }
    };

    static int utf8_char_len(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;
        if ((first_byte & 0xE0) == 0xC0) return 2;
        if ((first_byte & 0xF0) == 0xE0) return 3;
        if ((first_byte & 0xF8) == 0xF0) return 4;
        return 1;
    }

    static int utf8_decode(const std::string& s, size_t& i) {
        if (i >= s.size()) return -1;
        unsigned char c = (unsigned char)s[i];
        int cp = 0, len = utf8_char_len(c);
        if (len == 1) { cp = c; }
        else if (len == 2) { cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F); }
        else if (len == 3) { cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F); }
        else if (len == 4) { cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F); }
        i += len;
        return cp;
    }

    bool is_cjk_char(const std::string& utf8_char) const {
        if (utf8_char.empty()) return false;
        size_t pos = 0;
        int cp = utf8_decode(utf8_char, pos);
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x3040 && cp <= 0x309F) ||
               (cp >= 0x30A0 && cp <= 0x30FF) ||
               (cp >= 0xAC00 && cp <= 0xD7AF);
    }

    static std::string utf8_encode(int cp) {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    static constexpr int BL_UNMAPPED = 256;

    static const std::array<uint16_t, 324>& byte_table() {
        static std::array<uint16_t, 324> tbl;
        static bool ready = false;
        if (!ready) {
            tbl.fill((uint16_t)BL_UNMAPPED);
            for (int b = 33; b <= 126; b++) tbl[b] = (uint16_t)b;
            for (int b = 161; b <= 172; b++) tbl[b] = (uint16_t)b;
            for (int b = 174; b <= 255; b++) tbl[b] = (uint16_t)b;
            int n = 0;
            for (int b = 0; b < 256; b++)
                if (tbl[b] == BL_UNMAPPED)
                    tbl[256 + n++] = (uint16_t)b;
            ready = true;
        }
        return tbl;
    }

    static int byte_to_cp(uint8_t b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            return b;
        int n = 0;
        for (int x = 0; x < b; x++)
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255)))
                n++;
        return 256 + n;
    }

    static bool is_byte_level_cp(int cp) {
        return cp > 0 && cp <= 323 && byte_table()[cp] != BL_UNMAPPED;
    }

    static uint8_t cp_to_byte(int cp) {
        return (uint8_t)byte_table()[cp];
    }

    static std::string byte_to_char(uint8_t b) {
        return utf8_encode(byte_to_cp(b));
    }

    int get_merge_rank(const std::string& a, const std::string& b) const {
        auto it = merge_ranks.find(a + " " + b);
        if (it != merge_ranks.end()) return it->second;
        return -1;
    }

    std::vector<std::string> bpe_encode_word(const std::string& word) const {
        std::string bytes_str;
        for (size_t i = 0; i < word.size(); i++) {
            uint8_t b = (uint8_t)word[i];
            bytes_str += byte_to_char(b);
        }

        std::vector<std::string> symbols;
        for (size_t i = 0; i < bytes_str.size(); ) {
            int len = utf8_char_len((unsigned char)bytes_str[i]);
            symbols.push_back(bytes_str.substr(i, len));
            i += len;
        }

        while (symbols.size() > 1) {
            int best_rank = 1 << 30;
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                int r = get_merge_rank(symbols[i], symbols[i + 1]);
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                }
            }
            if (best_rank == (1 << 30)) break;

            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }
        return symbols;
    }

public:
    BPETokenizer() : bos_id(-1), eos_id(-1) {}

    bool load(const std::string& json_path) {
        std::ifstream f(json_path);
        if (!f.is_open()) return false;

        nlohmann::json j;
        f >> j;

        auto& model = j["model"];
        auto& v = model["vocab"];
        id_to_token.resize(v.size());
        for (auto it = v.begin(); it != v.end(); ++it) {
            std::string token = it.key();
            int id = it.value();
            vocab[token] = id;
            if (id < (int)id_to_token.size())
                id_to_token[id] = token;
        }

        auto& merges_arr = model["merges"];
        for (size_t i = 0; i < merges_arr.size(); i++) {
            std::string merge_str = merges_arr[i][0].get<std::string>() + " " +
                                    merges_arr[i][1].get<std::string>();
            merge_ranks[merge_str] = (int)i;
        }

        auto bos_it = vocab.find("<|im_start|>");
        if (bos_it != vocab.end()) bos_id = bos_it->second;
        auto eos_it = vocab.find("<|im_end|>");
        if (eos_it != vocab.end()) eos_id = eos_it->second;

        return true;
    }

    // Encode text. If with_special is true, wraps with BOS/EOS.
    std::vector<int> encode(const std::string& text, bool with_special = true) const {
        std::vector<int> ids;
        if (with_special && bos_id >= 0) ids.push_back(bos_id);

        std::string current;
        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = (unsigned char)text[i];
            int len = 1;
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            if (i + len > text.size()) len = (int)(text.size() - i);

            std::string ch = text.substr(i, len);
            if (ch == " " || ch == "\n" || ch == "\t" || ch == "\r") {
                if (!current.empty()) {
                    auto syms = bpe_encode_word(current);
                    for (auto& s : syms) {
                        auto it = vocab.find(s);
                        if (it != vocab.end()) ids.push_back(it->second);
                    }
                    current.clear();
                }
                auto syms = bpe_encode_word(ch);
                for (auto& s : syms) {
                    auto it = vocab.find(s);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            } else if (is_cjk_char(ch)) {
                if (!current.empty()) {
                    auto syms = bpe_encode_word(current);
                    for (auto& s : syms) {
                        auto it = vocab.find(s);
                        if (it != vocab.end()) ids.push_back(it->second);
                    }
                    current.clear();
                }
                auto syms = bpe_encode_word(ch);
                for (auto& s : syms) {
                    auto it = vocab.find(s);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            } else {
                current += ch;
            }
            i += len;
        }
        if (!current.empty()) {
            auto syms = bpe_encode_word(current);
            for (auto& s : syms) {
                auto it = vocab.find(s);
                if (it != vocab.end()) ids.push_back(it->second);
            }
        }

        if (with_special && eos_id >= 0) ids.push_back(eos_id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size()) continue;
            const auto& token = id_to_token[id];
            for (size_t i = 0; i < token.size(); ) {
                int cp = utf8_decode(token, i);
                if (is_byte_level_cp(cp)) {
                    result += (char)cp_to_byte(cp);
                }
            }
        }
        return result;
    }

    size_t vocab_size() const { return id_to_token.size(); }
    int get_bos_id() const { return bos_id; }
    int get_eos_id() const { return eos_id; }
};
