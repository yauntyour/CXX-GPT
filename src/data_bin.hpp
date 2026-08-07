#pragma once

#include <cstdint>
#include <cstddef>

// ============================================================
// Binary dataset "big block" format
//
// All documents are tokenized, wrapped with BOS/EOS markers,
// and concatenated into ONE flat token block so that training
// can slice contiguous batches with zero JSON/tokenizer work.
//
// Layout:
//   Header (40 bytes, see DataBlockHeader)
//   tokens: uint16_t[num_tokens]
//
// vocab_size must be <= 65535 so tokens fit in uint16_t.
// ============================================================

struct DataBlockHeader {
    char magic[8];        // "CXXBIN01"
    uint32_t version;     // 1
    uint32_t vocab_size;
    uint32_t bos_id;
    uint32_t eos_id;
    uint64_t num_tokens;
    uint64_t num_docs;
};

static_assert(sizeof(DataBlockHeader) == 40, "unexpected header size");

inline constexpr const char* DATA_BLOCK_MAGIC = "CXXBIN01";
inline constexpr uint32_t DATA_BLOCK_VERSION = 1;
