"""
Pre-encode dataset using the BPE tokenizer from tokenizer.json.
Saves encoded token sequences as binary files with document boundaries.

Binary format:
  [4B magic: 0x44505447 "GPTD" LE]
  [4B version: 1]
  [4B num_docs]
  Per doc: [4B num_tokens] [num_tokens * 4B token IDs]
"""
import json
import struct
import os
from tokenizers import Tokenizer

MAGIC = 0x44505447
VERSION = 1


def load_jsonl_texts(filepath):
    """Extract concatenated conversation text from JSONL file."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            parts = []
            for msg in data.get('messages', []):
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'system':
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
                elif role == 'user':
                    parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
                elif role == 'assistant':
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
            if parts:
                texts.append(''.join(parts))
    return texts


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_dir, 'tokenizer', 'tokenizer.json')
    dataset_dir = os.path.join(base_dir, 'dataset')

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    for split in ['train', 'val', 'test']:
        jsonl_path = os.path.join(dataset_dir, f'{split}.jsonl')
        bin_path = os.path.join(dataset_dir, f'{split}.bin')

        if not os.path.exists(jsonl_path):
            print(f"  {split}: file not found, skipping")
            continue

        print(f"Processing {split}...")
        texts = load_jsonl_texts(jsonl_path)
        print(f"  Loaded {len(texts)} conversations")

        all_docs = []
        total_tokens = 0
        for i, text in enumerate(texts):
            encoding = tokenizer.encode(text)
            ids = encoding.ids
            all_docs.append(ids)
            total_tokens += len(ids)
            if (i + 1) % 1000 == 0:
                print(f"  Encoded {i + 1}/{len(texts)} conversations")

        print(f"  {len(all_docs)} docs, {total_tokens} total tokens")

        with open(bin_path, 'wb') as f:
            f.write(struct.pack('<I', MAGIC))
            f.write(struct.pack('<I', VERSION))
            f.write(struct.pack('<I', len(all_docs)))
            for ids in all_docs:
                f.write(struct.pack('<I', len(ids)))
                for tid in ids:
                    f.write(struct.pack('<I', tid))

        print(f"  Saved to {bin_path}\n")

    print("Done! All splits pre-encoded.")


if __name__ == '__main__':
    main()
