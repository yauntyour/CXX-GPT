"""Cross-check CXX TwoStageGPT forward vs the PyTorch reference (model.py).

Loads the model.* tensors from the C++ GGUF checkpoint into the reference
model, feeds the identical token span, and compares logits elementwise.
"""
import sys
import struct
import numpy as np
import torch

sys.path.insert(0, r"D:\Developments\Python\XiaoXi-LLM")
from model import model

GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0

# C++ GGUF tensor name -> reference module attribute
MAPPING = {
    "embedding.weight": "embedding.weight",
    "attn_q.W": "q_proj.weight",
    "attn_k.W": "k_proj.weight",
    "attn_v.W": "v_proj.weight",
    "q_ffn_fc.W": "q_ffn.0.weight",
    "q_ffn_proj.W": "q_ffn.3.weight",
    "k_ffn_fc.W": "k_ffn.0.weight",
    "k_ffn_proj.W": "k_ffn.3.weight",
    "v_ffn_fc.W": "v_ffn.0.weight",
    "v_ffn_proj.W": "v_ffn.3.weight",
    "lq_proj.W": "lq_proj.weight",
    "lk_proj.W": "lk_proj.weight",
    "lv_proj.W": "lv_proj.weight",
    "ffn_fc.W": "ffn.0.weight",
    "ffn_proj.W": "ffn.3.weight",
}

META_TYPE_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}


def read_string(buf, pos):
    (n,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    s = buf[pos:pos + n].decode("utf-8")
    return s, pos + n


def skip_metadata_value(buf, pos, vtype):
    if vtype == 8:  # string
        return read_string(buf, pos)[1]
    if vtype == 9:  # array
        (n,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        (elem,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if elem == 8:
            for _ in range(n):
                pos = read_string(buf, pos)[1]
        else:
            pos += n * META_TYPE_SIZES[elem]
        return pos
    return pos + META_TYPE_SIZES.get(vtype, 0)


def load_gguf(path):
    with open(path, "rb") as f:
        data = f.read()

    assert data[:4] == GGUF_MAGIC
    pos = 4
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, pos)
    pos += 20

    for _ in range(kv_count):
        key, pos = read_string(data, pos)
        (vtype,) = struct.unpack_from("<I", data, pos)
        pos += 4
        pos = skip_metadata_value(data, pos, vtype)

    infos = []
    for _ in range(tensor_count):
        name, pos = read_string(data, pos)
        (n_dims,) = struct.unpack_from("<I", data, pos)
        pos += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, data, pos)
        pos += 8 * n_dims
        (ttype, offset) = struct.unpack_from("<IQ", data, pos)
        pos += 12
        infos.append((name, dims, ttype, offset))

    data_base = (pos + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1)

    tensors = {}
    for name, dims, ttype, offset in infos:
        assert ttype == GGML_TYPE_F32, f"unexpected type {ttype} for {name}"
        n = int(np.prod(dims))
        arr = np.frombuffer(data, dtype=np.float32, count=n,
                            offset=data_base + offset)
        tensors[name] = arr.reshape(dims)

    return tensors


def main():
    ckpt = r"D:\Developments\CXX\CXX-GPT\models\two_stage_checkpoint.gguf"
    tensors = load_gguf(ckpt)
    print(f"Parsed {len(tensors)} tensors from GGUF")

    vocab = 6400
    torch.manual_seed(0)
    m = model(vocab, 256, 8, 6, 1).eval().cuda()
    torch.backends.cuda.matmul.allow_tf32 = False

    for cpp_name, py_attr in MAPPING.items():
        key = "model." + cpp_name
        assert key in tensors, f"missing {key}"
        src = torch.from_numpy(tensors[key].astype(np.float32))
        dst = m
        for part in py_attr.split("."):
            if part.isdigit():
                dst = dst[int(part)]
            else:
                dst = getattr(dst, part)
        assert tuple(dst.shape) == tuple(src.shape), \
            f"{py_attr}: {tuple(dst.shape)} vs C++ {tuple(src.shape)}"
        dst.data.copy_(src)
    print("Weights loaded (15 tensors)")

    with open(r"D:\Developments\CXX\CXX-GPT\probe_tokens.bin", "rb") as f:
        x = torch.from_numpy(
            np.frombuffer(f.read(), dtype=np.int32)).long().view(2, 256).cuda()
    cpp_logits = np.fromfile(
        r"D:\Developments\CXX\CXX-GPT\probe_logits.bin", dtype=np.float32)
    cpp_logits = cpp_logits.reshape(512, vocab)

    with torch.no_grad():
        ref_logits = m(x).cpu().numpy().reshape(512, vocab)

    diff = np.abs(cpp_logits - ref_logits)
    rel = diff / (np.abs(ref_logits) + 1e-6)
    print(f"max abs logit diff   = {diff.max():.3e}")
    print(f"mean abs logit diff  = {diff.mean():.3e}")
    print(f"max rel logit diff   = {rel.max():.3e}")
    print(f"argmax agreement     = {(cpp_logits.argmax(1) == ref_logits.argmax(1)).mean() * 100:.2f}%")

    # CE loss comparison
    with open(r"D:\Developments\CXX\CXX-GPT\probe_targets.bin", "rb") as f:
        y = torch.from_numpy(
            np.frombuffer(f.read(), dtype=np.int32)).long().cuda()

    logits = m(x)
    ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), y.view(2, 256), reduction="mean")
    print(f"PyTorch CE loss = {ce.item():.4f}")

    # Per-group gradient norms (autograd) for comparison with the C++ probe
    ce.backward()
    names = [
        "embedding.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "q_ffn.0.weight", "q_ffn.3.weight", "k_ffn.0.weight", "k_ffn.3.weight",
        "v_ffn.0.weight", "v_ffn.3.weight",
        "lq_proj.weight", "lk_proj.weight", "lv_proj.weight",
        "ffn.0.weight", "ffn.3.weight",
    ]
    print("\nPer-group grad norms (PyTorch autograd):")
    rows = []
    for attr in names:
        p = m
        for part in attr.split("."):
            p = p[int(part)] if part.isdigit() else getattr(p, part)
        gn = p.grad.norm().item()
        rows.append((attr, gn))
    rows.sort(key=lambda r: -r[1])
    for attr, gn in rows:
        print(f"{attr:>24}  {gn:.6g}")


if __name__ == "__main__":
    main()
