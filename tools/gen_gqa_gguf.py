#!/usr/bin/env python3
"""Generate a tiny GGUF v3 model with grouped-query attention.

Dimensions:
  n_embd=32, n_ff=64, n_layer=1, n_head=4, n_head_kv=2, head_dim=8
  context=8, vocab=16

The weights are deterministic Q4_0 values so the kernel can be checked
against tools/verify_gqa.py. This is a test fixture, not a useful model.
"""
import random
import struct
import sys
import zlib

N_EMBD = 32
N_FF = 64
N_LAYER = 1
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
MAX_SEQ = 8
VOCAB = 16
EPS = 1e-5


def pack_str(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack('<Q', len(b)) + b


def kv(key, vtype, value):
    out = pack_str(key) + struct.pack('<I', vtype)
    if vtype == 8:
        out += pack_str(value)
    elif vtype == 9:  # array of strings
        out += struct.pack('<I', 8) + struct.pack('<Q', len(value))
        for s in value:
            out += pack_str(s)
    elif vtype == 4:
        out += struct.pack('<I', value)
    elif vtype == 6:
        out += struct.pack('<f', value)
    else:
        raise ValueError(vtype)
    return out


def q4_0(values, seed):
    """Quantize 32-element blocks and return their GGUF bytes."""
    rng = random.Random(seed)
    out = bytearray()
    for base in range(0, len(values), 32):
        block = values[base:base + 32]
        # Bias the data away from zero so the fixture is not degenerate.
        block = [v + rng.uniform(-0.05, 0.05) for v in block]
        scale = max(abs(v) for v in block) / 7.0 or 0.01
        q = []
        for v in block:
            qi = int(round(v / scale)) + 8
            q.append(max(0, min(15, qi)))
        out += struct.pack('<e', scale)
        for j in range(16):
            out.append((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
    return bytes(out)


def matrix(name, rows, cols, seed):
    vals = []
    rng = random.Random(seed)
    for i in range(rows * cols):
        vals.append(rng.uniform(-1.0, 1.0))
    return (name, [cols, rows], 2, q4_0(vals, seed + 1))


def norm(name, n, seed):
    rng = random.Random(seed)
    vals = [1.0 + rng.uniform(-0.1, 0.1) for _ in range(n)]
    return (name, [n], 0, struct.pack(f'<{n}f', *vals))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'build/gqa.gguf'

    tensors = [
        matrix('token_embd.weight', VOCAB, N_EMBD, 11),
        norm('blk.0.attn_norm.weight', N_EMBD, 12),
        matrix('blk.0.attn_q.weight', N_EMBD, N_EMBD, 13),
        matrix('blk.0.attn_k.weight', N_HEAD_KV * HEAD_DIM, N_EMBD, 14),
        matrix('blk.0.attn_v.weight', N_HEAD_KV * HEAD_DIM, N_EMBD, 15),
        matrix('blk.0.attn_output.weight', N_EMBD, N_EMBD, 16),
        norm('blk.0.ffn_norm.weight', N_EMBD, 17),
        matrix('blk.0.ffn_gate.weight', N_FF, N_EMBD, 18),
        matrix('blk.0.ffn_up.weight', N_FF, N_EMBD, 19),
        matrix('blk.0.ffn_down.weight', N_EMBD, N_FF, 20),
        norm('output_norm.weight', N_EMBD, 21),
        matrix('output.weight', VOCAB, N_EMBD, 22),
    ]

    tokens = [f'<{i}>' for i in range(VOCAB)]

    header = b'GGUF' + struct.pack('<I', 3)
    header += struct.pack('<Q', len(tensors)) + struct.pack('<Q', 10)

    metadata = b''
    metadata += kv('general.architecture', 8, 'llama')
    metadata += kv('llama.embedding_length', 4, N_EMBD)
    metadata += kv('llama.feed_forward_length', 4, N_FF)
    metadata += kv('llama.block_count', 4, N_LAYER)
    metadata += kv('llama.attention.head_count', 4, N_HEAD)
    metadata += kv('llama.attention.head_count_kv', 4, N_HEAD_KV)
    metadata += kv('llama.rope.dimension_count', 4, HEAD_DIM)
    metadata += kv('llama.context_length', 4, MAX_SEQ)
    metadata += kv('llama.attention.layer_norm_rms_epsilon', 6, EPS)
    metadata += kv('tokenizer.ggml.tokens', 9, tokens)

    # Lay out tensor data with 32-byte alignment.
    data = bytearray()
    infos = b''
    for name, dims, ttype, blob in tensors:
        while len(data) % 32 != 0:
            data.append(0)
        off = len(data)
        data += blob
        infos += pack_str(name) + struct.pack('<I', len(dims))
        for d in dims:
            infos += struct.pack('<Q', d)
        infos += struct.pack('<I', ttype) + struct.pack('<Q', off)

    body = header + metadata + infos
    while len(body) % 32 != 0:
        body += b'\x00'
    body += data

    with open(out_path, 'wb') as f:
        f.write(body)
    print(f'wrote {out_path}: {len(body)} bytes, {len(tensors)} tensors')


if __name__ == '__main__':
    main()
