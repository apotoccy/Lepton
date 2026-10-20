#!/usr/bin/env python3
"""Full reference forward pass for TinyStories.

Runs the tokenized prompt `Once upon a time` through Q/K/V attention,
RoPE, the KV cache, six layers, and the output projection. Pure Python,
no numpy; Q4_0/Q8_0/F32 tensors only.
"""
import math
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else 'models/tinystories.gguf'
with open(PATH, 'rb') as f:
    data = f.read()


def skip_value(d, o, t):
    if t == 8:
        return o + 8 + struct.unpack('<Q', d[o:o+8])[0]
    if t == 9:
        et = struct.unpack('<I', d[o:o+4])[0]
        n = struct.unpack('<Q', d[o+4:o+12])[0]
        o += 12
        if et in (8, 9):
            for _ in range(n):
                o = skip_value(d, o, et)
            return o
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        return o + n * sz[et]
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    return o + sz[t]


kc = struct.unpack('<Q', data[16:24])[0]
tc = struct.unpack('<Q', data[8:16])[0]

o = 24
meta = {}
for _ in range(kc):
    n = struct.unpack('<Q', data[o:o+8])[0]
    key = data[o+8:o+8+n].decode()
    o += 8 + n
    t = struct.unpack('<I', data[o:o+4])[0]
    vo = o + 4
    if t == 8:
        ln = struct.unpack('<Q', data[vo:vo+8])[0]
        meta[key] = data[vo+8:vo+8+ln].decode()
    elif t == 4:
        meta[key] = struct.unpack('<I', data[vo:vo+4])[0]
    elif t == 6:
        meta[key] = struct.unpack('<f', data[vo:vo+4])[0]
    o = skip_value(data, o + 4, t)

tensors = {}
for _ in range(tc):
    nlen = struct.unpack('<Q', data[o:o+8])[0]
    name = data[o+8:o+8+nlen].decode()
    o += 8 + nlen
    nd = struct.unpack('<I', data[o:o+4])[0]
    o += 4
    dims = [struct.unpack('<Q', data[o+i*8:o+(i+1)*8])[0] for i in range(nd)]
    o += nd * 8
    ttype = struct.unpack('<I', data[o:o+4])[0]
    o += 4
    tdoff = struct.unpack('<Q', data[o:o+8])[0]
    o += 8
    tensors[name] = (dims, ttype, tdoff)

while o % 32 != 0:
    o += 1
data_start = o

n_embd = meta['llama.embedding_length']
n_ff = meta['llama.feed_forward_length']
n_layer = meta['llama.block_count']
n_head = meta['llama.attention.head_count']
n_head_kv = meta.get('llama.attention.head_count_kv', n_head)
head_dim = meta['llama.rope.dimension_count']
max_seq = meta['llama.context_length']
eps = meta['llama.attention.layer_norm_rms_epsilon']
vocab = tensors['token_embd.weight'][0][1]
kv_dim = n_head_kv * head_dim


def dequant_q4(raw, n):
    out = [0.0] * n
    for b in range(n // 32):
        off = b * 18
        scale = struct.unpack('<e', raw[off:off+2])[0]
        for j in range(16):
            byte = raw[off + 2 + j]
            out[b*32 + j] = scale * ((byte & 0xF) - 8)
            out[b*32 + j + 16] = scale * (((byte >> 4) & 0xF) - 8)
    return out


def dequant_q8(raw, n):
    out = [0.0] * n
    for b in range(n // 32):
        off = b * 34
        scale = struct.unpack('<e', raw[off:off+2])[0]
        for i in range(32):
            v = raw[off + 2 + i]
            if v > 127:
                v -= 256
            out[b*32 + i] = scale * v
    return out


def load(name):
    dims, ttype, tdoff = tensors[name]
    rows = dims[1] if len(dims) > 1 else 1
    cols = dims[0]
    n = rows * cols
    base = data_start + tdoff
    if ttype == 2:
        flat = dequant_q4(data[base:base + (n // 32) * 18], n)
    elif ttype == 8:
        flat = dequant_q8(data[base:base + (n // 32) * 34], n)
    elif ttype == 0:
        flat = list(struct.unpack(f'<{n}f', data[base:base + n*4]))
    else:
        raise ValueError(f'unsupported tensor type {ttype}')
    return [flat[r*cols:(r+1)*cols] for r in range(rows)]


def matvec(m, x):
    return [sum(row[j] * x[j] for j in range(len(x))) for row in m]


def rmsnorm(x, w):
    ms = sum(v*v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [x[i] * inv * w[i] for i in range(len(x))]


def silu(x):
    return x / (1.0 + math.exp(-x))


def rope(x, pos, n_heads):
    half = head_dim // 2
    freqs = [math.exp(-2.0 * i / head_dim * math.log(10000.0))
             for i in range(half)]
    for h in range(n_heads):
        for i in range(half):
            angle = pos * freqs[i]
            c = math.cos(angle)
            s = math.sin(angle)
            base = h * head_dim + i * 2
            a = x[base]
            b = x[base + 1]
            x[base] = a * c - b * s
            x[base + 1] = a * s + b * c


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


emb = load('token_embd.weight')
cache_k = [[[0.0] * kv_dim for _ in range(max_seq)]
           for _ in range(n_layer)]
cache_v = [[[0.0] * kv_dim for _ in range(max_seq)]
           for _ in range(n_layer)]


def step(token, pos):
    hidden = list(emb[token])
    for L in range(n_layer):
        an = load(f'blk.{L}.attn_norm.weight')[0]
        normed = rmsnorm(hidden, an)
        q = matvec(load(f'blk.{L}.attn_q.weight'), normed)
        k = matvec(load(f'blk.{L}.attn_k.weight'), normed)
        v = matvec(load(f'blk.{L}.attn_v.weight'), normed)
        rope(q, pos, n_head)
        rope(k, pos, n_head_kv)
        cache_k[L][pos] = list(k)
        cache_v[L][pos] = list(v)

        attended = [0.0] * n_embd
        group = n_head // n_head_kv
        for h in range(n_head):
            q_off = h * head_dim
            kv_off = (h // group) * head_dim
            scores = []
            for p in range(pos + 1):
                acc = 0.0
                for d in range(head_dim):
                    acc += q[q_off + d] * cache_k[L][p][kv_off + d]
                scores.append(acc / math.sqrt(head_dim))
            probs = softmax(scores)
            for d in range(head_dim):
                acc = 0.0
                for p in range(pos + 1):
                    acc += probs[p] * cache_v[L][p][kv_off + d]
                attended[q_off + d] = acc

        attn = matvec(load(f'blk.{L}.attn_output.weight'), attended)
        hidden = [hidden[i] + attn[i] for i in range(n_embd)]

        fn = load(f'blk.{L}.ffn_norm.weight')[0]
        normed = rmsnorm(hidden, fn)
        gate = matvec(load(f'blk.{L}.ffn_gate.weight'), normed)
        up = matvec(load(f'blk.{L}.ffn_up.weight'), normed)
        interm = [silu(gate[i]) * up[i] for i in range(n_ff)]
        down = matvec(load(f'blk.{L}.ffn_down.weight'), interm)
        hidden = [hidden[i] + down[i] for i in range(n_embd)]

    onf = load('output_norm.weight')[0]
    normed = rmsnorm(hidden, onf)
    return matvec(load('output.weight'), normed)


# "Once upon a time" as tokenized by the TinyStories tokenizer.
prompt = [9038, 2501, 263, 931]
if len(sys.argv) > 2:
    prompt = [int(x) for x in sys.argv[2].split(',')]
logits = None
for pos, token in enumerate(prompt):
    logits = step(token, pos)
best = max(range(len(logits)), key=lambda i: logits[i])
print(f'forward reference: n_embd={n_embd} n_head={n_head} '
      f'n_head_kv={n_head_kv} n_layer={n_layer}')
print(f'prompt = {prompt}')
print(f'predicted token = {best}  logit = {logits[best]:.6f}')
top = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
print('top5 =', ' '.join(f'{i}:{logits[i]:.3f}' for i in top))
