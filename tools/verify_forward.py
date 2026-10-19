#!/usr/bin/env python3
"""Verify kernel's `forward` command against GGUF on disk.
Pure Python, no numpy. Handles Q4_0, Q8_0, and F32 tensors."""
import struct, sys, math

path = 'models/tinystories.gguf'
with open(path, 'rb') as f:
    data = f.read()

def skip_value(d, o, t):
    if t == 8: return o + 8 + struct.unpack('<Q', d[o:o+8])[0]
    if t == 9:
        et = struct.unpack('<I', d[o:o+4])[0]
        n = struct.unpack('<Q', d[o+4:o+12])[0]
        o += 12
        if et in (8, 9):
            for _ in range(n): o = skip_value(d, o, et)
            return o
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        return o + n * sz[et]
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    return o + sz[t]

kc = struct.unpack('<Q', data[16:24])[0]
tc = struct.unpack('<Q', data[8:16])[0]

o = 24
for _ in range(kc):
    n = struct.unpack('<Q', data[o:o+8])[0]
    o += 8 + n
    t = struct.unpack('<I', data[o:o+4])[0]
    o = skip_value(data, o+4, t)

tensors = {}
for _ in range(tc):
    nlen = struct.unpack('<Q', data[o:o+8])[0]
    name = data[o+8:o+8+nlen].decode()
    o += 8 + nlen
    nd = struct.unpack('<I', data[o:o+4])[0]
    o += 4
    dims = []
    for _ in range(nd):
        dims.append(struct.unpack('<Q', data[o:o+8])[0])
        o += 8
    ttype = struct.unpack('<I', data[o:o+4])[0]
    o += 4
    tdoff = struct.unpack('<Q', data[o:o+8])[0]
    o += 8
    tensors[name] = (dims, ttype, tdoff)

while o % 32 != 0: o += 1
data_start = o

def dequant_q4_0(raw, n):
    out = [0.0] * n
    for b in range(n // 32):
        off = b * 18
        scale = struct.unpack('<e', raw[off:off+2])[0]
        for j in range(16):
            byte = raw[off + 2 + j]
            out[b*32 + j]      = scale * ((byte & 0x0F) - 8)
            out[b*32 + j + 16] = scale * (((byte >> 4) & 0x0F) - 8)
    return out

def dequant_q8_0(raw, n):
    out = [0.0] * n
    for b in range(n // 32):
        off = b * 34
        scale = struct.unpack('<e', raw[off:off+2])[0]
        for i in range(32):
            v = raw[off + 2 + i]
            if v > 127: v -= 256
            out[b*32 + i] = scale * v
    return out

def load_tensor(name):
    dims, ttype, tdoff = tensors[name]
    if len(dims) == 1:
        rows, cols = 1, dims[0]
    else:
        rows, cols = dims[1], dims[0]
    n = rows * cols
    if ttype == 2:
        raw = data[data_start + tdoff : data_start + tdoff + (n // 32) * 18]
        flat = dequant_q4_0(raw, n)
    elif ttype == 8:
        raw = data[data_start + tdoff : data_start + tdoff + (n // 32) * 34]
        flat = dequant_q8_0(raw, n)
    elif ttype == 0:
        flat = list(struct.unpack(f'<{n}f',
            data[data_start + tdoff : data_start + tdoff + n * 4]))
    else:
        raise ValueError(f"unsupported type {ttype} for {name}")
    return [flat[r*cols:(r+1)*cols] for r in range(rows)]

def matvec(M, x):
    out = [0.0] * len(M)
    for i, row in enumerate(M):
        acc = 0.0
        for j in range(len(row)):
            acc += row[j] * x[j]
        out[i] = acc
    return out

def rmsnorm(x, w, eps=1e-5):
    ms = sum(xi * xi for xi in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    w_flat = w[0] if isinstance(w[0], list) else w
    return [x[i] * inv * w_flat[i] for i in range(len(x))]

def silu(x):
    return x / (1.0 + math.exp(-x))

# Token 1 = <s>
emb = load_tensor('token_embd.weight')
hidden = list(emb[1])
print(f"emb: {hidden[0]:.6f} {hidden[1]:.6f} {hidden[2]:.6f}")

def do_layer(L):
    global hidden
    an = load_tensor(f'blk.{L}.attn_norm.weight')
    av = load_tensor(f'blk.{L}.attn_v.weight')
    ao = load_tensor(f'blk.{L}.attn_output.weight')
    fn = load_tensor(f'blk.{L}.ffn_norm.weight')
    fg = load_tensor(f'blk.{L}.ffn_gate.weight')
    fu = load_tensor(f'blk.{L}.ffn_up.weight')
    fd = load_tensor(f'blk.{L}.ffn_down.weight')

    normed = rmsnorm(hidden, an)
    V = matvec(av, normed)
    attn = matvec(ao, V)
    hidden = [hidden[i] + attn[i] for i in range(len(hidden))]
    if L == 0:
        print(f"L0 attn: {hidden[0]:.6f} {hidden[1]:.6f} {hidden[2]:.6f}")

    normed = rmsnorm(hidden, fn)
    gate = matvec(fg, normed)
    up = matvec(fu, normed)
    interm = [silu(gate[i]) * up[i] for i in range(len(gate))]
    ffn = matvec(fd, interm)
    hidden = [hidden[i] + ffn[i] for i in range(len(hidden))]
    if L == 0:
        print(f"L0 ffn:  {hidden[0]:.6f} {hidden[1]:.6f} {hidden[2]:.6f}")

for L in range(6):
    do_layer(L)

print(f"final:  {hidden[0]:.6f} {hidden[1]:.6f} {hidden[2]:.6f}")

onf = load_tensor('output_norm.weight')
normed = rmsnorm(hidden, onf)
ow = load_tensor('output.weight')
logits = matvec(ow, normed)

best = max(range(len(logits)), key=lambda i: logits[i])
print(f"predicted token = {best}  logit = {logits[best]:.6f}")
