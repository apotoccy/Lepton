#!/usr/bin/env python3
"""Reference tokenizer — encode text and show token IDs.
Matches what the kernel's `encode` and `genprompt` commands do."""
import struct, sys

path = 'models/tinystories.gguf'
text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

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
o = 24
kv = {}
for _ in range(kc):
    klen = struct.unpack('<Q', data[o:o+8])[0]
    key = data[o+8:o+8+klen].decode('utf-8', errors='replace')
    o += 8 + klen
    ty = struct.unpack('<I', data[o:o+4])[0]
    kv[key] = (ty, o + 4)
    o = skip_value(data, o+4, ty)

# Tokens
tty, toff = kv['tokenizer.ggml.tokens']
et = struct.unpack('<I', data[toff:toff+4])[0]
cnt = struct.unpack('<Q', data[toff+4:toff+12])[0]
o = toff + 12
tokens = []
for _ in range(cnt):
    n = struct.unpack('<Q', data[o:o+8])[0]
    tokens.append(data[o+8:o+8+n])
    o += 8 + n
print(f"vocab: {cnt} tokens")

# Merges
mty, moff = kv['tokenizer.ggml.merges']
met = struct.unpack('<I', data[moff:moff+4])[0]
mcnt = struct.unpack('<Q', data[moff+4:moff+12])[0]
o = moff + 12
merges = []
for i in range(mcnt):
    n = struct.unpack('<Q', data[o:o+8])[0]
    merges.append(data[o+8:o+8+n])
    o += 8 + n
print(f"merges: {mcnt} entries")

# Build vocab ID dict
vocab = {t: i for i, t in enumerate(tokens)}
# Build merge rank dict: key "x y" -> rank
merge_rank = {m: i for i, m in enumerate(merges)}

def bpe(word: bytes) -> list:
    pieces = [bytes([c]) for c in word]
    while len(pieces) > 1:
        best_rank = len(merges) + 1
        best_idx = -1
        for i in range(len(pieces) - 1):
            pair = pieces[i] + b' ' + pieces[i+1]
            r = merge_rank.get(pair, None)
            if r is not None and r < best_rank:
                best_rank = r
                best_idx = i
        if best_idx == -1:
            break
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx+1]
        del pieces[best_idx+1]
    return pieces

def encode(txt: str) -> list:
    ids = []
    for word in txt.split():
        wb = b'\xe2\x96\x81' + word.encode('utf-8')
        for piece in bpe(wb):
            if piece in vocab:
                ids.append(vocab[piece])
            else:
                print(f"  warn: '{piece}' not in vocab")
                ids.append(0)
    return ids

ids = encode(text)
print(f"\nencode({text!r}) -> {len(ids)} tokens")
for i, tid in enumerate(ids):
    tok = tokens[tid]
    dec = tok.replace(b'\xe2\x96\x81', b' ')
    print(f"  [{i}] {tid} = {dec!r}")
