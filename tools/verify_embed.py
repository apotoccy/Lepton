#!/usr/bin/env python3
"""Verify kernel's `embed N` command against GGUF on disk."""
import struct, sys

path = 'models/tinystories.gguf'
token_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

# Skip KV block to reach tensor info table
o = 24
for _ in range(kc):
    n = struct.unpack('<Q', data[o:o+8])[0]
    o += 8 + n
    t = struct.unpack('<I', data[o:o+4])[0]
    o = skip_value(data, o+4, t)

# Walk ALL tensor infos; find token_embd
token_embd = None
for _ in range(tc):
    nlen = struct.unpack('<Q', data[o:o+8])[0]
    name = data[o+8:o+8+nlen]
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
    if name == b'token_embd.weight':
        token_embd = (dims, ttype, tdoff)

# Align to 32 for data section start
while o % 32 != 0: o += 1
data_section_start = o

if token_embd is None:
    print("token_embd.weight not found"); sys.exit(1)

dims, ttype, tdoff = token_embd
emb_dim = dims[0]        # 288
blocks_per_row = emb_dim // 32

start = data_section_start + tdoff + token_id * blocks_per_row * 18
row = data[start : start + blocks_per_row * 18]

out = []
for b in range(blocks_per_row):
    off = b * 18
    scale = struct.unpack('<e', row[off:off+2])[0]
    for j in range(16):
        out.append(scale * ((row[off+2+j] & 0x0F) - 8))
    for j in range(16):
        out.append(scale * (((row[off+2+j] >> 4) & 0x0F) - 8))

print(f"host token {token_id} d[0..3] = "
      f"{out[0]:.6f} {out[1]:.6f} {out[2]:.6f} {out[3]:.6f}")
