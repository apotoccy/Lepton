#!/usr/bin/env python3
import struct
with open('models/tinystories.gguf', 'rb') as f:
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
    if name.startswith('blk.0.') or name in ('token_embd.weight', 'output.weight', 'output_norm.weight'):
        print(f"{name:32s} dims={dims} type={ttype}")
