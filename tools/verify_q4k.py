#!/usr/bin/env python3
"""Independent Q4_K dequantization check for the tiny fixture."""
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'build/q4k.gguf'
data = open(path, 'rb').read()
# tensor info starts at 24 with no KV; data starts at the first 32-byte boundary
o = 24
n = struct.unpack('<Q', data[o:o+8])[0]
o += 8 + n
nd = struct.unpack('<I', data[o:o+4])[0]
o += 4 + nd * 8
ttype = struct.unpack('<I', data[o:o+4])[0]
o += 4
toff = struct.unpack('<Q', data[o:o+8])[0]
while o % 32 != 0:
    o += 1
start = o + toff
assert ttype == 12
block = data[start:start + 144]
d = struct.unpack('<e', block[0:2])[0]
dmin = struct.unpack('<e', block[2:4])[0]
scales = block[4:16]
qs = block[16:144]

def scale(j):
    if j < 4:
        return scales[j] & 63
    return (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)

def mn(j):
    if j < 4:
        return scales[j+4] & 63
    return (scales[j+4] >> 4) | (((scales[j-4] >> 4) & 3) << 4)

vals = []
for j4 in range(4):
    s0, m0 = scale(j4*2), mn(j4*2)
    s1, m1 = scale(j4*2+1), mn(j4*2+1)
    lo = []
    hi = []
    for l in range(32):
        byte = qs[j4*32 + l]
        lo.append(d * s0 * (byte & 0xF) - dmin * m0)
        hi.append(d * s1 * (byte >> 4) - dmin * m1)
    vals.extend(lo)
    vals.extend(hi)
total = sum(vals)
dot = sum(v * ((i % 7) - 3) for i, v in enumerate(vals))
print(f'q4k reference sum={total:.6f} dot={dot:.6f}')
assert abs(total - 928.0) < 1e-6, total
assert abs(dot - (-20.5)) < 1e-6, dot
