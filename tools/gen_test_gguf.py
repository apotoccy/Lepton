#!/usr/bin/env python3
"""Generate a GGUF v3 file with one 768x768 Q4_0 tensor whose rows have
distinct sums. Per-block scale depends on the row, so row sums are
distinguishable: row 0 = -96, row 1 = -192, ..., row 7 = -768."""
import struct, sys

out = sys.argv[1] if len(sys.argv) > 1 else 'test.gguf'

d = b'GGUF'
d += struct.pack('<I', 3)
d += struct.pack('<Q', 1)
d += struct.pack('<Q', 0)

name = b'weights'
d += struct.pack('<Q', len(name))
d += name
d += struct.pack('<I', 2)
d += struct.pack('<Q', 768)
d += struct.pack('<Q', 768)
d += struct.pack('<I', 2)          # Q4_0
d += struct.pack('<Q', 0)

while len(d) % 32 != 0:
    d += b'\x00'

n_blocks = 768 * 768 // 32        # 18432
for b in range(n_blocks):
    row = b // 24                 # 24 blocks per row of 768 elements
    scale = 0.25 * (1 + (row & 0x7))
    d += struct.pack('<e', scale) # f16 scale
    for j in range(16):
        lo = (b + j) & 0x0F
        hi = (b - j) & 0x0F
        d += bytes([lo | (hi << 4)])

with open(out, 'wb') as f:
    f.write(d)

print(f"wrote {out}: {len(d)} bytes, {n_blocks} Q4_0 blocks")
