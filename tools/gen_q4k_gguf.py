#!/usr/bin/env python3
"""Generate a tiny Q4_K dequantization fixture with one 256-element tensor."""
import struct, sys

out_path = sys.argv[1] if len(sys.argv) > 1 else 'build/q4k.gguf'

def pack_str(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack('<Q', len(b)) + b

header = b'GGUF' + struct.pack('<I', 3)
header += struct.pack('<Q', 1)  # tensors
header += struct.pack('<Q', 0)  # kv
name = b'test.weight'
info = pack_str(name) + struct.pack('<I', 2)
info += struct.pack('<Q', 256) + struct.pack('<Q', 1)
info += struct.pack('<I', 12) + struct.pack('<Q', 0)
body = header + info
while len(body) % 32 != 0:
    body += b'\x00'

block = bytearray()
block += struct.pack('<e', 0.5)   # d
block += struct.pack('<e', 0.25)  # dmin
# Eight scales = 1, eight mins = 0, encoded as 6-bit pairs.
scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
block += bytes(scales)
for i in range(128):
    block += bytes([(i * 7 + 3) & 0xFF])
body += block
with open(out_path, 'wb') as f:
    f.write(body)
print(f'wrote {out_path}: {len(body)} bytes')
