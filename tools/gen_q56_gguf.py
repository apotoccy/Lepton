#!/usr/bin/env python3
"""Generate a simple Q5_K or Q6_K fixture: qtype 13 or 14."""
import struct, sys
qtype = int(sys.argv[1]) if len(sys.argv) > 1 else 13
out = sys.argv[2] if len(sys.argv) > 2 else (f'build/q{qtype}.gguf')

def pack_str(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack('<Q', len(b)) + b

header = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 1) + struct.pack('<Q', 0)
name = b'test.weight'
info = pack_str(name) + struct.pack('<I', 2)
info += struct.pack('<Q', 256) + struct.pack('<Q', 1)
info += struct.pack('<I', qtype) + struct.pack('<Q', 0)
body = header + info
while len(body) % 32 != 0:
    body += b'\x00'
if qtype == 6:
    block = struct.pack('<e', 0.5) + bytes(4) + bytes([0x11]) * 16
elif qtype == 7:
    block = struct.pack('<e', 0.5) + struct.pack('<e', 0.25) + bytes(4) + bytes([0x11]) * 16
elif qtype == 13:
    block = struct.pack('<e', 0.5) + struct.pack('<e', 0.25)
    block += bytes([1,1,1,1,0,0,0,0,1,1,1,1])
    block += bytes(32) + bytes([0x11]) * 128
elif qtype == 14:
    block = bytes([0x11]) * 128 + bytes(64) + bytes([1]) * 16
    block += struct.pack('<e', 0.5)
else:
    raise SystemExit('qtype must be 13 or 14')
body += block
open(out, 'wb').write(body)
print(f'wrote {out}: {len(body)} bytes type={qtype}')
