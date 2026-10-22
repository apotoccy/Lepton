#!/usr/bin/env python3
"""Reference checks for the simple Q5_K/Q6_K fixtures."""
import struct, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'build/q5k.gguf'
data = open(path, 'rb').read()
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
block = data[o+toff:]
x = [((i % 7) - 3) for i in range(256)]
if ttype == 6:
    d = struct.unpack('<e', block[0:2])[0]
    qh = struct.unpack('<I', block[2:6])[0]
    qs = block[6:22]
    lo = []
    hi = []
    for j in range(16):
        byte = qs[j]
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        lo.append(d * (((byte & 0xF) | xh0) - 16))
        hi.append(d * (((byte >> 4) | xh1) - 16))
    vals = lo + hi
elif ttype == 7:
    d = struct.unpack('<e', block[0:2])[0]
    m = struct.unpack('<e', block[2:4])[0]
    qh = struct.unpack('<I', block[4:8])[0]
    qs = block[8:24]
    lo = []
    hi = []
    for j in range(16):
        byte = qs[j]
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        lo.append(d * ((byte & 0xF) | xh0) + m)
        hi.append(d * ((byte >> 4) | xh1) + m)
    vals = lo + hi
elif ttype == 13:
    d = struct.unpack('<e', block[0:2])[0]
    dmin = struct.unpack('<e', block[2:4])[0]
    scales = block[4:16]; qh = block[16:48]; ql = block[48:176]
    def scale(j):
        return (scales[j] & 63) if j < 4 else ((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4))
    def mn(j):
        return (scales[j+4] & 63) if j < 4 else ((scales[j+4] >> 4) | (((scales[j-4] >> 4) & 3) << 4))
    vals = []
    u1, u2 = 1, 2
    for j4 in range(4):
        s0, m0 = scale(j4*2), mn(j4*2)
        s1, m1 = scale(j4*2+1), mn(j4*2+1)
        lo = [d*s0*((ql[j4*32+l] & 0xF) + (16 if qh[l] & u1 else 0)) - dmin*m0 for l in range(32)]
        hi = [d*s1*((ql[j4*32+l] >> 4) + (16 if qh[l] & u2 else 0)) - dmin*m1 for l in range(32)]
        vals.extend(lo); vals.extend(hi)
        u1 <<= 2; u2 <<= 2
elif ttype == 14:
    d = struct.unpack('<e', block[208:210])[0]
    ql = block[0:128]; qh = block[128:192]; sc = block[192:208]
    vals = []
    for chunk in range(2):
        q1s, q2s, q3s, q4s = [], [], [], []
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[chunk*64+l] & 0xF) | (((qh[chunk*32+l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[chunk*64+l+32] & 0xF) | (((qh[chunk*32+l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[chunk*64+l] >> 4) | (((qh[chunk*32+l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[chunk*64+l+32] >> 4) | (((qh[chunk*32+l] >> 6) & 3) << 4)) - 32
            def s8(x):
                return x - 256 if x > 127 else x
            q1s.append(d*s8(sc[chunk*8+is_])*q1)
            q2s.append(d*s8(sc[chunk*8+is_+2])*q2)
            q3s.append(d*s8(sc[chunk*8+is_+4])*q3)
            q4s.append(d*s8(sc[chunk*8+is_+6])*q4)
        vals.extend(q1s); vals.extend(q2s); vals.extend(q3s); vals.extend(q4s)
else:
    raise SystemExit('not Q5_K/Q6_K')
total = sum(vals); dot = sum(v * x[i] for i, v in enumerate(vals))
print(f'q{ttype} reference sum={total:.6f} dot={dot:.6f}')
