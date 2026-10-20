#!/usr/bin/env python3
"""Verify the tiny merge-tokenizer fixture independently."""
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else 'build/merge.gguf'
with open(PATH, 'rb') as f:
    data = f.read()

kc = struct.unpack('<Q', data[16:24])[0]
meta = {}
o = 24
for _ in range(kc):
    n = struct.unpack('<Q', data[o:o+8])[0]
    key = data[o+8:o+8+n].decode()
    o += 8 + n
    t = struct.unpack('<I', data[o:o+4])[0]
    vo = o + 4
    if t == 8:
        ln = struct.unpack('<Q', data[vo:vo+8])[0]
        meta[key] = data[vo+8:vo+8+ln].decode()
        o = vo + 8 + ln
    elif t == 9:
        count = struct.unpack('<Q', data[vo+4:vo+12])[0]
        p = vo + 12
        vals = []
        for _ in range(count):
            ln = struct.unpack('<Q', data[p:p+8])[0]
            vals.append(data[p+8:p+8+ln].decode())
            p += 8 + ln
        meta[key] = vals
        o = p
    else:
        raise SystemExit(f'unsupported metadata type {t}')

tokens = meta['tokenizer.ggml.tokens']
merges = meta['tokenizer.ggml.merges']
pieces = [c.encode() for c in '▁ab']
for merge in merges:
    x, y = (p.encode() for p in merge.split(' ', 1))
    for i in range(len(pieces) - 1):
        if pieces[i] == x and pieces[i + 1] == y:
            pieces[i:i + 2] = [x + y]
            break
best = None
for i, tok in enumerate(tokens):
    if tok.encode() in pieces:
        best = i
print(f'merge reference: pieces={pieces!r} token={best} value={tokens[best]!r}')
assert best == 5, best
