#!/usr/bin/env python3
"""Generate a tiny GGUF tokenizer with a merge list.

Used to exercise the kernel's merge-based BPE path. The tokenizer maps
the word `ab` to the single token `<0x80>`? No: it maps `ab` to "▁ab".
Expected kernel command: `encode ab` -> token 5 (' ab').
"""
import struct
import sys


def pack_str(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def kv_array_strings(key, values):
    out = pack_str(key) + struct.pack('<I', 9)
    out += struct.pack('<I', 8) + struct.pack('<Q', len(values))
    for s in values:
        out += pack_str(s)
    return out


def kv_string(key, value):
    return pack_str(key) + struct.pack('<I', 8) + pack_str(value)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'build/merge.gguf'
    tokens = ['<unk>', '▁', 'a', 'b', '▁a', '▁ab', '<0x80>']
    merges = ['▁ a', '▁a b']

    header = b'GGUF' + struct.pack('<I', 3)
    header += struct.pack('<Q', 0)  # no tensors
    header += struct.pack('<Q', 3)  # KV count
    metadata = kv_string('general.architecture', 'llama')
    metadata += kv_array_strings('tokenizer.ggml.tokens', tokens)
    metadata += kv_array_strings('tokenizer.ggml.merges', merges)
    body = header + metadata
    while len(body) % 32 != 0:
        body += b'\x00'
    with open(out_path, 'wb') as f:
        f.write(body)
    print(f'wrote {out_path}: {len(body)} bytes, {len(tokens)} tokens, {len(merges)} merges')


if __name__ == '__main__':
    main()
