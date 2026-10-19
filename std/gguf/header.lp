/* Minimal GGUF v3 header parser.
   Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

   A GGUF file begins with 24 bytes:
     offset  size  field
     0       4     magic       "GGUF" = 0x46554747 in LE
     4       4     version     u32, currently 3
     8       8     tensor_count u64
     16      8     kv_count    u64

   The tensor table and metadata follow immediately. This module parses
   only the fixed-size header; tensor and metadata parsing are separate. */

struct GGUFHeader {
    version: u32;
    tensor_count: u64;
    kv_count: u64;
}

const GGUF_MAGIC: u32 = 0x46554747 as u32;

fn parse_header(path: str) -> GGUFHeader {
    let buf = alloc([u8; 24]);
    let n = read_file_at(path, 0, buf, 24);
    if (n != 24) {
        panic("gguf: file too small to be a valid header");
    }

    let magic = read_u32(buf, 0);
    if (magic != GGUF_MAGIC) {
        panic("gguf: bad magic (not a GGUF file)");
    }

    return GGUFHeader(version: read_u32(buf, 4),
                      tensor_count: read_u64(buf, 8),
                      kv_count: read_u64(buf, 16),
    );
}
