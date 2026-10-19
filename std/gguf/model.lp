/* GGUF whole-file loader.

   Loads a complete GGUF file into a caller-supplied byte buffer and
   extracts the offsets of its parts:

     [24 bytes]   header
     [N bytes]    metadata KV block (kv_count entries)
     [M bytes]    tensor info table (tensor_count entries)
     [padding]    to reach alignment
     [rest]       tensor data, laid out per each tensor's data_offset

   The buffer size is a compile-time constant because `alloc` requires
   it. 16 MB fits:
     - F32   up to ~4M parameters
     - Q8_0  up to ~16M parameters
     - Q4_0  up to ~32M parameters
   Larger models need dynamic allocation, on the roadmap.

   This module reads `general.alignment` from the KV block when present,
   defaulting to 32 per the GGUF spec. */

import "std/gguf/header.lp";
import "std/gguf/kv.lp";
import "std/gguf/tensor.lp";

struct GGUFModelInfo {
    version: u32;
    tensor_count: u64;
    kv_count: u64;
    tensor_table_offset: i64;
    data_section_offset: i64;
    file_size: i64;
}

fn read_general_alignment(buf: ptr[[u8; 262144]], kv_count: u64) -> u32 {
    if (kv_count == 0 as u64) {
        return 32 as u32;
    }
    let off = find_kv(buf, 0, kv_count, "general.alignment");
    if (off < 0) {
        return 32 as u32;
    }
    return read_u32(buf, off);
}

fn load_model_file(buf: ptr[[u8; 16777216]], path: str) -> GGUFModelInfo {
    let sz = file_size(path);
    if (sz < 24) {
        panic("gguf: file smaller than header");
    }
    if (sz > 16777216 as i64) {
        panic("gguf: model exceeds 16 MB buffer");
    }

    read_file_at(path, 0, buf, sz);

    let magic = read_u32(buf, 0);
    if (magic != 0x46554747 as u32) {
        panic("gguf: bad magic (not a GGUF file)");
    }

    let version = read_u32(buf, 4);
    let tensor_count = read_u64(buf, 8);
    let kv_count = read_u64(buf, 16);

    let kv_end = skip_kv_block(buf, 24, kv_count);
    let tensor_table_offset = kv_end;

    /* Walk the tensor info table to find where it ends. */
    var o = tensor_table_offset;
    for i in 0..(tensor_count as i64) {
        o = o + tensor_info_size(buf, o);
    }

    /* Align up to `general.alignment` (default 32). */
    let align = read_general_alignment(buf, kv_count);
    let a = align as i64;
    let rem = o % a;
    var data_section = o;
    if (rem != 0) {
        data_section = o + (a - rem);
    }

    return GGUFModelInfo(
        version:             version,
        tensor_count:        tensor_count,
        kv_count:            kv_count,
        tensor_table_offset: tensor_table_offset,
        data_section_offset: data_section,
        file_size:           sz,
    );
}

fn tensor_data_offset(info: GGUFModelInfo, t: GGUFTensorInfo) -> i64 {
    return info.data_section_offset + (t.data_offset as i64);
}
