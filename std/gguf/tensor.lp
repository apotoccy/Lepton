/* GGUF tensor table parser.

   Each tensor entry in the info table:
     u64  name_length
     u8[] name
     u32  n_dims         (max 4 in practice)
     u64  dims[n_dims]
     u32  ggml_type
     u64  data_offset    (relative to the start of the tensor data section)

   The tensor data section starts after the info table, aligned to
   general.alignment. This module parses the info table only; loading
   the tensor data is a separate step.

   ggml_type values (subset):
     0  F32     4 bytes per element
     1  F16     2 bytes per element
     2  Q4_0    32 elements per 18 bytes
     8  Q8_0    32 elements per 34 bytes
     10 Q2_K    32 elements per 84 bytes
     12 Q4_K    32 elements per 144 bytes */

struct GGUFTensorInfo {
    name_offset: i64;
    name_len: u64;
    n_dims: u32;
    dim0: u64;
    dim1: u64;
    dim2: u64;
    dim3: u64;
    ggml_type: u32;
    data_offset: u64;
}

fn tensor_info_size(buf: ptr[u8], o: i64) -> i64 {
    let name_len = read_u64(buf, o);
    let n_dims = read_u32(buf, o + 8 + (name_len as i64));
    return 8 + (name_len as i64) + 4 + 8 * (n_dims as i64) + 4 + 8;
}

fn parse_tensor_info(buf: ptr[u8], o: i64) -> GGUFTensorInfo {
    let name_len = read_u64(buf, o);
    let name_start = o + 8;
    var p = name_start + (name_len as i64);
    let n_dims = read_u32(buf, p);
    p = p + 4;

    var d0 = 1 as u64;
    var d1 = 1 as u64;
    var d2 = 1 as u64;
    var d3 = 1 as u64;
    for i in 0..(n_dims as i64) {
        let v = read_u64(buf, p);
        p = p + 8;
        if (i == 0) { d0 = v; }
        if (i == 1) { d1 = v; }
        if (i == 2) { d2 = v; }
        if (i == 3) { d3 = v; }
    }

    let ty = read_u32(buf, p);
    p = p + 4;
    let off = read_u64(buf, p);

    return GGUFTensorInfo(
        name_offset: name_start,
        name_len:    name_len,
        n_dims:      n_dims,
        dim0:        d0,
        dim1:        d1,
        dim2:        d2,
        dim3:        d3,
        ggml_type:   ty,
        data_offset: off,
    );
}

fn find_tensor_by_index(buf: ptr[u8], first_offset: i64,
                        idx: u64) -> GGUFTensorInfo {
    var o = first_offset;
    for i in 0..(idx as i64) {
        o = o + tensor_info_size(buf, o);
    }
    return parse_tensor_info(buf, o);
}

fn tensor_element_count(t: GGUFTensorInfo) -> u64 {
    return t.dim0 * t.dim1 * t.dim2 * t.dim3;
}

fn tensor_byte_size(t: GGUFTensorInfo) -> u64 {
    let n = tensor_element_count(t);
    if (t.ggml_type == 0 as u32) {           /* F32 */
        return n * 4 as u64;
    }
    if (t.ggml_type == 1 as u32) {           /* F16 */
        return n * 2 as u64;
    }
    if (t.ggml_type == 8 as u32) {           /* Q8_0 */
        return (n / 32 as u64) * 34 as u64;
    }
    if (t.ggml_type == 2 as u32) {           /* Q4_0 */
        return (n / 32 as u64) * 18 as u64;
    }
    if (t.ggml_type == 12 as u32) {          /* Q4_K */
        return (n / 32 as u64) * 144 as u64;
    }
    println("unsupported ggml_type = {t.ggml_type}");
    panic("gguf: unsupported tensor type for size computation");
        return 0 as u64;
}
