import "std/gguf/model.lp";

fn put_u8(b: ptr[[u8; 16777216]], o: i64, v: u8) {
    b[o] = v;
}
fn put_u32(b: ptr[[u8; 16777216]], o: i64, v: u32) {
    b[o]     = ( v        & 0xFF as u32) as u8;
    b[o + 1] = ((v >>  8) & 0xFF as u32) as u8;
    b[o + 2] = ((v >> 16) & 0xFF as u32) as u8;
    b[o + 3] = ((v >> 24) & 0xFF as u32) as u8;
}
fn put_u64(b: ptr[[u8; 16777216]], o: i64, v: u64) {
    put_u32(b, o,     (v & 0xFFFFFFFF as u64) as u32);
    put_u32(b, o + 4, (v >> 32) as u32);
}
fn put_f32(b: ptr[[u8; 16777216]], o: i64, v: f32) {
    let tmp = alloc([f32; 1]);
    tmp[0] = v;
    let bytes = tmp as ptr[[u8; 4]];
    b[o]     = bytes[0];
    b[o + 1] = bytes[1];
    b[o + 2] = bytes[2];
    b[o + 3] = bytes[3];
}
fn put_bytes(b: ptr[[u8; 16777216]], o: i64, s: str) -> i64 {
    var i = 0;
    while (s[i] != '\0') {
        b[o + (i as i64)] = s[i] as u8;
        i = i + 1;
    }
    return o + (i as i64);
}

fn main() {
    let buf = alloc([u8; 16777216]);

    /* Header: magic + version 3 + tensor_count 1 + kv_count 0. */
    put_u8(buf, 0, 71 as u8);
    put_u8(buf, 1, 71 as u8);
    put_u8(buf, 2, 85 as u8);
    put_u8(buf, 3, 70 as u8);
    put_u32(buf, 4, 3 as u32);
    put_u64(buf, 8, 1 as u64);
    put_u64(buf, 16, 0 as u64);

    /* Tensor info: name_len 7, "weights", n_dims 1, dims[0] 16,
       ggml_type 0 (F32), data_offset 0. Starts at offset 24. */
    var o = 24;
    put_u64(buf, o, 7 as u64); o = o + 8;
    o = put_bytes(buf, o, "weights");
    put_u32(buf, o, 1 as u32); o = o + 4;
    put_u64(buf, o, 16 as u64); o = o + 8;
    put_u32(buf, o, 0 as u32); o = o + 4;
    put_u64(buf, o, 0 as u64); o = o + 8;

    /* Tensor info table ends at offset 63. Align to 32 -> 64. Pad one byte. */
    put_u8(buf, 63, 0 as u8);

    /* Tensor data at offset 64: f32 values 1.0, 2.0, ..., 16.0. */
    for i in 0..16 {
        put_f32(buf, 64 + (i as i64) * 4, (i as f32) + 1.0f);
    }

    write_file("t.gguf", buf, 128);

    /* Load it back through the model API. */
    let model = alloc([u8; 16777216]);
    let info = load_model_file(model, "t.gguf");

    println("version = {info.version}");
    println("tensor_count = {info.tensor_count}");
    println("kv_count = {info.kv_count}");
    println("tensor_table_offset = {info.tensor_table_offset}");
    println("data_section_offset = {info.data_section_offset}");
    println("file_size = {info.file_size}");

    /* Locate the one tensor and check its info struct. */
    let t = parse_tensor_info(model, info.tensor_table_offset);
    println("name_len = {t.name_len}");
    println("n_dims = {t.n_dims}");
    println("dim0 = {t.dim0}");
    println("ggml_type = {t.ggml_type}");

    /* Read a couple of elements from the tensor data. */
    let data = tensor_data_offset(info, t);
    println("first = {read_f32(model, data)}");
    println("last = {read_f32(model, data + 60)}");
}
