/* GGUF metadata KV parser.

   Format per entry:
     u64  key_length
     u8[] key           (key_length bytes, no NUL)
     u32  value_type    (0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32,
                         7=bool, 8=string, 9=array, 10=u64, 11=i64, 12=f64)
     val  value         (depends on type)

   All operations take a ptr to a caller-allocated [u8; 262144] buffer
   holding the KV block (read starting at offset 24 of the GGUF file).
   The 256 KB limit is generous for metadata; tokenizer vocabularies
   exceed it and are loaded separately. */


fn kv_type_size(t: u32) -> i64 {
    if (t == 0) { return 1; }    /* u8 */
    if (t == 1) { return 1; }    /* i8 */
    if (t == 2) { return 2; }    /* u16 */
    if (t == 3) { return 2; }    /* i16 */
    if (t == 4) { return 4; }    /* u32 */
    if (t == 5) { return 4; }    /* i32 */
    if (t == 6) { return 4; }    /* f32 */
    if (t == 7) { return 1; }    /* bool */
    if (t == 10) { return 8; }   /* u64 */
    if (t == 11) { return 8; }   /* i64 */
    if (t == 12) { return 8; }   /* f64 */
    return -1;                   /* variable-length: string, array */
}

fn skip_kv_value(buf: ptr[u8], offset: i64, t: u32) -> i64 {
    if (t == 8) {
        let len = read_u64(buf, offset);
        return offset + 8 + (len as i64);
    }
    if (t == 9) {
        let elem_t = read_u32(buf, offset);
        let count  = read_u64(buf, offset + 4);
        var o = offset + 12;
        if (elem_t == 8 || elem_t == 9) {
            for i in 0..(count as i64) {
                o = skip_kv_value(buf, o, elem_t);
            }
            return o;
        }
        let sz = kv_type_size(elem_t);
        if (sz < 0) {
            panic("gguf kv: unknown array element type");
        }
        return o + (count as i64) * sz;
    }
    let sz = kv_type_size(t);
    if (sz < 0) {
        panic("gguf kv: unknown value type");
    }
    return offset + sz;
}

fn skip_kv_block(buf: ptr[u8], offset: i64,
                 count: u64) -> i64 {
    var o = offset;
    for i in 0..(count as i64) {
        let key_len = read_u64(buf, o);
        o = o + 8 + (key_len as i64);
        let t = read_u32(buf, o);
        o = o + 4;
        o = skip_kv_value(buf, o, t);
    }
    return o;
}

fn bytes_match(buf: ptr[u8], offset: i64,
               len: u64, key: str) -> bool {
    var i = 0;
    while ((i as u64) < len) {
        let a = read_u8(buf, offset + (i as i64));
        let b = key[i] as u8;
        if (a != b) {
            return false;
        }
        i = i + 1;
    }
    if (key[i] != '\0') {
        return false;
    }
    return true;
}

/* Return the byte offset of the value for a given key, or -1 if not
   found. Offsets are relative to the start of buf. */
fn find_kv(buf: ptr[u8], offset: i64, count: u64,
           key: str) -> i64 {
    var o = offset;
    for i in 0..(count as i64) {
        let key_len = read_u64(buf, o);
        let key_start = o + 8;
        let t = read_u32(buf, key_start + (key_len as i64));
        let val_off = key_start + (key_len as i64) + 4;
        if (bytes_match(buf, key_start, key_len, key)) {
            return val_off;
        }
        o = skip_kv_value(buf, val_off, t);
    }
    return -1;
}
