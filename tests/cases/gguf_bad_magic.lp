import "std/gguf/header.lp";

fn main() {
    /* 24 bytes of the wrong magic. parse_header should panic. */
    let b = alloc([u8; 24]);
    for i in 0..24 {
        b[i] = 0 as u8;
    }
    write_file("bad.gguf", b, 24);

    println("about to parse a non-GGUF file");
    let h = parse_header("bad.gguf");
    /* If we reach here, the magic check failed to fire. */
    println("bad: parser accepted garbage version={h.version}");
}
