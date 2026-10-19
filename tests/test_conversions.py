#!/usr/bin/env python3
"""Tests for the implicit-conversion and warning policy."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leptonc import compile_source, CompileError


def compile_src(src, strict=False):
    warnings = []
    compile_source(src, "<conversion-test>", strict_types=strict,
                   warnings_out=warnings)
    return warnings


def expect_strict_error(src, needle):
    try:
        compile_src(src, strict=True)
    except CompileError as e:
        text = str(e)
        if needle not in text:
            raise AssertionError(
                f"expected {needle!r} in error, got: {text}")
        return
    raise AssertionError(f"expected strict-type error containing {needle!r}")


def test_safe_integer_widening():
    src = '''
fn main() {
    let x: i64 = 1;
    println("{x}");
}
'''
    assert compile_src(src) == []
    assert compile_src(src, strict=True) == []


def test_lossy_numeric_warning():
    src = '''
fn main() {
    let x: i32 = 1.9;
    println("{x}");
}
'''
    warnings = compile_src(src)
    assert len(warnings) == 1, warnings
    text = str(warnings[0])
    assert "lossy numeric" in text
    assert "'f64' to 'i32'" in text


def test_strict_numeric_error():
    src = '''
fn main() {
    let x: i32 = 1.9;
    println("{x}");
}
'''
    expect_strict_error(src, "lossy numeric conversion")


def test_explicit_cast_no_warning():
    src = '''
fn main() {
    let x: i32 = 1.9 as i32;
    println("{x}");
}
'''
    assert compile_src(src) == []
    assert compile_src(src, strict=True) == []


def test_exact_float_constant():
    src = '''
fn main() {
    let x: f32 = 0.5;
    println("{x}");
}
'''
    assert compile_src(src) == []


def test_lossy_float_constant():
    src = '''
fn main() {
    let x: f32 = 0.1;
    println("{x}");
}
'''
    warnings = compile_src(src)
    assert len(warnings) == 1, warnings
    assert "lossy numeric" in str(warnings[0])


def test_byte_pointer_view():
    src = '''
fn take(p: ptr[u8]) -> void {
}

fn main() {
    let a = alloc([u8; 4]);
    take(a);
}
'''
    assert compile_src(src) == []
    assert compile_src(src, strict=True) == []


def test_unsafe_pointer_warning():
    src = '''
fn take(p: ptr[i32]) -> void {
}

fn main() {
    let a = alloc([f32; 4]);
    take(a);
}
'''
    warnings = compile_src(src)
    assert len(warnings) == 1, warnings
    assert "unsafe pointer" in str(warnings[0])


def test_strict_pointer_error():
    src = '''
fn take(p: ptr[i32]) -> void {
}

fn main() {
    let a = alloc([f32; 4]);
    take(a);
}
'''
    expect_strict_error(src, "unsafe pointer conversion")


def test_null_pointer_no_warning():
    src = '''
fn main() {
    let p: ptr[u8] = 0;
    println("{p}");
}
'''
    assert compile_src(src) == []


def test_zero_struct_fields():
    src = '''
struct S {
    a: u64;
    b: u32;
}

fn main() {
    let s = S(a: 0, b: 0);
    println("{s.a} {s.b}");
}
'''
    assert compile_src(src) == []


def test_int_to_float_allowed():
    src = '''
fn main() {
    let x: f32 = 1;
    println("{x}");
}
'''
    assert compile_src(src) == []


def main():
    tests = [v for name, v in sorted(globals().items())
             if name.startswith('test_') and callable(v)]
    for test in tests:
        test()
    print(f"Conversion policy: {len(tests)} tests passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
