# Lepton Language Guide

Lepton is a statically typed, AI-native systems language that compiles to C11.
It is designed for bare-metal execution, explicit memory management, and
hardware-native numeric work. The compiler, runtime, and Lepton OS live in the
same repository.

This guide is an overview. The canonical implementation reference, including
the full conversion policy and current gaps, is
[docs/LANGUAGE.md](docs/LANGUAGE.md). For the OS, see
[DOC-LEPTONOS.md](DOC-LEPTONOS.md).

## Current status

- **121/121** compiler regression tests on scalar and x86-64 SSE2.
- 13 IR-verifier tests and 12 conversion-policy tests.
- 64-bit and 32-bit userland builds, freestanding 32-bit kernel, and GRUB ISO.
- `let` immutability, compound assignment, field/index/nested lvalues.
- Content-based string comparison and structural struct/array equality.
- Complete vector min/max/comparison/select operations.
- Explicit element-wise array casts.
- `free` and `alloc_reset` allocator builtins.

## 1. Source files and comments

A source file uses the `.lp` extension and contains top-level declarations:

```lepton
// Single-line comment
/* Multi-line
   comment */

import "std/nn/matmul.lp";

const LIMIT: i32 = 16;

struct Point { x: i32; y: i32; }

enum Color { Red, Green, Blue }

fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

Imports are resolved relative to the importing file, then through `.`, `./std`,
and `$LEPTON_PATH`. Imports are flattened; circular imports are an error.

## 2. Variables and mutability

```lepton
let immutable = 1;        // inferred i32
let explicit: i64 = 1;    // initializer converted to i64
var counter: i32 = 0;     // mutable
var zeroed: i32;          // zero-initialized
```

- `let` bindings are immutable; assigning to one is an error.
- `var` bindings are mutable.
- Function parameters are mutable local copies.
- A pointer binding may be `let` while its pointee is mutable:

```lepton
let buf = alloc([u8; 8]);
buf[0] = 1;               // allowed
// buf = ...;             // rejected: the binding is immutable
```

- Locals without an initializer are zero-initialized. There is no
  definite-assignment analysis.
- Blocks introduce scopes; shadowing is allowed.

## 3. Types

| Category | Types |
|---|---|
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |
| Floats | `f16`, `bf16`, `f32`, `f64` |
| Fixed-point | `q7`, `q15`, `q31` (range `[-1, 1)`, saturating arithmetic) |
| Other scalars | `bool`, `char`, `str` |
| Vectors | `v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8` |
| Aggregates | `[T; N]`, `ptr[T]`, structs, enums |

`str` is a null-terminated `const char*`. String literals support `{expr}`
interpolation and compare by content.

Numeric literals are `i32` unless they contain a decimal point or exponent, in
which case they are `f64`. The `f` suffix makes an `f32` literal and `h` makes
an `f16` literal:

```lepton
let a = 42;          // i32
let b = 1.5;         // f64
let c = 1.5f;        // f32
let d = 1.5h;        // f16
let e = 0.5 as q15;  // fixed-point
```

## 4. Assignment

```lepton
var x: i32 = 1;
x = 2;
x += 3;                 // compound assignment

var a: [i32; 4] = {1, 2, 3, 4};
a[0] = 9;
a[1] += 2;

var p = Point(x: 1, y: 2);
p.x = 7;
p.y += 1;

var grid: [[i32; 2]; 2];
grid[0][0] = 4;         // nested index
```

Compound operators are `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`,
and `>>=`. Assignment targets may be variables, fields, array/vector elements,
pointer elements, or nested combinations. Base and index expressions are
evaluated once. An arbitrary expression such as `(p + 1)[0] = ...` is not an
accepted lvalue.

Nested aggregate updates use a copy-modify-write sequence internally. They are
correct but not yet optimized.

## 5. Operators and comparisons

Arithmetic: `+ - * / %`; shifts `<< >>`; bitwise `& | ^ ~`; comparisons
`== != < <= > >=`; logical `&& || !`. Precedence follows C. `&&` and `||`
short-circuit.

`if`/`while` conditions and `&&`/`||`/`!` operands must be `bool`.

| Operands | Allowed comparisons |
|---|---|
| Numeric values | all six operators |
| `bool` | `==`, `!=` |
| Same enum type | all six operators |
| Same pointer type | all six operators |
| `str` | all six operators, content-based |
| Structs, arrays | `==`, `!=` only, structural |
| Vectors | use `vcmpeq`/`vselect` or element access |

Strings compare contents, not addresses. Null equals null and sorts before
non-null strings. Struct/array equality is recursive: strings by content,
vectors lane-wise, pointers by address, and floats with IEEE `==` (NaN is
unequal to itself). Aggregate ordering is rejected.

## 6. Control flow

```lepton
if (x > 10) {
    println("Big");
} else if (x > 5) {
    println("Medium");
} else {
    println("Small");
}

var i = 0;
while (i < 5) {
    i += 1;
}

for j in 0..5 { ... }     // 0, 1, 2, 3, 4
for j in 1..=5 { ... }    // 1, 2, 3, 4, 5

switch (x) {
    case 0, 1 { ... }
    case 2 { ... }
    default { ... }
}
```

`break` and `continue` are supported. `for` bounds must be integers. `switch`
works on integers, `char`, and enums; cases take constant values and there is
no implicit fallthrough. `break` exits the switch, while `continue` still
applies to the enclosing loop. There is no `do`/`while` or ternary operator
yet.

## 7. Functions and modules

```lepton
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn log(x: f32) -> void {
    println("{x}");
}

extern fn serial_init() -> void;

fn main() {
    let result = add(2, 3);
    log(result as f32);
}
```

Parameters and return values are typed and checked. Non-void functions must
return a value on every reachable path. Calls check arity and argument types;
there is no overloading or default arguments. `main` may omit its return type
and returns `0` on fallthrough.

## 8. Arrays and vectors

### Arrays

```lepton
var a: [i32; 4];                     // zero-initialized
let b: [f32; 4] = {1.0f, 2.0f, 3.0f, 4.0f};
let first = b[0];
```

Arrays are values. A bare `[T; N]` parameter is passed by value; use
`ptr[[T; N]]` for by-reference access:

```lepton
fn by_value(a: [i32; 4]) -> void { a[0] = 99; }     // caller unchanged
fn by_ref(a: ptr[[i32; 4]]) -> void { a[0] = 99; }  // caller changed
```

Nested arrays, arrays of structs, and arrays as struct fields are supported.
Array conversion is explicit and element-wise:

```lepton
let a: [i32; 4] = {1, 2, 3, 4};
let b = a as [f32; 4];
let c = b as [q15; 4];
```

The arrays must have equal lengths and their element types must both be numeric
or both be pointers. There are no implicit array conversions. Arrays of
structs, strings, vectors, and nested arrays are not cast element-wise.

### Vectors

```lepton
var v: v4f32 = [1.0f, 2.0f, 3.0f, 4.0f];
let e = v[0];
v[1] = 5.0f;
```

Vector arithmetic is element-wise: `+ - * /`. Scalar operands broadcast. All
five vector types support:

```lepton
vmax2(a, b)     vmin2(a, b)
vcmpeq(a, b)    vcmpne(a, b)
vcmplt(a, b)    vcmple(a, b)
vcmpgt(a, b)    vcmpge(a, b)
vselect(mask, a, b)
```

Comparisons produce all-ones/all-zeros lane masks for `vselect`. Reductions are
`sum`, `vmax`, and `vmin`. The x86-64 target uses SSE2 where SSE2 has a direct
operation; other cases use scalar helpers.

## 9. Pointers and structs

```lepton
let p = alloc([u8; 64]);     // ptr[[u8; 64]]
p[0] = 1;

let q = p + 1;               // pointer arithmetic
let d = q - p;               // i64 element difference

var x = 10;
let r = &x;                  // address of a named local
```

Pointers are typed with `ptr[T]`, including `ptr[[T; N]]`. Pointer comparison is
allowed for the same pointer type. Pointer arithmetic uses element units.

```lepton
struct Point {
    x: i32;
    y: i32;
}

var p = Point(x: 3, y: 4);
p.x = 7;
p.y += 1;
println("{p.x}, {p.y}");
```

Structs use named-field initializers. Missing fields are zero-initialized.
Structs may contain arrays, nested structs, and pointers, including
self-referential pointers. Recursive by-value aggregates are rejected.

Enums:

```lepton
enum Color { Red, Green, Blue }

let c: Color = Color.Green;
if (c == Color.Green) { ... }
```

## 10. Memory management

```lepton
let p = alloc([i32; 4]);
p[0] = 7;

free(p);            // hosted: releases p; freestanding: no-op
alloc_reset();      // freestanding: reclaims the whole bump heap
```

`alloc([T; N])` returns `ptr[[T; N]]`. In hosted builds the allocator is
`malloc`-based, `free` releases a block, and `alloc_reset` is a no-op. In
freestanding builds the kernel uses a bump allocator: individual `free` is not
possible, so `free` is a no-op and `alloc_reset` reclaims everything.
`alloc_reset` invalidates every live allocation, so call it only at a safe
boundary.

## 11. Builtins

Output and panic:

```lepton
print("No newline");
println("With newline");
println("Interpolation: {x + 1}");
panic("fatal error");
```

Math intrinsics work on scalars and `v4f32`:

```lepton
sqrt(x)   rsqrt(x)
exp(x)    tanh(x)    erf(x)
cos(x)    sin(x)
```

Byte readers and strings:

```lepton
read_u8(buf, off)     read_u16(buf, off)
read_u32(buf, off)    read_u64(buf, off)
read_i8(buf, off)     read_i32(buf, off)    read_i64(buf, off)
read_f16(buf, off)    read_f32(buf, off)
read_str(buf, off, n)
```

Hosted file I/O:

```lepton
let sz = file_size("data.bin");
let n = read_file_at("data.bin", 0, buf, sz);
let w = write_file("out.bin", buf, sz);
```

Freestanding file I/O is declared but not defined, so a kernel program that
calls it fails at link time.

## 12. Conversions

Explicit `as` casts are always allowed for numeric and pointer conversions and
never warn. Array-to-array casts are explicit and element-wise, as described
above.

Implicit conversions at annotated boundaries:

- allowed without warning: exact types, integer widening, integer-to-float,
  `f16`/`bf16 -> f32/f64`, `f32 -> f64`, safe quantized-to-float conversions,
  byte/void pointer views, exact constants, and null pointer literals;
- warn by default and error with `--strict-types`: integer narrowing,
  signedness changes, float-to-integer, `f64 -> f32/f16/bf16`, `f32 ->
  f16/bf16`, `f16 <-> bf16`, unsafe quantized conversions, and unsafe pointer
  conversions.

## 13. Compiler options

```text
-S, --emit-c          emit C instead of compiling
-o FILE               output path
--m32                 build 32-bit
--freestanding        omit the hosted runtime
--target scalar|x86_64
--dump-ir             print lowered IR
--verify-ir           print IR verification summary
--strict-types        reject lossy numeric and unsafe pointer conversions
```

The pipeline is lexer/parser -> typed flat IR -> IR verification -> C11. The
scalar target uses portable loops; the x86-64 target adds SSE2 paths.

## 14. Current gaps

- `raw` blocks are not implemented.
- Assignment targets are limited to variables, fields, and indices;
  `(p + 1)[0] = ...` is rejected.
- Struct and array ordering (`<`, `<=`, `>`, `>=`) is rejected; equality works.
- Untyped array/vector literals infer a common element type; array conversions
  must be explicit.
- No generics, closures, or definite-assignment analysis.
- Freestanding file I/O is declared but not defined.
- Freestanding `free` is a no-op; `alloc_reset` reclaims the whole bump heap.
- Only scalar and x86-64 SSE2 backends exist.
