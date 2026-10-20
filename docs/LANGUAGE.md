# Lepton Language Reference

This document describes the Lepton language as implemented by the compiler in
this repository. It is a practical reference, not a formal specification.

## Source files and imports

A Lepton source file has the `.lp` extension and contains top-level
declarations:

```lepton
import "std/gguf/kv.lp";
import foo.bar;                 // resolves to foo/bar.lp

const LIMIT: i32 = 16;

extern fn lp_alloc(n: i64) -> ptr[u8];

struct Point { x: i32; y: i32; }

enum Color { Red, Green, Blue }

fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

Imports are resolved relative to the importing file, then through the search
paths (`.`, `./std`, and `$LEPTON_PATH`). Imports are flattened before lowering;
circular imports are an error.

## Types

| Category | Types |
|---|---|
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |
| Floats | `f16`, `bf16`, `f32`, `f64` |
| Fixed-point | `q7`, `q15`, `q31` (range `[-1, 1)`) |
| Other scalars | `bool`, `char`, `str` |
| Vectors | `v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8` |
| Aggregates | `[T; N]`, `ptr[T]`, `struct`, `enum` |

`str` is a null-terminated `const char*`. String literals support `{expr}`
interpolation.

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

## Declarations and scope

```lepton
let immutable = 1;       // type inferred from the initializer
let explicit: i64 = 1;   // initializer coerced to i64
var counter: i32 = 0;    // mutable
var zeroed: i32;         // zero-initialized
```

- `let` is immutable; assigning to it is an error. `var` is mutable.
- Function parameters are mutable local copies.
- Mutating a pointer's pointee is allowed even when the pointer binding is
  `let`: `let p = alloc([u8; 4]); p[0] = 1;` is valid, but `p = ...` is not.
- A declared type is enforced. The initializer is converted to that type.
- Declarations without an initializer are allowed for any non-`void` type and
  are **zero-initialized**. There is no definite-assignment analysis.
- Blocks introduce scopes. Shadowing is allowed; the compiler gives each
  binding a unique C name.
- Top-level `const` values support literals and casts of literals.

## Assignment

```lepton
var x: i32 = 1;
x = 2;
x += 3;                 // compound assignment

var a: [i32; 4] = {1, 2, 3, 4};
a[0] = 9;
a[1] += 2;

var p = Point(x: 1, y: 2);
p.x = 7;
p.y += 1;               // -> 3

var grid: [[i32; 2]; 2];
grid[0][0] = 4;         // nested index

let buf = alloc([u8; 8]);
buf[0] = 1;             // pointer pointee: allowed
```

Compound operators are `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`,
and `>>=`. The operation uses the same type rules and conversion checks as the
corresponding binary operator, then the result is converted back to the target
type.

Assignment targets may be a variable, a struct field, an array/vector element,
a pointer element, or a nested combination of these. The base and index
expressions are evaluated once. Nested aggregate updates use a
copy-modify-write sequence internally; this is correct but not yet optimized.

An arbitrary expression is not an assignment target: `(p + 1)[0] = ...` is
rejected.

## Expressions and operators

Arithmetic: `+ - * / %`, shifts `<< >>`, bitwise `& | ^ ~`, comparisons
`== != < <= > >=`, logical `&& || !`.

Precedence follows C. `&&` and `||` short-circuit.

### Conditions

`if` and `while` conditions must be `bool`:

```lepton
if (x > 0) { ... }       // ok
if (x) { ... }           // error: condition must be bool
```

Operands of `&&`, `||`, and `!` must also be `bool`.

### Comparisons

| Operands | Allowed comparisons |
|---|---|
| Numeric values (`i*`, `u*`, `f*`, `q*`, `char`) | all six operators |
| `bool` | `==`, `!=` only |
| Same enum type | all six operators |
| Same pointer type | all six operators |
| `str` | all six operators (content-based) |
| Structs, arrays | `==`, `!=` only (structural) |
| Vectors | rejected; use `vcmpeq`/`vselect` or element access |

Comparing different pointer types requires an explicit cast. To compare a
pointer with null, cast the null literal to the exact pointer type:

```lepton
if (p != 0 as ptr[[u8; 4]]) { ... }
```

String equality compares contents, not addresses. Ordering is lexicographic by
unsigned byte value. Null strings compare equal to each other and sort before
non-null strings:

```lepton
let a = "alpha";
if (a == "alpha") { ... }   // true
if (a < "beta") { ... }     // true
```

### Aggregate equality

Structs and arrays support recursive `==`/`!=`, but not ordering:

```lepton
let a = Point(x: 1, y: 2);
let b = Point(x: 1, y: 2);
if (a == b) { ... }          // true

let u: [i32; 3] = {1, 2, 3};
let v: [i32; 3] = {1, 2, 3};
if (u == v) { ... }          // true
```

- Struct equality compares every field.
- Array equality compares every element.
- String fields/elements compare by content.
- Vector fields/elements compare lane by lane.
- Pointer fields/elements compare addresses.
- Floating-point fields/elements use IEEE `==`, so NaN is unequal to itself.
- Ordering (`<`, `<=`, `>`, `>=`) is rejected for structs and arrays.

## Functions

```lepton
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn log(x: f32) -> void {
    println("{x}");
}

extern fn lp_alloc(n: i64) -> ptr[u8];
```

- Parameters and return values are typed and checked.
- A non-void function must return a value on every reachable path.
- `void` functions use `return;` or fall through.
- Calls check arity and argument types. Functions do not have overloading or
  default arguments.
- `main` may omit its return type; falling off the end returns `0`.

## Aggregates

### Arrays

```lepton
var a: [i32; 4];                 // zero-initialized
let b: [f32; 4] = {1.0f, 2.0f, 3.0f, 4.0f};
let first = b[0];
a[1] = 7;
```

Arrays are values. A bare `[T; N]` parameter is passed by value (a C struct
copy). Use `ptr[[T; N]]` for by-reference access:

```lepton
fn by_value(a: [i32; 4]) -> void { a[0] = 99; }     // caller unchanged
fn by_ref(a: ptr[[i32; 4]]) -> void { a[0] = 99; }  // caller changed
```

Nested arrays, arrays of structs, and arrays as struct fields are supported.
Differently typed array variables are not converted element-wise.

### Pointers

```lepton
let p = alloc([u8; 64]);     // ptr[[u8; 64]]
p[0] = 1;                    // ->e[0]
let q = p + 1;               // pointer arithmetic
let d = q - p;               // i64 element difference
let r = &some_local;         // address of a named local
```

`alloc` uses a bump allocator; there is no `free`.

### Structs

```lepton
struct Point {
    x: i32;
    y: i32;
}

let p = Point(x: 3, y: 4);
println("{p.x}, {p.y}");
```

Struct initializers use named fields. Missing fields are zero-initialized.
Structs may contain arrays, nested structs, and pointers, including
self-referential pointers. Recursive by-value aggregates are rejected.

Fields may be assigned directly (`p.x = 1`) and may be nested
(`p.inner.x += 1`); see [Assignment](#assignment).

### Enums

```lepton
enum Color { Red, Green, Blue }

let c = Color.Green;
if (c == Color.Green) { ... }
```

Enums are C enums and support equality and ordering within the same enum type.

## Allocators and I/O

- `alloc([T; N])` returns `ptr[[T; N]]`.
- Hosted file I/O: `file_size(path)`, `read_file_at(path, offset, buf, n)`,
  `write_file(path, buf, n)`.
- Byte readers: `read_u8`, `read_u16`, `read_u32`, `read_u64`, `read_i8`,
  `read_i32`, `read_i64`, `read_f16`, `read_f32`.
- `read_str(buf, offset, n)` allocates a null-terminated string.
- `print`, `println`, and `panic` accept a string or stringifiable value.
- Math: `sqrt`, `rsqrt`, `exp`, `tanh`, `erf`, `cos`, `sin` on scalars and
  `v4f32`.
- Vector helpers: `sum`, `vmax`, `vmin`, `vmax2`, `vmin2`, `vcmpeq`, `vcmpne`,
  `vcmplt`, `vcmple`, `vcmpgt`, `vcmpge`, `vselect`. Comparisons produce
  all-ones/all-zeros lane masks; pass them to `vselect`. All five vector types
  are supported; SSE2 is used where it has a direct operation.

## Conversion policy

Explicit `as` casts are always allowed for numeric and pointer conversions and
never produce conversion warnings.

Implicit conversions at annotated boundaries follow this policy:

**Allowed without a warning**

- exact types;
- integer widening (`i32 -> i64`, `u8 -> i32`) and unsigned-to-wider-signed;
- integer-to-float;
- `f16`/`bf16 -> f32/f64`, `f32 -> f64`;
- `q7`/`q15 -> f32`, any quantized type `-> f64`;
- byte/void pointer views (`ptr[u8]`, `ptr[char]`, `ptr[void]`,
  pointers to `[u8; N]`);
- exact constants that round-trip, and null pointer literals.

**Warn by default; error with `--strict-types`**

- integer narrowing and incompatible signedness changes;
- float-to-integer;
- `f64 -> f32/f16/bf16`, `f32 -> f16/bf16`, `f16 <-> bf16`;
- quantized conversions outside the safe cases;
- pointer conversions that are not exact and not byte/void views.

Exact constants are checked by value before warning, so `0 -> u64`,
`1.0 -> i32`, and `0.5 -> f32` are silent.

## Control flow

```lepton
if (cond) { ... } else if (cond) { ... } else { ... }

while (cond) {
    if (stop) { break; }
    if (skip) { continue; }
}

for i in 0..10 { ... }     // 0 .. 9
for i in 0..=10 { ... }    // 0 .. 10
```

`for` bounds must be integers. There is no `switch`, `do`/`while`, or ternary
operator.

## Compiler options

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

## Current gaps

- `raw` blocks parse but are not lowered.
- Assignment targets are limited to variables, fields, and indices; arbitrary
  expressions such as `(p + 1)[0] = ...` are not accepted.
- Struct and array ordering is rejected; only `==`/`!=` are supported.
- Untyped array/vector literals still infer a common element type.
- No `free`, `switch`, generics, closures, or definite-assignment analysis.
- Freestanding file I/O is declared but not defined.
