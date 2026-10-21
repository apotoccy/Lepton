# Lepton OS & Lepton Language Technical Specification

## 1. Executive summary

Lepton is a minimalist, AI-native programming language and bare-metal operating
system for edge inference. Its goal is to run quantized language models directly
on raw hardware - including legacy 32-bit PCs - without Linux, glibc, a Python
VM, PyTorch, CUDA, or another ML framework.

The system is deterministic, statically typed, and built around a compiler that
emits portable C11. The same compiler produces hosted test programs and
freestanding 32-bit kernel code.

Current status:

- **134/134** compiler tests on scalar and x86-64 SSE2.
- 13 IR-verifier tests and 12 conversion-policy tests.
- TinyStories-15M boots and runs in QEMU and on a 2008-era Lenovo IdeaPad S9.
- `fwdpos 0 1` predicts token **9038** (`Once`), matching the Python reference.

## 2. System architecture

The repository contains three primary subsystems:

1. **The compiler** (`leptonc.py` and `lepton_*.py`) translates Lepton source to
   C11.
2. **The standard library** (`std/`) implements GGUF parsing, tensor helpers,
   dequantization, matmul, normalization, and activations in Lepton.
3. **Lepton OS** (`kernel/`) is a 32-bit x86 Multiboot 1 kernel with a REPL,
   drivers, and the inference loop.

### 2.1 Compiler pipeline

The compiler is written in Python and uses a multi-pass pipeline:

1. **Lexer and parser.** A hand-written lexer and recursive-descent parser
   produce an AST with source spans.
2. **Type resolution and lowering.** The AST is lowered to a flat, typed
   intermediate representation (`lepton_ir.py`). Locals and temporaries are
   function-scoped and hoisted into C function scope.
3. **IR verification.** `lepton_verify.py` checks labels, jumps, terminators,
   local/temp types, operand and destination types, call signatures, aggregate
   shapes, and return paths before C emission. `--verify-ir` prints a summary.
4. **C emission.** `lepton_c_ir.py` emits C11. Aggregate C typedefs are
   forward-declared and emitted in dependency order. The target descriptions in
   `lepton_target.py` select scalar or x86-64 SSE2 code.
5. **Native compilation.** The generated C is compiled by the host C compiler
   for userland, or by `-m32 -ffreestanding` for the kernel.

The compiler supports:

- hosted and freestanding (`--freestanding`) modes;
- 32-bit (`--m32`) and native userland builds;
- scalar and x86-64 SSE2 targets;
- `--dump-ir` and `--verify-ir`;
- conversion diagnostics, with `--strict-types` turning lossy numeric and
  unsafe pointer conversions into errors.

## 3. Language

### 3.1 Type system

- **Integers:** `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
- **Floats:** `f16`, `bf16`, `f32`, `f64`.
- **Fixed-point:** `q7`, `q15`, `q31` in `[-1, 1)`, with saturating arithmetic.
- **Other scalars:** `bool`, `char`, `str` (null-terminated).
- **Vectors:** `v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8`.
- **Aggregates:** fixed-size arrays `[T; N]`, pointers `ptr[T]`, structs, enums.

Declared types are enforced at bindings, assignments, calls, returns, and
struct fields. Literals are lowered against their target type. Numeric and
pointer conversions are explicit in the IR, with warning/error policy controlled
by `--strict-types`.

### 3.2 Declarations and mutation

```lepton
let immutable = 1;
var counter: i32 = 0;
counter += 1;

var p = Point(x: 1, y: 2);
p.x = 3;
p.inner.x += 1;
```

- `let` is immutable; `var` is mutable.
- Parameters are mutable local copies.
- Pointer pointees remain mutable through an immutable pointer binding.
- Assignment supports identifiers, fields, array/vector elements, pointer
  elements, nested combinations, pointer-expression bases such as `(p + 1)[0]`,
  and compound operators (`+=`, `-=`, `*=`, `/=`, `%=`, bitwise, and shifts).
- Nested aggregate writes use copy-modify-write internally.

### 3.3 Control flow and comparisons

- `if`/`else`, `while`, and range-based `for` (`..`, `..=`).
- `switch` on integers, `char`, and enums, with multi-value cases and
  `default`; no implicit fallthrough.
- `raw` blocks for unary pointer loads and stores, including nested fields and
  array elements.
- `break` and `continue`; `break` exits the nearest loop or switch.
- Conditions and `&&`/`||`/`!` operands must be `bool`.
- Numeric, bool, enum, and same-typed pointer comparisons.
- Content-based string equality and ordering.
- Structural `==`/`!=` for structs and arrays.
- Aggregate ordering is rejected.
- No `do`/`while` or ternary operator.

### 3.4 Aggregates and conversions

- Arrays are values; bare `[T; N]` parameters pass by value, while
  `ptr[[T; N]]` gives by-reference access.
- Arrays as struct fields, arrays of structs, nested arrays, and
  self-referential structs through pointers are supported.
- Recursive by-value aggregates are rejected.
- Explicit array casts convert equal-length arrays element by element when both
  element types are numeric or both are pointers.
- There are no implicit array conversions.
- Struct/array equality compares fields/elements recursively; strings by
  content, vectors lane-wise, pointers by address, and floats with IEEE `==`.

### 3.5 Memory

```lepton
let p = alloc([i32; 4]);
p[0] = 7;
free(p);            // hosted: releases; freestanding: no-op
alloc_reset();      // freestanding: resets the bump heap

let mark = alloc_mark();
let q = alloc([i32; 4]);
alloc_release(mark);   // freestanding: rewinds to mark
```

- Hosted mode uses `malloc`/`free`; reset and mark/release are advisory no-ops.
- Freestanding mode uses a bump allocator. Individual `free` is not possible,
  so `free` is a no-op; `alloc_reset` reclaims the entire heap, and
  `alloc_release(mark)` rewinds to a previous `alloc_mark()` position. Both
  invalidate allocations made after the reset point.
- The kernel marks the heap before each REPL command and releases transient
  allocations afterward; long-lived buffers are allocated before the mark.
- The heap is sized from the Multiboot memory map after the loaded model,
  capped at 64 MB, with an 8 MB minimum.

## 4. Runtime and standard library

The generated C does not call libm. Math helpers (`sqrt`, `rsqrt`, `exp`,
`tanh`, `erf`, `cos`, `sin`), string helpers, allocation, and I/O are
hand-written in `lepton_runtime.py` for hosted builds and in
`kernel/src/runtime.c` for freestanding builds.

### 4.1 GGUF parsing

The standard library parses GGUF directly from memory:

- header: magic `GGUF`, version, tensor count, KV count;
- KV metadata, including arrays used by tokenizer vocabularies;
- tensor table: names, dimensions, data types, byte offsets;
- data alignment from `general.alignment` (default 32 bytes).

### 4.2 Quantization

- **Q4_0:** 18-byte blocks, 32 elements, f16 scale plus 16 packed bytes.
- **Q8_0:** 34-byte blocks, 32 elements, f16 scale plus 32 int8 values.
- F32 norm tensors are handled directly.

Q4_0/Q8_0/Q4_K are supported with F16/F32 fallbacks; Q5_K/Q6_K are not.
Grouped-query attention is supported.

### 4.3 Transformer

The forward pass implements a LLaMA-style network:

- token embeddings;
- multi-head attention with Q, K, V projections;
- RoPE on Q and K;
- a per-layer KV cache for autoregressive generation;
- RMSNorm;
- SwiGLU MLP;
- final norm and vocabulary projection.

The kernel re-dequantizes weights during each forward pass; a persistent
dequantized-weight cache is future work.

### 4.4 Tokenizer and sampling

- BPE vocabulary from GGUF, with bounded piece buffers and capacity checks.
- Greedy longest-match fallback for models without merge lists.
- Unknown bytes map to `<0xNN>` vocabulary entries when available.
- Temperature scaling and Top-K filtering with an xorshift32 PRNG.

The merge path is exercised by a synthetic fixture; TinyStories has no merges.

## 5. Lepton OS

### 5.1 Boot and memory layout

- Multiboot 1; loads via GRUB or QEMU `-kernel` at 1 MiB.
- Installs a flat GDT and IDT, remaps the PIC, and enables the x87 FPU and SSE.
- The GGUF model is a Multiboot module accessed directly from RAM.
- The bump heap is initialized after the module. Its range comes from the
  Multiboot memory map, capped at 64 MB, with an 8 MB minimum. Low-memory boot
  fails with a clear message.

### 5.2 Drivers

- VGA text console, 80x25 at `0xB8000`.
- PS/2 keyboard, scancode set 1, Shift/Ctrl/Alt support, Ctrl+Alt+Del reboot,
  and a 256-byte ring buffer.
- COM1 serial at 38400 baud for headless QEMU.
- PIT at 100 Hz.
- PIC remapped to `0x20`-`0x2F`.

### 5.3 REPL

The kernel presents a `lepton>` prompt with backspace editing. Commands
include:

- inspection: `info`, `config`, `tensors`, `modules`, `matvec`;
- tokenizer: `vocab`, `decodetok`, `encode`;
- generation: `generate [T] [K] [N]`, `genprompt "text" [T] [K] [N]`,
  `fwdpos`, `forward`; generation stops at EOS or context limit;
- maintenance: `reset` (KV cache), `reboot` (or Ctrl+Alt+Del), `poweroff`,
  `clear`, `filetest`; the prompt supports history, cursor editing, and tab
  completion.

Expected configuration/inference errors are soft errors that return to the
prompt. Configuration parsing is architecture-aware and falls back to tensor
shapes when keys are missing. Fatal panics print a Ctrl+Alt+Del reboot hint.
`poweroff` uses ACPI PM1 when available, with emulator port fallbacks.

`config` parses and prints GGUF hyperparameters and reports whether the
compiled inference path supports the shape. Activation, KV, logits, and
token-table workspaces are sized from the config within a 60 MB budget;
unsupported shapes are rejected.

## 6. Build and test

```bash
make test                 # hosted compiler tests (117)
make test-sse             # SSE2 target tests (117)
python3 tests/test_ir_verify.py
python3 tests/test_conversions.py

make kernel MODEL=models/tinystories.gguf
make run-kernel MODEL=models/tinystories.gguf

make iso MODEL=models/tinystories.gguf
make run-iso
```

`make kernel` only builds `build/kernel/kernel.elf`; `run-kernel` launches
QEMU with the model as a Multiboot module. `make flash USB=/dev/sdX` writes the
ISO to a device and requires `sudo`; verify the device name first.

## 7. Current limitations

- **Model dimensions.** Activation, KV, logits, token-table, and scratch
  workspaces are sized from GGUF metadata within a 60 MB budget; the context
  cap is 256 positions. Tokenizer merge/ID buffers still have fixed limits.
- **Context safety.** The configured `max_seq` is enforced by the forward guard
  and generation loops.
- **Quantization coverage.** Q4_0/Q8_0/F32 only; no K-quants.
- **Memory.** The bump allocator cannot reclaim individual blocks. Mark/release
  and `alloc_reset` invalidate allocations after their reset point, and
  `lp_alloc` null checks are incomplete.
- **Compiler gaps.** No generics or closures.
  Array/vector indexing and field assignment still require lvalue bases.
- **File I/O.** Freestanding file I/O returns `-1`; the kernel has no
  filesystem driver.
- **Backends.** Scalar and x86-64 SSE2 only; AVX2 and ARM/NEON are future work.
- **Performance.** Weights are re-dequantized on every token, and the inference
  loop is single-threaded.
