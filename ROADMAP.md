# Lepton Roadmap

Compiler and OS for edge AI inference, from 32-bit PCs to modern devices.

## Current status (v0.5.0, October 2026)

- **Compiler:** single IR pipeline (AST → IR → verify → C11); 134/134 compiler
  tests pass on scalar and SSE2, plus 13 IR-verifier and 12 conversion-policy
  unit tests.
- **OS:** 32-bit Multiboot kernel, GRUB ISO, QEMU boot, interactive REPL,
  multi-turn `chat`, and eight-page VGA scrollback.
- **Inference:** TinyStories-15M and SmolLM2-135M-Instruct Q4_0 run from a
  Multiboot module. Instruct `ask` works with the ChatML system/user prefix
  cache. Q8_0 is usable; mixed K-quant models are refused explicitly.
- **S9:** baseline recorded; 908.75 ms/token for `bench 8`, first Ask prefill
  16.62 s, cached second prefill 6.18 s, answer 6.42 s.
- **Platforms:** shared core, BIOS32 shim, and UEFI64 implementation. The
  UEFI64 kernel loads SmolLM2-135M-Instruct Q4_0 from the ESP, supports
  `ask`/`chat`, and accepts Ctrl+C cancellation. A BIOS32+UEFI64 hybrid ISO
  boots both paths.
- **Next focus:** native 64-bit mode (exit Boot Services), input polish, and
  CI coverage for the hybrid image.

## Verification checklist

Run these before claiming progress:

```bash
make test
make test-sse
make -B kernel
make iso MODEL=models/tinystories.gguf

# Boot + query the kernel headlessly
(sleep 3; printf 'config\ninfo\n'; sleep 2) | \
  timeout 15 qemu-system-i386 \
  -kernel build/kernel/kernel.elf \
  -initrd models/tinystories.gguf \
  -serial stdio -display none -no-reboot -m 256
```

Note: the Makefile does not track `std/*.lp` imports or `lepton_*.py` as
dependencies of the kernel/userland rules, so use `make -B` after changing
compiler or standard-library code.

---

## 🏆 Major Milestones Achieved

- **Compiler:** single IR pipeline, 134/134 tests passing on scalar and SSE2
  plus 13 IR-verifier and 12 conversion-policy tests, emitting portable C11.
- **Bare-Metal AI:** a 15M-parameter TinyStories model on a 2008 Lenovo IdeaPad
  S9 (Intel Atom), with no OS, libc, or Python; reproducible in QEMU and as a
  GRUB ISO.
- **Interactive OS:** VGA text console, PS/2 and serial keyboard, 100 Hz timer,
  interrupts, and a live REPL.

---

## Phase 0 — Refactor for growth ✅

- [x] Test infrastructure (`tests/`, `make test`, `make test-sse`)
- [x] Modules (`import "path.lp";` and `import foo.bar;`)
- [x] Source spans threaded through the pipeline
- [x] Typed IR (`lepton_ir.py`), AST → IR → C migration
- [x] Delete the legacy AST emitter (single pipeline; a commented-out copy and
      stale docstrings remain to clean up)
- [x] Split runtime strings into `lepton_runtime.py`
- [x] 64-bit and 32-bit userland build targets
- [x] Declared-type enforcement for `let`/`var`, assignments, call arguments,
      return values, and struct fields
- [x] Target-typed array and vector literals against annotations
- [x] Type-error regression tests and clean diagnostics
- [x] IR verifier: operand/destination types, label and jump validity,
      call signatures, aggregate shapes, and reachable fallthrough
- [x] Malformed-IR verifier unit tests
- [x] Conversion policy: lossy numeric and unsafe pointer warnings,
      constant-aware exactness, `--strict-types`, and unit tests
- [x] Bool-only `if`/`while`/logical conditions and unary `!`
- [x] Comparison rules for numeric, bool, enum, same-typed pointer, and string
      values; structural struct/array equality; aggregate ordering rejected
- [x] Integer-only `for` loop bounds
- [x] `switch` on integers, `char`, and enums with multi-value cases,
      `default`, no implicit fallthrough, and switch-aware `break`
- [x] `raw` blocks with unary pointer loads and stores, including nested
      fields and array elements
- [x] Flow-sensitive definite-assignment checks across branches, loops, and
      `switch`
- [x] Arbitrary pointer-expression lvalue bases such as `(p + 1)[0] = ...`
- [x] Mutable lvalues: `let` immutability, field/index/nested assignment, and
      compound assignment for identifiers, indices, fields, and vector lanes
- [x] Vector min/max/comparison/select for all vector types, with SSE2
      comparison/select paths and scalar fallbacks
- [x] Structural `==`/`!=` for structs and arrays via generated C helpers,
      including nested fields, strings, vectors, pointers, and NaN semantics
- [x] Explicit element-wise array casts for equal-length arrays with numeric
      or pointer elements
- [x] `free(ptr)` and `alloc_reset()`; freestanding free is a no-op and reset
      reclaims the bump heap
- [x] `alloc_mark()`/`alloc_release(mark)` and per-command REPL reclamation
- [x] Language reference (`docs/LANGUAGE.md`)
- [x] `pyflakes` step in CI and `pyflakes`-clean compiler modules
- [ ] Remove the remaining commented AST-emitter code and update docstrings

## Phase 1 — Numeric core ✅

Primitives & Vectors:

- [x] `f16`, `bf16`, `q7`, `q15`, `q31` with saturating arithmetic
  (`f16`/`bf16` use native compiler types when available and portable fallbacks
  otherwise)
- [x] Vector types (`v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8`)
- [x] Elementwise arithmetic, min/max, comparison masks, reductions
- [x] SSE2 lowering for x86-64 (scalar fallbacks included)
- [x] Scalar-to-vector broadcast

Math (scalar and `v4f32`):

- [x] Self-contained implementations (no libm) for `sqrt`, `rsqrt`, `exp`,
      `tanh`, `erf`, `cos`, `sin`
- [x] SSE2 intrinsics for `sqrt` and `rsqrt`

## Phase 2 — Tensors and matmul ✅

- [x] Array types as first-class values: `[T; N]`
- [x] Array literal `{...}`, indexing, zero-init
- [x] Pointer arithmetic: `ptr[T] ± int`, `ptr[T] - ptr[T]`
- [x] Scalar pointer indexing (`q[i]` where `q: ptr[u8]`)
- [x] Heap allocation via `alloc([T; N])` returning `ptr[[T; N]]`
- [x] By-reference array access via pointer-to-array parameters
      (`ptr[[T; N]]`); a bare `[T; N]` parameter is passed as a C struct by
      value, not by reference
- [x] `matmul4`, `matvec4`, `identity4` in `std/nn/matmul.lp`
- [x] Arrays as struct fields, arrays of structs, and nested arrays
- [x] Forward declarations and dependency-ordered C typedefs for aggregates
- [x] Self-referential structs through pointer fields
- [x] Tests for by-value arrays and `ptr[[T; N]]` by-reference parameters
- [ ] Symbolic shapes with const-generic element counts
- [ ] Batched matmul for attention (`Q @ K^T`); currently inline in the kernel
- [ ] Autotuned loop order (`i, k, j`) for large matrices

## Phase 3 — Inference Runtime ✅

File I/O & Bytes:

- [x] Hosted file I/O builtins: `file_size`, `read_file_at`, `write_file`
- [x] Little-endian byte readers: `read_u8..u64`, `read_i32/i64`, `read_f16`,
      `read_f32`
- [x] `read_str` builtin (heap-allocated null-terminated string)
- [x] Pointer-to-pointer casts for typed views of byte buffers
- [x] Freestanding file I/O stubs (`lp_file_size`/`lp_read_file_at`/
      `lp_write_file` return -1; the kernel has no filesystem)

GGUF Parsing:

- [x] GGUF header parser (magic, version, tensor_count, kv_count)
- [x] GGUF metadata KV parser (u32-length-prefixed strings, typed values)
- [x] GGUF tensor table parser (name, dims, type, offset)
- [x] GGUF whole-file loader (alignment handled, `general.alignment`) with a
      16 MB compile-time buffer cap; the kernel's Multiboot-module path is
      independent of this loader
- [x] Tensor lookup by name (`tensor_by_name`)

Dequantization & Math:

- [x] Q4_0 dequantization (18-byte blocks, 32 elements)
- [x] Q8_0 dequantization (34-byte blocks, 32 elements)
- [x] Full Llama forward pass, now config-sized for embedding/FFN widths,
      layers, heads, head dimension, vocabulary, and context
- [x] `ModelConfig` parsing (`std/gguf/config.lp`) and shape validation
- [x] Config-driven activation, KV, logits, token-table, and scratch sizing
- [x] Context guard against the configured `max_seq` (256-position cache cap)
- [x] Q, K, V matvecs, RoPE, multi-head attention, softmax, KV cache
- [x] RMSNorm, SwiGLU MLP (inline in kernel)

Generation & Tokenizer:

- [x] Autoregressive generation with configurable length and EOS stopping
      (`generate`/`genprompt` default to 30 tokens)
- [x] Streamed output, stop sequences, Ctrl+C cancellation, and `selftest`
- [x] Sampling: temperature, top-k, xorshift32 PRNG
- [x] Tokenizer decode (BPE vocab offsets, U+2581 substitution)
- [x] Tokenizer encode (BPE merges path plus greedy fallback) with bounded
      vocab/piece buffers, `<0xNN>` byte fallback, and a synthetic merge-path
      test
- [x] Prompt-based generation (`genprompt "..." T K N`), with temperature
      `0` selecting greedy output and streamed tokens
- [x] Tokenizer byte fallback to `<0xNN>` vocabulary entries
- [x] Bounds-safe token table, decode IDs, merge table, and piece buffers
- [x] Test the BPE merge path with a synthetic merge fixture

## Phase 4 — Backends (Deferred)

Trigger to resume: run on ARM (smartphones/edge) or want AVX2 speed. Do this
after the config-driven runtime work.

- [ ] AVX2 target (`--target avx2`)
- [ ] NEON target for ARM (`arm-linux-gnueabihf-gcc`)
- [ ] `--target native` (auto-detect from host compiler)
- [ ] LLVM IR backend
- [ ] WASM target

## Phase 5 — Lepton OS Integration ✅ (Foundation complete)

Hardware & Drivers:

- [x] 32-bit Multiboot entry point, GDT, IDT
- [x] PIC remap, PIT timer at 100 Hz
- [x] Keyboard driver (PS/2 scancode set 1 + serial input)
- [x] VGA text-mode console driver (0xB8000)
- [x] Multiboot module access (`mb_module_count`, `mb_module_start`,
      `mb_module_size`)

Kernel Features:

- [x] Interactive REPL loop (line accumulation, backspace, Enter)
- [x] Multiboot memory-map validation and dynamic heap sizing (64 MB cap,
      8 MB minimum after the module)
- [x] Checked `alloc` emission and cleaner `lp_alloc` failure handling
- [x] Real hardware test (Lenovo IdeaPad S9, 2008 Intel Atom)
- [x] Context-length guard from GGUF metadata
- [ ] Config-driven KV/logits/token-table sizing
- [ ] Split `kernel.lp` into `repl.lp`, `tokenizer.lp`, `infer.lp`, and
      `model.lp` modules

## Phase 6 — AI-native language features (Research)

- [ ] `grad` transform (reverse-mode AD at the IR level)
- [ ] Probabilistic primitives (`sample`, `observe`)
- [ ] Symbolic tensor simplification
- [ ] Compile-time graph fusion

---

## 🚀 Vision & Next Steps (Prioritized)

### 1. Performance: quantized matvec before a bigger f32 cache (High Impact)

Currently, `generate` dequantizes all 57 tensors from Q4_0/Q8_0 to `f32` for
**every token**. On the Atom netbook this is why it takes ~30 seconds per token.

The GGUF stores ~24.4M tensor elements: 9.2M in `token_embd`, 9.2M in `output`,
and 5.97M in the six transformer blocks. A 24 MB persistent f32 cache covers
only the six transformer blocks (~23.9 MB). The stored tensors are ~97.6 MB at
f32, and the 32000×288 output projection alone is ~36.9 MB. A 24 MB buffer is
therefore not enough for "all tensors"; size the plan against the stored
tensors, not the 15M model label.

- [ ] Add per-phase timing (using `ticks()`) and a baseline tokens/sec
      measurement in QEMU and on the S9
- [x] Implement Q4_0/Q8_0 dot products directly against f32 activations,
      keeping the block form throughout the hot path; QEMU improved from
      ~299.4 to ~272.5 ms/token
- [ ] If a cache is used, budget explicitly: layers ≈24 MB, output ≈37 MB,
      embedding rows only; raise and validate the heap before allocating
- [ ] Vectorize `matvec`, `rmsnorm`, `softmax`, and `silu` with `v4f32`
      (288, 768, and 48 are multiples of 4)
- [ ] Use `i32` for dimensions/indices in hot loops; keep `i64` for file offsets
- [ ] Cache tensor indices and offsets at boot instead of rescanning the tensor
      table per token

### 2. Model support: config-driven runtime first, then modern quantization

- [x] Build a `ModelConfig` from GGUF metadata: `embedding_length`,
      `feed_forward_length`, `attention.head_count`, `block_count`,
      `rope.dimension_count`, `context_length`, `layer_norm_rms_epsilon`
- [x] Derive `head_dim`/vocabulary and validate the supported shape; reject
      unsupported shapes instead of running with wrong dimensions
- [x] Validate matvec tensor dimensions/types at inference; unsupported types
      fail softly
- [x] Tokenizer buffers are fixed-size but capacity-checked for the current
      line/prompt limits
- [x] Add grouped-query attention (`n_head_kv`) and validate it against a
      two-position synthetic GQA reference
- [x] Validate real GQA with SmolLM2-135M (9/3 heads)
- [x] Add GPT-2 byte-level tokenizer support and tied output embeddings
- [x] Parse model-specific RoPE frequency base and validate SmolLM in `selftest`
- [x] Add SmolLM2 ChatML `ask` path and validate it with Instruct Q4_0
- [x] Add ask prefill/answer timing and a raw keyboard event diagnostic
- [x] Cache the fixed ChatML system/user prefix across asks
- [x] Layer-major batched prompt prefill
- [x] SSE2 Q4_0 matvec
- [x] Multi-turn `chat` mode with conversation KV state
- [x] Eight-page VGA scrollback with PageUp/PageDown
- [x] Rolling chat context with oldest-turn eviction, `/new`, and `/ctx`
- [x] Context-aware chat prompt with prefill/answer/heap status
- [x] Add Q5_0/Q5_1 block support and fixtures
- [x] Refuse mixed K-quant models with a clear compatibility error
- [ ] Root-cause mixed K-quant forward corruption
- [x] Add Q4_K dequantization and direct matvec, plus F16/F32 fallbacks
- [x] Add Q5_K/Q6_K dequantization and direct matvecs
- [ ] Plan memory for 1B+ models. They do not fit the current 64 MB heap;
      this likely needs ATA/SD streaming or a larger-RAM target, and should be
      designed before adding the formats
- [x] Add a full forward reference (Q/K/RoPE/attention) and a deterministic
      `selftest`

### 3. Compiler: verify the IR and finish the remaining type gaps

- [x] `let` annotations: check assignability, insert standard numeric casts, and
      bind the declared type
- [x] Function arguments and returns: arity and type checking
- [x] Struct fields and array element types: check and coerce
- [x] Add negative tests for every new rule
- [x] Conversion policy: warnings by default, `--strict-types` errors, and
      tests for numeric/pointer cases
- [x] Add an IR verifier (operand types, terminators, unique C names)
- [ ] Add a regression test for shadowing; manual tests currently pass for
      `if`/`else` and `for` scopes with differing types
- [x] Fix C typedef generation for arrays in struct fields and extern parameters
- [x] Clean up duplicate definitions and unused imports reported by `pyflakes`

### 4. OS UX: scrollback, history, and introspection

- [x] Command history ring buffer (Up/Down arrows in the REPL)
- [x] `reset` command to clear the KV cache without rebooting
- [x] Soft configuration/inference errors return to the prompt
- [x] Fatal panic screen with Ctrl+Alt+Del reboot path
- [x] `mem` command showing heap usage and limits
- [ ] `perf` command showing per-phase timings
- [x] Prompt/generation length guard tied to the configured context length
- [x] Keyboard Ctrl/Alt handling; caps-lock LED control remains

### 5. Phase 4: ARM / smartphones

The original goal was 32-bit PCs *and* modern devices.

- [ ] Cross-compile the kernel and compiler to `armv7-a` (Raspberry Pi / older
      phones)
- [ ] Add NEON intrinsics to `lepton_target.py`

---

## Known Limitations (Carried Forward)

**Compiler and language**

- [ ] Integer-to-float conversions are permitted without a warning by policy;
      very wide integers can still lose precision
- [ ] Untyped array/vector literals still infer a common element type; array
      conversions are explicit (`a as [f32; N]`)
- [x] Kernel linear layers support Q4_0/Q8_0/Q4_K/Q5_K/Q6_K plus F16/F32
      fallbacks and F32 norms
- [ ] `parse_header` leaks a 24-byte buffer (fine for one call, bad in a loop)
- [x] `lp_panic` is marked `noreturn` in freestanding declarations and the
      kernel definition

**Kernel and OS**

- [x] Q4_K/Q5_K/Q6_K and F16/F32 fallbacks are supported.
      Inference workspaces are config-sized within a 60 MB budget and a
      256-position context cap.
- [ ] `free` remains a no-op in the bump allocator
- [x] Ctrl/Alt tracking, Ctrl+Alt+Del reboot, and `reboot` command
- [x] ACPI `poweroff` command with emulator fallbacks
- [x] Architecture-aware GGUF config keys and tensor-shape fallbacks
- [ ] No keyboard LED control (caps lock); no scrollback
- [ ] Ring buffer drops input silently on overflow (128 chars max)

**Tooling and testing**

- [x] `tools/verify_forward.py` covers Q/K, attention, and all six layers
- [ ] `tools/verify_tokenize.py` requires `tokenizer.ggml.merges` and crashes on
      models without merges (TinyStories)
- [ ] `tools/gen_test_gguf.py` produces a single-tensor file, not a runnable
      model; `make run-kernel` without `MODEL=` cannot run AI commands
- [ ] The tools hardcode `models/tinystories.gguf`; add path arguments
- [ ] No automated QEMU/kernel smoke test in CI
- [x] Deterministic generation regression via `selftest` and the full
      forward reference (first greedy token 29892)

---

## Design Decisions (Running Log)

- **Scalar-first, then SIMD per target.** Every vector op has a scalar fallback.
- **Array and vector aggregates use an `e` member.** Element access is `.e[i]`
  on every target, including SSE2 (using unions); structs use named fields.
- **Aggregate typedefs are dependency-ordered.** Forward declarations cover
  pointer-only cycles, while by-value struct/array dependencies are
  topologically sorted; recursive by-value aggregates are rejected.
- **Conversion diagnostics are warnings by default.** Lossy numeric and unsafe
  pointer conversions warn with a source span; `--strict-types` turns them into
  errors. Integer-to-float and byte/void pointer views are allowed without a
  warning, and exact constants do not warn.
- **Conditions are bool.** `if`, `while`, `&&`, `||`, and `!` require bool
  operands; numeric or pointer conditions are rejected.
- **Comparisons are explicit.** Numeric, bool, enum, same-typed pointer, and
  string comparisons are allowed, and structs/arrays support structural
  `==`/`!=`. Aggregate ordering is rejected. Strings compare by content
  (`lp_str_eq` / `lp_str_cmp`), not address; aggregate equality compares
  fields/elements recursively, strings by content, vectors lane-wise, and
  pointers by address, with IEEE float/NaN semantics.
- **Definite assignment is checked.** A local declaration without an
  initializer must be assigned on every path before it is read; the C backend
  still emits zero initialization as a safety net.
- **`let` is immutable; `var` is mutable.** Parameters are mutable local
  copies. Mutating a pointer's pointee is allowed even when the pointer
  binding is `let`.
- **Nested lvalues copy-modify-write.** Updating a field or element inside an
  aggregate copies the aggregate to a temporary, updates it, and writes it
  back through the parent; this is correct but not yet optimized.
- **No libm.** All math is self-contained, enabling freestanding kernels.
  Generated code does not call libm; hosted programs use the host C library.
- **Heap is dynamically initialized after the model.** It is bounded by the
  Multiboot memory map, capped at 64 MB, and requires at least 8 MB.
- **The IR is flat.** Locals and temporaries are function-scoped and hoisted.
- **The IR is verified before C emission.** Type, control-flow, and signature
  invariants are checked after lowering; `--verify-ir` prints a summary.
- **Runtime helpers are prefixed `lp_`.** User functions are emitted as
  `lp_<name>` except `main` and externs.
- **Type annotations are contracts.** Locals, assignments, arguments, returns,
  and struct fields are checked; numeric conversions and pointer casts are
  explicit `IRCast` nodes in the lowered program.
- **Model shape should come from GGUF, not source forks.** The current kernel
  is a TinyStories-specific implementation moving toward a `ModelConfig`.
- **Keep weights quantized in the hot path.** Q4_0/Q8_0 dot products are the
  intended performance direction; f32 expansion is a fallback/debug path.

## Notes for Future Sessions

- **Compiler checks:** run `python3 -m pyflakes lepton_*.py leptonc.py` after
  changes. It is currently clean; keep it that way. `make test` also runs
  `tests/test_ir_verify.py`.
- **Build dependencies:** use `make -B kernel` after touching `lepton_*.py` or
  `std/*.lp`; the Makefile does not track those as kernel dependencies.
- **Real hardware:** when testing on the Lenovo, always check `[dbg: start]`
  prints. If the screen goes black, it is usually an SSE/x87 miscompile or a
  stack overflow.
- **Paste safety:** use `nano /tmp/patch.py` for large pastes and splice.
- **Model asset:** record the source and SHA-256 of `models/tinystories.gguf`
  when sharing; the binary is not reproducible from the repository alone.
