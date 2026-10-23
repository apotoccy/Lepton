# LEPTON LANGUAGE AND LEPTON OS

Lepton is a minimalist, AI-native programming language and bare-metal operating
system for edge inference. It compiles Lepton source through a typed IR to
portable C11, and runs quantized GGUF models directly on the CPU with no Linux,
glibc, Python VM, PyTorch, or CUDA.

The reference target is a 15M-parameter TinyStories model. The kernel boots on a
2008 Lenovo IdeaPad S9 (Intel Atom) and in QEMU; the same kernel can be packaged
as a GRUB ISO.

## Status (as of October 2026, v0.2.1)

- **Compiler:** 134/134 regression tests pass with both the scalar and x86-64
  SSE2 targets (`make test`, `make test-sse`), plus 13 IR-verifier and 12
  conversion-policy unit tests.
- **Build:** 64-bit and 32-bit userland programs build; the freestanding 32-bit
  kernel builds; the GRUB ISO builds.
- **Boot:** the kernel boots via QEMU `-kernel` and via the ISO `-cdrom` path.
- **Inference:** TinyStories-15M loads from a Multiboot module. `fwdpos 0 1`
  predicts token 9038, matching an independent Python reference. After a
  four-token "Once upon a time" prefill, greedy generation starts with token
  29892 (`,`), matching a full reference that includes Q/K, RoPE, and attention.
- **Tokenizer:** `encode Once upon a time` produces `[9038, 2501, 263, 931]`,
  matching a greedy reference.
- **Scope:** inference is currently specialized to the TinyStories-15M
  architecture and Q4_0/Q8_0 weights. See [Current limitations](#current-limitations).

## Key features

- **Bare-metal AI:** runs a GGUF model directly on the CPU with no underlying OS
  and no ML framework. Q4_0/Q8_0 matvecs consume quantized blocks directly, so
  linear layers do not dequantize full weight matrices on every token.
- **Custom compiler pipeline:** lexer/parser → typed flat IR → verification →
  C11, with source spans and one lowering pipeline.
- **IR verification:** every lowered module is checked for well-typed operands,
  valid control flow and call signatures, aggregate shapes, and reachable
  fallthrough before C emission. `python3 leptonc.py file.lp -S --verify-ir`
  prints a summary.
- **Enforced declared types:** `let`/`var` annotations, assignments, call
  arguments, return values, and struct fields must be assignable to their
  declared types. Numeric conversions and pointer casts are inserted explicitly
  in the IR, and array/vector literals are lowered against their target type.
- **Aggregate types:** arrays as struct fields, arrays of structs, nested arrays,
  self-referential structs through pointers, and by-value or pointer-to-array
  parameters are supported. C typedefs are forward-declared and emitted in
  dependency order.
- **Conversion diagnostics:** lossy numeric conversions and unsafe pointer
  conversions warn by default. `--strict-types` turns them into errors.
  Integer-to-float conversions and byte/void pointer views are permitted
  without a warning.
- **Control-flow and comparison typing:** `if`/`while` conditions and
  `&&`/`||`/`!` require `bool`. Integer, `char`, and enum `switch` statements
  support multi-value cases and `default`, with no implicit fallthrough.
  Numeric, bool, enum, same-typed pointer, and string comparisons are
  supported, plus structural `==`/`!=` for structs and arrays. Aggregate
  ordering is rejected with source-located errors. Strings compare by content,
  not address.
- **Definite assignment:** every local read must be initialized on all paths.
  Declarations without an initializer are allowed but must be assigned before
  use through branches, loops, and `switch`. The C backend still emits zero
  initialization as a safety net.
- **Mutable lvalues:** `let` bindings are immutable and `var` is mutable.
  Assignment supports identifiers, fields, indices, vector lanes, nested
  lvalues, pointer-expression bases such as `(p + 1)[0] = ...`, and compound
  operators (`+=`, `-=`, etc.). Pointer pointees remain mutable through an
  immutable pointer binding.
- **Raw pointer boundary:** `raw { ... }` enables unary `*p` loads and stores,
  including compound stores, nested fields, and array elements. Type checking
  still applies; the block only marks the unsafe dereference boundary.
- **Hardware-oriented numerics:** `f16`, `bf16`, fixed-point `q7`/`q15`/`q31`,
  and SIMD vectors (`v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8`). `f16`/`bf16`
  use native compiler types where available and portable fallbacks otherwise.
- **Vector operations:** `vmax2`, `vmin2`, `vcmpeq`, `vcmpne`, `vcmplt`,
  `vcmple`, `vcmpgt`, `vcmpge`, and `vselect` work for every vector type.
  The x86-64 target uses SSE2 for direct comparisons, floating min/max, and
  bitwise lane select; other cases use scalar helpers.
- **Structural equality:** structs and arrays support recursive `==`/`!=`.
  Struct fields and array elements compare by value, strings by content,
  vectors lane-wise, and pointers by address. Floating-point fields use IEEE
  `==`, so NaNs compare unequal.
- **Explicit array casts:** `as` converts arrays with the same length and
  numeric or pointer element types element by element, including `f16`/`bf16`
  and quantized conversions. There are no implicit array conversions.
- **GGUF configuration:** the kernel parses `ModelConfig` from GGUF metadata
  using `general.architecture` plus tensor-shape fallbacks (embedding/FFN
  widths, layers, heads, KV heads, RoPE dimension, context length, vocabulary,
  and RMS epsilon), displays it via `config`, and sizes
  its activation, KV, logits, and token table workspaces from it. Inference is
  rejected if the shape is unsupported or exceeds the 60 MB workspace budget.
  Grouped-query attention (`n_head_kv < n_head`) is supported.
- **Allocator control:** `free(ptr)` releases a hosted allocation;
  `alloc_reset()` reclaims the whole freestanding bump heap;
  `alloc_mark()`/`alloc_release(mark)` reclaim transient allocations. The kernel
  marks the heap before each REPL command and releases it afterward, and has a
  `reset` command that clears the KV cache. In freestanding builds `free` is a
  no-op.
- **Self-contained runtime:** math (`sqrt`, `exp`, `tanh`, `erf`, `cos`, `sin`),
  sampling, GGUF parsing, and print/string helpers are hand-written. Generated
  code does not call libm. Hosted programs use the host C library; the kernel
  links only libgcc for 64-bit integer helpers.
- **Interactive OS:** VGA text console, PS/2 and serial keyboard, 100 Hz PIT,
  interrupts, PS/2 controller initialization (scancode set 1), IRQ1 plus a
  polling fallback, and a line-edited REPL with colored prompts, eight-entry
  history, cursor editing, tab completion, help topics, `stop` sequences,
  Ctrl+C cancellation, `ask` ChatML questions with timing/progress, `keytest`,
  `selftest`, `mem`, `clear`,
  `reset`, `reboot`, `poweroff`, Ctrl+Alt+Del, and ACPI shutdown. Expected
  inference/configuration errors return to the prompt; fatal panics print a
  reboot hint.
- **Verified target:** TinyStories-15M on a 2008 Lenovo IdeaPad S9 (Intel Atom),
  plus a reproducible QEMU/ISO path.

## Documentation

- [Language reference](docs/LANGUAGE.md) — syntax, types, aggregates,
  conversions, control flow, and current gaps.
- [Language guide](DOCS.md) — practical overview of the implemented language.
- [Lepton OS guide](DOC-LEPTONOS.md) — architecture, drivers, REPL, and build
  and deployment instructions.
- [Technical specification](LEPTONSPEC.md) — compiler, runtime, OS, and
  inference design.
- [Roadmap](ROADMAP.md) — status, history, and next work.
- [Model notes](models/README.md) — expected TinyStories file and checksum.

## Requirements

- Python 3.8+.
- A C11 compiler (`cc`, GCC, or Clang).
- `gcc-multilib` for `-m32` builds (kernel and 32-bit userland).
- `qemu-system-i386` for `make run-kernel` and `make run-iso`.
- `grub-pc-bin`, `grub-common`, `xorriso`, and `mtools` for `make iso`.
- A GGUF model for the AI commands. `models/tinystories.gguf` is the reference
  model used by the QEMU/ISO demos and the verification tools.

## Project structure

```text
Lepton/
├── leptonc.py              # Driver, lexer, parser
├── lepton_ir.py            # Typed IR classes and renderer
├── lepton_lower.py         # AST -> IR lowering, scoping, type inference
├── lepton_verify.py        # IR verification before C emission
├── lepton_types.py         # Primitives, vectors, fixed-point, C mappings
├── lepton_target.py        # Scalar and x86-64 SSE2 target descriptions
├── lepton_c_ir.py          # IR -> C11 emitter
├── lepton_runtime.py       # C preamble, runtime, and math strings
├── Makefile                # Userland, kernel, ISO, flash, and test targets
├── README.md
├── ROADMAP.md              # Development history, status, and future work
├── DOC-LEPTONOS.md          # Lepton OS guide
├── DOCS.md                  # Practical language guide
├── LEPTONSPEC.md            # Technical specification
│
├── src/hello.lp            # Sample userland program
│
├── kernel/                 # The Lepton OS (bare metal)
│   ├── arch/
│   │   ├── boot.S          # 32-bit Multiboot entry, GDT/IDT, FPU/SSE init
│   │   ├── isr.S           # Interrupt stubs and GDT loader
│   │   ├── interrupts.c    # PIC remap, PIT timer, PS/2 + serial input
│   │   ├── multiboot.c     # Multiboot module access
│   │   ├── vga.c           # VGA text-mode console
│   │   └── kernel.ld       # Linker script (loads at 1 MiB)
│   └── src/
│       ├── kernel.lp       # Kernel entry, REPL, tokenizer, and inference
│       └── runtime.c       # Freestanding runtime (serial, VGA, alloc, strings)
│
├── std/                    # Lepton standard library
│   ├── gguf/
│   │   ├── header.lp       # GGUF header parsing
│   │   ├── kv.lp           # Metadata key-value parsing
│   │   ├── tensor.lp       # Tensor table parsing and lookup
│   │   ├── config.lp       # ModelConfig from GGUF metadata
│   │   ├── dequant.lp      # Q4_0 and Q8_0 dequantization
│   │   └── model.lp        # Whole-file loader (16 MB buffer cap)
│   └── nn/
│       ├── matmul.lp       # 4x4 matmul/matvec helpers
│       ├── norm.lp         # RMSNorm and softmax helpers
│       └── activations.lp  # GELU, ReLU, vector variants
│
├── tests/
│   ├── cases/              # Success cases (.lp with .out/.exit)
│   ├── errors/             # Expected compilation failures
│   ├── test_ir_verify.py   # Malformed-IR verifier tests
│   ├── test_conversions.py # Conversion-policy tests
│   └── run_tests.py        # Test harness (make test)
│
├── docs/
│   └── LANGUAGE.md         # Language reference
│
├── tools/                  # Python utilities
│   ├── gen_test_gguf.py    # Synthetic single-tensor GGUF for smoke tests
│   ├── gen_gqa_gguf.py     # Tiny synthetic GQA model fixture
│   ├── gen_merge_gguf.py   # Tiny tokenizer fixture with BPE merges
│   ├── gen_q4k_gguf.py     # Tiny Q4_K dequantization fixture
│   ├── gen_q56_gguf.py     # Tiny Q5_K/Q6_K dequantization fixture
│   ├── dump_dims.py        # Dump GGUF tensor dimensions
│   ├── verify_embed.py     # Embedding reference check
│   ├── verify_forward.py   # Full Q/K/attention reference for TinyStories
│   ├── verify_gqa.py       # Two-position GQA reference check
│   ├── verify_merge.py     # Merge-tokenizer reference check
│   ├── verify_q4k.py       # Q4_K dequantization reference check
│   ├── verify_q56.py       # Q5_K/Q6_K reference check
│   └── verify_tokenize.py  # Tokenizer reference (requires merges)
│
├── models/                 # Place downloaded .gguf models here
├── READFILES/              # 52 .docx development transcripts (~43 MB)
└── AI-WORKSPACE/           # Scratch space for AI-assisted development
```

## Quick start

All commands assume the repository root.

### 1. Build the compiler and run the tests

```bash
make test
make test-sse
```

Expected results: `make test` reports `134 passed, 0 failed`,
`IR verifier: 13 tests passed`, and `Conversion policy: 12 tests passed`;
`make test-sse` reports `134 passed, 0 failed`.

### 2. Build and run a userland program

```bash
make all                  # 64-bit and 32-bit builds
./build/x86_64/hello
```

Or invoke the compiler directly:

```bash
python3 leptonc.py src/hello.lp -o hello
./hello
```

To see the IR verification summary:

```bash
python3 leptonc.py src/hello.lp -S -o hello.c --verify-ir
```

To reject lossy numeric and unsafe pointer conversions instead of warning:

```bash
python3 leptonc.py src/hello.lp -S -o hello.c --strict-types
```

### 3. Build and run Lepton OS in QEMU

```bash
make clean-kernel
make run-kernel MODEL=models/tinystories.gguf
```

Without `MODEL=`, `run-kernel` uses `build/kernel/test.gguf`, a synthetic
single-tensor file used for tensor/dequant smoke tests. The language-model and
tokenizer commands (`fwdpos`, `generate`, `genprompt`, `encode`, `vocab`) will
not work with that file.

At the `lepton>` prompt, useful commands include:

```text
config
info
tensors
fwdpos 0 1
encode Once upon a time
generate 0
genprompt "Once upon a time" 0.001 1
reset
reboot
poweroff
filetest
```

`generate 0` and `genprompt ... 0` select the greedy path. Both commands accept
an optional third numeric argument for the token count, defaulting to 30:

```text
generate 0 1 5
genprompt "Once upon a time" 0 1 5
```

Generation streams tokens as they are produced and stops at EOS, a configured
stop sequence, Ctrl+C, or when the configured context is full. Set a stop
sequence with `stop "text"`; `stop` with no text clears it.

To quit QEMU: press `Ctrl-A`, release, then `X`.

### 4. Build and run the GRUB ISO

```bash
make iso MODEL=models/tinystories.gguf
make run-iso
```

The ISO is written to `build/kernel/lepton.iso` (about 30 MB with the TinyStories
model). To flash it, use balenaEtcher or `make flash USB=/dev/sdX`; the latter
runs `sudo dd`, so verify the device name carefully.

The kernel is a 32-bit Multiboot 1 image. On real hardware, use legacy BIOS boot
(disable UEFI/Secure Boot or enable CSM).

### 5. Validate the GQA fixture

```bash
python3 tools/gen_gqa_gguf.py build/gqa.gguf
python3 tools/verify_gqa.py build/gqa.gguf
make run-kernel MODEL=build/gqa.gguf
```

At the prompt, run:

```text
gqapair
```

The reference and the kernel should both report `pos1 token = 3`.

### 6. Validate the merge tokenizer

```bash
python3 tools/gen_merge_gguf.py build/merge.gguf
python3 tools/verify_merge.py build/merge.gguf
make run-kernel MODEL=build/merge.gguf
```

At the prompt:

```text
encode ab
```

Both should resolve to token **5** (`▁ab`).

### 7. Validate Q4_K

```bash
python3 tools/gen_q4k_gguf.py build/q4k.gguf
python3 tools/verify_q4k.py build/q4k.gguf
make run-kernel MODEL=build/q4k.gguf
```

At the prompt, `q4ktest` should print `sum=928 dot=-20.5`.

Q5_K/Q6_K fixtures use the same `qktest` command:

```bash
python3 tools/gen_q56_gguf.py 13 build/q5k.gguf
python3 tools/verify_q56.py build/q5k.gguf
python3 tools/gen_q56_gguf.py 14 build/q6k.gguf
python3 tools/verify_q56.py build/q6k.gguf
```

Expected: Q5_K `sum=128 dot=-3`; Q6_K `sum=-3968 dot=93`.

### 8. Measure forward latency

```text
bench 16
```

`bench [N]` reports sequential forward time from the 100 Hz PIT. See
[RELEASE.md](RELEASE.md) for the current baseline.

### 9. Validate the full forward reference

With `models/tinystories.gguf` present:

```bash
python3 tools/verify_forward.py models/tinystories.gguf
```

The reference runs the tokenized prompt `Once upon a time` through Q/K/V
attention, RoPE, six layers, and the output projection. It should predict
token **29892** (`,`), matching the first greedy token from:

```text
genprompt "Once upon a time" 0 1 1
```

## Model profiles

TinyStories remains the fast deterministic regression model. SmolLM2-135M is
the realistic model profile: 30 layers, 576 hidden, GQA 9/3, GPT-2 tokenizer,
49k vocabulary, tied output. See [MODELS.md](MODELS.md) for dimensions,
checksums, expected outputs, and test recipes.

```bash
make iso MODEL=models/smallm.gguf
```

The kernel caps SmolLM2's 8192-position metadata context to 256 positions for
the current 60 MB workspace budget.

## Model naming note

The reference model is called TinyStories-15M, but the GGUF stores about 24.4M
tensor elements: `token_embd.weight` (9.2M), `output.weight` (9.2M), and 6
transformer layers (5.97M). The memory/performance arithmetic in
[ROADMAP.md](ROADMAP.md) uses the stored tensors, not the 15M label.

## Current limitations

These are the boundaries of the current implementation, not a wish list; see
[ROADMAP.md](ROADMAP.md) for the work planned against them.

- **Model dimensions.** Activation, KV, logits, and token-table workspaces are
  sized from GGUF metadata, subject to a 60 MB budget and a 256-position
  context cap. Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K weights and F16/F32
  fallbacks are supported. Tokenizer buffers are capacity-checked for the line and prompt
  limits.
- **Context safety.** The configured `max_seq` is enforced by the forward guard
  and generation loops, with a hard cap of 256 KV positions. The KV cache is
  allocated for exactly `max_seq` positions.
- **Conversion policy.** Lossy numeric conversions and unsafe pointer
  conversions warn by default; `--strict-types` makes them errors. Explicit
  `as` casts never warn. Integer-to-float is allowed without warning (wide
  integers may lose precision), and byte/void pointer views are always
  allowed. Untyped array/vector literals still infer a common type; array
  conversions are explicit (`a as [f32; N]`), not implicit.
- **Memory management.** The kernel uses a bump allocator. `free` is a no-op
  in freestanding builds; `alloc_reset` reclaims the whole heap but invalidates
  every live allocation. Each REPL command marks the heap and releases its
  transient allocations afterward, so `vocab`, `encode`, `tensors`, and token
  decoding no longer leak between commands. Long-lived buffers are allocated
  before the mark. Heap allocation failures are checked, and the heap is
  bounded by the Multiboot memory map (capped at 64 MB, minimum 8 MB after the
  loaded module).
- **Quantization coverage.** Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, and Q6_K
  are implemented, plus F16 and F32 fallbacks.
- **Freestanding file I/O.** `lp_file_size`, `lp_read_file_at`, and
  `lp_write_file` are defined in the kernel and return `-1`; there is no
  filesystem driver. Hosted builds use real file I/O.
- **Tokenizer edge cases.** The BPE merge path is exercised by a synthetic
  fixture; piece buffers and vocab/count checks are bounded, and unknown bytes
  map to GGUF `<0xNN>` tokens when present.
- **Generation controls.** Stop sequences match decoded text after
  normalizing the sentencepiece/GPT-2 space markers. `selftest` has profiles
  for TinyStories and SmolLM2; other models are skipped.
- **Q&A.** `ask` uses SmolLM2 Instruct ChatML (`<|im_start|>` /
  `<|im_end|>`) and requires an Instruct GGUF. Prefill is ~37 tokens and
  prints progress; Ctrl+C cancels. Ordinary REPL typing is unaffected once
  `ask` returns. `chat` starts a multi-turn ChatML mode with `/exit`, `/new`,
  `/ctx`, `/help`, and Ctrl+C cancellation; it keeps conversation KV state
  and drops the oldest turn when the context fills. Chat prompts show
  `ctx` usage and answers report prefill/answer time and heap. PageUp/PageDown or
  Ctrl+P/Ctrl+N review up to eight previous VGA pages. The system/user ChatML
  prefix is cached after the first ask:
  QEMU baseline after batched prefill/SSE2 was ~58 s for the first prefill and
  ~21 s for later questions. The cache is invalidated by KV-clearing commands.
  See [S9_BASELINE.md](S9_BASELINE.md) for the clean S9 measurement sequence. The tested file is
  `models/smollm2-instruct-q4_0.gguf` (Q4_0); Q4_K_S/Q4_K_M are refused by
  the mixed-K-quant compatibility guard. Use Q8_0 if a larger model is
  acceptable. The base `smallm.gguf`
  remains for completion tests only.
- **One backend.** Only scalar and x86-64 SSE2 are implemented. There is no
  AVX2, NEON/ARM, LLVM IR, or WASM backend yet.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pushes and pull
requests: pyflakes, scalar compiler tests, SSE2 tests, IR-verifier tests,
conversion policy tests, 64-bit and 32-bit userland builds, and the freestanding
kernel build. It does not run QEMU
or inference, because the QEMU path requires a GGUF model that is not tracked in
Git.

## Development history and scratch space

- **`READFILES/`** contains 52 `.docx` transcripts (~43 MB) from the
  AI-assisted development sessions. They are historical rationale, not a build
  requirement and not a current specification. The latest entries explain the
  kernel/ISO debugging that led to the current Makefile rules.
- **`AI-WORKSPACE/`** is scratch space for experiments. It is intentionally
  empty in the repository.

When sharing the project, also record the TinyStories model source and SHA-256
next to `models/tinystories.gguf`; the binary is not reproducible from the
repository alone.

## License

MIT. See [LICENSE](LICENSE).
