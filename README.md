# LEPTON LANGUAGE AND LEPTON OS

Lepton is a minimalist, AI-native programming language and bare-metal operating
system for edge inference. It compiles Lepton source through a typed IR to
portable C11, and runs quantized GGUF models directly on the CPU with no Linux,
glibc, Python VM, PyTorch, or CUDA.

The reference target is a 15M-parameter TinyStories model. The kernel boots on a
2008 Lenovo IdeaPad S9 (Intel Atom) and in QEMU; the same kernel can be packaged
as a GRUB ISO.

## Status (as of October 2026)

- **Compiler:** 82/82 regression tests pass with both the scalar and x86-64 SSE2
  targets (`make test`, `make test-sse`).
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
  and no ML framework.
- **Custom compiler pipeline:** lexer/parser → typed flat IR → C11, with source
  spans and one lowering pipeline.
- **Enforced declared types:** `let`/`var` annotations, assignments, call
  arguments, return values, and struct fields must be assignable to their
  declared types. Numeric conversions and pointer casts are inserted explicitly
  in the IR, and array/vector literals are lowered against their target type.
- **Hardware-oriented numerics:** `f16`, `bf16`, fixed-point `q7`/`q15`/`q31`,
  and SIMD vectors (`v2f64`, `v4f32`, `v4i32`, `v8i16`, `v16i8`). `f16`/`bf16`
  use native compiler types where available and portable fallbacks otherwise.
  Every implemented vector operation has a scalar fallback; the x86-64 target
  adds SSE2 paths for the operations SSE2 provides.
- **Self-contained runtime:** math (`sqrt`, `exp`, `tanh`, `erf`, `cos`, `sin`),
  sampling, GGUF parsing, and print/string helpers are hand-written. Generated
  code does not call libm. Hosted programs use the host C library; the kernel
  links only libgcc for 64-bit integer helpers.
- **Interactive OS:** VGA text console, PS/2 and serial keyboard, 100 Hz PIT,
  interrupts, and a live REPL.
- **Verified target:** TinyStories-15M on a 2008 Lenovo IdeaPad S9 (Intel Atom),
  plus a reproducible QEMU/ISO path.

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
├── lepton_types.py         # Primitives, vectors, fixed-point, C mappings
├── lepton_target.py        # Scalar and x86-64 SSE2 target descriptions
├── lepton_c_ir.py          # IR -> C11 emitter
├── lepton_runtime.py       # C preamble, runtime, and math strings
├── Makefile                # Userland, kernel, ISO, flash, and test targets
├── README.md
├── ROADMAP.md              # Development history, status, and future work
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
│   │   ├── tensor.lp       # Tensor table parsing
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
│   └── run_tests.py        # Test harness (make test)
│
├── tools/                  # Python utilities
│   ├── gen_test_gguf.py    # Synthetic single-tensor GGUF for smoke tests
│   ├── dump_dims.py        # Dump GGUF tensor dimensions
│   ├── verify_embed.py     # Embedding reference check
│   ├── verify_forward.py   # Partial forward reference (V/output; skips Q/K)
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

Expected result for each: `82 passed, 0 failed`.

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
```

`generate 0` selects the greedy path. `genprompt ... 0` is currently coerced to
temperature 1.0, so use a small positive temperature such as `0.001` for
deterministic generation. Both `generate` and `genprompt` currently produce 30
tokens.

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

## Model naming note

The reference model is called TinyStories-15M, but the GGUF stores about 24.4M
tensor elements: `token_embd.weight` (9.2M), `output.weight` (9.2M), and 6
transformer layers (5.97M). The memory/performance arithmetic in
[ROADMAP.md](ROADMAP.md) uses the stored tensors, not the 15M label.

## Current limitations

These are the boundaries of the current implementation, not a wish list; see
[ROADMAP.md](ROADMAP.md) for the work planned against them.

- **One model shape.** The kernel hardcodes `n_embd=288`, `n_ff=768`,
  `n_layer=6`, `n_head=6`, `head_dim=48`, vocab 32000, and a 256-position KV
  cache. Only Q4_0/Q8_0 weights and F32 norms are handled. Other Llama-family
  sizes require source changes; the GGUF metadata is not yet used to configure
  the inference path.
- **Context safety.** The model metadata declares `context_length=128`, while
  the kernel reserves 256 positions, and it does not yet check prompt or
  generation positions against the cache. Stay within the cache until the
  bound is enforced.
- **Implicit conversions.** At an annotated boundary, numeric conversions are
  inserted automatically, including narrowing conversions such as `i32` from
  `f64`; pointer-to-pointer casts are also allowed for byte-buffer views. There
  is no lossy-conversion warning yet. Array/vector literals are target-typed
  when an annotation is present; untyped literals still infer a common type.
- **Memory management.** The kernel uses a bump allocator with no `free` or
  reset. Commands that allocate per item (`vocab`, `encode`, `tensors`, token
  decoding) leak across REPL invocations, and not every `lp_alloc` result is
  checked for null. The heap is a fixed 64 MB reservation after the loaded
  module, not dynamically sized from the Multiboot memory map.
- **Quantization coverage.** Q4_K/Q5_K/Q6_K and grouped-query attention (GQA)
  are not implemented. The kernel's linear layers handle Q4_0/Q8_0 weights plus
  F32 norm tensors; other tensor types are not supported.
- **Freestanding file I/O.** `lp_file_size`, `lp_read_file_at`, and
  `lp_write_file` are declared for freestanding builds but are not defined in
  `kernel/src/runtime.c`; a freestanding program that uses them fails to link.
- **Tokenizer edge cases.** The BPE merges path exists but is untested on this
  model (TinyStories has no merges), its fixed piece buffers need bounds work,
  and byte-fallback handling is missing.
- **Compiler gaps.** Compound assignment (`+=`, etc.) parses but is not lowered;
  `raw` blocks parse but are not lowered; arrays inside struct fields currently
  fail to compile because the emitter does not generate their C typedefs;
  `vcmpeq`/`vcmple`/`vcmpge` and an SSE2 `vselect` are missing.
- **One backend.** Only scalar and x86-64 SSE2 are implemented. There is no
  AVX2, NEON/ARM, LLVM IR, or WASM backend yet.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pushes and pull
requests: scalar compiler tests, SSE2 tests, 64-bit and 32-bit userland builds,
and the freestanding kernel build. It does not run QEMU or inference, because
the QEMU path requires a GGUF model that is not tracked in Git.

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
