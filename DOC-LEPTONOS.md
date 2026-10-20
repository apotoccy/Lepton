# Lepton OS

Lepton OS is a minimalist, AI-native operating system that runs quantized
language models directly on x86 hardware. It is a 32-bit Multiboot 1 kernel with
no Linux, glibc, Python VM, or ML framework underneath it.

The kernel boots on QEMU, on the GRUB ISO path, and on legacy-BIOS hardware. It
has been tested on a 2008-era Lenovo IdeaPad S9 (Intel Atom) as well as modern
x86 machines in QEMU.

For the language itself, see [DOCS.md](DOCS.md) and the canonical
[docs/LANGUAGE.md](docs/LANGUAGE.md). For development history and future work,
see [ROADMAP.md](ROADMAP.md).

## Current status

- Compiler: **134/134** regression tests on scalar and x86-64 SSE2, plus 13
  IR-verifier tests and 12 conversion-policy tests.
- Build: 64-bit and 32-bit userland programs, the freestanding 32-bit kernel,
  and the GRUB ISO all build.
- Boot: tested through QEMU `-kernel`, QEMU ISO `-cdrom`, and real hardware.
- Inference: TinyStories-15M loads from a Multiboot module. `fwdpos 0 1`
  predicts token **9038** (`Once`), matching the Python reference.
- Tokenizer: `encode Once upon a time` returns `[9038, 2501, 263, 931]`.
- Scope: inference workspaces are sized from GGUF metadata, within a 60 MB
  budget and 256-position context cap. Q4_0/Q8_0, F32 norms, and GQA are
  supported; modern K-quants are not.

## 1. Architecture

Lepton OS is a monolithic 32-bit kernel using the **Multiboot 1** specification.

- **Bootloader.** GRUB or QEMU's `-kernel` flag loads the kernel at 1 MiB.
- **CPU setup.** The boot path installs a flat GDT and an IDT, remaps the PIC,
  and enables the x87 FPU and SSE (CR0/CR4) before floating-point code runs.
- **Model loading.** The GGUF file is passed as a Multiboot module and accessed
  directly from RAM; the kernel does not copy the whole model.
- **Heap.** A bump heap is initialized after the loaded module. Its range is
  bounded by the Multiboot memory map, capped at 64 MB, and requires at least
  8 MB. A low-memory boot prints a fatal message instead of overrunning RAM.
- **Interrupts.** The PIT runs at 100 Hz and drives ticks and the PRNG. Keyboard
  and serial input are interrupt-driven.

## 2. Hardware drivers

- **VGA text console.** 80x25 text at `0xB8000`, with scrolling, cursor
  tracking, and color attributes.
- **PS/2 keyboard.** Scancode Set 1 with Shift, Ctrl, and Alt tracking;
  Ctrl+Alt+Del reboots. Keystrokes go into a 256-byte ring buffer. The REPL
  line buffer is 128 bytes.
- **Serial (COM1).** 38400 baud, mirroring screen output and accepting input.
  Used for headless QEMU sessions.
- **PIT.** IRQ 0 at 100 Hz.
- **PIC.** Remapped to vectors `0x20`-`0x2F` so they do not collide with CPU
  exceptions.

## 3. Inference runtime

The model runtime is self-contained and written in Lepton plus a small
freestanding C runtime.

- **GGUF.** Parses the header, metadata KV pairs, tensor table, and alignment.
- **Quantization.** Dequantizes Q4_0 (18-byte blocks) and Q8_0 (34-byte blocks).
  F32 norm tensors are handled directly.
- **Transformer.** LLaMA-style forward pass with token embeddings, multi-head
  attention, RoPE, a per-layer KV cache, RMSNorm, and a SwiGLU MLP.
- **Tokenizer.** Loads the BPE vocabulary from GGUF. A greedy longest-match
  fallback handles models without merge lists, and unknown bytes fall back to
  `<0xNN>` vocabulary entries. Piece buffers and vocab counts are bounded. The
  merge path is present but untested on TinyStories, which has no merges.
- **Sampling.** Temperature scaling, Top-K filtering, and an xorshift32 PRNG.

The kernel currently re-dequantizes weights during each forward pass. A
persistent dequantized-weight cache is a future performance step.

## 4. REPL

The kernel enters an idle loop and presents a `lepton>` prompt with backspace
line editing.

### Inspection

- `info` — GGUF version, tensor count, and KV count.
- `config` — model hyperparameters parsed from GGUF metadata (`n_embd`,
  `n_ff`, layers, heads, KV heads, head dimension, context length, vocabulary,
  RMS epsilon), plus whether this build supports the shape.
- `tensors` — tensor names, dimensions, and quantization types.
- `modules` — physical addresses and sizes of loaded Multiboot modules.
- `matvec` — diagnostic matrix-vector multiplication on the first tensor.
- `filetest` — exercise the freestanding file I/O failure contract.
- `reboot` — reboot the machine; Ctrl+Alt+Del does the same from the keyboard.
- `poweroff` — shut down via ACPI PM1, with QEMU/emulator port fallbacks.

### Tokenizer

- `vocab <start> <end>` — print a vocabulary range.
- `decodetok <id>` — decode one token ID.
- `encode "<text>"` — encode text to token IDs.

### Generation and reset

- `generate [temp] [top_k]` — 30 tokens from BOS. `generate 0` selects the
  greedy path.
- `genprompt "<text>" [temp] [top_k]` — encode a prompt, prefill, and generate
  30 continuation tokens. Use a small positive temperature such as `0.001` for
  deterministic output; a literal `0` is currently coerced to `1.0`.
- `reset` — clear the KV cache without rebooting.
- `reboot` — reset the machine; Ctrl+Alt+Del also works.
- `poweroff` — power down through ACPI when the firmware exposes it.

Both generation commands are fixed at 30 tokens. `generate`, `genprompt`, and
`fwdpos` clear the KV cache before they run, and `reset` provides an explicit
cleanup command. Each REPL command also reclaims its transient heap allocations
with an allocator mark/release around the command.

Configuration parsing is architecture-aware and falls back to tensor shapes
when metadata keys are missing. Configuration and inference problems
(unsupported shapes, workspace budget, context bounds) are reported as soft
errors with a specific reason and return to the prompt. A fatal
panic prints `Ctrl+Alt+Del to reboot` and enables interrupts so the keyboard
reset combination can recover the machine.

## 5. Building and running

All commands run from the repository root.

### Build only

```bash
make clean-kernel
make kernel MODEL=models/tinystories.gguf
```

`make kernel` compiles `build/kernel/kernel.elf`. It does **not** launch QEMU.

### Run in QEMU

```bash
make run-kernel MODEL=models/tinystories.gguf
```

Without `MODEL=`, `run-kernel` uses `build/kernel/test.gguf`, a synthetic
one-tensor file for smoke tests. Model and tokenizer commands will not work with
that file.

Quit QEMU with `Ctrl-A`, release, then `X`.

### ISO and hardware

```bash
make iso MODEL=models/tinystories.gguf
make run-iso
```

The ISO is written to `build/kernel/lepton.iso`. To flash it:

```bash
make flash USB=/dev/sdX
```

`make flash` runs `sudo dd`; verify the device name first. For legacy hardware,
disable UEFI/Secure Boot or enable CSM. If hybrid USB boot fails, a FAT32 USB
with GRUB installed and `kernel.elf` plus the model in `/boot` also works.

## 6. Limitations

- **Dimensions.** Activation, KV, logits, token-table, and scratch buffers are
  sized from GGUF metadata. The shape must fit a 60 MB workspace budget;
  tokenizer merge/ID buffers still have fixed limits.
- **One quantization set.** Q4_0, Q8_0, and F32 norms only; Q4_K/Q5_K/Q6_K are
  not implemented.
- **Context safety.** The configured `max_seq` is enforced by the forward guard
  and generation loops, with a hard cap of 256 KV positions.
- **Memory.** `free` is a no-op in the freestanding bump allocator. The REPL
  marks the heap before each command and releases transient allocations after
  it; `alloc_reset()` reclaims the whole heap but invalidates long-lived
  buffers. Not every `lp_alloc` result is null-checked.
- **File I/O.** Freestanding `file_size`, `read_file_at`, and `write_file`
  return `-1`; the kernel has no filesystem driver.
- **Compiler gaps.** Array/vector indexing and field assignment require lvalue
  bases.
- **One backend.** Scalar and x86-64 SSE2 only; no AVX2, NEON/ARM, LLVM IR, or
  WASM backend.
