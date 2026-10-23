# Changelog

## Unreleased

- Add a hybrid MBR/GPT table to `lepton-hybrid.iso` so Etcher/USB writers
  accept it and UEFI can boot from the FAT ESP. BIOS USB still requires the
  BIOS-only or CD path.

- Compile all UEFI64 kernel/runtime/platform objects with `-O$(OPT)` and SSE2.
  Native `bench 1` improved from 5060 to 2140 ms/token; native ask from
  126.55/34.47 s to 59.03/14.94 s. Reduce native PE section count so OVMF can
  load the optimized image.
- Added a native exception IDT, PIC remap, and 100 Hz PIT timer interrupt.
  Verified native `chat` and `poweroff` under OVMF.

- Added an optional UEFI64 native build that loads the model, calibrates TSC,
  exits Boot Services, installs a minimal exception IDT, and runs the kernel
  with direct GOP output and serial input.
  Native `fwdpos 0 1` returns 28 (`,`); native `ask` returns `2 + 2 = 4.`
  (prefill 126.55 s, answer 34.47 s under OVMF).
- Fixed PS/2 command ACK handling and added scancode-set selection; PS/2 input
  remains unreliable and serial input is the native test path.

## v0.5.0 — 2026-10-23

- Split the kernel onto a shared platform contract with a BIOS32 shim.
- Added `kernel32` and `kernel64-check` targets; the shared core now compiles
  as 64-bit C, exposing pointer-width work for the UEFI64 port.
- Added `PLATFORMS.md` and CI coverage for the 64-bit compile check.
- Added a minimal UEFI64 PE application that reports serial/GOP/memory-map
  information under OVMF.
- Added a UEFI64 platform shell with a GOP 8x16 console, ConIn line editing
  and history, a 16 MB heap, RTC time, and `help`/`mem`/`config`/`time`/
  `alloc`/`clear`/`echo`/`quit` commands.
- Linked the shared Lepton kernel core as a 64-bit UEFI application.
- Load a Q4_0 GGUF from the EFI System Partition instead of embedding it.
  OVMF with SmolLM2-135M-Instruct Q4_0 returns `fwdpos 0 1` = 28 (`,`).
- Build a hybrid ISO with a BIOS32 GRUB/El Torito entry and a direct UEFI64
  FAT ESP entry. Both boot the SmolLM2 model.
- Add UEFI64 keyboard queueing and Ctrl+C cancellation, fix platform tick
  units, and verify `ask`/`chat` with SmolLM2 Q4_0 under OVMF.

## v0.4.0 — 2026-10-23

- Added `chat`, a multi-turn ChatML mode with a persistent system prefix,
  conversation KV state, `ctx` status, `/exit`, and Ctrl+C cancellation.
- Added chat-aware prompts and line redraws (`you>`) while preserving the
  raw `lepton>` REPL.
- Added an eight-page VGA scrollback ring with PageUp/PageDown and
  Ctrl+P/Ctrl+N; typing returns to the live screen.
- Added rolling chat memory: track turns, drop the oldest turn when context
  fills, rebuild the retained KV, and add `/new`, `/ctx`, and `/help`.
- Chat prompts show context usage; prefill/answer timing and heap usage are
  reported after each turn.
- Refuse mixed K-quant models (Q4_K/Q5_K/Q6_K/Q5_0/Q5_1) before inference
  instead of producing corrupt output; Q4_0 and Q8_0 remain supported.
- Kept the per-token SSE2 Q4_0 prefill path after the batched variant proved
  slower on the S9; the experiment is recorded in `S9_BASELINE.md`.

## v0.3.1 — 2026-10-22

- Fixed REPL backspace echo: `\b` is not a Lepton string escape, so the raw
  backspace character is emitted directly instead of printing `b b`.
- Added SmolLM2-135M compatibility: GPT-2 byte-level tokenizer, BOS handling,
  tied token embedding/output, GQA 9/3, configurable RoPE base (100000), a
  256-position context cap, and a SmolLM `selftest` profile.
- Added `MODELS.md` with TinyStories and SmolLM2 profiles and test recipes.
- Decode `<0xNN>` byte tokens in story output and document TinyStories
  prompt-based story requests.
- Added `ask` with SmolLM2 Instruct ChatML tokens and `<|im_end|>` stopping.
  Ask prefill prints progress/timing and is Ctrl+C-cancellable; the fixed
  ChatML system/user prefix is cached across questions; `keytest` diagnoses
  raw keyboard events.
  Validated with `SmolLM2-135M-Instruct.Q4_0.gguf`
  (`ask "What is 2+2?" 8` → `2 + 2 = 4.`). Q5_0/Q5_1 block support was added
  and fixture-validated, but Q4_K_M still produced incorrect output and
  remains under investigation.

## v0.3.0 — 2026-10-21

- PS/2 keyboard is explicitly placed in scancode set 2 with translation to
  set 1; polling stops once IRQ1 is seen.
- REPL line editing: Left/Right, Home/End, Delete, Ctrl+C, Ctrl+L.
- Eight-entry command history with Up/Down.
- Tab completion for command names.
- Colored prompt, `clear`, and `help <command>` topics.
- `bench` progress markers.
- Streamed generation, stop sequences, Ctrl+C cancellation, context warnings,
  `selftest`, and `mem`.
- Direct Q4_0/Q8_0 matvecs; QEMU baseline improved from ~299.4 to
  ~272.5 ms/token.
- Q4_0 SSE2 matvec and layer-major batched prompt prefill. TinyStories
  `bench 8` is ~252.5 ms/token; SmolLM `bench 1` is ~2080 ms/token.
- Ask prefill fell to ~58 s first / ~21 s cached in QEMU.
- Q4_K/Q5_K/Q6_K dequantization/matvec, F16/F32 fallbacks, tensor
  shape/type checks, and K-quant fixtures validated against independent
  references.

## v0.2.2 — 2026-10-21

- PS/2 keyboard controller initialization and polling fallback for the S9.
- ACPI poweroff now reads `_S5` from the DSDT and uses the FADT SMI/PM1
  fields instead of treating PM1 lengths as sleep types.

## v0.2.1 — 2026-10-21

Tooling and validation release.

### Kernel

- Grouped-query attention (`n_head_kv < n_head`), including K/V projection,
  KV-cache, RoPE, and query-head grouping.
- Heap sizing from the Multiboot memory map, capped at 64 MB with an 8 MB
  minimum and a clear low-memory boot failure.
- Checked `alloc` emission; `lp_alloc_init` ABI corrected.
- Tokenizer bounds checks, UTF-8 piece initialization, `<0xNN>` byte
  fallback, and a working merge-based BPE path.
- Configurable generation length (`[n]`), EOS stopping, context-aware caps,
  and greedy temperature `0` for `genprompt`.

### Tooling and CI

- Tiny GQA model generator and two-position reference.
- Merge-tokenizer fixture and reference.
- Pure GGUF fixture checks in CI.
- Kernel build and both compiler test suites remain in CI.

### Validation

- 134 compiler tests passed on scalar and SSE2 targets.
- GQA, merge-tokenizer, and full-forward references pass.
- QEMU 128 MB and 32 MB boot and inference verified.
- QEMU baseline: about 299 ms/token for TinyStories.
- S9 field test remains pending.

### Breaking changes

None.

## v0.2.0

First tagged public snapshot after the type-system and kernel inference work.

- Declared-type enforcement, conversion diagnostics/strict mode, bool-only
  conditions, lvalues, compound assignment, string comparison, aggregate
  equality, vector comparison/select, array casts, `switch`, `raw`, and
  definite-assignment verification.
- GGUF `ModelConfig`, config-driven workspaces, soft inference errors,
  `reboot`, Ctrl+Alt+Del, and an ACPI `poweroff` command.
- Q4_0/Q8_0/F32 inference, tokenizer encode/decode, generation, and
  interactive REPL on the Lenovo IdeaPad S9.
