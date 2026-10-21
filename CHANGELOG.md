# Changelog

## v0.3.1 — 2026-10-21

- Fixed REPL backspace echo: `\b` is not a Lepton string escape, so the raw
  backspace character is emitted directly instead of printing `b b`.
- Added SmolLM2-135M compatibility: GPT-2 byte-level tokenizer, BOS handling,
  tied token embedding/output, GQA 9/3, configurable RoPE base (100000), a
  256-position context cap, and a SmolLM `selftest` profile.
- Added `MODELS.md` with TinyStories and SmolLM2 profiles and test recipes.
- Added `ask` with SmolLM2 Instruct ChatML tokens and `<|im_end|>` stopping.
  Validated with `SmolLM2-135M-Instruct.Q4_0.gguf`
  (`ask "What is 2+2?" 8` → `2 + 2 = 4.`). The Q4_K_S/Q4_K_M variants use
  Q5_0 tensors and are not supported yet.

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
