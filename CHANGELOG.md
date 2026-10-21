# Changelog

## Unreleased — v0.2.2

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
