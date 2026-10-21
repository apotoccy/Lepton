# Lepton v0.3.0 Release Validation

Date: 2026-10-21

## Release contents

- REPL 2.0: cursor editing, eight-entry history, tab completion, colored
  prompt, `clear`, and `help <command>` topics.
- Grouped-query attention with a synthetic two-position reference.
- Multiboot memory-map heap sizing, allocation checks, and low-memory failure.
- Tokenizer bounds, `<0xNN>` byte fallback, and a working merge-based BPE path.
- Configurable generation length, EOS stopping, and greedy temperature `0`.
- GQA, merge-tokenizer, and full-forward reference tooling.
- Pure GGUF fixture checks in CI.

## Validation matrix

| Check | Result |
|---|---|
| `make test` | 134 passed, 0 failed |
| `make test-sse` | 134 passed, 0 failed |
| Clean `make all` | Passes |
| `make -B kernel` | Builds `build/kernel/kernel.elf` |
| GQA fixture | Reference and kernel both predict token 3 at position 1 |
| Q4_K fixture | Reference and kernel both report sum 928, dot -20.5 |
| Q5_K/Q6_K fixtures | Reference and kernel agree on sums and dots |
| Merge fixture | Reference and kernel both resolve `ab` to token 5 (`▁ab`) |
| Full forward reference | Predicts token 29892 (`,`) for `Once upon a time` |
| QEMU 128 MB | Boots; `fwdpos 0 1` returns 9038 (`Once`) |
| QEMU 32 MB | Boots with a 10.8 MB heap; same prediction |
| QEMU 24 MB | Clean low-memory fatal message |
| REPL editor | History, Tab, Left/Delete verified in QEMU |
| `pyflakes` / `git diff --check` | Clean |

## Performance baseline

Model: TinyStories-15M, Q4_0 weights, 6 layers, `n_embd=288`, 128-position
context.

Host: AMD Ryzen 7 5800HS, Ubuntu 6.8 kernel, QEMU 6.2.0, `-m 128`.

```text
lepton> bench 16
  bench: 16 tokens in 4360 ms (272.5 ms/token)
```

With `-m 32`:

```text
lepton> bench 8
  bench: 8 tokens in 2240 ms (280 ms/token)
```

`bench [N]` measures sequential forward passes at 100 Hz PIT resolution. It is
a CPU baseline, not a wall-clock S9 measurement.

## S9 field test

Still required on the Lenovo IdeaPad S9:

1. Build the tagged ISO with `make iso MODEL=models/tinystories.gguf`.
2. Boot it and confirm the prompt accepts typing. Then run `config`,
   `fwdpos 0 1`, `bench 8`, and `genprompt "Once upon a time" 0 1 20`.
3. Record the S9 `bench` result and whether `poweroff` completes after the
   ACPI `_S5` fix.

The host currently has no physical S9 access, so this remains a release
checklist item rather than a completed validation.

## Known limitations

- S9 keyboard input uses controller initialization plus a polling fallback;
  field testing passed after the v0.2.1 keyboard fix.
- REPL 2.0 is QEMU-tested; an S9 line-editing/history retest is pending.
- SmolLM2-135M is validated in QEMU: config, GPT-2 tokenization, tied output,
  GQA, RoPE base 100000, prompt prefill, generation, and `selftest`
  (BOS 29, prompt 28). `ask` is validated with the Instruct Q4_0 variant
  (`2 + 2 = 4.`); Q4_K_S/Q4_K_M from that repository need Q5_0 support.
- One S9 report showed a black screen with the REPL build; the boot path is
  being triaged separately from the keyboard fix.
- `poweroff` now uses the DSDT `_S5` sleep types and the FADT SMI/PM1 fields;
  S9 retest is required.
- Heap cap is 64 MB and the inference workspace budget is 60 MB; models above
  the TinyStories class do not fit.
- Context is capped at 256 positions.
- Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F16, and F32 weights are supported.
- Tensor shape/type checks now fail softly for incompatible matvecs.
- GQA is implemented and synthetic-tested; no real GQA model has been run.
- The tokenizer merge path is synthetic-tested; TinyStories has no merges.
- Generation defaults to 30 tokens and streams output; stop sequences and
  Ctrl+C cancellation are supported.
- No scrollback; command history is limited to eight entries.
- 272.5 ms/token is the current QEMU baseline (about 9% faster than the
  previous 299.4 ms/token); S9 performance is expected to be slower.

## Next-phase priorities

1. S9 benchmark and poweroff investigation (ACPI/APM).
2. K-quant support for broader GGUF compatibility.
3. Real GQA and merge-tokenizer model validation.
4. Streaming generation and stop sequences.
5. Larger-context / lower-memory inference work.
