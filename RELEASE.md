# Lepton v0.4.0 Release Validation

Date: 2026-10-23

## Release contents

- Multi-turn ChatML `chat` mode with a persistent system prefix.
- Rolling conversation memory: oldest-turn eviction, KV rebuild, `/new`,
  `/ctx`, and `/help`.
- Context-aware `you [used/max]>` prompt with prefill/answer timing, token
  count, and heap usage.
- Eight-page VGA scrollback with PageUp/PageDown and Ctrl+P/Ctrl+N.
- Mixed K-quant compatibility guard: Q4_K_M and other mixed K-quant models
  are refused instead of producing corrupt text.
- Q4_0 and Q8_0 Instruct inference remain supported.
- Per-function annotations across `kernel/src/kernel.lp` and `leptonc.py`.
- S9 baseline and the reverted batched-Q4 experiment documented.

## Validation matrix

| Check | Result |
|---|---|
| `make test` | 134 passed, 0 failed |
| `make test-sse` | 134 passed, 0 failed |
| `make all` | 64-bit and 32-bit userland builds pass |
| `make -B kernel` | Builds `build/kernel/kernel.elf` |
| `make iso MODEL=models/smollm2-instruct-q4_0.gguf` | Builds 100 MB ISO |
| `pyflakes` / `git diff --check` | Clean |
| TinyStories `fwdpos 0 1` | Token 9038 (`Once`) |
| TinyStories `genprompt` | Correct |
| SmolLM `selftest` | PASS |
| Instruct Q4_0 `ask "What is 2+2?" 8` | `2 + 2 = 4.` |
| Q4_K_M `config` / `fwdpos` | Clear unsupported K-quant error |
| QEMU Q4_0 Ask prefill/answer | 59.08 s / 14.96 s for 8 tokens |
| Chat `/new`, `/ctx`, `/exit` | Verified in QEMU |
| Scrollback | Verified in QEMU via Ctrl+P |
| Ctrl+C prefill/generation | Verified |

The S9 functional test passed for the v0.3.1 model path. Chat, scrollback, and
the mixed-K-quant guard still need an S9 smoke test before field sign-off.

## S9 baseline

Recorded in [S9_BASELINE.md](S9_BASELINE.md):

```text
heap:              16,036,176 / 67,108,864 bytes
bench 8:           7270 ms (908.75 ms/token)
first ask prefill: 16620 ms
first ask answer:   6420 ms
second prefill:     6180 ms
second answer:      6420 ms
2+2:               2 + 2 = 4
3+3:               3 + 3 = 3 +   (model-quality error)
```

The batched Q4_0 matvec experiment was faster in QEMU but slower on the S9
(21.78 s / 8.73 s prefills) and was reverted. Per-token SSE2 remains the
release path.

## Known limitations

- Mixed K-quant models are refused; the root cause of their corrupt forward
  output is unresolved.
- Q8_0 is supported but larger and slower than Q4_0.
- Context is capped at 256 positions and the heap at 64 MB.
- SmolLM2-135M is unreliable at arithmetic.
- No filesystem, graphical UI, or persistent chat transcript.
- The S9 is a single-core Atom; first-token prefill remains noticeable.

## Upgrade notes

- Use `models/smollm2-instruct-q4_0.gguf` for Instruct Q&A.
- Q4_K_S/Q4_K_M now fail fast with:
  `mixed K-quant model; use a Q4_0 or Q8_0 conversion`.
- `chat` uses `/new` to reset, `/ctx` for usage, and `/exit` to leave.
- PageUp/PageDown or Ctrl+P/Ctrl+N review previous VGA pages.
