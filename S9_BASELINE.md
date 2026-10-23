# S9 Performance Baseline

This is the clean measurement sequence for the Lenovo IdeaPad S9 (1 GB RAM).
Run it after a reboot so the ChatML prefix cache is cold.

## Build

```bash
cd /home/alex/Lepton
make clean-kernel
make iso MODEL=models/smollm2-instruct-q4_0.gguf
```

Use the resulting `build/kernel/lepton.iso`.

## Capture the baseline

```text
config
mem
keytest          # press a few keys during the five-second window
bench 8
selftest
```

Then the Q&A path:

```text
ask "What is 2+2?" 8
ask "What is 3+3?" 8
```

The first ask builds the cached ChatML prefix. The second ask reuses it.

Record:

| Measurement               | S9 result                            |
|---------------------------|--------------------------------------|
| `config` model            | smollm2-instruct-q4_0.gguf           |
| `mem` Heap used / limit   | 16036176 / 67108864 bytes            |
| `keytest` events          | a - 97, f - 102, k - 107             |
| `bench 8` ms/token        | 7270 ms (908.75 ms/token)            |
| `selftest`                | skipped (unrecognized model profile) |
| First ask prefill ms      | 16620 ms                             |
| First ask answer ms       | 6420 ms                              |
| Second ask prefill ms     | 6180 ms                              |
| Second ask answer ms      | 6420 ms                              |
| `2+2` answer correct      | 2 + 2 = 4                            |
| `3+3` answer correct      | 3 + 3 = 3 +                          |

## Batched Q4_0 experiment (reverted)

The batched Q4_0 matvec was faster in QEMU but slower on the S9:

| Measurement | Per-token SSE2 | Batched Q4_0 |
|---|---:|---:|
| First ask prefill | 16620 ms | 21780 ms |
| First ask answer | 6420 ms | 6280 ms |
| Second ask prefill | 6180 ms | 8730 ms |
| Second ask answer | 6420 ms | 6280 ms |

Answer generation was unchanged; only prefill regressed. The batch kernel was
reverted in favor of the per-token SSE2 path.

## QEMU reference after SSE2 + layer-major prefill

TinyStories, 6 layers:

```text
bench 8: ~252.5 ms/token
```

SmolLM2-135M base, 30 layers:

```text
bench 1: ~2080 ms/token
```

Instruct Q4_0 Q&A:

```text
first ask:  prefix built;  37-token prefill  ~58 s
second ask: prefix cached; 13-token prefill  ~21 s
answer:     ~10.7 s for 6 tokens
```

These are QEMU reference numbers, not S9 expectations. The S9 will be slower.

## Notes

- `bench [n]` clears the KV cache and measures sequential forward passes.
- `ask` uses ChatML and caches the system/user prefix after the first call.
- `reset`, `generate`, `genprompt`, `fwdpos`, `bench`, `gqapair`, `selftest`, and `forward` invalidate the Ask prefix cache.
- Use Ctrl+C to cancel batched prefill or generation.
