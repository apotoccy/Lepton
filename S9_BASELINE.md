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

| Measurement | S9 result |
|---|---|
| `config` model | Qwen/SmolLM values |
| Heap used / limit | |
| `keytest` events | |
| `bench 8` ms/token | |
| `selftest` | |
| First ask prefill ms | |
| First ask answer ms | |
| Second ask prefill ms | |
| Second ask answer ms | |
| `2+2` answer correct | |
| `3+3` answer correct | |

## QEMU reference after SSE2 + batched prefill

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
