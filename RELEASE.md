# Lepton v0.5.0 Release Validation

Date: 2026-10-23

## Release contents

- Shared kernel core with a BIOS32 platform shim.
- UEFI64 hello application and GOP/ConIn platform shell.
- 64-bit Lepton kernel linked against the UEFI platform.
- GGUF model loading from the EFI System Partition.
- UEFI64 `ask` and `chat`, including context status and Ctrl+C cancellation.
- BIOS32 + UEFI64 hybrid ISO with two El Torito entries.
- Retained v0.4.0 features: rolling chat memory, VGA scrollback, mixed-K guard.

## Validation matrix

| Check | Result |
|---|---|
| `make test` | 134 passed, 0 failed |
| `make test-sse` | 134 passed, 0 failed |
| `make all` | 64-bit and 32-bit userland builds pass |
| `make kernel32` | BIOS32 kernel builds |
| `make kernel64-check` | 64-bit core compiles |
| `make uefi64` | UEFI hello application builds |
| `make uefi64-shell` | UEFI platform shell builds |
| `make uefi64-kernel` | 64-bit Lepton kernel builds |
| `make iso-hybrid` | 276 MB hybrid ISO builds |
| BIOS hybrid boot | Lepton kernel self-checks run |
| UEFI hybrid boot | 64-bit Lepton kernel self-checks run |
| UEFI `fwdpos 0 1` | 28 (`,`) with SmolLM2 Q4_0 |
| UEFI `ask "what is 2+2?" 8` | `2 + 2 = 4.` |
| UEFI `chat` one turn | `2 + 2 = 4.`, `ctx 47/256` |
| UEFI Ctrl+C during prefill | Cancels and returns to prompt |
| `pyflakes` / `git diff --check` | Clean |

QEMU UEFI timings are slow:

```text
ask prefill: 133000 ms, answer: 35000 ms
chat system prefill: 73000 ms, question prefill: 57000 ms
```

These are QEMU/64-bit codegen timings, not native Ryzen expectations.

## S9 baseline

From [S9_BASELINE.md](S9_BASELINE.md):

```text
heap: 16,036,176 / 67,108,864 bytes
bench 8: 7270 ms (908.75 ms/token)
first ask prefill: 16620 ms
first ask answer: 6420 ms
second prefill: 6180 ms
second answer: 6420 ms
```

The S9 BIOS32 path was field-tested for v0.3.1 and remains the same kernel
path. The UEFI64 path still needs a Ryzen field test.

## Hybrid ISO layout

```text
/boot/lepton32.elf             BIOS32 kernel
/models/smollm2...gguf         BIOS Multiboot model
/boot/efi.img                  FAT ESP El Torito image
  EFI/BOOT/BOOTX64.EFI         64-bit kernel
  models/smollm2...gguf        UEFI model
```

The model is duplicated because BIOS Multiboot and UEFI file access use
different filesystems. A hybrid MBR/GPT table allows USB writers and UEFI
boot; BIOS USB still needs the BIOS-only/CD path.

For the S9, use `make iso-s9` and write `build/kernel/lepton-s9.iso`.

## Known limitations

- UEFI64 uses Boot Services; native 64-bit mode is not implemented.
- The 64-bit core still has 32-bit pointer assumptions, bounded by a low
  model/heap region.
- Hybrid ISO creation requires GRUB, xorriso, mtools, and OVMF; it is not yet
  in CI.
- Mixed K-quant models are refused explicitly.
- SmolLM2-135M remains unreliable at arithmetic.

## Upgrade notes

- BIOS32: `make kernel32`, `make iso`.
- UEFI64: `make uefi64-kernel`, `make run-uefi64-kernel`.
- Both: `make iso-hybrid`.
- Recommended model: `models/smollm2-instruct-q4_0.gguf`.
