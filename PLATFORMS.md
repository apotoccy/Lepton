# Platform targets

Lepton uses one shared kernel core with per-platform boot and driver shims.

```text
kernel/src/kernel.lp       shared core: GGUF, inference, REPL, chat
kernel/src/runtime.c       shared heap, serial/VGA helpers, SSE2 kernels
kernel/platform.h          platform contract and BootInfo
kernel/arch/bios32/        BIOS32 implementation (current release)
kernel/arch/uefi64/        UEFI64 platform and boot
```

## Current targets

| Target | Status | Use |
|---|---|---|
| `make kernel32` | Stable | BIOS/Multiboot kernel for the S9 and QEMU |
| `make kernel` | Alias for `kernel32` | Existing build path |
| `make kernel64-check` | Compile check | Verifies the shared core builds as 64-bit C |
| `make uefi64` | Hello milestone | Minimal 64-bit EFI application |
| `make run-uefi64` | Hello milestone | Boots the EFI hello app under OVMF |
| `make uefi64-shell` | Shell milestone | GOP console + ConIn platform shell |
| `make run-uefi64-shell` | Shell milestone | Boots the platform shell under OVMF |
| `make uefi64-kernel` | Stable milestone | 64-bit Lepton core; model loaded from ESP |
| `make run-uefi64-kernel` | Inference milestone | Boots the 64-bit kernel under OVMF |
| `make uefi64-native` | Native milestone | Exits Boot Services; TSC + serial input |
| `make run-uefi64-native` | Native milestone | Boots native mode under OVMF |
| `make iso-hybrid` | Release image | BIOS32 + UEFI64 hybrid ISO |
| `make run-iso-hybrid` | Release image | Boots the BIOS hybrid under QEMU |
| `make run-iso-hybrid-uefi` | Release image | Boots the UEFI hybrid under OVMF |
| `make iso` | Stable | BIOS GRUB ISO |

`kernel64-check` is a compile check; the bootable target is `uefi64-kernel`.
The shared core still contains 32-bit pointer assumptions, but the low-memory
UEFI model/heap region keeps addresses below 4 GB.

The UEFI64 hello application boots under OVMF, prints over serial and GOP,
reports the UEFI memory map, and locates the Graphics Output Protocol.

The UEFI64 platform shell adds a framebuffer console with an embedded 8x16
font, UEFI ConIn keyboard input, line editing/history, a 16 MB UEFI-backed
heap, an RTC command, and `help`/`mem`/`config`/`clear`/`echo`/`quit`.

The UEFI64 Lepton kernel links the shared kernel core with the UEFI platform,
loads a Q4_0 GGUF from the EFI System Partition, copies it into low allocated
memory, and runs inference. Under OVMF, SmolLM2-135M-Instruct Q4_0 reports
`fwdpos 0 1` as token 28 (`,`).

The model path is `\models\smollm2-instruct-q4_0.gguf`; the FAT image is
built by `run-uefi64-kernel`. Loading a model no longer requires rebuilding
the EFI binary.

A native build (`UEFI_NATIVE`) loads the model while Boot Services are
available, calls `ExitBootServices`, installs an exception IDT, remaps the
PIC, and runs a 100 Hz PIT timer interrupt. It uses direct GOP output,
interrupt-driven ticks, and serial input.

Native OVMF results after `-O2`:

```text
bench 1: 2110 ms/token
ask "what is 2+2?" 8: prefill 59.03 s, answer 14.94 s
chat system prefill: 33.35 s
chat question prefill: 25.74 s
chat answer: 21.49 s
```

Native `chat` returns `2 + 2 = 4.` and `poweroff` exits QEMU. PS/2 input still
needs scancode-set work; serial input is the supported native path. APIC
timer/interrupt ownership and USB HID remain future work.

The 64-bit Boot Services kernel supports one-shot `ask` and multi-turn
`chat`, with ChatML context status. OVMF verification:

```text
ask "what is 2+2?" 8
  prefill: 133000 ms
  2 + 2 = 4.
  answer: 35000 ms for 8 token(s)

chat
  system prefill: 73000 ms
  prefilling 16 tokens...
  prefill: 57000 ms
  2 + 2 = 4.
  ctx 47/256 | turns 1 | 8 tok / 50000 ms | heap 16/64 MB
```

Ctrl+C cancels 64-bit prefill. These QEMU timings are slow; native Ryzen
hardware and later optimization should be much faster.

## Hybrid ISO

`make iso-hybrid` creates `build/kernel/lepton-hybrid.iso` with a hybrid
MBR/GPT partition table and two El Torito entries:

- BIOS32 GRUB standalone -> `/boot/lepton32.elf` + `/models/...gguf`;
- UEFI64 FAT ESP -> `/EFI/BOOT/BOOTX64.EFI` + `/models/...gguf`.

The model is present in both the ISO9660 tree and the FAT ESP because the
BIOS module and UEFI file APIs read different filesystems. The image is about
276 MB.

The hybrid MBR makes the image acceptable to USB writers such as Etcher and
bootable through the UEFI ESP. BIOS USB boot still lands in the GRUB rescue
prompt because the custom El Torito MBR is not a full GRUB core image; use the
BIOS-only `make iso` image or boot the hybrid via CD for the S9.

## Platform contract

The shared core calls only these platform entry points:

- console: init, putc, clear, color, cursor, page up/down;
- input: poll, read, cancel pending;
- time: ticks, halt;
- boot: module count/start/size/end, memory end;
- reset: reboot, poweroff.

`kernel/arch/bios32/platform.c` maps them to the existing VGA, PS/2, PIT,
Multiboot, and ACPI implementations.

## Design constraint

The model and inference code are shared; the boot binaries are not. The
hybrid ISO carries the model twice because BIOS Multiboot and the UEFI file
protocol read different filesystems.
