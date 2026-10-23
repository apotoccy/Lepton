# Platform targets

Lepton uses one shared kernel core with per-platform boot and driver shims.

```text
kernel/src/kernel.lp       shared core: GGUF, inference, REPL, chat
kernel/src/runtime.c       shared heap, serial/VGA helpers, SSE2 kernels
kernel/platform.h          platform contract and BootInfo
kernel/arch/bios32/        BIOS32 implementation (current release)
kernel/arch/uefi64/        UEFI64 port (next phase)
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
| `make uefi64-kernel` | Inference milestone | 64-bit Lepton core + embedded Q4_0 model |
| `make run-uefi64-kernel` | Inference milestone | Boots the 64-bit kernel under OVMF |
| `make iso` | Stable | BIOS GRUB ISO |

`kernel64-check` is not a bootable UEFI kernel yet. It currently exposes
32-bit pointer assumptions in the shared core, which the UEFI port will fix.

The UEFI64 hello application boots under OVMF, prints over serial and GOP,
reports the UEFI memory map, and locates the Graphics Output Protocol.

The UEFI64 platform shell adds a framebuffer console with an embedded 8x16
font, UEFI ConIn keyboard input, line editing/history, a 16 MB UEFI-backed
heap, an RTC command, and `help`/`mem`/`config`/`clear`/`echo`/`quit`.

The UEFI64 Lepton kernel milestone links the shared kernel core with the UEFI
platform, copies an embedded Q4_0 TinyStories model into low allocated memory,
and runs inference. Under OVMF, `fwdpos 0 1` predicts token 9038 (` Once`).
The model is embedded for this milestone; loading it from the ESP and running
SmolLM2-135M are next.

## Platform contract

The shared core calls only these platform entry points:

- console: init, putc, clear, color, cursor, page up/down;
- input: poll, read, cancel pending;
- time: ticks, halt;
- boot: module count/start/size/end, memory end;
- reset: reboot, poweroff.

`kernel/arch/bios32/platform.c` maps them to the existing VGA, PS/2, PIT,
Multiboot, and ACPI implementations.

## Planned hybrid ISO

```text
/boot/lepton32.elf          BIOS32 kernel
/EFI/BOOT/BOOTX64.EFI      UEFI64 kernel
/models/...gguf             shared model
```

The model and inference code are shared; the boot binaries are not.
