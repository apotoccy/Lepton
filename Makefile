# =============================================================================
#  Lepton build system
# =============================================================================

PYTHON   ?= python3
LEPTONC  ?= $(PYTHON) leptonc.py
CC       ?= cc
OPT      ?= 2

SRC_DIR  ?= src
BUILD    ?= build

SRC      ?= $(wildcard $(SRC_DIR)/*.lp)
NAMES    := $(basename $(notdir $(SRC)))

OUT64    := $(BUILD)/x86_64
OUT32    := $(BUILD)/x86_32
BIN64    := $(addprefix $(OUT64)/,$(NAMES))
BIN32    := $(addprefix $(OUT32)/,$(NAMES))

# ---------------------------------------------------------------------------
#  Kernel
# ---------------------------------------------------------------------------
KERNEL_DIR    ?= kernel
KERNEL_OUT    := $(BUILD)/kernel

KERNEL_LP     := $(KERNEL_DIR)/src/kernel.lp
KERNEL_RT     := $(KERNEL_DIR)/src/runtime.c
KERNEL_BOOT   := $(KERNEL_DIR)/arch/boot.S
KERNEL_LD     := $(KERNEL_DIR)/arch/kernel.ld
KERNEL_PLATFORM := $(KERNEL_DIR)/arch/bios32/platform.c
KERNEL64_OUT  := $(BUILD)/kernel64
UEFI64_OUT    := $(BUILD)/uefi64
UEFI64_EFI    := $(UEFI64_OUT)/hello.efi
UEFI64_OBJ    := $(UEFI64_OUT)/hello.o
UEFI64_ESP    := $(UEFI64_OUT)/esp.img
UEFI64_DISK   := $(UEFI64_OUT)/disk.img
QEMU64        ?= qemu-system-x86_64
OVMF          ?= /usr/share/ovmf/OVMF.fd

KERNEL_CFLAGS := -m32 -msse2 -mfpmath=sse -mstackrealign -std=c11 -O$(OPT) \
                 -ffreestanding -fno-stack-protector -fno-pic -fno-builtin \
                 -fno-asynchronous-unwind-tables -Wall -Wextra -Wno-unused-parameter

KERNEL_LDFLAGS := -m32 -no-pie -nostdlib -nostartfiles -nodefaultlibs \
                  -Wl,-T,$(KERNEL_LD) -Wl,--build-id=none                  

KERNEL_LIBS   := -lgcc

KERNEL_C      := $(KERNEL_OUT)/kernel.c
KERNEL_ELF    := $(KERNEL_OUT)/kernel.elf
KERNEL_BIN    := $(KERNEL_OUT)/kernel.bin
KERNEL_ISO    := $(KERNEL_OUT)/lepton.iso

QEMU          ?= qemu-system-i386

USB ?= /dev/sdX    # override with: make flash USB=/dev/sdb

# Probe 32-bit toolchain availability once, at parse time.
HAVE_M32 := $(shell printf 'int main(void){return 0;}\n' > /tmp/.lp_m32.c; \
                   $(CC) -m32 /tmp/.lp_m32.c -o /tmp/.lp_m32 2>/dev/null \
                   && echo yes || echo no; \
                   rm -f /tmp/.lp_m32.c /tmp/.lp_m32)

.PHONY: all 64 32 asm run run32 \
        kernel kernel32 kernel64-check uefi64 uefi64-esp run-uefi64 \
        kernel-c kernel-bin iso run-kernel \
        clean clean-kernel check-m32 help

all: 64 32

64: $(BIN64)

32: check-m32 $(BIN32)

# ---------------------------------------------------------------------------
#  Userland rules
# ---------------------------------------------------------------------------
$(OUT64)/%: $(SRC_DIR)/%.lp leptonc.py
	@mkdir -p $(OUT64)
	$(LEPTONC) $< -o $@ -O$(OPT) --cc $(CC)

$(OUT32)/%: $(SRC_DIR)/%.lp leptonc.py
	@mkdir -p $(OUT32)
	$(LEPTONC) $< -o $@ -O$(OPT) --m32 --cc $(CC)

asm: 64
	@for f in $(SRC); do \
		out=$(OUT64)/$$(basename $$f .lp).c; \
		echo "  C     $$out"; \
		$(LEPTONC) $$f -S -o $$out || exit 1; \
	done

run: 64
	@for b in $(BIN64); do echo "==> $$b"; $$b || exit 1; done

run32: check-m32 32
	@for b in $(BIN32); do echo "==> $$b"; $$b || exit 1; done

# ---------------------------------------------------------------------------
#  Kernel rules
# ---------------------------------------------------------------------------
kernel: kernel32

kernel32: check-m32 $(KERNEL_ELF)
	@echo "kernel32 ready: $(KERNEL_ELF)"

# Compile the shared kernel core as 64-bit C. This is a build check for the
# UEFI port, not yet a bootable kernel.
kernel64-check:
	@mkdir -p $(KERNEL64_OUT)
	$(LEPTONC) $(KERNEL_LP) --freestanding -S -o $(KERNEL64_OUT)/kernel.c
	$(CC) -m64 -ffreestanding -fno-stack-protector -fno-pic -fno-builtin \
	    -fno-asynchronous-unwind-tables -Wall -Wextra -Wno-unused-parameter \
	    -c $(KERNEL64_OUT)/kernel.c -o $(KERNEL64_OUT)/kernel.o
	@echo "kernel64 compile check: $(KERNEL64_OUT)/kernel.o"

# Minimal UEFI64 application; not yet the Lepton kernel.
$(UEFI64_OBJ): kernel/arch/uefi64/hello.c
	@mkdir -p $(UEFI64_OUT)
	$(CC) -m64 -std=c11 -ffreestanding -fno-stack-protector -fno-pic \
	    -mno-red-zone -fno-asynchronous-unwind-tables -fno-builtin \
	    -fno-ident -Wall -Wextra -Wno-unused-parameter -c $< -o $@

$(UEFI64_EFI): $(UEFI64_OBJ)
	ld -m i386pep -nostdlib --subsystem 10 --entry efi_main \
	    --image-base 0x10000000 -o $@ $<
	objcopy --remove-section .comment --remove-section .note.gnu.property $@
	@echo "uefi64: $@"

uefi64: $(UEFI64_EFI)

$(UEFI64_ESP): $(UEFI64_EFI)
	@mkdir -p $(UEFI64_OUT)/esp/EFI/BOOT
	cp $(UEFI64_EFI) $(UEFI64_OUT)/esp/EFI/BOOT/BOOTX64.EFI
	rm -f $@
	dd if=/dev/zero of=$@ bs=1M count=32 status=none
	mkfs.fat -F16 $@ >/dev/null
	mcopy -i $@ -s $(UEFI64_OUT)/esp/EFI ::/
	@echo "uefi64 esp: $@"

uefi64-esp: $(UEFI64_ESP)

$(UEFI64_DISK): $(UEFI64_ESP)
	@truncate -s 64M $@
	parted -s $@ mklabel msdos mkpart primary fat16 1MiB 63MiB set 1 boot on >/dev/null
	sfdisk --part-type $@ 1 ef >/dev/null
	dd if=$(UEFI64_ESP) of=$@ bs=1M seek=1 conv=notrunc status=none
	@echo "uefi64 disk: $@"

run-uefi64: $(UEFI64_DISK)
	$(QEMU64) -bios $(OVMF) -drive format=raw,file=$(UEFI64_DISK) \
	    -serial stdio -display none -no-reboot -m 256

kernel-c: check-m32 $(KERNEL_C)

kernel-bin: check-m32 $(KERNEL_BIN)

# Lepton source -> freestanding C
$(KERNEL_C): $(KERNEL_LP) leptonc.py
	@mkdir -p $(KERNEL_OUT)
	$(LEPTONC) $< --freestanding --m32 -S -o $@

$(KERNEL_OUT)/kernel.o: $(KERNEL_C)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(KERNEL_OUT)/runtime.o: $(KERNEL_RT)
	@mkdir -p $(KERNEL_OUT)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(KERNEL_OUT)/interrupts.o: kernel/arch/interrupts.c
	@mkdir -p $(KERNEL_OUT)
	$(CC) $(KERNEL_CFLAGS) -mno-sse -mno-sse2 -mfpmath=387 -c $< -o $@

$(KERNEL_OUT)/vga.o: kernel/arch/vga.c
	@mkdir -p $(KERNEL_OUT)
	$(CC) $(KERNEL_CFLAGS) -mno-sse -mno-sse2 -mfpmath=387 -c $< -o $@


$(KERNEL_OUT)/platform.o: $(KERNEL_PLATFORM) $(KERNEL_DIR)/platform.h
	@mkdir -p $(KERNEL_OUT)
	$(CC) $(KERNEL_CFLAGS) -mno-sse -mno-sse2 -mfpmath=387 -c $< -o $@

$(KERNEL_OUT)/multiboot.o: kernel/arch/multiboot.c
	@mkdir -p $(KERNEL_OUT)
	$(CC) $(KERNEL_CFLAGS) -mfpmath=387 -c $< -o $@

$(KERNEL_OUT)/isr.o: kernel/arch/isr.S
	@mkdir -p $(KERNEL_OUT)
	$(CC) -m32 -c -fno-pic -fno-pie $< -o $@
	
$(KERNEL_OUT)/boot.o: $(KERNEL_BOOT)
	@mkdir -p $(KERNEL_OUT)
	$(CC) -m32 -c -fno-pic -fno-pie $< -o $@	

MODEL ?= $(KERNEL_OUT)/test.gguf

$(KERNEL_OUT)/test.gguf: tools/gen_test_gguf.py
	@mkdir -p $(KERNEL_OUT)
	$(PYTHON) tools/gen_test_gguf.py $@

KERNEL_TEST_MODULE := $(MODEL)

$(KERNEL_ELF): $(KERNEL_OUT)/boot.o \
               $(KERNEL_OUT)/isr.o \
               $(KERNEL_OUT)/kernel.o \
               $(KERNEL_OUT)/runtime.o \
               $(KERNEL_OUT)/interrupts.o \
               $(KERNEL_OUT)/multiboot.o \
               $(KERNEL_OUT)/vga.o \
               $(KERNEL_OUT)/platform.o \
               $(KERNEL_LD)
	$(CC) $(KERNEL_LDFLAGS) -o $@ \
	    $(KERNEL_OUT)/boot.o $(KERNEL_OUT)/isr.o \
	    $(KERNEL_OUT)/kernel.o $(KERNEL_OUT)/runtime.o \
	    $(KERNEL_OUT)/interrupts.o $(KERNEL_OUT)/multiboot.o \
	    $(KERNEL_OUT)/vga.o $(KERNEL_OUT)/platform.o \
	    $(KERNEL_LIBS)
	@echo "kernel built: $@"

# Flat binary, useful for direct QEMU -kernel loading and for dd-ing to a disk.
$(KERNEL_BIN): $(KERNEL_ELF)
	objcopy -O binary $< $@
	@echo "flat binary: $@"

# Bootable ISO via GRUB. Needs: grub-pc-bin grub-common xorriso mtools
$(KERNEL_ISO): $(KERNEL_ELF) $(MODEL)
	@mkdir -p $(KERNEL_OUT)/iso/boot/grub
	cp $(KERNEL_ELF) $(KERNEL_OUT)/iso/boot/kernel.elf
	cp $(MODEL) $(KERNEL_OUT)/iso/boot/model.gguf
	printf 'set timeout=0\nset default=0\nmenuentry "Lepton OS" {\n' \
	    > $(KERNEL_OUT)/iso/boot/grub/grub.cfg
	printf '  multiboot /boot/kernel.elf\n' \
	    >> $(KERNEL_OUT)/iso/boot/grub/grub.cfg
	printf '  module /boot/model.gguf\n  boot\n}\n' \
	    >> $(KERNEL_OUT)/iso/boot/grub/grub.cfg
	grub-mkrescue -o $@ $(KERNEL_OUT)/iso
	@echo "iso: $@"

iso: check-m32 $(KERNEL_ISO)

# Ctrl-A (release), then X — quits QEMU
# Ctrl-A (release), then C — switches to the QEMU monitor
# Ctrl-A (release), then H — prints the full mux help

run-kernel: check-m32 $(KERNEL_ELF) $(KERNEL_TEST_MODULE)
	$(QEMU) -kernel $(KERNEL_ELF) -initrd $(KERNEL_TEST_MODULE) \
	    -serial mon:stdio -display none -no-reboot

run-iso: check-m32 $(KERNEL_ISO)
	$(QEMU) -cdrom $(KERNEL_ISO) -serial mon:stdio -display none -no-reboot \
	    -boot d

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
.PHONY: test test-v test-verifier test-conversions

test:
	$(PYTHON) tests/run_tests.py
	$(PYTHON) tests/test_ir_verify.py
	$(PYTHON) tests/test_conversions.py

test-verifier:
	$(PYTHON) tests/test_ir_verify.py

test-conversions:
	$(PYTHON) tests/test_conversions.py

test-v:
	$(PYTHON) tests/run_tests.py -v
	$(PYTHON) tests/test_ir_verify.py
	$(PYTHON) tests/test_conversions.py
	
.PHONY: test-sse

test-sse:
	LEPTON_TARGET=x86_64 $(PYTHON) tests/run_tests.py	
	
.PHONY: test-all

test-all: test test-sse



# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
.PHONY: dump-ir

dump-ir:
	@test -n "$(SRC)" || { echo "usage: make dump-ir SRC=tests/cases/foo.lp"; exit 1; }
	$(PYTHON) leptonc.py $(SRC) --dump-ir

.PHONY: lint-tabs
lint-tabs:
	@for f in leptonc.py lepton_types.py lepton_ir.py lepton_lower.py lepton_c_ir.py; do \
	    if grep -qP '^\t| \t' "$$f"; then echo "TABS in $$f"; fi; \
	done
	@echo "tab lint done"


check-m32:
ifeq ($(HAVE_M32),no)
	@echo "note: 32-bit toolchain unavailable."; \
	echo "      install with: sudo apt install gcc-multilib"; \
	exit 1
endif

clean:
	rm -rf $(BUILD)

clean-kernel:
	rm -rf $(KERNEL_OUT)

help:
	@echo "Lepton build system"
	@echo ""
	@echo "  Userland:"
	@echo "    make               build 64-bit binaries"
	@echo "    make all           build 64-bit and 32-bit"
	@echo "    make asm           also emit generated C"
	@echo "    make run / run32   build and run binaries"
	@echo ""
	@echo "  Kernel:"
	@echo "    make kernel        build build/kernel/kernel.elf"
	@echo "    make kernel32      explicit BIOS32 kernel target"
	@echo "    make kernel64-check compile the shared core as 64-bit C"
	@echo "    make uefi64        build the minimal UEFI64 app"
	@echo "    make run-uefi64    boot it with OVMF"
	@echo "    make kernel-c      emit only the Lepton-generated C"
	@echo "    make kernel-bin    also produce a flat .bin"
	@echo "    make iso           build a GRUB-bootable lepton.iso"
	@echo "    make run-kernel    boot the ELF in QEMU (serial on stdout)"
	@echo "    make run-iso       boot the ISO in QEMU"
	@echo ""
	@echo "  Variables: CC=$(CC) OPT=$(OPT) QEMU=$(QEMU)"
	@echo "  Detected:  HAVE_M32=$(HAVE_M32)"
	
	
# ---------------------------------------------------------------------------
#  Write an ISO to USB stick. See above -> USB ?= /dev/sdX    # override with: make flash USB=/dev/sdb
# ---------------------------------------------------------------------------

.PHONY: flash
flash: check-m32 $(KERNEL_ISO)
	@test "$(USB)" != "/dev/sdX" || { \
	    echo "set USB first: make flash USB=/dev/sdX"; exit 1; }
	@echo "About to overwrite $(USB). Ctrl-C within 10s to abort."
	# 10 sec delay
	@sleep 10
	sudo dd if=$(KERNEL_ISO) of=$(USB) bs=4M status=progress oflag=sync
	sync
	@echo "done: $(KERNEL_ISO) -> $(USB)"
	

