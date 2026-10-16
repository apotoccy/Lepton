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

KERNEL_CFLAGS := -m32 -std=c11 -O$(OPT) \
                 -ffreestanding -fno-stack-protector -fno-pic \
                 -fno-builtin -fno-asynchronous-unwind-tables \
                 -Wall -Wextra -Wno-unused-parameter

KERNEL_LDFLAGS := -m32 -nostdlib -nostartfiles -nodefaultlibs \
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
        kernel kernel-c kernel-bin iso run-kernel \
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
kernel: check-m32 $(KERNEL_ELF)

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

$(KERNEL_OUT)/boot.o: $(KERNEL_BOOT)
	@mkdir -p $(KERNEL_OUT)
	$(CC) -m32 -c $< -o $@

$(KERNEL_ELF): $(KERNEL_OUT)/boot.o \
               $(KERNEL_OUT)/kernel.o \
               $(KERNEL_OUT)/runtime.o \
               $(KERNEL_LD)
	$(CC) $(KERNEL_LDFLAGS) -o $@ \
	    $(KERNEL_OUT)/boot.o $(KERNEL_OUT)/kernel.o $(KERNEL_OUT)/runtime.o \
	    $(KERNEL_LIBS)
	@echo "kernel built: $@"

# Flat binary, useful for direct QEMU -kernel loading and for dd-ing to a disk.
$(KERNEL_BIN): $(KERNEL_ELF)
	objcopy -O binary $< $@
	@echo "flat binary: $@"

# Bootable ISO via GRUB. Needs: grub-pc-bin grub-common xorriso mtools
$(KERNEL_ISO): $(KERNEL_ELF)
	@mkdir -p $(KERNEL_OUT)/iso/boot/grub
	cp $(KERNEL_ELF) $(KERNEL_OUT)/iso/boot/kernel.elf
	printf 'set timeout=0\nset default=0\nmenuentry "Lepton OS" {\n' \
	    > $(KERNEL_OUT)/iso/boot/grub/grub.cfg
	printf '  multiboot /boot/kernel.elf\n  boot\n}\n' \
	    >> $(KERNEL_OUT)/iso/boot/grub/grub.cfg
	grub-mkrescue -o $@ $(KERNEL_OUT)/iso >/dev/null 2>&1
	@echo "iso: $@"

iso: check-m32 $(KERNEL_ISO)

run-kernel: check-m32 $(KERNEL_ELF)
	$(QEMU) -kernel $(KERNEL_ELF) -serial stdio -display none -no-reboot

run-iso: check-m32 $(KERNEL_ISO)
	$(QEMU) -cdrom $(KERNEL_ISO) -serial stdio -no-reboot

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
.PHONY: test test-v

test:
	$(PYTHON) tests/run_tests.py

test-v:
	$(PYTHON) tests/run_tests.py -v
	
.PHONY: test-sse

test-sse:
	LEPTON_TARGET=x86_64 $(PYTHON) tests/run_tests.py	
	
.PHONY: test-all

test-all: test test-sse



# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
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
	

