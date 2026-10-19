/* ==========================================================================
   Lepton OS — 32-bit x86 Multiboot entry point.

   GRUB loads the kernel ELF at 1 MiB, checks the Multiboot 1 header
   below, and jumps to _start with:
     %eax = 0x2BADB002     (multiboot magic — proves GRUB did this)
     %ebx = pointer to multiboot_info_t  (memory map, modules, etc.)
   ========================================================================== */

/* Multiboot 1 header. Must appear in the first 8 KiB of the file.
   FLAGS bit 0: align modules on page boundaries.
   FLAGS bit 1: provide memory info (basic mem_lower/mem_upper).
   The checksum makes (magic + flags + checksum) == 0. */
.set MB_MAGIC,    0x1BADB002
.set MB_FLAGS,    0x00000003
.set MB_CHECKSUM, -(MB_MAGIC + MB_FLAGS)

.section .multiboot, "a"
.align 4
.long MB_MAGIC
.long MB_FLAGS
.long MB_CHECKSUM

/* ==========================================================================
   .bss — stack and multiboot globals.
   ========================================================================== */
.section .bss
.align 16
stack_bottom:
    .skip 262144                      /* 256 KiB kernel stack */
stack_top:

/* GRUB's multiboot values, stashed here so C or Lepton code can read
   them later. Both are zero-initialized by the multiboot loader. To
   access from C:  extern uint32_t mb_magic, mb_info; */
.global mb_magic
.global mb_info
.align 4
mb_magic:
    .long 0
mb_info:
    .long 0

/* ==========================================================================
   .text
   ========================================================================== */
.section .text
.global _start
.type   _start, @function
_start:
    /* ---- 1. Set up the stack ---- */
    movl $stack_top, %esp
    movl $stack_bottom, %ebp
    cld                               /* forward direction for string ops */

    /* ---- 2. Stash multiboot values ---- */
    /* Do this before anything that could clobber %eax or %ebx. */
    movl %eax, mb_magic
    movl %ebx, mb_info

    /* ---- 3. Clear CR0.TS ---- */
    /* If the Task Switched bit is set, every x87/SSE instruction raises
       #NM (Device Not Available). GRUB usually clears it, but not always;
       clts is the dedicated instruction for clearing it. */
    clts

    /* ---- 4. Configure CR0 for x87 ---- */
    /* Bit 1 (MP): monitor coprocessor — set it.
       Bit 2 (EM): emulation — MUST be clear, or x87 instructions trap.
       Bit 5 (NE): numeric error — set it to get #MF on x87 errors
                   instead of the legacy IRQ13 path (which we lack). */
    movl %cr0, %eax
    andl $~(1 << 2), %eax             /* clear EM */
    orl  $((1 << 1) | (1 << 5)), %eax /* set MP, NE */
    movl %eax, %cr0

    /* ---- 5. Configure CR4 for SSE ---- */
    /* Bit 9  (OSFXSR):      enable FXSAVE/FXRSTOR and SSE instructions.
       Bit 10 (OSXMMEXCPT):  route unmasked SSE exceptions through #XF
                             rather than #UD. Without it, an unmasked
                             SSE exception looks like an invalid opcode. */
    movl %cr4, %eax
    orl  $((1 << 9) | (1 << 10)), %eax
    movl %eax, %cr4

    /* ---- 6. Initialize the x87 FPU ---- */
    /* fninit resets the FPU: empty stack, round-to-nearest, all
       exceptions masked. Must come after CR0.EM is cleared, or fninit
       itself traps. */
    fninit

    /* ---- 7. Call the Lepton entry point ---- */
    /* x87 and SSE are now live. Lepton code can use f32, f64, f16,
       bf16, q7/q15/q31, and vector types with real hardware. */
    call main

    /* ---- 8. If main returns, halt forever ---- */
1:  cli
    hlt
    jmp 1b

.size _start, . - _start
