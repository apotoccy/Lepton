/* Multiboot header (must be in the first 8 KiB) */
.set MB_MAGIC,    0x1BADB002
.set MB_FLAGS,    0x00000003          /* align modules + memory info */
.set MB_CHECKSUM, -(MB_MAGIC + MB_FLAGS)

.section .multiboot, "a"
.align 4
.long MB_MAGIC
.long MB_FLAGS
.long MB_CHECKSUM

.section .bss
.align 16
stack_bottom:
    .skip 16384                       /* 16 KiB kernel stack */
stack_top:

.section .text
.global _start
.type   _start, @function
_start:
    movl $stack_top, %esp
    movl $stack_bottom, %ebp
    cld

    /* Multiboot leaves magic in %eax and info ptr in %ebx.
       For now we just discard them and call the Lepton entry. */
    call main

    /* If main ever returns, halt forever. */
1:  cli
    hlt
    jmp 1b

.size _start, . - _start
