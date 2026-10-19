/* Interrupt service routine stubs and GDT loading.
 *
 * Layout of the stack at isr_common:
 *   [esp+0]  pointer to saved register frame (arg to irq_handler)
 *   [esp+4]  pusha: 8 registers (32 bytes)
 *   [esp+36] interrupt number    (pushed by stub)
 *   [esp+40] error code or 0     (pushed by stub or CPU)
 *   [esp+44] eip
 *   [esp+48] cs
 *   [esp+52] eflags
 */

.section .text

/* ---- GDT loading ---- */
.global gdt_load
.type gdt_load, @function
gdt_load:
    movl 4(%esp), %eax
    lgdt (%eax)
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %gs
    movw %ax, %ss
    ljmp $0x08, $1f
1:  ret

/* ---- Stub macros ---- */
.macro ISR_NOERR num
.global isr\num
isr\num:
    cli
    pushl $0
    pushl $\num
    jmp isr_common
.endm

.macro ISR_ERR num
.global isr\num
isr\num:
    cli
    pushl $\num
    jmp isr_common
.endm

/* ---- CPU exceptions (vectors 0..31) ---- */
ISR_NOERR  0
ISR_NOERR  1
ISR_NOERR  2
ISR_NOERR  3
ISR_NOERR  4
ISR_NOERR  5
ISR_NOERR  6
ISR_NOERR  7
ISR_ERR    8
ISR_NOERR  9
ISR_ERR   10
ISR_ERR   11
ISR_ERR   12
ISR_ERR   13
ISR_ERR   14
ISR_NOERR 15
ISR_NOERR 16
ISR_ERR   17
ISR_NOERR 18
ISR_NOERR 19
ISR_NOERR 20
ISR_NOERR 21
ISR_NOERR 22
ISR_NOERR 23
ISR_NOERR 24
ISR_NOERR 25
ISR_NOERR 26
ISR_NOERR 27
ISR_NOERR 28
ISR_NOERR 29
ISR_ERR   30
ISR_NOERR 31

/* ---- Hardware IRQs (PIC-remapped to 0x20..0x2F) ---- */
ISR_NOERR 32
ISR_NOERR 33
ISR_NOERR 34
ISR_NOERR 35
ISR_NOERR 36
ISR_NOERR 37
ISR_NOERR 38
ISR_NOERR 39
ISR_NOERR 40
ISR_NOERR 41
ISR_NOERR 42
ISR_NOERR 43
ISR_NOERR 44
ISR_NOERR 45
ISR_NOERR 46
ISR_NOERR 47

/* ---- Common tail ---- */
isr_common:
    pusha
    movl %esp, %eax
    pushl %eax
    call irq_handler
    addl $4, %esp
    popa
    addl $8, %esp
    iret
