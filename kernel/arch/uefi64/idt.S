/* Minimal exception IDT + PIT IRQ0 for native UEFI mode. */
.text
.global isr_default
.type isr_default,@function
isr_default:
    cli
    call native_exception
1:  hlt
    jmp 1b
.size isr_default, .-isr_default

.global irq0_stub
.type irq0_stub,@function
irq0_stub:
    push %rax
    incq native_tick(%rip)
    movb $0x20, %al
    outb %al, $0x20
    pop %rax
    iretq
.size irq0_stub, .-irq0_stub
