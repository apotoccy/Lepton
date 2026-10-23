/* Minimal 64-bit exception IDT for native UEFI mode. */
.text
.global isr_default
.type isr_default,@function
isr_default:
    cli
    call native_exception
1:  hlt
    jmp 1b
.size isr_default, .-isr_default
