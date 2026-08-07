# turbovec x86 permute-dot inner loop: one 64-byte code register, NQ=8.
# LLVM-MCA-BEGIN loop
vmovdqu64 (%rdi), %zmm4
vpandd    %zmm4, %zmm8, %zmm5
vgf2p8affineqb $0, %zmm9, %zmm4, %zmm6
vpshufb   %zmm5, %zmm10, %zmm5
vpshufb   %zmm6, %zmm10, %zmm6
vpdpbusd  (%rsi){1to16}, %zmm5, %zmm0
vpdpbusd  4(%rsi){1to16}, %zmm6, %zmm0
vpdpbusd  8(%rsi){1to16}, %zmm5, %zmm1
vpdpbusd  12(%rsi){1to16}, %zmm6, %zmm1
vpdpbusd  16(%rsi){1to16}, %zmm5, %zmm2
vpdpbusd  20(%rsi){1to16}, %zmm6, %zmm2
vpdpbusd  24(%rsi){1to16}, %zmm5, %zmm3
vpdpbusd  28(%rsi){1to16}, %zmm6, %zmm3
vpdpbusd  32(%rsi){1to16}, %zmm5, %zmm11
vpdpbusd  36(%rsi){1to16}, %zmm6, %zmm11
vpdpbusd  40(%rsi){1to16}, %zmm5, %zmm12
vpdpbusd  44(%rsi){1to16}, %zmm6, %zmm12
vpdpbusd  48(%rsi){1to16}, %zmm5, %zmm13
vpdpbusd  52(%rsi){1to16}, %zmm6, %zmm13
vpdpbusd  56(%rsi){1to16}, %zmm5, %zmm14
vpdpbusd  60(%rsi){1to16}, %zmm6, %zmm14
# LLVM-MCA-END
