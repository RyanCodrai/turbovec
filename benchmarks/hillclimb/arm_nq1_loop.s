# turbovec arm nq=1 inner loop: one 16-byte code register, 4 per group.
# LLVM-MCA-BEGIN loop
ldr  q4, [x1]
and  v5.16b, v4.16b, v8.16b
ushr v6.16b, v4.16b, #4
tbl  v5.16b, {v9.16b}, v5.16b
tbl  v6.16b, {v9.16b}, v6.16b
smmla v0.4s, v10.16b, v6.16b
smmla v0.4s, v11.16b, v5.16b
# LLVM-MCA-END
