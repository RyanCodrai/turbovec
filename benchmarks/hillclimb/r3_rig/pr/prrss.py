"""Steady-state resident bytes per vector of a built, searched 2-bit index (fresh process, heap trimmed)."""
import os, gc, ctypes, numpy as np
from turbovec import TurboQuantIndex
libc = ctypes.CDLL("libc.so.6")
def rss(): gc.collect(); libc.malloc_trim(0); return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
n, dim = 400_000, 1536
rng = np.random.default_rng(0); q = rng.standard_normal((100, dim), dtype=np.float32)
r0 = rss()
ix = TurboQuantIndex(dim, bit_width=2)
for _ in range(8):
    x = rng.standard_normal((n // 8, dim), dtype=np.float32); ix.add(x); del x
ix.search(q, k=10); ix.search(q[:1], k=10)
r1 = rss(); ix.add(q); ix.search(q, k=10); r2 = rss()
print(f"planes={os.environ.get('TURBOVEC_2BIT_PLANES','0')} bytes_per_vector after build+search={(r1 - r0) / n:.1f} after a further add+search={(r2 - r0) / (n + 100):.1f} (codes {dim // 4} + scale 4)")
