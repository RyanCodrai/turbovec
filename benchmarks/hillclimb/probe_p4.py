#!/usr/bin/env python3
"""P4 probe: what does the device actually do for a 77 MB durable write?

Reproduces the save path's shape without any turbovec code — temp file,
payload, fsync, atomic rename, parent-directory fsync — so the measured
number is the floor `write()` is working against on this filesystem.

Variants:
  serial    one write(2) of the whole payload, then fsync
  parallel  the payload split across N threads with pwrite, then fsync
            (what write_atomic_parallel does)
  nofsync   the parallel write with the fsync removed — isolates how much
            of the time is the device commit rather than the page-cache fill

Usage: python3 probe_p4.py [mb] [reps] [threads]
"""

import os
import statistics
import sys
import tempfile
import threading
import time


def med(xs):
    return statistics.median(xs) * 1e3


def run(payload, tmpdir, mode, nthreads):
    tmp = os.path.join(tmpdir, "probe.tmp")
    dst = os.path.join(tmpdir, "probe.out")
    t0 = time.perf_counter()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        if mode == "serial":
            os.write(fd, payload)
        else:
            os.ftruncate(fd, len(payload))
            chunk = -(-len(payload) // nthreads)

            def worker(i):
                off = i * chunk
                view = payload[off:off + chunk]
                while view:
                    n = os.pwrite(fd, view, off)
                    view = view[n:]
                    off += n

            ts = [threading.Thread(target=worker, args=(i,))
                  for i in range(nthreads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        if mode != "nofsync":
            os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp, dst)
    if mode != "nofsync":
        dfd = os.open(tmpdir, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    dt = time.perf_counter() - t0
    os.unlink(dst)
    return dt


def main():
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 77
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    nthreads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    payload = os.urandom(mb * 1024 * 1024)
    with tempfile.TemporaryDirectory() as d:
        print(f"dir={d} payload={mb} MB reps={reps} threads={nthreads}")
        for mode in ("serial", "parallel", "nofsync"):
            ts = []
            for _ in range(reps):
                time.sleep(0.15)  # drain the device queue, as the bench does
                ts.append(run(payload, d, mode, nthreads))
            ms = med(ts)
            print(f"{mode:9s} {ms:8.1f} ms   {mb / (ms / 1e3):7.1f} MB/s")


if __name__ == "__main__":
    main()
