#!/usr/bin/env python3
"""
Is the kernel's __nanosleep padding load-bearing for the fingerprint, or just
removable runtime an attacker can reclaim as relay slack?

Builds the original kernel and a variant with the reset-block __nanosleep calls
removed (grid.sync kept), runs the same seed on both, and reports:
  - runtime delta (ms saved by stripping the sleeps)
  - fingerprint L1 between original and no-sleep, vs the same-seed noise floor
If stripped fingerprint ~ noise floor: sleeps are pure padding -> free relay slack.
If stripped fingerprint >> noise floor: sleeps shape the signal (less exploitable).

Usage:  python3 strip_test.py /path/to/atomic_race_probe.cu
"""
import subprocess, sys, os, tempfile, time
import numpy as np

SEED = 12345; RUNS = 8

def make_nosleep(src):
    out = []
    for line in src.splitlines(keepends=True):
        if "__nanosleep(" in line:          # drop the sleep statements
            continue
        if "for (int i = 0; i < 2; i++)" in line:   # drop their (now-empty) loop header
            continue
        out.append(line)
    return "".join(out)

def build(cu, d, tag):
    binary = os.path.join(d, tag)
    subprocess.run(["nvcc", "-O0", "-arch=sm_90", cu, "-o", binary], check=True)
    return binary

def run(binary):
    out = subprocess.run([binary, str(SEED)], capture_output=True, text=True, check=True).stdout
    rows = [[int(x) for x in ln.split(",")] for ln in out.splitlines() if ln and not ln.startswith("#")]
    return np.array(rows, dtype=float)

def timed(binary):
    t0 = time.perf_counter(); run(binary); return (time.perf_counter()-t0)*1000.0

def main(cu):
    d = tempfile.mkdtemp()
    orig_src = open(cu).read()
    nos_cu = os.path.join(d, "nosleep.cu"); open(nos_cu, "w").write(make_nosleep(orig_src))
    print("compiling original + no-sleep variant...")
    b_orig = build(cu, d, "orig"); b_nos = build(nos_cu, d, "nos")

    for b in (b_orig, b_nos): run(b)  # warm up
    t_orig = np.median([timed(b_orig) for _ in range(RUNS)])
    t_nos  = np.median([timed(b_nos)  for _ in range(RUNS)])

    L1 = lambda a, b: np.abs(a - b).sum()
    o1, o2 = run(b_orig), run(b_orig)
    n1      = run(b_nos)
    floor   = L1(o1, o2)                 # same binary, same seed, run-to-run noise
    cross   = L1(o1, n1)                 # original vs no-sleep, same seed

    print(f"\nruntime  original {t_orig:7.1f} ms   no-sleep {t_nos:7.1f} ms   saved {t_orig-t_nos:7.1f} ms")
    print(f"fingerprint L1  noise floor (orig vs orig) {floor:10.0f}")
    print(f"fingerprint L1  original vs no-sleep       {cross:10.0f}")
    ratio = cross / max(floor, 1)
    print(f"  ratio {ratio:.1f}x noise  ->", 
          "sleeps are REMOVABLE PADDING (fingerprint survives; attacker reclaims the saved ms)" if ratio < 3
          else "sleeps SHAPE the fingerprint (stripping changes the answer)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "atomic_race_probe.cu")
