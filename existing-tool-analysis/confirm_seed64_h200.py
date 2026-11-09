#!/usr/bin/env python3
"""
Direct confirmation of the `seed % 64` collapse. REQUIRES a real GPU (H200).
NOTE: not run by the author of this script (no GPU available at write time);
logic is straightforward but please sanity-check on first run.

Compiles the project's own kernel, then runs several seeds chosen so that some
are congruent mod 64 and some are not, and prints pairwise L1 distances.

Prediction if the collapse is real:
  - seeds that are EQUAL mod 64 (e.g. 0, 64, 128, 64000000) -> small L1,
    comparable to re-running the SAME seed (just run-to-run noise).
  - seeds that DIFFER mod 64 (e.g. 0 vs 1) -> large L1.

Usage:
    python3 confirm_seed64_h200.py /path/to/atomic_race_probe.cu
"""
import subprocess, sys, os, tempfile, itertools
import numpy as np

CONGRUENT   = [0, 64, 128, 64_000_000]   # all == 0 (mod 64)
NONCONGRU   = [1, 2, 7, 33]              # distinct mod 64
RUNS = 3                                  # repeats per seed to gauge noise floor

def run(binary, seed):
    out = subprocess.run([binary, str(seed)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        rows.append([int(x) for x in line.split(",")])
    return np.array(rows, dtype=float)   # (n_sms/2, 200)

def main(cu):
    binary = os.path.join(tempfile.mkdtemp(), "probe")
    print("compiling...")
    # H200 = Hopper = sm_90. (Their build uses -O0; keep it, since results may depend on it.)
    subprocess.run(["nvcc", "-O0", "-arch=sm_90", cu, "-o", binary], check=True)

    data = {}
    for s in CONGRUENT + NONCONGRU:
        data[s] = [run(binary, s) for _ in range(RUNS)]

    L1 = lambda a, b: np.abs(a - b).sum()
    # noise floor: same seed, different runs
    floor = np.mean([L1(data[s][i], data[s][j])
                     for s in data for i, j in itertools.combinations(range(RUNS), 2)])
    print(f"\nnoise floor (same seed, diff run):           {floor:10.0f}")

    cong = np.mean([L1(data[a][0], data[b][0])
                    for a, b in itertools.combinations(CONGRUENT, 2)])
    print(f"mean L1, seeds EQUAL mod 64:                  {cong:10.0f}   (expect ~noise floor)")

    nonc = np.mean([L1(data[a][0], data[b][0])
                    for a, b in itertools.combinations(NONCONGRU, 2)])
    print(f"mean L1, seeds DIFFERENT mod 64:              {nonc:10.0f}   (expect >> floor)")

    print("\nVERDICT:",
          "seed%64 COLLAPSE CONFIRMED" if cong < 3 * floor < nonc
          else "inconclusive - inspect values above")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "atomic_race_probe.cu")
