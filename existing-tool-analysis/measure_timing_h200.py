#!/usr/bin/env python3
"""
Measure the fingerprint's runtime and run-to-run variance on a real GPU.

Why it matters: the location protocol infers position from response latency,
so any slack in the runtime is slack an attacker can hide a relay in. Rule of
thumb in fiber: ~100 km of separation between decoy and hidden real chip per
1 ms of timing slack (one-way; a relay pays the round trip).

Caveat: this times the whole process invocation, so it includes per-call CUDA
context init. That is what a verifier sees if it launches the probe fresh each
challenge; if the verifier keeps a warm context, the relevant number is smaller
and would need a cudaEvent measurement inside the kernel. Treat the SPREAD, not
the absolute mean, as the headline.

Usage:
    python3 measure_timing_h200.py /path/to/atomic_race_probe.cu
"""
import subprocess, sys, os, tempfile, time
import numpy as np

N = 30
SEED = 12345

def main(cu):
    binary = os.path.join(tempfile.mkdtemp(), "probe")
    print("compiling...")
    subprocess.run(["nvcc", "-O0", "-arch=sm_90", cu, "-o", binary], check=True)
    subprocess.run([binary, str(SEED)], capture_output=True, check=True)  # warm-up
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        subprocess.run([binary, str(SEED)], capture_output=True, check=True)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    rng = ts.max() - ts.min()
    spread = np.percentile(ts, 95) - np.percentile(ts, 5)
    print(f"\nruntime over {N} runs (seed {SEED}):")
    print(f"  mean    {ts.mean():8.1f} ms")
    print(f"  std     {ts.std():8.1f} ms")
    print(f"  min     {ts.min():8.1f} ms")
    print(f"  max     {ts.max():8.1f} ms")
    print(f"  max-min {rng:8.1f} ms   ~ {rng*100:6.0f} km relay slack")
    print(f"  p95-p5  {spread:8.1f} ms   ~ {spread*100:6.0f} km relay slack")
    print("\nInterpretation: the larger the spread, the more room a relayed response")
    print("has to hide inside normal timing variation. Compare against the location")
    print("protocol's claimed timing resolution.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "atomic_race_probe.cu")
