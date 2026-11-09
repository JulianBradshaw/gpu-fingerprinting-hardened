#!/usr/bin/env python3
"""
Replication of fingerprint analysis for waynetee/gpu-fingerprinting.
Runs entirely on the released data (no GPU needed).

Usage:
    python3 replicate.py /path/to/gpu-fingerprinting/data/atomic_race_probe

Reproduces:
  1. Distance separation (genuine / impostor / cross-seed)
  2. Genuine-chip identification ceiling under a single-run registry
  3. Decoy spoofing attack + residual-amplification sweep (closest-match exploit)
  4. Structural evidence that the seed enters only as seed % 64
"""
import json, glob, sys, os, itertools
import numpy as np

def load(path):
    d = json.load(open(path))
    out = []
    for fp in d["fingerprints"]:
        res = sorted(fp["results"], key=lambda x: x["sm"])
        X = np.array([x["results"] for x in res], dtype=float)  # (66, 200)
        out.append(dict(gpu=fp["gpu_uuid"], seed=int(fp["seed"]),
                        run=fp["run"], X=X, mach=d["machine_id"]))
    return out

def main(data_dir):
    main_files = sorted(glob.glob(os.path.join(data_dir, "2026*.json")))
    seeds_file = os.path.join(data_dir, "seeds", "16seeds.json")
    M = sum([load(f) for f in main_files], [])     # 3 machines x 8 gpu x 2 seed x 10 run
    S = load(seeds_file)                            # 8 gpu x 16 seed x 1 run
    L1 = lambda a, b: np.abs(a - b).sum()

    # ---- 1. distance separation (main data: same seed only) ----
    gen, imp = [], []
    for a, b in itertools.combinations(M, 2):
        if a["seed"] != b["seed"]:
            continue
        (gen if a["gpu"] == b["gpu"] else imp).append(L1(a["X"], b["X"]))
    gen, imp = np.array(gen), np.array(imp)
    thr95 = np.percentile(gen, 95)
    print("== 1. distance separation ==")
    print(f"  genuine  (same gpu/seed, diff run)  median {np.median(gen):>9.0f}  p95 {thr95:>9.0f}")
    print(f"  impostor (diff gpu, same seed)       median {np.median(imp):>9.0f}  p5  {np.percentile(imp,5):>9.0f}")

    sg, ds = [], []
    for a, b in itertools.combinations(S, 2):
        d = L1(a["X"], b["X"])
        if a["gpu"] == b["gpu"]:        sg.append(d)   # same gpu, diff seed
        elif a["seed"] == b["seed"]:    ds.append(d)   # diff gpu, same seed
    print(f"  same gpu, DIFFERENT seed             median {np.median(sg):>9.0f}  <- seed dominates")
    print(f"  diff gpu, same seed                  median {np.median(ds):>9.0f}")

    # ---- 2. genuine identification ceiling (single-run registry, 8 gpus) ----
    print("\n== 2. genuine-chip ID ceiling (single-run registry) ==")
    for f in main_files:
        m = load(f); gp = sorted(set(x["gpu"] for x in m)); ok = []
        for seed in sorted(set(x["seed"] for x in m)):
            R = {(x["gpu"], x["run"]): x["X"] for x in m if x["seed"] == seed}
            runs = sorted(set(x["run"] for x in m if x["seed"] == seed))
            for a in runs:
                for b in runs:
                    if a == b: continue
                    for g in gp:
                        d = [L1(R[(h, a)], R[(g, b)]) for h in gp]
                        ok.append(gp[int(np.argmin(d))] == g)
        print(f"  {os.path.basename(f):32s} {np.mean(ok):6.1%}")

    # ---- 3. decoy attack + amplification (uses 16-seed data) ----
    gpus = sorted(set(x["gpu"] for x in S)); G = len(gpus)
    A = np.zeros((G, 16, 66, 200))
    for x in S: A[gpus.index(x["gpu"]), x["seed"]] = x["X"]
    L1b = lambda a, b: np.abs(a - b).sum(axis=(-2, -1))

    def attack(lam):
        ok, dd = [], []
        for t in range(16):
            tr = [s for s in range(16) if s != t]
            for g in range(G):
                dec = [h for h in range(G) if h != g]
                # decoy supplies stream realization at seed t; correction transfers chip identity
                P = [A[h, t] + lam * np.median(A[g, tr] - A[h, tr], 0) for h in dec]
                p = np.median(P, 0)
                d = L1b(A[:, t], p)
                ok.append(np.argmin(d) == g); dd.append(d[g])
        return np.mean(ok), np.median(dd), np.mean(np.array(dd) <= thr95)

    print("\n== 3. decoy spoofing attack (chance = 1/8 = 12.5%) ==")
    print(f"  {'amplification':>14s}  {'match':>6s}  {'medL1':>9s}  {'<=genuine p95':>13s}")
    for lam in [1.0, 1.5, 2.0, 3.0]:
        m_, med, pz = attack(lam)
        print(f"  {('x%.1f'%lam):>14s}  {m_:6.1%}  {med:9.0f}  {pz:13.1%}")

    # ---- 4. seed enters only as seed % 64 ----
    print("\n== 4. structural: seed acts only through seed % 64 ==")
    def lanes(seed):
        out = np.zeros((66, 200), dtype=int)
        for k, sm in enumerate(range(0, 132, 2)):
            s = (seed ^ sm) & 0xffffffff; buf = []
            for _ in range(250):
                s = (s * 1103515245 + 12345) & 0x7fffffff; buf.append(s % 64)
            out[k] = buf[50:]
        return out
    r = np.arange(50, 250); base = (r - 50) * 66
    cons = A.mean(0) - base                                   # chip-averaged, base-removed
    # (a) LCG lane explains per-round motion; the %17 "delay" does not
    free = ~(r % 70 <= 10); mask = free[1:] & free[:-1]
    d = np.diff(cons, axis=-1)
    lv = np.stack([lanes(s) for s in range(16)])
    y = np.concatenate([d[s][:, mask].ravel() for s in range(16)])
    xl = np.concatenate([lv[s][:, 1:][:, mask].ravel() for s in range(16)])
    def r2(x):
        pred = np.zeros_like(y, dtype=float)
        for v in np.unique(x): pred[x == v] = y[x == v].mean()
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  R^2 of per-round motion from lane (lcg%64):  {r2(xl):.3f}")
    # (b) seed k == seed 0 with SMs XOR-permuted by k  (chip-independent)
    cc = []
    for k in [2, 4, 6, 8, 10, 12, 14]:
        ok_idx = [j for j in range(66) if (2 * j ^ k) < 132]
        perm = [(2 * j ^ k) // 2 for j in ok_idx]
        cc.append(np.corrcoef(cons[k][ok_idx].ravel(), cons[0][perm].ravel())[0, 1])
    print(f"  corr(seed k, seed 0 XOR-permuted) over k=2..14:  mean {np.mean(cc):.3f}")
    print(f"  corr(seed k, seed 0 UNpermuted):                 {np.corrcoef(cons[2].ravel(), cons[0].ravel())[0,1]:.3f}")
    print("  => only seed%64 matters; full 2^32 seed space collapses to 64 distinct challenges per chip.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
