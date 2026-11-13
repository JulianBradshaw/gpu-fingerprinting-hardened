# Adversarial Evaluation and Hardening of GPU Fingerprinting for Location Verification

> **Disclaimer:** the paper and this README are drafts written by a large language model (LLM). They were not written by a human.

Code, derived results and the draft paper for an adversarial evaluation of software GPU fingerprinting (atomic-contention fingerprints on NVIDIA H200s), and for a hardened probe that resists the replay and decoy-modelling attacks found against the original design.

**Status:** draft. The paper is in [`paper/gpu_fingerprinting_hardened_paper.pdf`](paper/gpu_fingerprinting_hardened_paper.pdf). Rates in this README are copied from the paper; they are descriptive results from short-gap, single-host trials and have not been independently replicated.

This work builds on Tee and Happel (see [Reference](#reference)). Their probe is called the **legacy probe** below.

## Why this matters

Delay-based location verification needs to know that a challenge response came from the registered GPU, not a decoy chip or a replay cache. The legacy probe reports up to 100% closed-set re-identification across 24 H200s, but closed-set identification is not the same as authentication: an open-set verifier must accept genuine responses across collections *and* reject spoofs synthesised from other GPUs of the same model. This repository tests that second property.

## Key findings

### 1. The legacy probe can be spoofed

Measured on eight plain H200s across four collections (10,240 fingerprints):

- **6-bit seed bottleneck.** The legacy challenge LCG selects racing threads with `lcg_state % 64`, so seeds congruent modulo 64 give identical thread-selection sequences. Under one collection's p95 cutoff, related-seed replay passed **91.7%** of attempts.
- **Decoy modelling.** Coordinate-wise median offset modelling with a seven-decoy H200 ensemble reached **87.5% (224/256)** acceptance at the pooled p95 cutoff.
- **Unstable calibration.** Nearest-chip identification stayed at 94.5–100%, but p95 cutoffs swung 3.3×–5.7× across calibration splits, and three of eight chips accounted for 97.8% of genuine rejections.

### 2. The hardened probe closes most of the gap

The hardened probe combines:

1. **20-round ChaCha20 challenge expansion** of a public 256-bit challenge with separate SM and round domains.
2. **Sleep-free, 16-address contention**: challenge-dependent delay work in place of sleeps, racing across 16 counters.
3. **Per-round SM rank permutations** as the output format.
4. **Two-of-three micro-burst voting**: three launches per warm burst; at least two must be uniquely nearest to the claimed GPU and within its cutoff.

Matched head-to-head on a separate cohort of eight plain H200s (256 held-out trials per arm):

| Metric | Legacy | Hardened | Hardened + variation floor (exploratory) |
|---|---|---|---|
| Genuine accept, pooled p95 | 85.5% (219/256) | 86.3% (221/256) | — |
| Genuine accept, per-chip p95 | 78.5% (201/256) | 89.1% (228/256) | 87.9% (225/256) |
| Predetermined single-decoy spoof | 53.9% (138/256) | **0.0% (0/256)** | 0.0% (0/256) |
| Seven-decoy ensemble spoof, pooled p95 | 87.5% (224/256) | **9.8% (25/256)** | — |
| Seven-decoy ensemble spoof, per-chip p95 | 47.7% (122/256) | 9.8% (25/256) | **0.0% (0/256)** |
| Wrong physical chip accepted, per-chip p95 | 0.7% (12/1,792) | 0.8% (15/1,792) | 0.8% (15/1,792) |

In a separate matched two-H200 NVL run, the hardened implementation cut median cold process runtime by **36%** (772 ms → 492 ms). A warm three-launch burst took 38.3 ms in a single-H200 NVL pilot.

### 3. Exploratory: a launch-to-launch variation floor

19 of the 25 remaining seven-decoy accepts targeted one chip (GPU C). The accepted spoofs scored *closer* than every genuine GPU C burst, but had abnormally low launch-to-launch variation (median 0.188 vs. 0.340 genuine), probably because a median across decoys damps single-chip jitter. Adding a calibration-derived minimum-variation floor reduced seven-decoy acceptance to 0/256, at a cost of three genuine bursts. One single-decoy pairing still passed 3/32 against GPU C. This rule was formulated after inspecting the data and **needs prospective validation**.

## Limitations

- Captures used short gaps (5–13 minutes) on single hosts, without controlled power cycles or separate target/decoy hosts.
- Trials reuse GPUs, challenges and registration templates, so rates are descriptive rather than independent statistical trials.
- The legacy and hardened probes use different native response budgets; matched-budget ablations are needed to attribute gains to individual components.
- All scoring is offline. Network deadlines, live challenge state and hardware attestation, needed for real location verification, are out of scope.

## Repository layout

```
paper/                       Draft paper (PDF)
reports/                     Combined security report (long-form write-up behind the paper)
existing-tool-analysis/      Legacy probe: kernels, collection harness, offline analyses, results (paper §3)
new-tool-analysis/           Hardened probe and matched experiments (paper §4)
  gpu_fingerprinting_hardened_v1/   Hardened probe source (CUDA + Python verifier, tests)
  eight_h200_runs/.../offline-results/   Frozen rule, summaries and score arrays for the matched eight-H200 run
data/RAW_DATA_SHA256SUMS.txt Checksums for the raw archives (not stored in git)
```

### Paper-to-code map

| Paper item | Files |
|---|---|
| §3 legacy collections (Runs 1–4) | `existing-tool-analysis/campaign_v4.py` (collector), `campaign_v5_data_*/` (per-run metadata, `analysis.json`, collector and kernel snapshots). Runs 1–2 = instance 59660701, Runs 3–4 = 59661478, in timestamp order. |
| Legacy kernel | `existing-tool-analysis/atomic_race_probe.cu`, `atomic_race_probe_batch_v2.cu` |
| §3 seed-64 replay, timing, sleep removal | `confirm_seed64_h200.py`, `measure_timing_h200.py`, `strip_test.py`, with outputs in `test1_seed64.log`, `test2_timing.log`, `test3_strip.log` |
| Fig. 2 / Table 2 frozen cross-collection verification | `frozen_registry_tradeoff.py` → `frozen_registry_tradeoff/`; `cross_collection_verification.py` → `cross_collection_verification.{json,md}`; `zero_observed_spoof_bound.py` |
| §3 split sensitivity, chip concentration, defences | `extended_offline_analysis.py` → `extended_offline_analysis/` |
| Data integrity audit | `audit_four_datasets.py`, `recompute_four_runs.py` → `four_run_data_audit.json`, `four_run_reanalysis.json` |
| §4.1 hardened probe | `new-tool-analysis/gpu_fingerprinting_hardened_v1/` (`probe_hardened.cu`, `chacha20_core.h`, `fingerprint.py`, `collect.py`, `evaluate.py`, `manage.py`, `test_hardening.py`) |
| §4.2 single-NVL sanity run (59832716) | `single_h200_sanity_plan.md`, `single_h200_sanity_analysis.py`, `host_run.sh` |
| §4.2 two-NVL pilot (59835170) and matched timing run (59934299) | `two_h200_host.sh`, `two_h200_analysis.py`, `matched_h200.py`, `check_matched_decoys.py`, `compare_h200_calibration.py` |
| §4.2 matched eight-H200 benchmark (59951659), Fig. 1, Table 1 | `eight_h200_ensemble.py` → `eight_h200_ensemble_59951659_results.json`, `eight_h200_runs/.../offline-results/` |
| §4.3 GPU C follow-up, Fig. 3, Table 3 | `analyze_eight_h200_followups.py` → `eight_h200_ensemble_followups.json` |
| Cloud orchestration (Vast.ai) | `provision_*.ps1`, `run_*.ps1` |

## Environment

- **GPUs:** plain NVIDIA H200 (132 SMs, compute capability 9.0) for the legacy collections and the matched eight-H200 run; H200 NVL for runs 59832716, 59835170 and 59934299. All were rented single hosts.
- **CUDA:** 12.4 (nvcc in the run archives); the hardened package was also built locally with CUDA 12.4.99 for `sm_75`/`sm_90`.
- **Build flags:** `-arch=sm_90`; legacy compiled at `-O0`, hardened at `-O2`.
- **Python:** 3.11 with numpy 1.26 for the legacy collections (see `metadata.json`); analysis scripts need only numpy.
- **GPU driver version:** not recorded in the archived metadata.

## Raw data

The raw captures (~630 MB: 32 per-GPU `.npz` arrays from the four legacy collections, and four `run.tar.gz` archives from the hardened/matched runs) are **not stored in this repository**. Their SHA-256 checksums and the paths they belong at are listed in [`data/RAW_DATA_SHA256SUMS.txt`](data/RAW_DATA_SHA256SUMS.txt). The per-run summaries (`analysis.json`, `metadata.json`, per-GPU JSON headers that include each `.npz` hash), derived result files and score arrays are included, so the paper's tables can be checked against the stored results without the raw captures.

To re-run the legacy offline analyses once the raw `.npz` files are restored into `existing-tool-analysis/campaign_v5_data_*/`:

```bash
cd existing-tool-analysis
python frozen_registry_tradeoff.py --data-root . --output out_frozen
python extended_offline_analysis.py --data-root . --reference frozen_registry_tradeoff/results.json \
    --output out_extended --cache work/raw_pairwise_distances.npz
python cross_collection_verification.py
```

The hardened-run scripts (`eight_h200_ensemble.py analyze`, `analyze_eight_h200_followups.py`, `two_h200_analysis.py`) run from inside an extracted `run.tar.gz` directory.

## Notes on provenance

- Scripts that originally used absolute local paths now resolve paths relative to the repository. No analysis logic was changed.
- `new-tool-analysis/gpu_fingerprinting_hardened_v1/MANIFEST.json` records hashes from the original package, which included a prebuilt Windows binary (`probe_hardened_sm75.exe`). That binary is not included, and one wording change in that folder's `README.md` means its hash no longer matches the manifest entry.
- The three `test*.log` files were converted from UTF-16 to UTF-8.

## Reference

W. Tee and J. Happel. GPU fingerprinting for location verification. *arXiv:2605.01930v2*, June 2026. Code: <https://github.com/waynetee/gpu-fingerprinting>.
