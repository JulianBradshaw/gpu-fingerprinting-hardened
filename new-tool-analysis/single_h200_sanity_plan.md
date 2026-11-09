# First single-H200 validation plan

## Objective and decision

Run the hardened-v1 probe on one dedicated H200 to establish whether it produces valid measurements, survives repeated requests, records useful timing, and maintains its own-distance consistency across fresh processes. This is the engineering gate before spending on a multi-GPU identification and attack study.

The result will be **advance**, **investigate**, or **technical failure**, separately for `all` and `subset`. Passing this plan establishes readiness for the next experiment. It does not establish chip discrimination, spoof resistance, attestation, geographic location, or a secure relay deadline.

This document is a protocol, not a record of an executed hardware experiment. The companion analysis helper has been checked with synthetic CPU fixtures; neither that check nor the existing unit tests measures GPU fingerprint quality.

## 1. Freeze the experiment before collection

- Use `gpu_fingerprinting_hardened_v1` unchanged, with its source and binary hashes saved. Rebuild on the Linux rental host; the Windows executable cannot run there.
- Use one full, dedicated H200, identified by its complete GPU UUID. Record the provider's exclusivity claim and any visible competing processes. If it is partitioned, shared, or the intended device cannot be selected, stop this protocol and document that limitation.
- Set a rental wall-time and spending cap before starting. Estimate whether the full collection fits only after the smoke check; account for telemetry and capture overhead as well as kernel time. This plan does not provision a machine or authorize an open-ended rental.
- Fix both profiles at 128 recorded rounds, 32 internal warm-up rounds, 64 threads per block, and three launches per saved burst. Retain the two startup warm-up launches per capture process.
- Use raw mean-active L1 as the only primary metric, alpha 0.05 for calibration, and the 58/64 evaluation screen below. Do not tune the metric, thresholds, rounds, or seed pool after looking at evaluation results.
- Do not change clocks, power limits, cooling, or other GPU settings during this first check. Record observable conditions. A normal host is sufficient; this is not a thermal or power-cycle experiment.
- Keep all failed attempts. Any source, binary, device, or profile change starts a new dated experiment; do not splice the replacement data into this one.

The `all` profile is an ablation within the new implementation, not a reproduction of the old probe. Old datasets and numerical thresholds do not transfer to this format.

## 2. Collection design and exact budget

Generate two fresh challenge pools: two challenges for smoke checks and eight different challenges for the main experiment. Reuse the main pool across modes and phases to permit like-for-like within-challenge comparisons. These are disclosed laboratory challenges, not a future deployment pool.

| Stage | Modes | Challenges per mode | Repeats per challenge | Saved bursts | Saved kernel fingerprints |
|---|---:|---:|---:|---:|---:|
| Smoke | 2 | 2 | 1 | 4 | 12 |
| Registration | 2 | 8 | 8 | 128 | 384 |
| Calibration | 2 | 8 | 8 | 128 | 384 |
| Held-out evaluation | 2 | 8 | 8 | 128 | 384 |
| Total | | | | **388** | **1,164** |

There are eight capture processes: two smoke and six main. Their two startup warm-up launches add 16 unsaved launches, giving **1,180 total probe kernel launches** if every stage completes once. Every launch also contains 32 unrecorded warm-up rounds; those are not additional launches. Retries would increase this budget and must be reported separately.

Each main phase/mode has 64 bursts and uses a fresh probe process/context. Within each capture, requests are shuffled by a recorded seed. Run main captures sequentially in this predeclared order:

1. `all` registration
2. `subset` registration
3. `subset` calibration
4. `all` calibration
5. `all` evaluation
6. `subset` evaluation

This interleaves the modes while retaining registration → calibration → evaluation within each mode. It does not eliminate thermal or order effects. Record actual start/end times and any gaps; do not call separate processes independent physical conditions.

## 3. Prepare the Linux host

Upload the source package and `single_h200_sanity_analysis.py`. Use a new directory such as `/workspace/fp-sanity-YYYYMMDDTHHMMSSZ`; preserve earlier directories. The commands below are for Bash on the rental host, and **will execute GPU work at the smoke/collection stages**. They are not PowerShell commands. Do not use the old campaign runners unchanged: they target the earlier experiment.

Start a shell that stops on errors and preserves failures through logging pipelines:

```bash
set -euo pipefail
RUN="/workspace/fp-sanity-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$RUN"
mkdir "$RUN/code" "$RUN/data" "$RUN/logs" "$RUN/results"
cd "$RUN"
```

Place the package's source files directly in `$RUN/code`, and the analysis helper in `$RUN/`. Keep the uploaded archive too. Do not execute a Windows binary or silently substitute an installed older probe.

Record host metadata and select the exact UUID shown by `nvidia-smi`:

```bash
nvidia-smi -L | tee logs/devices.txt
nvidia-smi -q -x > logs/host-before.xml
nvcc --version > logs/nvcc.txt
g++ --version > logs/host-compiler.txt
python3 --version > logs/python.txt
uname -a > logs/system.txt
export GPU_UUID='GPU-REPLACE-WITH-FULL-UUID'
nvidia-smi -i "$GPU_UUID" -q -x > logs/selected-before.xml
```

Replace the UUID placeholder before proceeding. Inspect the metadata: intended H200, no unintended partition/sharing, usable driver/toolkit, enough disk space, and no unexpected workloads. Save the rental ID, advertised configuration, rate, budget, and any known condition in `logs/session-notes.md`. Observed UUID and telemetry are experimental bookkeeping, not trusted hardware identity evidence.

Use an isolated Python environment and retain dependency/build logs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt 2>&1 | tee logs/dependencies.txt
python -m pip freeze > logs/python-packages.txt
cd code
python -W error::ResourceWarning -m unittest -v test_hardening 2>&1 | tee ../logs/cpu-tests.txt
nvcc -std=c++17 -O2 -arch=sm_90 probe_hardened.cu -o probe_hardened 2>&1 | tee ../logs/build.txt
cd ..
find code -maxdepth 1 -type f -exec sha256sum {} + > logs/build-input-hashes.txt
sha256sum single_h200_sanity_analysis.py >> logs/build-input-hashes.txt
```

A compile or CPU-test failure is a stop condition, not a reason to skip that check. Save any skipped host-compiler test explicitly.

## 4. Smoke check: stop early on engineering failures

```bash
python code/collect.py pool --count 2 --output data/smoke-pool.json
for MODE in all subset; do
  python code/collect.py capture \
    --binary "$RUN/code/probe_hardened" --device "$GPU_UUID" \
    --pool data/smoke-pool.json --phase evaluation --repeats 1 \
    --mode "$MODE" --rounds 128 --timeout 120 \
    --output "data/smoke-$MODE.jsonl" \
    2>&1 | tee "logs/smoke-$MODE.txt"
done
```

Inspect both manifests and stderr files before collecting more. Require:

- Both manifests say `complete`, with exactly two measurements each, the intended UUID, and the requested profile.
- Every response validates: three integer arrays, each shaped `128 × number_of_SMs`; stable sorted unique physical SM mapping; correct challenge-derived active mask; inactive entries `-1`; active ranks exactly `0…k−1` without ties.
- `all` activates every mapped SM; `subset` activates the expected 4–8 SMs in each round. The collector checks the expected mask, not merely the count.
- No CUDA error, mapping failure, crash, invalid JSON, timeout, or lost request. The probe deliberately refuses unsupported block placement; treat that as a real technical result.
- Finite nonnegative kernel and host times. Investigate systematically zero or implausible times and any unexplained stderr output.

The collector validates schema and mapping during capture. It saves raw telemetry before/after each burst, or records telemetry unavailability. Missing telemetry alone is not a rank-schema failure, but report it and avoid claims about stable operating conditions.

Use the smoke manifests' wall duration to estimate the remaining 384 main bursts. Startup costs, per-burst telemetry calls, Python validation, output size, and disk flushing matter. Proceed only if the remaining budget is adequate. The 120-second timeout is a response guard, not a performance target or geographic bound.

## 5. Collect the main dataset

Run only after the smoke gate passes. Generate the eight-challenge pool once. Do not regenerate it per phase.

```bash
python code/collect.py pool --count 8 --output data/main-pool.json
for ITEM in all:registration subset:registration subset:calibration all:calibration all:evaluation subset:evaluation; do
  MODE="${ITEM%%:*}"
  PHASE="${ITEM##*:}"
  python code/collect.py capture \
    --binary "$RUN/code/probe_hardened" --device "$GPU_UUID" \
    --pool data/main-pool.json --phase "$PHASE" --repeats 8 \
    --mode "$MODE" --rounds 128 --timeout 120 \
    --output "data/$MODE-$PHASE.jsonl" \
    2>&1 | tee "logs/$MODE-$PHASE.txt"
done
nvidia-smi -i "$GPU_UUID" -q -x > logs/selected-after.xml
```

Each capture must finish completely before the next begins. After collection, require exactly eight observations per challenge per phase and mode, matching manifests and launch order, one binary hash throughout, stable UUID/SM mapping, distinct capture IDs, and no duplicated sample IDs. Retain all JSONL, manifest, and stderr files.

On failure, stop the loop and preserve the incomplete attempt. Do not top it up, silently drop bad rows, or overwrite it. Diagnose the issue, then run a new complete experiment under a new directory if needed. Do not label a partial run as passing.

## 6. Frozen-calibration analysis

The companion helper is deliberately single-device. It does not bypass the enrollment tool's minimum-two-device requirement or create a dummy second chip. For each mode it:

1. Builds 24 reference kernel fingerprints per challenge from its eight registration bursts.
2. Scores each new kernel by its minimum raw mean-active L1 distance to that challenge's registration templates. Inactive coordinates are excluded.
3. Uses the second-smallest of three kernel scores as the burst score, corresponding to two-of-three own-distance consistency.
4. Sets the threshold from the 64 calibration burst scores: sorted order statistic `ceil((64+1) × 0.95) = 62`, using one-based indexing. This is the corrected quantile used by the prototype, not an interpolated empirical percentile.
5. Freezes the references and threshold, then counts evaluation bursts with at least two scores at or below that threshold. Evaluation never contributes templates, normalization, or threshold selection.

The corrected quantile does not provide a distribution-free guarantee here without its exchangeability assumptions. Shared device, repeated challenges, and operating-condition changes limit that interpretation.

Run analysis on the rental or later offline on the downloaded files:

```bash
for MODE in all subset; do
  python single_h200_sanity_analysis.py \
    --code code --pool data/main-pool.json --device "$GPU_UUID" \
    --registration "data/$MODE-registration.jsonl" \
    --calibration "data/$MODE-calibration.jsonl" \
    --evaluation "data/$MODE-evaluation.jsonl" \
    --mode "$MODE" --rounds 128 --repeats 8 \
    --output "results/$MODE-summary.json" \
    2>&1 | tee "logs/$MODE-analysis.txt"
done
```

Exit code 0 means the technical checks and provisional consistency screen passed. Exit code 2 means a valid summary was written but the consistency screen failed; with `set -e`, analyze the other mode separately after recording this result. Other failures require diagnosis. Do not loosen the criterion and rerun against the same held-out evaluation data.

The summaries include per-challenge evaluation counts, registration within-burst and between-burst distance distributions, leave-one-burst-out registration scores, calibration/evaluation score distributions, thresholds, per-position kernel timing, host timing, capture wall duration, telemetry availability, and input hashes. The distance-pair counts are correlated descriptive observations, not independent sample sizes.

## 7. Predeclared interpretation

| Gate | Criterion | Consequence |
|---|---|---|
| Technical validity | Complete exact counts, valid ranks/masks, stable UUID/map/profile/binary, distinct samples, no capture errors | Any failure: technical failure; investigate before expanding |
| Own-distance consistency | At least **58 of 64** evaluation bursts within the frozen threshold, per mode | Below this: investigate scoring or probe stability before relying on that mode |
| Timing and conditions | Report distributions, gaps, telemetry availability, and anomalies | Explain limitations; do not infer a relay deadline |
| Advance | A mode passes technical validity and the consistency screen, with no unresolved correctness problem | That mode is a candidate for multi-GPU evaluation |

The 58/64 floor is a predeclared engineering screen of at least 90% observed consistency (58/64 = 90.625%). It is not a 95% genuine-acceptance claim, an authentication error bound, or a statistically powered security result. Report the actual count rather than only pass/fail. Also report the worst challenge; a high pooled count can conceal a weak seed. Neither mode's outcome substitutes for the other's.

Very low distances, identical responses, or strong variation between challenges require interpretation: public deterministic scheduling can produce those patterns without chip-specific information. Repeatability alone is insufficient. A good timing result also does not prove that the workload is irreducible or cannot be relayed.

Keep CUDA-event kernel time separate from host measurement time and total capture duration. Events include the 32 warm-up rounds inside a launch; host measurement includes launch/copy work but excludes JSON serialization, telemetry, network transport, and process startup. Capture duration includes additional overhead. Do not compare these directly to an old cold-start/end-to-end timing and claim a speedup.

## 8. Preserve results and close the rental

Download the complete run directory, including source, binary, challenge pools, raw observations, failed attempts, manifests, stderr, metadata, build/test logs, summaries, and session notes. Verify downloaded file hashes before deallocating. Preserve an untouched raw copy. Run the offline helper again locally if useful, using the downloaded code and identical inputs.

The short result memo should state: device/session and elapsed span; exact attempted/completed counts; failures and exclusions (normally none); each mode's threshold and evaluation count; per-challenge weakness; timing and metadata limitations; and the advance/investigate decision. Do not call the one-device result genuine identification or report a false-accept rate.

## Possible follow-ups after this check

- Collect multiple distinct H200s with independent registration, calibration, and held-out evaluation; measure nearest-chip accuracy and joint identity-plus-threshold acceptance.
- Test frozen registries across later sessions and controlled conditions, including repeated registration/calibration splits. Evaluate chip-specific or registration-trained normalized thresholds against both genuine responses and spoofs.
- Evaluate replay, decoy, and model-based responses against the new profiles with explicit attacker data budgets and held-out challenges where the protocol supports them.
- Compare same-host versus different-host devices and thermal/power/restart conditions with controls. A separate trusted timing/transport design is required for geographic or relay claims.

These are future experiments, not conclusions or prerequisites silently added to the first single-H200 sanity check.
