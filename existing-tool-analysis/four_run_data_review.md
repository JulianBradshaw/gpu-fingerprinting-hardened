# Four-run data review

## Dataset identity and completeness

The folder contains four complete v5 dataset downloads. They represent four distinct collection sessions across two rental instance IDs. All four identify the same eight H200 GPU UUIDs, in the same order.

| Run | Collection start, November 7, 2026 (Mountain time) | Rental instance |
|---|---|---|
| 1 | 8:47:06 PM | 59660701 |
| 2 | 8:51:46 PM | 59660701 |
| 3 | 8:56:41 PM | 59661478 |
| 4 | 9:00:27 PM | 59661478 |

Every collection has a different session UUID and different raw fingerprint arrays. They are not duplicate downloads. The two rental IDs do not indicate different physical chips: all eight GPU IDs overlap across all four collections. The metadata does not establish whether hardware was power cycled or other operating conditions changed between allocations.

All four use the same seed design, CUDA source hash and analysis-script hash. Each contains 32 seeds × 10 measurements × 8 GPUs = 2,560 fingerprints, for 10,240 across the four collections. Repeated measurements share hardware, seeds and training sets; these are not 10,240 independent security trials.

These eight GPUs share no UUIDs with the earlier incomplete v4 download associated with instance 59654478.

## Verification performed

- Checked all 32 compressed GPU archives against the saved SHA-256 checksums.
- Loaded the archives and checked seed IDs, measurement counts and integer data types.
- Confirmed dataset and analysis session IDs match.
- Recomputed all per-trial scores and operating-point summaries from the raw arrays using the reviewed local analysis implementation.
- Confirmed every recomputed trial and summary matches its saved analysis.
- Compared hashes of the decompressed arrays to distinguish new measurements from duplicated archives.

All checks passed. No remote operations or new GPU experiments were performed.

## Results at each collection's separately calibrated p95 threshold

| Run | Threshold | Genuine joint acceptance | Single decoy ×1 | Seven-decoy ensemble ×1 | Recorded replay | Mean-of-eight replay |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 41,725.5 | 118/128 (92.2%) | 0/128 (0%) | 11/128 (8.6%) | 0/96 (0%) | 0/96 (0%) |
| 2 | 67,486.9 | 118/128 (92.2%) | 31/128 (24.2%) | 87/128 (68.0%) | 1/96 (1.0%) | 0/96 (0%) |
| 3 | 153,213.9 | 122/128 (95.3%) | 87/128 (68.0%) | 112/128 (87.5%) | 88/96 (91.7%) | 80/96 (83.3%) |
| 4 | 79,429.9 | 124/128 (96.9%) | 37/128 (28.9%) | 96/128 (75.0%) | 3/96 (3.1%) | 3/96 (3.1%) |

Joint acceptance requires both correct target matching and distance within the threshold. Replay rates combine the four tested residue bases. Each collection calibrates its own threshold using its separate calibration measurements; this is not a comparison at one common fixed threshold.

## Interpretation

The results vary sharply despite using the same physical GPU IDs and the same seed design. Run 3 is the outlier: its p95 threshold is approximately 3.7 times Run 1's, and replay acceptance is much higher. The data support a strong dependence of observed attack acceptance on the collection and calibrated acceptance rule; they do not by themselves identify which operating condition caused the variation.

These collections do not justify one universal attack success rate or a claim of consistent parity with genuine chips. They do demonstrate substantial spoof acceptance in several evaluated collections, including high replay acceptance in Run 3.

The reported results still register and verify within each collection. This review did not test registration from one collection against challenges from another. The available matching GPU IDs and seed designs now make that comparison possible, but four collections over approximately thirteen minutes would remain a short-timescale study rather than evidence about long-term or post-transport stability.
