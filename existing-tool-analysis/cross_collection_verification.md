# Cross-collection genuine verification

All analysis used saved local measurements. No GPU experiments or remote operations were performed.

## Method

For each registration collection, use its eight registration measurements per chip/seed and calibrate thresholds solely from its separate calibration measurement (index 9). Verify the separate genuine challenge measurement (index 8) from each collection against that fixed registry. Match chips by UUID and use the 16 uniform seeds common to all four collections. Reject equal-distance identification ties.

Each comparison contains 128 challenges: eight chips × 16 seeds. Cross-collection thresholds are not recalibrated using the verification collection. The diagonal is a within-collection control and reproduces the existing genuine results at all three thresholds.

## Collections

| Run | Start, November 7, 2026 (Mountain time) | p95 threshold |
|---|---|---:|
| 1 | 08:47:06 PM | 41,725.5 |
| 2 | 08:51:46 PM | 67,486.9 |
| 3 | 08:56:41 PM | 153,213.9 |
| 4 | 09:00:27 PM | 79,429.9 |

## Joint acceptance at the registration collection’s p95 threshold

Requires correct nearest-chip identification and distance within the fixed threshold. Rows supply registration/calibration; columns supply verification responses.

| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |
|---|---:|---:|---:|---:|
| 1 | 118/128 (92.2%) | 87/128 (68.0%) | 98/128 (76.6%) | 90/128 (70.3%) |
| 2 | 126/128 (98.4%) | 118/128 (92.2%) | 123/128 (96.1%) | 117/128 (91.4%) |
| 3 | 127/128 (99.2%) | 117/128 (91.4%) | 122/128 (95.3%) | 115/128 (89.8%) |
| 4 | 121/128 (94.5%) | 121/128 (94.5%) | 121/128 (94.5%) | 124/128 (96.9%) |

## Identification without a distance threshold

| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |
|---|---:|---:|---:|---:|
| 1 | 128/128 (100.0%) | 125/128 (97.7%) | 124/128 (96.9%) | 122/128 (95.3%) |
| 2 | 126/128 (98.4%) | 128/128 (100.0%) | 127/128 (99.2%) | 128/128 (100.0%) |
| 3 | 127/128 (99.2%) | 128/128 (100.0%) | 128/128 (100.0%) | 127/128 (99.2%) |
| 4 | 121/128 (94.5%) | 126/128 (98.4%) | 128/128 (100.0%) | 128/128 (100.0%) |

## Distance gate alone at p95

| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |
|---|---:|---:|---:|---:|
| 1 | 118/128 (92.2%) | 87/128 (68.0%) | 98/128 (76.6%) | 90/128 (70.3%) |
| 2 | 126/128 (98.4%) | 118/128 (92.2%) | 123/128 (96.1%) | 117/128 (91.4%) |
| 3 | 128/128 (100.0%) | 117/128 (91.4%) | 122/128 (95.3%) | 115/128 (89.8%) |
| 4 | 124/128 (96.9%) | 121/128 (94.5%) | 121/128 (94.5%) | 124/128 (96.9%) |

## Joint acceptance at p90

| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |
|---|---:|---:|---:|---:|
| 1 | 110/128 (85.9%) | 83/128 (64.8%) | 88/128 (68.8%) | 78/128 (60.9%) |
| 2 | 118/128 (92.2%) | 113/128 (88.3%) | 116/128 (90.6%) | 113/128 (88.3%) |
| 3 | 111/128 (86.7%) | 100/128 (78.1%) | 107/128 (83.6%) | 102/128 (79.7%) |
| 4 | 121/128 (94.5%) | 119/128 (93.0%) | 120/128 (93.8%) | 121/128 (94.5%) |

## Joint acceptance at p99

| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |
|---|---:|---:|---:|---:|
| 1 | 128/128 (100.0%) | 105/128 (82.0%) | 111/128 (86.7%) | 103/128 (80.5%) |
| 2 | 126/128 (98.4%) | 123/128 (96.1%) | 126/128 (98.4%) | 124/128 (96.9%) |
| 3 | 127/128 (99.2%) | 124/128 (96.9%) | 127/128 (99.2%) | 119/128 (93.0%) |
| 4 | 121/128 (94.5%) | 126/128 (98.4%) | 128/128 (100.0%) | 128/128 (100.0%) |

## Findings

Cross-collection nearest-chip identification remains high: 121–128 of 128 challenges (94.5–100%). Applying the registration collection's fixed p95 threshold yields joint acceptance of 87–127 of 128 (68.0–99.2%) across the off-diagonal comparisons.

Using Run 1 registration and calibration, genuine acceptance is 92.2% within Run 1, then 68.0%, 76.6% and 70.3% on Runs 2, 3 and 4. In the chronological comparison across rental instance IDs, Run 2 registration accepts 96.1% of Run 3 challenges. These results show that transferring a registry and its acceptance threshold is sensitive to the chosen collection; they do not show a uniform collapse of chip identification.

## Interpretation limits

Only genuine-chip verification is evaluated here. These results do not establish cross-collection spoofing performance. Threshold differences can make forward and reverse comparisons asymmetric. Repeated challenges share chips, seeds and training measurements, so the rates are descriptive rather than independent-trial confidence estimates.

Run 1→2 and Run 3→4 remain within their respective rental instance IDs; Run 2→3 crosses rental instance IDs while retaining the same physical GPU UUIDs. Different rental IDs do not establish a power cycle or environmental change. Collection starts span about thirteen minutes; timestamps describe collection starts, not precise per-challenge timing. Comparisons to earlier collections are retrospective checks, not chronological deployment simulations.

The raw archive checksums and dimensions were validated before scoring. An independent integer-distance implementation reproduced every within-collection genuine control and calibration threshold. Full distances, target margins, seed IDs and results at all three thresholds are in cross_collection_verification.json.
