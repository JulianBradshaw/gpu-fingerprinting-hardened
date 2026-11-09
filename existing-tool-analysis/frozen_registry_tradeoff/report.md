# Frozen-registry joint genuine/spoof tradeoff

Offline analysis of the four saved datasets. No GPU experiments, rentals, or network operations were performed.

## Findings

On these observed samples, none of the three registries with later collections admits a single threshold that accepts at least 90% of genuine challenges in every later collection and rejects every tested spoof. At zero observed spoof acceptance across both training modes, the best minimum genuine acceptance across later collections is 68.0%, 83.6%, and 80.5% for registries 1, 2, and 3 respectively.

Requiring at least 95% genuine acceptance in every later collection forces worst-case observed source-trained spoof acceptance of at least 100%, 75.0%, and 87.5% respectively. These source-trained attacks use corrections learned only from the original registration collection. They include single-decoy and seven-decoy ensemble variants with multipliers 1, 1.5, and 2; these rates must not be attributed to the single-decoy model alone.

This establishes an empirical tradeoff for the evaluated scalar-distance threshold rule and specified attacks on this dataset. It does not establish impossibility for other scoring rules, defenses, devices, or future data. The retrospective threshold bounds are optimistic: thresholds are allowed to see later genuine evaluation scores when finding the best achievable observed tradeoff.

## Method

Rows below supply the frozen verifier: eight registration runs per seed/chip and a threshold calibrated only from that collection’s separate calibration run. Columns supply genuine responses and fresh decoy responses from their independent evaluation run. Chips are matched by UUID. Each entry has 128 challenges (eight chips × 16 uniform seeds).

Source-trained corrections use only the registration collection’s training seeds. Destination-trained corrections use the later collection’s registration samples on other seeds; this models an attacker able to refresh its target/decoy training data under those conditions. Both exclude the challenged seed. The single decoy is fixed in advance; the ensemble uses all seven other GPUs. Synthetic responses are explicitly truncated before scoring.

## Fixed source-calibrated p95 results

### Genuine joint acceptance

| Register ↓ / Verify → | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| 1 | 118/128 (92.2%) | 87/128 (68.0%) | 98/128 (76.6%) | 90/128 (70.3%) |
| 2 | 126/128 (98.4%) | 118/128 (92.2%) | 123/128 (96.1%) | 117/128 (91.4%) |
| 3 | 127/128 (99.2%) | 117/128 (91.4%) | 122/128 (95.3%) | 115/128 (89.8%) |
| 4 | 121/128 (94.5%) | 121/128 (94.5%) | 121/128 (94.5%) | 124/128 (96.9%) |

### Single decoy ×1, source-trained correction

| Register ↓ / Verify → | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| 1 | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) |
| 2 | 32/128 (25.0%) | 31/128 (24.2%) | 30/128 (23.4%) | 32/128 (25.0%) |
| 3 | 97/128 (75.8%) | 84/128 (65.6%) | 87/128 (68.0%) | 82/128 (64.1%) |
| 4 | 37/128 (28.9%) | 36/128 (28.1%) | 38/128 (29.7%) | 37/128 (28.9%) |

### Seven-decoy ensemble ×1, source-trained correction

| Register ↓ / Verify → | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| 1 | 11/128 (8.6%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) |
| 2 | 82/128 (64.1%) | 87/128 (68.0%) | 82/128 (64.1%) | 84/128 (65.6%) |
| 3 | 112/128 (87.5%) | 112/128 (87.5%) | 112/128 (87.5%) | 112/128 (87.5%) |
| 4 | 96/128 (75.0%) | 96/128 (75.0%) | 96/128 (75.0%) | 96/128 (75.0%) |

### Single decoy ×1, destination-trained correction

| Register ↓ / Verify → | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| 1 | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) |
| 2 | 35/128 (27.3%) | 31/128 (24.2%) | 26/128 (20.3%) | 31/128 (24.2%) |
| 3 | 127/128 (99.2%) | 64/128 (50.0%) | 87/128 (68.0%) | 68/128 (53.1%) |
| 4 | 59/128 (46.1%) | 34/128 (26.6%) | 36/128 (28.1%) | 37/128 (28.9%) |

### Seven-decoy ensemble ×1, destination-trained correction

| Register ↓ / Verify → | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---:|---:|---:|
| 1 | 11/128 (8.6%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) |
| 2 | 98/128 (76.6%) | 87/128 (68.0%) | 75/128 (58.6%) | 88/128 (68.8%) |
| 3 | 127/128 (99.2%) | 96/128 (75.0%) | 112/128 (87.5%) | 96/128 (75.0%) |
| 4 | 117/128 (91.4%) | 96/128 (75.0%) | 102/128 (79.7%) | 96/128 (75.0%) |

## Retrospective bound: one threshold for every later collection

For each source, find the smallest single threshold that accepts at least the stated percentage of genuine challenges in **each** later collection. Since acceptance is monotone in the threshold, these are the lowest achievable spoof rates at that genuine requirement on these exact samples. Thresholds are selected using evaluation data here: they are optimistic diagnostic bounds, not deployment-calibrated thresholds.

Worst spoof rate is the maximum observed acceptance across the tested multipliers, single/ensemble models, and later collections. It is not a pooled average or a newly validated attacker-selection strategy.

| Register | Later runs | Genuine requirement per later run | Smallest threshold | Worst source-trained spoof | Worst destination-trained spoof |
|---|---|---:|---:|---:|---:|
| 1 | 2, 3, 4 | 90% | 273,348 | 100.0% | 93.0% |
| 1 | 2, 3, 4 | 95% | 473,140 | 100.0% | 93.0% |
| 1 | 2, 3, 4 | 99% | unattainable due to identification errors | — | — |
| 2 | 3, 4 | 90% | 49,720 | 12.5% | 10.2% |
| 2 | 3, 4 | 95% | 113,648 | 75.0% | 86.7% |
| 2 | 3, 4 | 99% | 221,218 | 89.1% | 89.1% |
| 3 | 4 | 90% | 163,564 | 87.5% | 78.1% |
| 3 | 4 | 95% | 293,020 | 87.5% | 99.2% |
| 3 | 4 | 99% | 459,150 | 87.5% | 99.2% |

## Validation

All input array checksums were verified. Every within-collection single-decoy/ensemble result at p90/p95/p99 reproduces the saved analysis. The genuine cross-collection results also reproduce the prior independent analysis. All score breakpoints, per-trial distances/margins, fixed-threshold results and retrospective pairwise bounds are retained in results.json.

## Limits

- Finite observed challenges and specified attack models only; cannot prove security or universal impossibility.
- Rates share seeds, chips and training data; no independent-trial confidence claims.
- Source-trained corrections exclude the challenge seed; actual verifier registry reuse for training seeds is an evaluation assumption.
- Destination-trained mode is not proof that an attacker can obtain same-condition target training data chronologically before each challenge.
- Four collections on same eight H200s over about thirteen minutes; no established power-cycle/transport intervention.
- No attack latency, network relay, or replay-table study is performed here.

## Retrospective zero-observed-spoof bound

For completeness, the largest threshold rejecting every tested spoof in every chronological later collection gives the following minimum genuine acceptance across those later collections. These are observed-sample bounds, not guarantees of zero future spoof probability.

| Register | Source-trained models only | Destination-trained models only | Both sets of models |
|---|---:|---:|---:|
| 1 | 68.0% | 69.5% | 68.0% |
| 2 | 83.6% | 86.7% | 83.6% |
| 3 | 80.5% | 80.5% | 80.5% |
