# Additional offline analyses

Four saved collections on the same eight H200s. No rentals, GPU experiments or remote operations were performed. All raw archives were checksum-validated.

## Findings

- **Sample-role choice matters greatly.** Run 3's p95 threshold ranges from 41.7k to 236.7k when the same ten measurements are reassigned. Its original threshold of 153.2k is above the split median of 76.7k. The between-collection threshold differences therefore cannot be attributed solely to physical drift over time.
- **The main failure concentration is by chip.** Three of eight chips account for 135 of 138 genuine rejection events (97.8%) across the original six forward comparisons. No single seed dominates similarly.
- **The tradeoff often persists, but its magnitude is split-dependent.** In the balanced x1-only spoof study, 20 of 30 source/split configurations can attain 95% genuine acceptance in every later collection. Their minimum achievable worst observed spoof rates range from 14.1% to 99.2%. The other ten configurations cannot attain that genuine goal because of identification errors. These are overlapping sensitivity checks, not 30 independent experiments.
- **Seven decoys are not always necessary.** At the original Run 3→4 p95 operating point, source-trained single-decoy assignments accept 62.5–64.1% of spoofs; four-decoy assignments accept 82.8–87.5%; all seven accept 87.5%. Results differ sharply for tighter registries, and adding decoys is not strictly monotonic.
- **Simple defenses can help locally, without a consistent repair.** In Run 2→4, chip normalization increases genuine acceptance from 91.4% to 92.2% while reducing the highest tested spoof rate from 68.8% to 12.5%. The same rule still accepts 73.4% of the strongest tested spoofs in Run 3→4 and reduces genuine acceptance in the Run 1 comparisons.

The defensible interpretation is fragile calibration with strong chip-specific heterogeneity and empirically exploitable tradeoffs. These analyses do not isolate a physical time-drift mechanism or establish impossibility for all calibrated verifiers.

## 1. Sample-split sensitivity

Genuine verification uses all 90 ordered calibration/evaluation assignments; the other eight measurements register. The 10 cyclic splits additionally evaluate source-trained x1 single-decoy and seven-decoy ensemble spoofs. These splits overlap heavily and are sensitivity checks, not independent replications.

| Registry collection | Original p95 | p95 across all 90 splits: min / median / max |
|---|---:|---:|
| 1 | 41,726 | 41,726 / 47,378 / 171,011 |
| 2 | 67,487 | 58,287 / 101,961 / 192,961 |
| 3 | 153,214 | 41,683 / 76,672 / 236,679 |
| 4 | 79,430 | 53,724 / 92,985 / 188,793 |

| Registry | Balanced splits with attainable 95% genuine goal in every later collection | Minimum / median / maximum worst x1 spoof rate at that goal |
|---|---:|---:|
| 1 | 2/10 | 92.2% / 95.7% / 99.2% |
| 2 | 8/10 | 57.8% / 75.0% / 75.8% |
| 3 | 10/10 | 14.1% / 87.5% / 90.6% |

The 95% bounds are retrospective choices using later genuine scores; they are not held-out calibrations. Rotating roles changes registry samples as well as calibration/evaluation samples, so this does not isolate physical drift from sampling effects.

## 2. Failure concentration

Genuine rejection events below pool the six chronological forward comparisons at their original source p95 thresholds. A later response can be rejected by multiple source registries; event counts are not independent challenges.

| Chip (UUID suffix) | Genuine rejection events | Share of rejection events |
|---|---:|---:|
| a73006f53df4 | 66 | 47.8% |
| b8d095aa659e | 41 | 29.7% |
| 538d1beb6b0a | 28 | 20.3% |
| e8bbd5710bfc | 2 | 1.4% |
| 1a0c9250ef09 | 1 | 0.7% |

Top seeds by genuine rejection events: 1602297017: 13, 694443915: 11, 1524289825: 11, 1789368711: 10, 1859447115: 10.

The JSON includes per-pair, per-chip and per-seed genuine rejection and spoof-acceptance breakdowns, plus each collection’s largest calibration scores. These identify concentration, not a proven hardware cause.

## 3. Decoy-count ablation

Ranges are across seven predetermined assignments, not confidence intervals. Each assignment stays fixed across targets/seeds by cyclic GPU order. Values use the frozen pooled source p95 and x1 correction. The seven-decoy ensemble has one assignment.

| Register → later verify | Training | 1 decoy | 2 decoys | 4 decoys | 7 decoys |
|---|---|---:|---:|---:|---:|
| 1→2 | source_trained | 0.0–0.0% | 0.0–0.0% | 0.0–7.0% | 0.0% |
| 1→2 | destination_trained | 0.0–0.0% | 0.0–0.0% | 0.0–5.5% | 0.0% |
| 1→3 | source_trained | 0.0–0.0% | 0.0–0.0% | 0.0–8.6% | 0.0% |
| 1→3 | destination_trained | 0.0–0.0% | 0.0–0.8% | 0.0–0.0% | 0.0% |
| 1→4 | source_trained | 0.0–0.0% | 0.0–0.0% | 0.0–3.9% | 0.0% |
| 1→4 | destination_trained | 0.0–0.0% | 0.0–0.0% | 0.0–0.0% | 0.0% |
| 2→3 | source_trained | 9.4–28.1% | 12.5–35.2% | 31.2–60.2% | 64.1% |
| 2→3 | destination_trained | 7.0–22.7% | 12.5–32.0% | 28.1–44.5% | 58.6% |
| 2→4 | source_trained | 10.9–31.2% | 12.5–31.2% | 32.8–59.4% | 65.6% |
| 2→4 | destination_trained | 12.5–28.1% | 12.5–35.2% | 39.1–59.4% | 68.8% |
| 3→4 | source_trained | 62.5–64.1% | 51.6–67.2% | 82.8–87.5% | 87.5% |
| 3→4 | destination_trained | 52.3–65.6% | 60.9–73.4% | 75.0–78.1% | 75.0% |

## 4. Source-trained defense experiments

Four rules: original pooled threshold; per-chip calibration threshold with raw nearest-chip matching; chip-normalized matching and threshold; and seed-normalized threshold with raw matching. Chip/seed scales are median leave-one-out distances from source registration samples, floored at one. All cutoffs use only source calibration. No defense is tuned on later evaluation outcomes.

The table uses p95. Spoof rate is the highest observed across the tested fixed-single and seven-decoy families, both training modes and multipliers 1/1.5/2. It is not a pooled average or a deployable best-attack selection policy.

| Rule | Minimum / median / maximum genuine acceptance across six forward pairs | Minimum / median / maximum worst spoof acceptance |
|---|---:|---:|
| pooled | 68.0% / 83.2% / 96.1% | 0.0% / 32.0% / 87.5% |
| chip_threshold | 57.0% / 75.0% / 87.5% | 12.5% / 18.4% / 24.2% |
| chip_normalized | 59.4% / 77.3% / 93.0% | 0.0% / 6.2% / 73.4% |
| seed_normalized | 70.3% / 85.2% / 96.1% | 3.1% / 32.8% / 87.5% |

Paired defense results at all three source-calibrated percentiles are in defense_experiments.json. The ranges above do not imply that minimum genuine and maximum spoof values occur in the same comparison. Per-chip p95 is particularly uncertain because it uses 16 calibration scores. These candidate defenses are exploratory and evaluated against the existing attacks, not adaptively redesigned attacks.

## Scope

All analyses share the same four short-timescale collections, chips and seeds. They do not establish transport, power-cycle or long-term stability. Split and assignment ranges are descriptive, not confidence intervals. The balanced-split spoof study uses x1 source-trained attacks only; the defense study includes the original three multipliers and both training modes.

## Validation

All 16 default genuine cross-collection comparisons, all 390 pooled-defense model/percentile comparisons, and all 30 default balanced-split model controls reproduce the previously verified analyses. All role-separation checks pass. Counts and checks are saved in validation.json. These are computational consistency checks, not new independent measurements.
