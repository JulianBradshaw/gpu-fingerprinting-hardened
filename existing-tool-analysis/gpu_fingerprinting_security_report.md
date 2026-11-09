# GPU Fingerprinting for Location Verification: Empirical Security Assessment

7 November 2026

## Summary

This report examines the fingerprinting function released with *GPU Fingerprinting for Location Verification* and evaluates whether its responses can support reliable authentication against replay and modeling attacks.

The primary evidence consists of four complete collections on the same eight NVIDIA H200 GPUs, obtained over approximately thirteen minutes across two rental instance IDs. Offline analysis verified the downloaded arrays, reproduced the recorded results, evaluated registration across collections, swept acceptance thresholds, varied sample splits, compared decoy ensembles, and tested several simple defenses.

The principal findings are:

- **Challenge generation has a structural weakness.** Racing-thread selection depends on only six seed bits. Higher bits affect a delay term, so the complete response does not collapse to 64 fingerprints. Responses for related seeds nevertheless frequently win identification, and pass a distance threshold in some evaluated settings.
- **Genuine-chip identification remains strong under the original split.** Nearest-match accuracy is 94.5–100% across cross-collection comparisons, while adding an absolute-distance threshold can substantially reduce genuine acceptance. This distinguishes successful identification from reliable acceptance calibration; it does not establish resistance to forged responses.
- **Calibration is highly sensitive to sample choice.** The original p95 thresholds range from 41.7k to 153.2k across collections. Reassigning registration, calibration and evaluation roles within a single collection produces equally substantial variation. This does not isolate physical drift over time as the cause.
- **Genuine failures are concentrated.** Three of eight chips account for 135 of 138 genuine rejection events in the original chronological cross-collection comparisons.
- **Modeling attacks can achieve substantial acceptance.** A learned correction applied to live decoy responses sometimes passes a frozen verifier at high rates. Single-decoy and multi-decoy results differ materially; high ensemble success must not be attributed to one decoy.
- **The evaluated verifier exhibits security–usability tradeoffs whose severity depends on sample assignment.** The original split produces severe tradeoffs. In the narrower balanced-split study of source-trained ×1 attacks, the lowest achievable worst spoof acceptance at a 95% genuine requirement ranges from 14.1% to 99.2% where that requirement is attainable. These results support calibration fragility without establishing a universal failure of threshold-based verification.

These results identify weaknesses in the evaluated matching and threshold rules. They do not demonstrate an end-to-end location-verification bypass or establish that every fingerprint-based verifier must fail.

## 1. Scope and evidence

The paper presents a proof-of-concept GPU fingerprint and reports small-scale re-identification results. It explicitly leaves environmental stability and adversarial security as research questions. Identification among registered chips is a necessary component of authentication, but does not establish that an adversary cannot synthesize an acceptable response or conceal a relocated chip's response latency.

The primary datasets are:

| Collection | Start on 7 November, Mountain time | Rental instance |
|---|---|---|
| Run 1 | 8:47:06 PM | 59660701 |
| Run 2 | 8:51:46 PM | 59660701 |
| Run 3 | 8:56:41 PM | 59661478 |
| Run 4 | 9:00:27 PM | 59661478 |

All four identify the **same eight GPU UUIDs**, in the same order, and contain different raw measurements. They are four collections, not four independent sets of chips. The two rental IDs do not establish a power cycle or other environmental intervention.

Each collection contains 32 seeds and ten measurements per GPU/seed: **2,560 fingerprints per collection, 10,240 total**. Sixteen seeds sample the effective 31-bit LCG state space. The remaining sixteen form four residue groups, each containing `b`, `b+64`, `b+2^16` and `b+2^24`, for bases 0–3. The same seed design is reused across collections.

The primary collections use the same batched implementation and source hashes. The device-function source matches the supplied original kernel; the batched host loop keeps a CUDA context alive and resets the global counter before every launch. Comparisons here do not establish numerical equivalence to every cold-start implementation or hardware configuration.

Additional evidence includes the released research code and data-analysis scripts, single-H200 seed and timing logs, and an earlier collection with a complete saved analysis but incomplete raw-data download. That earlier collection used different GPU UUIDs and is excluded from the fully recomputed four-collection results.

Hardware measurements were supplied with the study. The subsequent analyses described here used saved local files; no new GPU experiments or remote operations were performed to produce this assessment.

## 2. Verification and attack methods

### Registration and acceptance

The default split assigns measurements 0–7 to registration, measurement 8 to evaluation, and measurement 9 to calibration. These are distinct launches under the collection's conditions, not independent environmental trials.

A candidate's score for a chip is its minimum L1 distance to that chip's eight registered fingerprints for the challenge seed. Joint acceptance requires:

1. The claimed chip is the uniquely closest registered chip; equal-distance ties are rejected.
2. The candidate's distance to that chip is within the acceptance threshold.

Pooled p90, p95 and p99 thresholds are estimated from the source collection's genuine calibration scores. The threshold gate is part of the evaluated verifier, not an established complete security protocol from the paper. A calibration percentile does not guarantee that percentage of joint acceptance on separate evaluation data.

For cross-collection tests, collection X supplies the registry and calibration. Collection Y supplies evaluation responses. Chips are aligned by UUID, and **Y is not used to recalibrate the fixed operating points**.

Each uniform-seed comparison contains 128 target/seed challenges. Chronological forward comparisons are 1→2, 1→3, 1→4, 2→3, 2→4 and 3→4. Reverse comparisons are useful diagnostics but are not chronological deployment simulations.

### Decoy modeling

For each target/decoy pair, the attacker learns an elementwise correction: the median difference between their registration means over training seeds. The challenged seed is excluded, leaving fifteen training seeds.

A single-decoy response is the fresh decoy evaluation response plus that correction. Ensemble responses take the elementwise median of corrected predictions from several decoys. Multipliers of 1, 1.5 and 2 are evaluated. Synthetic responses are explicitly truncated toward zero before L1 scoring.

Two training conditions are distinguished:

- **Source-trained:** corrections use X's registration data and remain fixed when evaluating later collections.
- **Destination-trained:** corrections use Y's registration data on other seeds. This models access to refreshed target/decoy training responses under Y's conditions; it does not establish that such access is chronologically available before every challenge.

The default single decoy is selected in advance by cyclic GPU order. The default ensemble uses all seven other GPUs. Separate ablations vary that hardware requirement. Training uses the verifier's registration samples on the training seeds; independently collected attacker profiles remain an evaluation gap. The held-out target response for the challenged seed is not used to construct its spoof.

## 3. Seed structure and related-seed replay

The kernel initializes an LCG with `seed ^ smid`, masks updates to 31 bits, and selects a racing thread with `lcg_state % 64`. The low six bits evolve independently of higher bits. Seeds equal modulo 64 therefore produce the same thread-selection sequence on a given SM.

The delay-loop length, `100000 + lcg_state % 17`, still depends on higher state bits. Consequently, the code does **not** establish only 64 complete responses. The highest uint32 seed bit is discarded at the first masked update, which is why the primary random challenges sample effective 31-bit states.

Single-H200 measurements report mean L1 distances of:

| Comparison | L1 distance |
|---|---:|
| Same seed, repeated runs | 12,385 |
| Different seeds equal modulo 64 | 116,620 |
| Seeds with different residues | 1,875,854 |

These sampled comparisons support low-bit dominance while showing a higher-bit effect above repeat noise. They do not quantify the number of challenges distinguishable by a calibrated verifier.

Replay tests submit either an unchanged recorded response for base seed `b`, or the truncated mean of eight such responses, for a different challenge seed sharing its residue. Only four residue bases and their specified variants are tested: a complete 64-entry replay table on arbitrary challenges is not demonstrated.

## 4. Within-collection results

The following results use each collection's original split and its own p95 calibration. Genuine and decoy percentages are joint acceptance over 128 challenges. Each replay column combines 96 attempts across the four residue bases.

| Run | p95 threshold | Genuine | Single decoy ×1 | Seven-decoy ×1 | Recorded replay | Mean-of-eight replay |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 41,725.5 | 92.2% | 0.0% | 8.6% | 0.0% | 0.0% |
| 2 | 67,486.9 | 92.2% | 24.2% | 68.0% | 1.0% | 0.0% |
| 3 | 153,213.9 | 95.3% | 68.0% | 87.5% | 91.7% | 83.3% |
| 4 | 79,429.9 | 96.9% | 28.9% | 75.0% | 3.1% | 3.1% |

Success varies sharply. For example, replay frequently identifies the right chip even when the distance gate rejects it. Nearest-match success alone is therefore insufficient to establish successful authentication.

Higher calibrated thresholds coincide with higher attack acceptance in these collections, but threshold movement is not isolated as the sole cause: registry samples, learned corrections and challenge realizations also differ.

## 5. Frozen verification across collections

With the source registry, correction and p95 threshold fixed, genuine and source-trained ×1 spoof acceptance are:

| Register → verify | Genuine | Single decoy | Seven-decoy ensemble |
|---|---:|---:|---:|
| 1→2 | 68.0% | 0.0% | 0.0% |
| 1→3 | 76.6% | 0.0% | 0.0% |
| 1→4 | 70.3% | 0.0% | 0.0% |
| 2→3 | 96.1% | 23.4% | 64.1% |
| 2→4 | 91.4% | 25.0% | 65.6% |
| 3→4 | 89.8% | 64.1% | 87.5% |

Across all twelve directed cross-collection comparisons, nearest-chip identification without a distance threshold remains 94.5–100%. Genuine joint acceptance can nevertheless be substantially lower. Run 1's registry rejects approximately 23–32% of genuine challenges from later collections.

The comparisons are asymmetric because changing the source changes both the registry and the threshold. The findings establish sensitivity of the transferred acceptance rule, not a uniform collapse of identification or a proven environmental cause.

## 6. The observed security–usability tradeoff

An exhaustive threshold sweep evaluates every observed score breakpoint. For each source registry, a retrospective bound finds the smallest **single** threshold accepting at least 95% of genuine challenges in **every** later collection.

This threshold is allowed to inspect later genuine evaluation scores. It is an optimistic diagnostic bound, not independently validated deployment calibration. Since acceptance is monotone in the threshold, raising it further cannot reduce any tested attack's acceptance.

| Frozen registry | Later collections | Smallest threshold meeting the genuine requirement | Worst source-trained spoof | Worst destination-trained spoof |
|---|---|---:|---:|---:|
| Run 1 | 2, 3, 4 | 473,140 | 100.0% | 93.0% |
| Run 2 | 3, 4 | 113,648 | 75.0% | 86.7% |
| Run 3 | 4 | 293,020 | 87.5% | 99.2% |

“Worst” is the maximum observed rate across the tested single/seven-decoy models, multipliers and later collections. It is not an average or a validated strategy for selecting attacks after deployment. These numbers must not be described as single-decoy success rates.

Conversely, choosing the largest threshold rejecting every tested spoof in all later collections leaves minimum genuine acceptance of **68.0%, 83.6% and 80.5%**, respectively. Thus, under the original split, no threshold achieves both 90% genuine acceptance in every later collection and zero observed acceptance of all tested spoofs.

This is a finite-sample result for the specified scalar-distance rule and attack set. It is not an impossibility theorem for other scoring rules, calibration methods or hardware.

## 7. Sample-split sensitivity

All ninety ordered choices of calibration and evaluation measurements were tested for genuine verification, leaving the other eight for registration. The p95 threshold varies substantially even within a single stored collection:

| Collection | Original p95 | Minimum / median / maximum across 90 splits |
|---|---:|---:|
| Run 1 | 41,726 | 41,726 / 47,378 / 171,011 |
| Run 2 | 67,487 | 58,287 / 101,961 / 192,961 |
| Run 3 | 153,214 | 41,683 / 76,672 / 236,679 |
| Run 4 | 79,430 | 53,724 / 92,985 / 188,793 |

In the original split, calibration medians remain approximately 19.3–22.0k while p95 ranges from 41.7–153.2k. The prominent variation is in upper-tail behavior, not an established uniform rescaling of fingerprints.

Ten balanced cyclic splits additionally evaluate source-trained ×1 single/seven-decoy attacks. Each measurement serves once as calibration and once as evaluation.

| Source registry | Splits able to reach 95% genuine acceptance in every later collection | Minimum / median / maximum worst ×1 spoof rate at that requirement |
|---|---:|---:|
| Run 1 | 2/10 | 92.2% / 95.7% / 99.2% |
| Run 2 | 8/10 | 57.8% / 75.0% / 75.8% |
| Run 3 | 10/10 | 14.1% / 87.5% / 90.6% |

The other ten source/split configurations cannot meet the genuine requirement because of identification errors, even without a distance gate. The original extreme attack rates are therefore not stable properties of the chips.

These overlapping splits are sensitivity checks, not independent replications. They change registry samples as well as calibration and evaluation samples, and cannot separate sampling effects from within-collection operating-state variation. The balanced spoof study is narrower than the original threshold sweep: it tests source-trained ×1 attacks only.

## 8. Failures concentrate on a few chips

At the original fixed p95 operating points, the six forward comparisons contain 138 genuine rejection events:

| Chip UUID suffix | Rejection events | Share |
|---|---:|---:|
| `a73006f53df4` | 66 | 47.8% |
| `b8d095aa659e` | 41 | 29.7% |
| `538d1beb6b0a` | 28 | 20.3% |
| Other five chips combined | 3 | 2.2% |

Three chips account for **97.8%** of the rejection events. The same small group dominates the largest calibration scores. The most frequent individual seed contributes thirteen rejection events, approximately 9.4%, so no one seed dominates similarly.

A later challenge can be scored against multiple source registries; these counts are repeated rejection events, not 138 independent physical failures. Concentration identifies where further investigation would be useful, not the underlying hardware cause or a justification for excluding difficult chips.

## 9. How many decoys are needed?

The default split was evaluated with one, two, four and seven decoys at ×1 correction. One-, two- and four-decoy cases use seven predetermined cyclic assignments; the seven-decoy ensemble has one assignment. All seven individual decoy choices are covered, but multi-decoy subsets are not exhaustive. Assignments were not selected using their test performance.

For source-trained attacks at the original frozen p95 thresholds:

| Register → verify | One decoy | Two decoys | Four decoys | Seven decoys |
|---|---:|---:|---:|---:|
| 2→3 | 9.4–28.1% | 12.5–35.2% | 31.2–60.2% | 64.1% |
| 2→4 | 10.9–31.2% | 12.5–31.2% | 32.8–59.4% | 65.6% |
| 3→4 | 62.5–64.1% | 51.6–67.2% | 82.8–87.5% | 87.5% |

Run 1's tighter frozen threshold rejects all tested one- and two-decoy source-trained spoofs in later collections, although some four-decoy assignments pass at rates up to 8.6%.

Additional decoys can improve acceptance substantially, but the effect is not strictly monotonic. High attack acceptance does not always require seven decoys, and neither a favorable single assignment nor a strong ensemble should be generalized to every hardware pairing. These results do not measure attack latency or compute cost.

## 10. Simple source-trained defenses

Four rules were evaluated using only the source collection to determine scales and thresholds:

- **Pooled threshold:** the baseline raw nearest-chip rule and pooled calibration cutoff.
- **Chip-specific threshold:** raw nearest-chip identification, with a separate calibration cutoff for each claimed chip.
- **Chip-normalized distance:** divide each chip's score by its source registration variability scale, then identify and threshold in normalized space.
- **Seed-normalized threshold:** normalize by source registration variability for the seed while retaining raw nearest-chip identification.

Normalization scales are median registration-only leave-one-out distances. Thresholds use the separate source calibration measurements. Later evaluation outcomes are not used to tune them. All original fixed-single/seven-decoy models, both training conditions and multipliers 1/1.5/2 are evaluated at calibration p90, p95 and p99.

No tested rule consistently repairs the weakness. Examples at p95 illustrate the mixed results:

| Comparison and rule | Genuine acceptance | Highest tested spoof acceptance |
|---|---:|---:|
| 2→4, pooled | 91.4% | 68.8% |
| 2→4, chip-normalized | 92.2% | 12.5% |
| 3→4, pooled | 89.8% | 87.5% |
| 3→4, chip-normalized | 89.1% | 73.4% |
| 3→4, chip-specific threshold | 84.4% | 23.4% |

Chip normalization can improve both genuine acceptance and spoof rejection in a particular comparison. Elsewhere it reduces genuine acceptance or still admits many spoofs. Chip-specific thresholds can reduce spoofing while introducing substantial genuine rejection. Seed normalization provides modest improvements in some comparisons without consistently resolving the tradeoff.

Per-chip p95 uses only sixteen calibration scores and is particularly uncertain. These are exploratory candidate defenses tested against the existing attack set, not evidence of robustness against adaptively redesigned attacks or all decoy assignments.

## 11. Timing and sleep-removal measurements

Thirty single-H200 process-level measurements report mean runtime **848.6 ms**, standard deviation **98.8 ms**, minimum/maximum **764.7/1,345.0 ms**, and p95−p5 spread **120.6 ms**. These include CUDA initialization and do not isolate kernel runtime or a warm-context verifier. The difference from the paper's approximately 2.9 seconds per run needs a matched measurement-boundary and configuration comparison.

Removing reset-round sleep calls reduces the measured median process runtime by approximately **42.9 ms** and changes a fingerprint by L1 **180,086**, compared with a genuine repeat distance of **11,624**. The fingerprint comparison uses one original/modified pair and one repeat-noise pair; it is exploratory.

The sleep calls affect the fingerprint, but this does not determine whether the modified response would pass a calibrated verifier. Removing them also changes race structure and is not a clean proxy for temperature or clock variation.

No usable relay-time budget or geographic bypass distance is established. That requires a specified verifier deadline, kernel-only timing and an end-to-end adversarial measurement.

## 12. Evidence quality and limits

All 32 primary compressed GPU archives passed their saved SHA-256 checks. Recomputed per-trial scores and operating-point summaries match the saved analyses. Default controls also match throughout the extended analyses: all sixteen genuine source/verification combinations (including within-collection controls), 390 pooled-defense model/percentile comparisons, and thirty balanced-split model controls.

The complete downloads permit recomputation, unlike the earlier partial transfer. A remote metadata status of `complete` is not evidence that a local download contains every required array. No runners were executed during the offline reviews; connection and transfer reliability are operational issues separate from the measured fingerprint-security results.

The principal limits are:

- Eight primary chips on one host, one GPU model, sixteen random uniform-seed challenges, and approximately thirteen minutes between collection starts.
- No established power-cycle, transport or controlled environmental intervention between primary collections.
- Shared chips, seeds, training data and overlapping sample splits. Rates and split ranges are descriptive, not independent-trial confidence estimates or statistical-equivalence results.
- Source-trained attacker profiles reuse registration samples on training seeds. Independent attacker profiling and separate-host decoys remain untested.
- Retrospective threshold bounds deliberately inspect evaluation scores; they demonstrate an observed tradeoff, not a successful calibration procedure for future use.
- Tests cover the stated distance rules and attacks, not every response-consistency check, defense or adaptive adversary.
- Only four replay residue bases are tested. No complete challenge-space enumeration or arbitrary-seed replay guarantee is established.
- Fingerprint acceptance is evaluated separately from response deadlines; no practical location-protocol bypass is demonstrated.

## 13. Possible followups

- Collect more repeats under steady conditions and compare larger enrollment/calibration sets against a separate later evaluation block.
- Study the three difficult chips alongside stable controls while recording temperature, clocks, power and background activity.
- Freeze registration and evaluate genuine and spoof responses after deliberate operating-condition changes or a confirmed same-chip power cycle.
- Repeat decoy modeling with targets and decoys on separate hosts, including predetermined small ensembles and independently collected attacker profiles.
- Compare the original thread selector with a well-mixed seed/SM/round generator, re-enrolling each implementation and measuring genuine verification, replay and modeling attacks.

## References and supporting material

- Tee and Happel, [*GPU Fingerprinting for Location Verification*](https://arxiv.org/html/2605.01930v2), particularly the requirements, evaluation and limitations sections.
- [Released code and data](https://github.com/waynetee/gpu-fingerprinting).
- Primary local collections: `campaign_v5_data_59660701_20261108T034630Z`, `campaign_v5_data_59660701_20261108T035137Z`, `campaign_v5_data_59661478_20261108T035611Z`, and `campaign_v5_data_59661478_20261108T040018Z`.
- Reproducible analyses: `cross_collection_verification.json`; `frozen_registry_tradeoff/results.json`; and the split, attribution, ablation, defense and validation JSON files in `extended_offline_analysis`.
- Analysis implementations: `frozen_registry_tradeoff.py` and `extended_offline_analysis.py`. The accompanying analysis bundle contains the extended results, scripts and reference scores; raw GPU datasets are retained separately.
