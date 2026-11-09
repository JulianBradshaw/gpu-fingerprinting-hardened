# GPU Fingerprinting for Location Verification: Red-Team Findings and Hardened Probe Evaluation

9 November 2026

## Summary

This study evaluates whether a GPU race fingerprint can distinguish physical chips, accept genuine responses across collections, and reject modeled spoofs. It covers the legacy probe and a hardened research implementation. The main security comparison runs both implementations on the same eight H200s with separate enrollment, calibration, and later evaluation.

- **Hardened substantially reduced the tested decoy attack at comparable genuine acceptance.** At the pooled p95 cutoff on the matched eight-GPU cohort, seven-decoy acceptance fell from **224/256 (87.5%) to 25/256 (9.8%)**. Acceptance of the predetermined single-decoy attack fell from **138/256 to 0/256**. Genuine acceptance was **219/256 versus 221/256**.
- **The remaining hardened failures are concentrated.** GPU C accepted **19/32** seven-decoy responses while accepting **32/32** genuine responses. An exploratory subset sweep found two- and four-decoy combinations that each passed **26/32** on that target.
- **Legacy identification and acceptance calibration behave differently.** In the original four collections, nearest-chip accuracy was **94.5–100%** across collection pairs. Frozen joint acceptance fell as low as **68.0%** for later genuine responses. Original p95 cutoffs spanned **41.7k–153.2k** and varied substantially when the same stored samples were assigned different roles. Three GPUs accounted for **135 of 138** genuine rejection events in the six chronological comparisons.
- **The legacy challenge schedule has a structural low-bit weakness.** Equal residues modulo 64 produce the same racing-thread selections on each SM. Higher seed bits still influence the delay and the complete response. Recorded related-seed replay acceptance varied from **0% to 91.7%** in the original experiments.
- **Hardened cold local responses were faster in a separate matched two-GPU run.** Median process-start-to-exit time fell from **772 ms to 492 ms**, a **36% reduction**. Hardened was faster in all eight paired timing attempts.
- **An exploratory variability check is promising.** Adding a lower bound on variation among a burst's three launches reduced hardened seven-decoy acceptance from **25/256 to 0/256** on the saved eight-GPU data. Genuine acceptance changed from **228/256 to 225/256**. A single decoy still passed **3/32** on GPU C in the subset sweep.

The evidence supports a meaningful improvement in the evaluated fingerprint stack against the specified attacks. The hardware measurements cover short gaps and small cohorts. Location verification, trusted attestation, and adaptive attacks against the variability check require separate experiments.

## 1. Systems and decision rules

The underlying study presents a proof-of-concept GPU fingerprint and leaves environmental stability and adversarial security open. This report evaluates a verifier that requires both correct identification and a calibrated distance cutoff.

The **legacy** probe is the released race kernel exercised with a batched host harness. The harness keeps a CUDA context open across a batch and resets the global counter before each launch. A challenge uses an effective 31-bit LCG state. A response is one fingerprint. Its score is the minimum summed L1 distance to eight registration fingerprints for that challenge and claimed GPU.

The **hardened** probe expands a public 256-bit challenge with the 20-round ChaCha20 block function using separate SM and round domains. It replaces deliberate sleeps with challenge-dependent delay work and synchronization, varies contention across sixteen counter addresses, records the physical SM map, and emits per-round rank permutations. The `all` profile includes every mapped SM. The `subset` profile selects 4–8 active SMs per round. Public challenge expansion provides diffusion and uses no secret authentication key.

A hardened response contains three launches in the same warm process. The default score for each launch is the minimum mean active-coordinate L1 distance to the claimed GPU's registration templates for that challenge. Eight registration bursts provide 24 templates. Joint acceptance requires at least two launches on which the claimed GPU is both uniquely nearest and within its cutoff. Legacy applies the same two conditions to its single fingerprint. Ties are rejected. The cutoff is calibrated on genuine responses separate from registration and evaluation.

The hardened toolkit also implements local single-use challenge state bound to the registry and target. The experiments in this report score captured responses offline. The replay and modeling results describe the fingerprint decision, with live challenge state, network deadlines, and attestation outside the measurement.

## 2. Original eight-GPU assessment

### Collections and attacks

Four complete collections used the same eight plain H200 UUIDs in the same order over roughly thirteen minutes. Each collection contains 32 seeds and ten fingerprints per GPU and seed, for **2,560 fingerprints per collection and 10,240 total**. Sixteen seeds sample the effective 31-bit state space. Sixteen additional seeds form four groups of `b`, `b+64`, `b+2^16`, and `b+2^24`. The default split uses eight fingerprints for registration, one for evaluation, and one for calibration. The two rental IDs establish separate rentals. Power state and operating conditions were unverified.

| Collection | Start on 7 November, Mountain time | Rental instance |
|---|---|---|
| Run 1 | 8:47:06 p.m. | 59660701 |
| Run 2 | 8:51:46 p.m. | 59660701 |
| Run 3 | 8:56:41 p.m. | 59661478 |
| Run 4 | 9:00:27 p.m. | 59661478 |

The LCG uses `seed ^ smid` and chooses a racing thread through `lcg_state % 64`. The low six bits evolve independently of the higher bits. The delay term still depends on higher bits through `100000 + lcg_state % 17`. Single-H200 measurements gave mean L1 **12,385** for a repeated seed, **116,620** for different seeds with the same residue, and **1,875,854** for seeds with different residues. The measurements support low-bit dominance while showing a higher-bit effect above repeat noise. Related-seed replay submits either a recorded fingerprint or the truncated mean of eight recorded fingerprints for a different challenge in the same residue group. The replay experiment covers four bases and their specified variants.

The decoy model learns the coordinate-wise median difference between target and decoy registration means over the fifteen other uniform seeds. It adds that correction to a held-out decoy response for the challenged seed. The ensemble takes the coordinate-wise median of predictions from multiple GPUs. The legacy scorer truncates synthetic values toward zero. Source-trained attacks learn from the collection that supplied the frozen registry. Destination-trained attacks learn on other seeds in the later collection and model access to refreshed profiles.

### Acceptance and calibration

The within-collection results below use the original split and each collection's pooled p95. Genuine and decoy denominators are 128. Each replay column uses 96 attempts across the four residue groups.

| Run | p95 cutoff | Genuine | Single decoy ×1 | Seven decoys ×1 | Recorded replay | Mean-of-eight replay |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 41,725.5 | 92.2% | 0.0% | 8.6% | 0.0% | 0.0% |
| 2 | 67,486.9 | 92.2% | 24.2% | 68.0% | 1.0% | 0.0% |
| 3 | 153,213.9 | 95.3% | 68.0% | 87.5% | 91.7% | 83.3% |
| 4 | 79,429.9 | 96.9% | 28.9% | 75.0% | 3.1% | 3.1% |

For later verification, the earlier collection supplies registration and the cutoff. The verifier aligns GPUs by UUID and scores later responses with the frozen registry. The attack correction in the following table is also source-trained.

| Register → verify | Genuine | Single decoy ×1 | Seven decoys ×1 |
|---|---:|---:|---:|
| 1→2 | 68.0% | 0.0% | 0.0% |
| 1→3 | 76.6% | 0.0% | 0.0% |
| 1→4 | 70.3% | 0.0% | 0.0% |
| 2→3 | 96.1% | 23.4% | 64.1% |
| 2→4 | 91.4% | 25.0% | 65.6% |
| 3→4 | 89.8% | 64.1% | 87.5% |

Across all twelve directed cross-collection comparisons, nearest-chip accuracy ranged from 94.5% to 100%. Genuine rejection under the joint rule was concentrated. UUID suffixes `a73006f53df4`, `b8d095aa659e`, and `538d1beb6b0a` account for 66, 41, and 28 of the 138 forward-comparison rejection events. One later response can be scored against several registries, so these are repeated events.

### Threshold and sample-split sensitivity

For the original split, a retrospective sweep chose the smallest single cutoff that accepted at least 95% of genuine responses in every later collection. Even with access to those later genuine scores, the resulting worst source-trained spoof rates were **100.0%, 75.0%, and 87.5%** for Runs 1–3. The worst destination-trained rates were **93.0%, 86.7%, and 99.2%**. In the reverse calculation, the largest cutoff rejecting every tested spoof left minimum genuine acceptance of **68.0%, 83.6%, and 80.5%**. These bounds apply to the observed split, metric, and attack set.

Across all 90 ordered choices of calibration and evaluation fingerprint within a collection, p95 ranged from **41.7k–171.0k** in Run 1, **58.3k–192.0k** in Run 2, **41.7k–236.7k** in Run 3, and **53.7k–188.8k** in Run 4. Original calibration medians were much closer, at roughly **19.3k–22.0k**. Upper-tail behavior changed most visibly. The experiment leaves physical drift, operating conditions, and sample choice as possible contributors.

Ten balanced splits also tested source-trained ×1 spoofing. Of 30 source/split combinations, 20 could achieve 95% genuine acceptance in every later collection. Among those, the smallest achievable worst spoof rate ranged from **14.1% to 99.2%**. Identification errors prevented the other ten from reaching the genuine requirement even with an open distance gate. The original split demonstrates a severe tradeoff, and the balanced splits show that its severity depends on the samples.

Legacy ablations show why decoy count matters. For 3→4, tested one-decoy choices passed **62.5–64.1%**, two-decoy combinations **51.6–67.2%**, four-decoy combinations **82.8–87.5%**, and the seven-decoy ensemble **87.5%**. Other source registries gave lower rates. These subset choices used predetermined cyclic assignments.

Chip-specific cutoffs, chip-normalized distance, and seed-normalized thresholds were fit on source data. Their tradeoffs were mixed. On 2→4, chip normalization moved genuine acceptance from **91.4% to 92.2%** and the highest tested spoof rate from **68.8% to 12.5%**. On 3→4, the same rule yielded **89.1% genuine and 73.4% spoof**. A chip-specific cutoff in that comparison reduced the highest spoof rate to **23.4%**, with genuine acceptance of **84.4%**.

### Legacy timing controls

Thirty single-H200 cold-process measurements averaged **848.6 ms**, with standard deviation **98.8 ms**, range **764.7–1,345.0 ms**, and a p95–p5 spread of **120.6 ms**. Removing reset-round sleep calls reduced the measured median by approximately **42.9 ms** and shifted one tested fingerprint by L1 **180,086**, versus a genuine repeat distance of **11,624**. The latter comparison uses one modified pair and one genuine repeat pair.

## 3. Hardened engineering pilots

The single-H200 NVL run on 8 November completed **388 bursts including smoke** with a stable 132-SM map. Both modes met the prespecified **58/64** engineering screen for own-distance consistency across fresh processes. `all` accepted **61/64**, and `subset` accepted **58/64**. The weakest challenges accepted **6/8** and **4/8**, respectively. Median warm probe time per burst was **38.29 ms** for `all` and **36.18 ms** for `subset`. These times exclude process startup and transport.

The two-H200 NVL pilot on 8 November completed **776 bursts including smoke**. Evaluation began approximately **10–11½ minutes** after calibration was frozen. Both modes selected the correct nearest chip on all **128** held-out bursts and accepted **0/128** wrong-chip trials. Genuine joint acceptance was **123/128** for `all` and **127/128** for `subset`. Five of the six misses came from one GPU in `all`. Its highest genuine score was **0.349**, its cutoff **0.343**, and the closest wrong-chip score **0.946**.

In an exploratory rescore of that pair, the raw per-chip calibration maximum increased genuine acceptance to **127/128** for `all` and **128/128** for `subset`. A pooled p95 improved `all` to **128/128** but reduced `subset` to **122/128**. Registration-trained MAD normalization with the same per-chip p95 rule changed neither mode's genuine count. All tested variants accepted **0/128** wrong-chip trials per mode.

## 4. Matched legacy versus hardened comparison on two H200 NVLs

On 9 November, both implementations ran on the same two H200 NVLs with alternating order. Each arm captured **516 responses including smoke**. The probes retained their native response sizes and were enrolled independently. Evaluation began **11½–13 minutes** after the freeze. Cutoffs used calibration data only.

| Outcome | Legacy | Hardened `all` |
|---|---:|---:|
| Genuine joint acceptance, per-chip p95 | 118/128 (92.2%) | 123/128 (96.1%) |
| Genuine joint acceptance, per-chip calibration maximum | 123/128 (96.1%) | 126/128 (98.4%) |
| Correct nearest chip | 128/128 | 128/128 |
| Wrong-chip acceptance, either cutoff | 0/128 | 0/128 |
| Paired cross-challenge replay acceptance, either cutoff | 0/64 | 0/64 |
| Median cold local response, eight paired attempts | 772 ms | 492 ms |

Cold timing ran from process start through exit for one native response, including CUDA setup and stdout. Hardened was faster in all eight paired attempts. These timings cover the one-launch and three-launch implementations, with SSH and verifier scoring outside the measurement.

Both arms accepted most cached responses to the original challenge at the offline scorer. The cross-challenge test relabeled an independently captured early response to a paired challenge. The corresponding legacy seeds differed by bit 6. Both arms rejected all those attempts. In an offline single-decoy check, both arms also accepted **0/128** at each frozen cutoff and tested correction multiplier. At ×1, the legacy prediction ranked the target first **128/128** times versus **0/128** for hardened. Larger corrections could win nearest for both arms, and the distance gate rejected those cases on this pair.

## 5. Matched eight-H200 ensemble test

The main security comparison used eight **plain H200s** on one host on 9 November. Both probes ran on the same eight UUIDs. This cohort differs from the historical four collections. Both arms used sixteen challenge slots, preserving the original campaign's sixteen uniform legacy seeds and mapping them to full 256-bit hardened challenges. Each GPU and challenge supplied eight registration responses, three calibration responses, and two evaluation responses per arm. Each arm captured **1,728 responses including smoke**. Registration and calibration inputs were sealed before the five-minute wait. Held-out capture started approximately **300–396 seconds** after sealing.

The ×1 attack uses the other fifteen seeds for each correction and a held-out decoy response on the challenged seed. The predetermined single-decoy assignment uses the next GPU in cohort order. The seven-decoy attack uses all seven alternatives. Legacy predictions use truncation. Clipping to the legacy counter bounds gave the same seven-decoy counts. Hardened predictions are projected to valid per-round rank permutations and evaluated under the normal two-of-three rule. Each arm retains its native response and matching rules, so the comparison measures the stack as a whole.

| Held-out outcome, pooled p95 | Legacy | Hardened `all` |
|---|---:|---:|
| Genuine joint acceptance | 219/256 (85.5%) | 221/256 (86.3%) |
| Correct nearest chip | 219/256 (85.5%) | 232/256 (90.6%) |
| Single-decoy acceptance | 138/256 (53.9%) | 0/256 (0.0%) |
| Seven-decoy acceptance | 224/256 (87.5%) | 25/256 (9.8%) |
| Wrong physical chip accepted | 37/1,792 (2.1%) | 17/1,792 (0.9%) |

The pooled p95 uses the historical interpolated percentile separately in each arm. The per-chip p95 uses the 47th-smallest of 48 calibration scores for each chip. Both rules use only the sealed calibration data. At pooled p95, ensemble acceptance fell by **77.7 percentage points** with a **0.8 percentage-point increase** in genuine acceptance.

At the per-chip p95 rule, genuine acceptance was **201/256 legacy versus 228/256 hardened**, and seven-decoy acceptance was **122/256 versus 25/256**. Wrong-chip acceptance was **12/1,792 versus 15/1,792** at that rule. At each chip's calibration maximum, genuine acceptance was **207/256 versus 229/256** and ensemble acceptance was **181/256 versus 25/256**. The ensemble improvement persists across the tested frozen cutoff policies. Wrong-chip acceptance depends on the policy.

At the hardened per-chip p95, GPU C accepted **19/32** ensemble responses and all **32/32** genuine bursts. GPUs E and G accounted for the remaining **2** and **4** accepted ensemble responses. GPU E accepted **20/32** genuine bursts under that hardened rule. Some failures arose before applying a cutoff. The legacy arm selected the wrong nearest GPU for all 32 held-out responses from E.

## 6. Exploratory offline follow-up

### Decoy subsets and response structure

An exhaustive sweep evaluated all **71** subsets of sizes 1, 2, 4, 6, and 7 against GPU C. Counts use C's frozen per-chip p95 and 32 held-out requests per subset. The best observed subset at each size is a descriptive worst case.

| Decoys | Subsets | Minimum / median / maximum accepted | Maximum after variability check |
|---:|---:|---:|---:|
| 1 | 7 | 0 / 0 / 3 | 3 |
| 2 | 21 | 0 / 0 / 26 | 0 |
| 4 | 35 | 8 / 16 / 26 | 0 |
| 6 | 7 | 18 / 20 / 24 | 0 |
| 7 | 1 | 19 / 19 / 19 | 0 |

Removing any single decoy from the seven-card ensemble still gave at least 18 accepts. The strongest two-card combination, B and F, passed **26/32**. The strongest individual card, E, passed **3/32**. The original single-decoy result covers one predetermined assignment per target.

On C, all **19** successful seven-decoy bursts had a lower joint distance statistic than every one of its **32** genuine bursts. The smallest attack score was **0.265**, and the smallest genuine score was **0.307**. On this sample, lowering the upper cutoff enough to reject all seven-decoy spoofs also rejects all genuine C bursts. The ensemble was closer to typical templates as well, with median distance **0.287** versus **0.340**. Nearest-chip margins overlapped.

Launch-to-launch variation separated those successful spoofs from genuine bursts. For each burst, the variation score was the median of its three pairwise mean L1 distances. Across bursts, the median score was **0.188** for accepted spoofs versus **0.340** for genuine bursts. The ranges were **0.183–0.194** and **0.323–0.356**, respectively. The tested ensemble produced unusually similar launches.

### Candidate lower bounds

Two additional checks kept the existing upper cutoff and nearest-chip rule. The first rejected a burst when its own-distance statistic was below the smallest genuine calibration score for the target. The second rejected a burst when its launch-to-launch variation was below the smallest genuine calibration value for the target. The floors used genuine calibration only. The hypotheses came from examining the same pilot, so the results remain exploratory.

| Hardened rule | Genuine accepted across eight GPUs | Seven-decoy accepted across eight GPUs | Worst two-decoy result on C |
|---|---:|---:|---:|
| Per-chip p95 baseline | 228/256 | 25/256 | 26/32 |
| Add minimum distance | 221/256 | 0/256 | 21/32 |
| Add minimum variation | 225/256 | 0/256 | 0/32 |

The variation check blocked every tested multi-decoy subset against C, while GPU E acting alone still passed **3/32**. It retained **31/32** genuine C responses. Wrong-chip acceptance across the cohort stayed at **15/1,792**. The distance floor blocked the full ensemble but admitted some smaller ensembles. The variation check is the stronger candidate for prospective testing.

## 7. Evidence quality, limits, and possible follow-ups

All 32 original compressed GPU datasets passed their saved SHA-256 checks, and local scoring reproduced the reported operating points. The single-GPU, two-GPU, and matched two-GPU archives were rescored locally. The matched eight-GPU archive matched the host's SHA-256, contained all 48 main captures, and passed local frozen-input and evaluation hash checks. Its captured probe error logs were empty.

The reported rates are descriptive. Trials share devices, challenges, and training samples, and the original split sweeps reuse data. The short collection gaps cover near-term repeatability on one host at a time. Controlled power cycles, transport, and longer-term conditions remain open. The native legacy and hardened response budgets differ, so the matched test evaluates the combined implementation. The two-GPU latency measurement covers the local cold process. Relay resistance and a geographic cutoff require verifier-side timing and separate controls.

1. **Prospectively test the variation check.** Freeze its statistic and threshold on a fresh enrollment. Evaluate independently captured genuine bursts and spoof variants that imitate genuine launch-to-launch variation. Report target-level genuine rejection and wrong-chip acceptance.
2. **Test stability and calibration over longer sessions.** Reuse identified physical chips across controlled idle, warm, restart, and longer-gap conditions. Compare fixed calibration policies at the same registration budget. Track the chips and challenges with concentrated genuine misses.
3. **Replicate the ensemble result on additional hosts.** Separate target and decoy hosts, collect independent attacker profiles, and predeclare small ensembles alongside the seven-card case. Report genuine and spoof acceptance under frozen cutoffs and label any held-out threshold sweep as descriptive.
4. **Separate the contribution of the hardening changes.** Use matched response budgets and ablations of challenge expansion, contention schedule, rank format, and burst voting. Use verifier-side timing and direct-versus-relayed controls for a separate location-protocol experiment.

## References and reproducibility

- Wayne Tee and Jonathan Happel, [*GPU Fingerprinting for Location Verification*](https://arxiv.org/html/2605.01930v2) and the [released code and data](https://github.com/waynetee/gpu-fingerprinting).
- Legacy raw collections are `campaign_v5_data_59660701_20261108T034630Z`, `campaign_v5_data_59660701_20261108T035137Z`, `campaign_v5_data_59661478_20261108T035611Z`, and `campaign_v5_data_59661478_20261108T040018Z`. Supporting analyses include `cross_collection_verification.json`, `frozen_registry_tradeoff`, and `extended_offline_analysis`.
- Hardened and matched raw archives are indexed by rental IDs **59832716** (single-GPU sanity), **59835170** (two-GPU pilot), **59934299** (matched two-GPU), and **59951659** (matched eight-GPU). Each archive retains the source, capture metadata, and logs relevant to its protocol.
- The matched eight-GPU scorer is `eight_h200_ensemble.py`. Its result summary and the 71-subset follow-up are in `eight_h200_ensemble_59951659_results.json` and `eight_h200_ensemble_followups.json`.
