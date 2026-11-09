# Hardening checklist: implementation decisions

The supplied checklist contains useful engineering suggestions, but several proposed root causes and “fixes” go beyond the evidence. This variant implements a testable subset and keeps uncertain interventions explicit.

| Suggestion | Decision |
|---|---|
| Replace low-bit LCG | Implemented with full 20-round ChaCha20 block expansion and 256-bit random challenges. SplitMix/Murmur finalizers are not treated as cryptographic PRFs, and reduced-round ChaCha is not substituted for the standard block function. |
| Full-bit avalanche | CPU tests check that every challenge bit changes sampled lane/delay trajectories. This does not prove fingerprint independence, collision freedom, or hardware spoof resistance. |
| Match all bit flips to old L1 baseline | Requires new hardware data; the old numeric baseline is inapplicable to the new rank format. |
| Race-winner feedback | Deferred. Noisy feedback could destroy repeatability; a stable discriminator must be demonstrated before adding it. |
| Random SM subsets and addresses | Implemented as a 4–8-SM profile and sixteen spaced counter addresses. Software does not claim to select physical L2 slices. An all-SM profile provides a separate ablation. |
| Defeat additive/ensemble modeling | Not established. Varied contention is an experiment, not proof of nonlinear unclonability. Offline evaluation is supplied; new, valid-format adaptive attacks are required. |
| Zero spoof acceptance at p99 | A possible measured outcome, not an engineering guarantee. Sample size, false rejection and confidence limits matter. |
| Lock clocks | Not performed automatically. Hardware settings require a deliberate experiment and suitable supported frequencies. They may affect identification and cannot be guessed from this dataset. |
| Telemetry | Collection records before/after nvidia-smi XML and sensor availability. HBM/ECC/PCIe fields depend on the driver and device; unavailable counters are not invented. Telemetry is observational, not attestation. |
| Thermal/power-cycle enrollment | Supported through independently captured labeled phases and collection IDs, but requires real controlled measurements. No such experiment was run here. |
| Robust scoring | Optional registration-trained MAD-scaled clipped L1. Raw mean-active L1 remains the baseline. Existing evidence does not establish that clipping/weighting fixes security. Full Mahalanobis fitting is avoided with sparse, high-dimensional data. |
| Two-of-three burst voting | Implemented and calibrated on the second own-distance order statistic, with held-out joint validation. Launches share a context/conditions; their error probabilities are not assumed independent. |
| Remove sleeps | The new profile has no deliberate sleeps and requires new enrollment. The existing sleep-removal result did not establish usable relay slack or irreducible runtime. |
| Warm context / timing | Persistent buffers and context, CUDA-event kernel timing, separate host measurement time. No assumed “tens of milliseconds” figure; actual speed must be measured. |
| Hardware attestation | Not implemented. Requiring it fails explicitly. A real integration needs genuine challenge/channel binding, trusted verification and hardware support; an identity label is insufficient. |

Additional mechanisms include strict response semantics, complete physical-SM mapping checks, versioned scoring configuration, compressed templates, per-record durable capture, separate calibration/evaluation, held-out genuine eligibility gates, nonce/registry/target binding, transactionally enforced single-use challenges, and refusal to recycle exhausted challenge pools.

The checklist's statements that static linear offsets, thermal transitions or removable idle time explain the observed failures are hypotheses, not established root causes. The empirical calibration study uses 128 pooled scores per original collection, not sixteen; the sixteen-score limitation applies to the per-chip calibration experiment. This distinction matters when selecting an adequate follow-up collection size.

The deployment trust boundary remains unresolved: this toolkit is not an attested execution environment or a secure network location protocol. Windows CUDA compilation has passed for sm_75 and sm_90; Linux/H200 runtime validation and fresh held-out genuine/attack measurements, including cross-condition validation, remain necessary. Keep unchanged baselines for comparison.
