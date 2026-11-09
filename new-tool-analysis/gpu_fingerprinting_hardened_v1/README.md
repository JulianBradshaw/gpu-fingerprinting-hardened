# GPU fingerprinting: hardening research variant v1

This is a **new experimental probe and local verification toolkit**, built in response to the empirical assessment and supplied hardening checklist. It leaves the original probes, analyses and runners unchanged.

It implements concrete hardening mechanisms. It is **not a validated secure location-verification system**. The CUDA probe builds on Windows for both sm_75 and sm_90, but has not been executed on a GPU. Its genuine identification quality and spoof resistance require fresh measurements. The old fingerprints and thresholds are incompatible with this format.

## What changes

- Full 256-bit challenges from `secrets`, expanded with the standard 20-round ChaCha20 block function and separate SM/round domains. No 31-bit LCG or low-six-bit seed partition is retained.
- Explicit `all` and `subset` profiles. Subset mode selects 4–8 physical SMs per round using challenge-derived priorities. Each round also selects one of sixteen spaced atomic-counter addresses. This varies contention; it does not give software control of physical L2 slices.
- A persistent process, preallocated device buffers, two startup warm-up launches, and 32 warm-up rounds per launch. Each request produces three independent launches in the same warm context, with CUDA-event times and host measurement time recorded separately.
- No deliberate sleep calls. The revised race includes public challenge-dependent delay work and synchronization. It is not claimed to be irreducible or immune to faster implementation.
- Physical SM mapping is recorded, aligned by actual ID, checked for uniqueness and checked for changes. Unsupported block placement fails instead of writing overlapping or misindexed results.
- Responses must have the expected shape, active-SM mask, inactive sentinel and per-round rank permutation. Invalid values, fractional ranks, mismatched profiles and ties are rejected.
- Separate registration, calibration and evaluation records; duplicate sample identities are rejected. The minimum registration budget is eight **bursts** per chip/challenge, i.e. 24 kernel fingerprints. Calibration needs at least 32 bursts per chip and coverage of every enrolled challenge. More calibration is advisable for tail estimation.
- The actual two-of-three burst statistic is calibrated. A target is eligible for research verification only after separate held-out genuine evaluation: at least 32 evaluation bursts, coverage of every challenge, and calibration/evaluation joint acceptance meeting the configured nominal target. This is a genuine-quality gate, not a spoof-resistance certificate.
- Raw mean-active L1 scoring is the default. A registration-trained, MAD-scaled, clipped L1 variant is opt-in. Its parameters are versioned in the registry. Neither is assumed secure against modeling.
- Single-use, registry-bound, target-bound challenges tracked transactionally in SQLite. The first verification attempt consumes its ticket, even for malformed or incorrect responses. Issued nonces are never automatically reused.
- Telemetry snapshots, per-record flushing, failed-capture manifests, compressed templates, and offline evaluation that separates schema-invalid spoofs from valid-format attack failures.

The challenge is public after issuance. Using it as a ChaCha20 key provides diffusion, not a secret MAC or proof of hardware identity. Different challenges are not mathematically guaranteed to have distinct fingerprints.

## Files

| File | Purpose |
|---|---|
| `probe_hardened.cu` | Persistent CUDA probe; stdin challenge lines, JSON replies |
| `chacha20_core.h` | Portable CPU/CUDA ChaCha20 block implementation |
| `collect.py` | CPU-only challenge-pool generation or explicit local GPU collection |
| `fingerprint.py` | Schema checks, enrollment, metrics and two-of-three decision |
| `manage.py` | Local enrollment, single-use issuance and verification |
| `evaluate.py` | Offline genuine/attack scoring without live challenge-state changes |
| `crypto_selftest.cpp` | Host C++ crypto test and Python agreement helper |
| `test_hardening.py` | CPU-only functional tests with synthetic rank data and mocked collection |
| `build_windows.cmd` | Windows build helper selecting the CUDA-compatible VS2022 toolset |
| `CHECKLIST_REVIEW.md` | Which suggestions were implemented, modified or deferred |

## Build and local checks

Python 3.10+ and NumPy are required. A CUDA environment with a supported C++17 host compiler is needed to build the probe. For the intended H200 research host:

```text
python -m pip install -r requirements.txt
nvcc -std=c++17 -O2 -arch=sm_90 probe_hardened.cu -o probe_hardened
python -m unittest -v test_hardening
```

These are instructions, not commands executed on remote hardware during preparation. The unit suite invokes no CUDA or nvidia-smi calls. If a host C++ compiler is available, it compiles only `crypto_selftest.cpp`, never the CUDA probe. It tests the RFC 8439 vector and cross-checks CPU C++/Python block outputs. Tests use synthetic ranks; they do not establish PUF quality or security. Temporary test files are written beneath the OS temporary directory, or `FP_TEST_WORK` if set.

On the development Windows machine, Visual Studio 2022 Build Tools and CUDA 12.4 are installed. From PowerShell in this folder, use `./build_windows.cmd` for the local GTX 1660 Ti target (sm_75), or `./build_windows.cmd 90` for an H200-targeted Windows compile check. The helper selects MSVC 14.38 explicitly; the newer installed toolset remains available for other work. These commands compile only. For a later local capture, the binary name is `probe_hardened_sm75.exe`. A Windows executable does not run on a Linux H200 host; build the CUDA source there separately.

## Enrollment example

Generate the pool locally; this does not touch a GPU:

```text
python collect.py pool --count 32 --output challenges.json
```

Keep unissued challenges, templates and the verifier state private on the trusted enrollment/verifier side. The finite enrolled pool is an operational limitation: arbitrary unseen challenges cannot be verified without enrollment. A nonce issued for any target is burned globally in that state database, because revealing it would otherwise enable advance queries against another target.

The following examples **run GPU measurements if executed**. Replace `GPU-UUID` with the actual selected GPU UUID, create output directories beforehand, and repeat for each compatible GPU. They do not rent hardware or change clocks/power settings.

```text
python collect.py capture --binary ./probe_hardened --device GPU-UUID --pool challenges.json --phase registration --repeats 8 --output registration-gpu1.jsonl
python collect.py capture --binary ./probe_hardened --device GPU-UUID --pool challenges.json --phase calibration --repeats 1 --output calibration-gpu1.jsonl
python collect.py capture --binary ./probe_hardened --device GPU-UUID --pool challenges.json --phase evaluation --repeats 1 --output evaluation-gpu1.jsonl
```

Every capture keeps one process/context alive across its shuffled requests. It saves a manifest and stderr sidecar. NVML-exposed conditions are recorded through `nvidia-smi -q -x` before and after each three-launch measurement; sensor availability is reported, not fabricated. These snapshots do not establish constant conditions within a burst or trusted remote telemetry.

Use separate sessions/conditions for held-out validation as appropriate. The code does not synthesize thermal brackets or prove physical independence merely from phase labels. Cohorts currently require the same profile and sorted physical-SM ID set for direct coordinate comparison. Mismatched cohorts are rejected and must not be silently padded or remapped.

Enroll with explicit file paths for all GPUs:

```text
python manage.py enroll --input registration-gpu1.jsonl calibration-gpu1.jsonl evaluation-gpu1.jsonl registration-gpu2.jsonl calibration-gpu2.jsonl evaluation-gpu2.jsonl --output registry.json
```

For a separately evaluated robust-scoring ablation, use `--metric clipped_mad_l1` and a different registry output. Registration-only scales have a floor of one rank unit; normalized per-bin residuals are capped at six. Source calibration uses an order-statistic correction for the requested alpha, default 0.05. Its exchangeability assumptions can fail across operating conditions, and it provides no adversarial false-accept guarantee.

## Reference local challenge flow

Issue a challenge from an eligible target's enrolled unused pool:

```text
python manage.py issue --registry registry.json --state challenges.sqlite --target GPU-UUID --output request.json
```

The reference client can respond to that single request:

```text
python collect.py capture --binary ./probe_hardened --device GPU-UUID --request request.json --phase evaluation --repeats 1 --output response.json
python manage.py verify --registry registry.json --state challenges.sqlite --request request.json --response response.json
```

Request mode uses the profile in the request and emits one JSON object. The probe also supports a long-lived stdin/stdout integration for multiple requests; the one-request CLI is not a deployed network daemon. Its process startup must not be hidden when measuring end-to-end response latency.

The verifier chooses the expected target from its own issued state. It does not accept the response's `device_uuid` label or timing as proof of identity. The 120-second default expiry is only a freshness TTL, **not** a geographic or RTT bound. SQLite state must persist; resetting it can permit reuse of previously disclosed challenges. Keep it on the trusted verifier, not on the measured host.

There is no network listener, authenticated transport, hardware attestation verifier, server-side ping timing or geographic decision here. `--require-attestation` fails explicitly because an attestation provider is not implemented. Results always mark `attestation_verified` and `location_verified` false.

## Red-team evaluation

`evaluate.py` scores saved data without issuing/consuming live challenges. Genuine records use their trusted local `device_uuid` as the analysis target. Attack records should include `claimed_target` and `case_label` fields. Those labels are not trusted by the live verifier.

```text
python evaluate.py --registry registry.json --input later-genuine.jsonl modeled-attacks.jsonl --output evaluation-results.json
```

For a meaningful comparison, freeze a source registry and calibration, then evaluate later genuine responses and independent single-/multi-decoy attacks under the new format. Learn attacker corrections from training seeds only, including an independent-profile variant. Re-test the one/two/four/seven-decoy cases and the multipliers previously examined.

Schema-invalid ensemble medians are counted separately. A capable attacker can project predictions to valid rank permutations; rejecting unprojected malformed responses is not evidence of modeling resistance. Report both validity and joint acceptance, with genuine error rates at the same threshold. Do not interpret an observed zero attack count as a security guarantee.

## What was deliberately not done

- No noisy race-winner feedback: it could amplify genuine measurement noise. The checklist does not establish that it would preserve identification.
- No automatic clock locking, power limits, privileged host changes or cloud provisioning.
- No claim that the 3-chip tail behavior was caused by temperature, host interrupts or a specific SM mechanism.
- No requirement that new fingerprint distances equal the old 1.88M baseline; dimensions and score semantics changed.
- No fake hardware attestation, physical-location claim, irreversible-time assumption, or promise that zero sleeps eliminates relay opportunities.

## Validation status

CPU tests check cryptographic block correctness, sampled dependence on all 256 challenge bits, parsing and rank constraints, scoring configuration binding, disjoint sample identities, held-out evaluation isolation, two-of-three voting, single-use issuance including concurrency, expiry, request binding, pool exhaustion, attestation failure behavior, and mocked persistent collection/failure preservation.

The local Windows environment now has Visual Studio 2022 Build Tools and NVCC 12.4.99. The portable crypto core passed its MSVC build/test, and **the CUDA source compiled successfully for sm_75 and sm_90**. No GPU probe was executed. No hardware enrollment, spoofing, accuracy, timing or stability claim is made for this version. A Linux H200 build and runtime validation are still needed for the intended hardware experiments.

## References

- [RFC 8439: ChaCha20 block function and test vector](https://www.rfc-editor.org/rfc/rfc8439)
- [NVIDIA Cooperative Groups documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [Original GPU-fingerprinting research code](https://github.com/waynetee/gpu-fingerprinting) and [paper](https://arxiv.org/html/2605.01930v2)

This is a new research implementation informed by those sources and the empirical assessment, not a claim of endorsement by their authors.
