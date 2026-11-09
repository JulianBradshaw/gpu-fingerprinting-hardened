# Validation performed

Prepared locally on 7 November 2026. No GPU experiment, cloud job, clock/power change or remote command was run.

## Passed

- **23 CPU-only unit tests** with resource warnings treated as errors.
- Host C++ compilation of the portable crypto core using the available GCC 4.8.1 compiler in C++11 mode.
- RFC 8439 block-function vector in both C++ and Python, plus eight deterministic cross-language block comparisons.
- Sampled lane/delay-trajectory dependence on every one of the 256 input challenge bits.
- Response shape/rank/mask validation, scoring-version binding, duplicate-sample rejection, and isolation of evaluation data from fitted templates/thresholds.
- Two-of-three voting, ties, single-use and concurrent issuance, expiry, request binding, exhaustion, and explicit refusal of unavailable hardware attestation.
- Mocked persistent collection, failed-response manifest preservation, and a complete issued-request/capture/verification cycle using an in-memory fake probe.
- Separate CPU-only command-line smoke: challenge pool → synthetic enrollment → issuance → verification → offline evaluation.
- Subsequent Windows installation/build checks: Visual Studio 2022 Build Tools 17.14, compatible MSVC toolset 14.38.33130, Windows SDK 10.0.22621, and NVCC 12.4.99.
- Portable crypto self-test built with MSVC and passed the RFC block vector.
- `probe_hardened.cu` compiled and linked successfully for sm_75 and sm_90 on Windows. Neither executable was used for GPU measurements.

Environment used: Python 3.14, NumPy 2.4.2, Windows. Graphics driver 551.76 was preserved. Synthetic fixtures and successful compilation validate software behavior/buildability only; they do not establish physical fingerprint quality or attack resistance.

## Not performed

- Linux/H200 compilation and runtime validation.
- H200 execution, enrollment quality, replay/modeling red-teaming, timing, telemetry collection on hardware, thermal/power-cycle stability, or cross-host evaluation.
- Hardware attestation or geographic verification.

The package therefore remains an unvalidated hardening experiment. Fresh held-out hardware measurements are required before drawing security conclusions. Old probe datasets and thresholds cannot validate this new fingerprint format.
