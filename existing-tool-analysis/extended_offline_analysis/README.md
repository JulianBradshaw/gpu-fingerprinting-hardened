# Offline analysis bundle

Start with `report.md`. The JSON files contain the full numeric summaries and breakdowns; `validation.json` records consistency checks against the earlier analyses.

Analyses performed:

1. All 90 genuine registration/calibration/evaluation role assignments, plus source-trained x1 spoofing on ten balanced cyclic assignments.
2. Genuine rejection and spoof acceptance by chip and seed, including large calibration-score attribution.
3. One-, two-, four- and seven-decoy comparisons using predetermined cyclic assignments.
4. Pooled, chip-specific, chip-normalized and seed-normalized verification rules with source-only calibration at p90, p95 and p99.

To reproduce, keep both Python files together, install numpy in your local analysis environment, and run:

```text
python extended_offline_analysis.py --data-root "PATH_TO_GPU_TEST_FOLDER" --reference reference_results.json --output new_analysis_output --cache work/raw_pairwise_distances.npz
```

The output directory must be new. `--resume` reuses existing phase outputs from an interrupted run; use it only with the same inputs and analysis version. The cache stores pairwise raw distances and verifies its input-data signature before reuse. No GPU or network operation is invoked.

The raw datasets are not bundled; use the four downloaded `campaign_v5_data_*` folders. `reference_results.json` is the earlier frozen-registry analysis used for attribution and consistency checks.

Split and assignment ranges are sensitivity summaries, not confidence intervals or independent replications. The report's opening findings and validation section summarize the completed computation; regenerated tables are produced by the analysis script.
