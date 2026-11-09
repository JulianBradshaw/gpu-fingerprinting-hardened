#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

stage=${1:?stage required}
run=${2:?run directory required}
gap=${3:?gap in seconds required}
[[ $run =~ ^/workspace/fp-pilot-[0-9a-f]{32}$ && -d $run && ! -L $run ]]
[[ $gap =~ ^[1-9][0-9]{1,3}$ ]]
(( gap >= 60 && gap <= 3600 ))
variant=pilot
if (( $# >= 4 )); then variant=$4; fi
[[ $variant == pilot || $variant == matched || $variant == ensemble ]]
cd "$run"
trap 'rc=$?; printf "PILOT_FAIL: %s (exit %s)\n" "$stage" "$rc" >&2' ERR
exec 9>.stage.lock
flock -n 9

setup_host() {
    nvidia-smi -L | tee logs/devices.txt
    nvidia-smi -q -x > logs/host-before.xml
    inventory=$(nvidia-smi --query-gpu=name,uuid,mig.mode.current --format=csv,noheader,nounits)
    printf '%s\n' "$inventory" > logs/inventory.csv
    mapfile -t devices <<< "$inventory"
    count=2
    model='H200 NVL'
    if [[ $variant == ensemble ]]; then count=8; model=H200; fi
    if (( ${#devices[@]} != count )); then echo "Expected exactly $count visible GPUs." >&2; return 1; fi
    for device in "${devices[@]}"; do
        if [[ ! $device =~ ^(NVIDIA\ )?$model,\ (GPU-[0-9a-fA-F-]{36}),\ (Disabled|\[N/A\])$ ]]; then
            echo "Expected a full $model with MIG off: $device" >&2
            return 1
        fi
        printf '%s\n' "${BASH_REMATCH[2]}"
    done > logs/device-uuids.txt
    sort -o logs/device-uuids.txt logs/device-uuids.txt
    [[ $(sort -u logs/device-uuids.txt | wc -l) -eq $count ]]
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits > logs/compute-processes-before.txt
    if grep -q '[0-9]' logs/compute-processes-before.txt; then
        echo 'Other GPU compute processes are visible.' >&2
        return 1
    fi
    nvcc --version > logs/nvcc.txt
    g++ --version > logs/host-compiler.txt
    python3 --version > logs/python.txt
    uname -a > logs/system.txt
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r code/requirements.txt 2>&1 | tee logs/dependencies.txt
    python -m pip freeze > logs/python-packages.txt
    (
        cd code
        python -W error::ResourceWarning -m unittest -v test_hardening 2>&1 | tee ../logs/cpu-tests.txt
        nvcc -std=c++17 -O2 -arch=sm_90 probe_hardened.cu -o probe_hardened 2>&1 | tee ../logs/build.txt
        if [[ $variant != pilot ]]; then
            nvcc -O0 -arch=sm_90 atomic_race_probe_batch_v2.cu -o probe_legacy 2>&1 | tee ../logs/legacy-build.txt
        fi
    )
    find code -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > logs/build-input-hashes.txt
    sha256sum two_h200_host.sh two_h200_analysis.py >> logs/build-input-hashes.txt
    if [[ $variant != pilot ]]; then
        sha256sum matched_h200.py >> logs/build-input-hashes.txt
    fi
    if [[ $variant == ensemble ]]; then
        sha256sum eight_h200_ensemble.py >> logs/build-input-hashes.txt
    fi
}

capture() {
    local mode=$1 label=$2 phase=$3 pool=$4 repeats=$5 output=$6 device line saved total
    if [[ $label == A ]]; then device=${uuids[0]}; else device=${uuids[1]}; fi
    (( ++capture_index ))
    printf '[%s capture %d/%d] %s / GPU %s / %s\n' "$stage" "$capture_index" "$capture_total" "$mode" "$label" "$phase"
    python -u code/collect.py capture --binary code/probe_hardened --device "$device" \
        --pool "$pool" --phase "$phase" --repeats "$repeats" --mode "$mode" \
        --rounds 128 --timeout 120 --output "data/$output.jsonl" 2>&1 | tee "logs/$output.log" |
        while IFS= read -r line || [[ -n $line ]]; do
            if [[ $line =~ ^([1-9][0-9]*)/([1-9][0-9]*)\ saved$ ]]; then
                saved=${BASH_REMATCH[1]}
                total=${BASH_REMATCH[2]}
                if (( saved == total || (total >= 16 && saved % 16 == 0) )); then
                    printf '  %s\n' "$line"
                fi
            else
                printf '%s\n' "$line"
            fi
        done
}

capture_legacy() {
    local label=$1 phase=$2 pool=$3 repeats=$4
    (( ++capture_index ))
    printf '[%s capture %d/%d] legacy / GPU %s / %s\n' "$stage" "$capture_index" "$capture_total" "$label" "$phase"
    python -u matched_h200.py legacy-capture --device "$label" --phase "$phase" \
        --pool "$pool" --repeats "$repeats" 2>&1 | tee "logs/legacy-$label-$phase.log"
}

run_matched() {
    local phase item mode label native_phase
    local -a order
    capture_total=16
    for phase in registration calibration replay evaluation; do
        case "$phase" in
        registration) order=(legacy:A all:A all:B legacy:B) ;;
        calibration) order=(all:B legacy:B legacy:A all:A) ;;
        replay)
            python -u matched_h200.py freeze --gap-seconds "$gap" 2>&1 | tee logs/freeze.log
            order=(all:A legacy:A legacy:B all:B)
            ;;
        evaluation)
            printf 'Both registries frozen. Early replay captures saved. Waiting %s seconds before held-out collection.\n' "$gap"
            sleep "$gap"
            sha256sum --check --quiet logs/build-input-hashes.txt
            order=(legacy:B all:B all:A legacy:A)
            ;;
        esac
        for item in "${order[@]}"; do
            IFS=: read -r mode label <<< "$item"
            if [[ $mode == legacy ]]; then
                capture_legacy "$label" "$phase" data/main-pool.json 8
            else
                native_phase=$phase
                if [[ $phase == replay ]]; then native_phase=evaluation; fi
                capture all "$label" "$native_phase" data/main-pool.json 8 "all-$label-$phase"
            fi
        done
    done
    nvidia-smi -q -x > logs/host-after.xml
    python -u matched_h200.py evaluate --gap-seconds "$gap" 2>&1 | tee logs/evaluation.log
    python -u matched_h200.py timing 2>&1 | tee logs/timing.log
}

case "$stage" in
setup)
    mkdir data logs results
    setup_host 2>&1 | tee logs/setup.log
    touch .setup-ok
    ;;
smoke|pilot)
    [[ -f .setup-ok ]]
    source .venv/bin/activate
    mapfile -t uuids < logs/device-uuids.txt
    sha256sum --check --quiet logs/build-input-hashes.txt
    if [[ $variant == ensemble ]]; then
        if [[ $stage == smoke ]]; then
            mkdir .smoke-started
            python -u eight_h200_ensemble.py smoke --gap-seconds "$gap" 2>&1 | tee logs/ensemble-smoke.log
            touch .smoke-ok
        else
            [[ -f .smoke-ok ]]
            mkdir .pilot-started
            python -u eight_h200_ensemble.py collect --gap-seconds "$gap" 2>&1 | tee logs/ensemble-collect.log
            nvidia-smi -q -x > logs/host-after.xml
            touch .pilot-ok
        fi
        printf 'PILOT_OK %s\n' "$stage"
        exit 0
    fi
    capture_index=0
    if [[ $stage == smoke ]]; then
        capture_total=4
        mkdir .smoke-started
        start=$SECONDS
        python code/collect.py pool --count 2 --output data/smoke-pool.json
        if [[ $variant == matched ]]; then
            for label in A B; do
                capture_legacy "$label" smoke data/smoke-pool.json 1
                capture all "$label" evaluation data/smoke-pool.json 1 "smoke-all-$label"
            done
        else
            for mode in all subset; do
                for label in A B; do
                    capture "$mode" "$label" evaluation data/smoke-pool.json 1 "smoke-$mode-$label"
                done
            done
        fi
        python - "$variant" <<'PY'
import json
import math
import sys
from pathlib import Path

sm_maps = set()
modes = ('all',) if sys.argv[1] == 'matched' else ('all', 'subset')
for uuid, label in zip(Path('logs/device-uuids.txt').read_text().splitlines(), 'AB'):
    for mode in modes:
        path = Path(f'data/smoke-{mode}-{label}.jsonl')
        manifest = json.loads(Path(f'{path}.manifest.json').read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (manifest['status'] != 'complete' or manifest['expected_measurements'] != 2
                or manifest['received_measurements'] != 2 or manifest['device_uuid'] != uuid
                or len(rows) != 2 or Path(f'{path}.stderr.log').stat().st_size):
            raise SystemExit(f'Inspect smoke capture: {path}')
        sm_maps.add(tuple(manifest['sm_ids']))
        for row in rows:
            if not all(math.isfinite(t) and t > 0 for t in [row['host_measurement_ms'], *row['kernel_ms']]):
                raise SystemExit(f'Invalid timing: {path}')
if len(sm_maps) != 1:
    raise SystemExit('The GPUs have incompatible SM maps for the current scorer.')
PY
        elapsed=$((SECONDS - start))
        if (( elapsed < 1 )); then elapsed=1; fi
        multiplier=96
        if [[ $variant == matched ]]; then multiplier=128; fi
        printf 'est_collection_seconds=%s\n' "$((elapsed * multiplier))"
        touch .smoke-ok
    else
        capture_total=12
        [[ -f .smoke-ok ]]
        mkdir .pilot-started
        if [[ $variant == matched ]]; then
            python matched_h200.py pool
            run_matched
        else
            python code/collect.py pool --count 8 --output data/main-pool.json
            for phase in registration calibration evaluation; do
                if [[ $phase == evaluation ]]; then
                    python -u two_h200_analysis.py freeze --gap-seconds "$gap" 2>&1 | tee logs/freeze.log
                    printf 'Calibration frozen after %d/%d captures. Waiting %s seconds before held-out capture.\n' "$capture_index" "$capture_total" "$gap"
                    sleep "$gap"
                    sha256sum --check --quiet logs/build-input-hashes.txt
                fi
                if [[ $phase == calibration ]]; then
                    order=(subset:B subset:A all:B all:A)
                else
                    order=(all:A all:B subset:A subset:B)
                fi
                for item in "${order[@]}"; do
                    mode=${item%:*}
                    label=${item#*:}
                    capture "$mode" "$label" "$phase" data/main-pool.json 8 "$mode-$label-$phase"
                done
            done
            nvidia-smi -q -x > logs/host-after.xml
            python -u two_h200_analysis.py evaluate --gap-seconds "$gap" 2>&1 | tee logs/evaluation.log
        fi
        touch .pilot-ok
    fi
    ;;
archive)
    tar --exclude='./.venv' --exclude='__pycache__' -czf "$run.tar.gz.tmp" .
    mv "$run.tar.gz.tmp" "$run.tar.gz"
    digest=$(sha256sum "$run.tar.gz")
    printf 'ARCHIVE_SHA256=%s\n' "${digest%% *}"
    ;;
*) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac
printf 'PILOT_OK %s\n' "$stage"
