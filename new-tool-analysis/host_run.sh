#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

stage=${1:?stage required}
RUN=${2:?run directory required}
[[ $RUN =~ ^/workspace/fp-sanity-[0-9a-f]{32}$ && -d $RUN && ! -L $RUN ]] || exit 1
cd "$RUN"
trap 'rc=$?; printf "HOSTRUN_FAIL: %s (exit %s)\n" "$stage" "$rc" >&2' ERR
# A dropped SSH connection can leave collection running. Refuse to archive a live run.
exec 9>.stage.lock
flock -n 9

capture() {
    local mode=$1 phase=$2 pool=$3 repeats=$4 output=$5
    python -u code/collect.py capture --binary code/probe_hardened --device "$GPU_UUID" \
        --pool "$pool" --phase "$phase" --repeats "$repeats" --mode "$mode" \
        --rounds 128 --timeout 120 --output "data/$output.jsonl"
}

setup_host() {
    cp -a /workspace/gpu_fingerprinting_hardened_v1/. code/
    cp /workspace/single_h200_sanity_analysis.py /workspace/single_h200_sanity_plan.md ./
    nvidia-smi -L | tee logs/devices.txt
    nvidia-smi -q -x > logs/host-before.xml
    gpu=$(nvidia-smi --query-gpu=name,uuid,mig.mode.current --format=csv,noheader,nounits)
    if [[ ! $gpu =~ ^(NVIDIA\ )?H200[^,]*,\ (GPU-[0-9a-fA-F-]{36}),\ (Disabled|\[N/A\])$ ]] || [[ $gpu == *$'\n'* ]]; then
        echo 'Expected one full H200 with MIG disabled or unsupported.' >&2
        exit 1
    fi
    GPU_UUID=${BASH_REMATCH[2]}
    printf '%s\n' "$GPU_UUID" > logs/device-uuid.txt
    nvidia-smi -i "$GPU_UUID" -q -x > logs/selected-before.xml
    nvidia-smi -i "$GPU_UUID" --query-compute-apps=pid --format=csv,noheader,nounits > logs/compute-processes-before.txt
    if grep -q '[0-9]' logs/compute-processes-before.txt; then
        echo 'Competing GPU processes are visible. Inspect the host before collecting.' >&2
        exit 1
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
    )
    find code -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > logs/build-input-hashes.txt
    sha256sum host_run.sh single_h200_sanity_analysis.py >> logs/build-input-hashes.txt
}

case "$stage" in
setup)
    mkdir code data logs results
    # Wait for the full transcript before marking setup complete.
    setup_host 2>&1 | tee logs/setup.log
    touch .setup-ok
    ;;
smoke|main|analyze)
    [[ -f .setup-ok ]]
    source .venv/bin/activate
    read -r GPU_UUID < logs/device-uuid.txt
    sha256sum --check --quiet logs/build-input-hashes.txt
    case "$stage" in
    smoke)
        mkdir .smoke-started
        started=$SECONDS
        python code/collect.py pool --count 2 --output data/smoke-pool.json
        for mode in all subset; do
            capture "$mode" evaluation data/smoke-pool.json 1 "smoke-$mode" 2>&1 | tee "logs/smoke-$mode.txt"
        done
        python - "$GPU_UUID" <<'PY'
import json
import math
import sys
from pathlib import Path

for mode in ('all', 'subset'):
    path = Path(f'data/smoke-{mode}.jsonl')
    manifest = json.loads(Path(f'{path}.manifest.json').read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (manifest['status'] != 'complete' or manifest['expected_measurements'] != 2
            or manifest['received_measurements'] != 2 or manifest['device_uuid'] != sys.argv[1]
            or len(rows) != 2 or Path(f'{path}.stderr.log').stat().st_size):
        raise SystemExit(f'Smoke capture needs inspection: {path}')
    for row in rows:
        host = row['host_measurement_ms']
        if not math.isfinite(host) or host <= 0 or not all(t > 0 for t in row['kernel_ms']):
            raise SystemExit(f'Unexpected smoke timing: {path}')
PY
        elapsed=$((SECONDS - started))
        if (( elapsed < 1 )); then elapsed=1; fi
        # Four saved smoke bursts versus 384 main bursts. This includes smoke startup overhead.
        printf 'est_main_seconds=%s\n' "$((elapsed * 96))"
        touch .smoke-ok
        ;;
    main)
        [[ -f .smoke-ok ]]
        mkdir .main-started
        python code/collect.py pool --count 8 --output data/main-pool.json
        for item in all:registration subset:registration subset:calibration all:calibration all:evaluation subset:evaluation; do
            mode=${item%%:*}
            phase=${item##*:}
            capture "$mode" "$phase" data/main-pool.json 8 "$mode-$phase" 2>&1 | tee "logs/$mode-$phase.txt"
        done
        nvidia-smi -i "$GPU_UUID" -q -x > logs/selected-after.xml
        touch .main-ok
        ;;
    analyze)
        [[ -f .main-ok ]]
        status=0
        for mode in all subset; do
            rc=0
            python single_h200_sanity_analysis.py --code code --pool data/main-pool.json \
                --device "$GPU_UUID" --registration "data/$mode-registration.jsonl" \
                --calibration "data/$mode-calibration.jsonl" --evaluation "data/$mode-evaluation.jsonl" \
                --mode "$mode" --output "results/$mode-summary.json" \
                2>&1 | tee "logs/$mode-analysis.txt" || rc=$?
            if (( rc != 0 )); then
                if (( rc == 2 )) && [[ -s results/$mode-summary.json ]]; then
                    if (( status == 0 )); then status=2; fi
                else
                    status=1
                fi
            fi
        done
        if (( status == 2 )); then
            printf 'HOSTRUN_TRIAGE analyze\n'
            exit 2
        fi
        if (( status != 0 )); then exit "$status"; fi
        ;;
    esac
    ;;
archive)
    # Seal the evidence outside the run directory; omit the recreatable Python environment.
    tar --exclude='./.venv' --exclude='__pycache__' -czf "$RUN.tar.gz.tmp" .
    mv "$RUN.tar.gz.tmp" "$RUN.tar.gz"
    digest=$(sha256sum "$RUN.tar.gz")
    printf 'ARCHIVE_SHA256=%s\n' "${digest%% *}"
    ;;
*) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac
printf 'HOSTRUN_OK %s\n' "$stage"
