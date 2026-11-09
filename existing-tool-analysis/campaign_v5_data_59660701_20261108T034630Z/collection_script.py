#!/usr/bin/env python3
"""Reproducible within-session GPU fingerprint evaluation (v4).

Collect (uses GPUs): python campaign_v4.py atomic_race_probe_batch_v2.cu --output session
Analyze saved data only: python campaign_v4.py --analyze session --output analysis

Requires numpy; collection also requires nvcc, nvidia-smi and dedicated access to
the selected GPUs. This script does not rent hardware. Collection keeps the
original 10-run budget: 8 registration, index 8 evaluation, index 9 calibration.
Calibration/evaluation are distinct launches, not independent environments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
import uuid

import numpy as np

VERSION = "4.0"
RUNS = 10
REGISTRATION = 8
CHALLENGE = 8
CALIBRATION = 9
OFFSETS = [0, 64, 1 << 16, 1 << 24]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_design(n_uniform=16, n_residues=4, seed=12345):
    if n_uniform < 2 or not 1 <= n_residues <= 64:
        raise ValueError("need >=2 uniform seeds and 1..64 residue bases")
    groups = {str(b): [b + o for o in OFFSETS] for b in range(n_residues)}
    occupied = {s for group in groups.values() for s in group}
    rng = random.Random(seed)
    uniform = []
    while len(uniform) < n_uniform:
        # Full effective 31-bit state space, including zero. The kernel discards
        # the highest uint32 bit at its first LCG update.
        s = rng.randrange(1 << 31)
        if s not in occupied:
            uniform.append(s)
            occupied.add(s)
    return dict(uniform=uniform, residue_groups=groups, seeds=sorted(occupied),
                seed_generation_seed=seed, seed_space="effective 31-bit LCG states")


def parse_batch(out, expected):
    """Validate echoed seed, SM identity, dimensions and integer values."""
    blocks, headers = [], {}
    current, rows, sms, pending_sm = None, [], [], None

    def finish():
        if current is None:
            return
        if pending_sm is not None:
            raise ValueError("SM header without a fingerprint row")
        n_sms = int(headers["n_sms"])
        if n_sms <= 0 or n_sms % 2 or sms != list(range(0, n_sms, 2)):
            raise ValueError("missing, reordered, or duplicate SM rows")
        arr = np.asarray(rows, dtype=np.int64)
        if arr.shape != (n_sms // 2, 200):
            raise ValueError(f"unexpected fingerprint shape: {arr.shape}")
        if np.any(arr < 0) or np.any(arr >= (n_sms // 2) * 200):
            raise ValueError("fingerprint counter outside expected range")
        blocks.append((current, arr.astype(np.int32)))

    for line in out.splitlines():
        if line.startswith("# seed="):
            finish()
            current, rows, sms, pending_sm = int(line.split("=", 1)[1]), [], [], None
        elif line.startswith("# sm="):
            if current is None or pending_sm is not None:
                raise ValueError("unexpected SM header")
            pending_sm = int(line.split("=", 1)[1])
        elif line.startswith("#"):
            if "=" in line:
                k, v = line[1:].strip().split("=", 1)
                headers[k] = v
        elif line.strip():
            if current is None or pending_sm is None:
                raise ValueError("fingerprint row without seed/SM header")
            sms.append(pending_sm)
            pending_sm = None
            rows.append([int(x) for x in line.split(",")])
    finish()
    if pending_sm is not None or [s for s, _ in blocks] != list(expected):
        raise ValueError("output seeds/count do not match requested launch order")
    if not headers.get("gpu_uuid"):
        raise ValueError("missing GPU UUID")
    return blocks, headers


def available_devices(selection):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=30)
    devices = []
    for line in result.stdout.splitlines():
        gpu_uuid, name = line.split(",", 1)
        devices.append(dict(uuid=gpu_uuid.strip(), name=name.strip()))
    if selection:
        chosen = []
        for token in selection.split(","):
            token = token.strip()
            matches = ([devices[int(token)]] if token.isdigit() else
                       [d for d in devices if d["uuid"] == token])
            if len(matches) != 1:
                raise ValueError(f"unknown GPU: {token}")
            chosen.append(matches[0])
        devices = chosen
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CUDA_VISIBLE_DEVICES is set; specify --devices explicitly (physical indices or UUIDs)")
    if len(devices) < 2 or len({d["uuid"] for d in devices}) != len(devices):
        raise ValueError("select at least two distinct GPUs")
    return devices


def collect(cu, output, args):
    design = make_design(args.uniform_seeds, args.residues, args.seed)
    devices = available_devices(args.devices)
    metadata = dict(schema_version=1, script_version=VERSION,
                    session_id=str(uuid.uuid4()), created_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__,
                    source_sha256=sha256(cu), script_sha256=sha256(__file__),
                    devices=devices, design=design, runs=RUNS,
                    roles=dict(registration=list(range(8)), evaluation=8, calibration=9),
                    order_seed=args.order_seed, status="collecting",
                    scope="within-session; no temperature/power/transport intervention")
    write_json(output / "metadata.json", metadata)
    # Preserve the actual source alongside hashes; do not capture environment variables.
    (output / "kernel.cu").write_bytes(Path(cu).read_bytes())
    (output / "collection_script.py").write_bytes(Path(__file__).read_bytes())
    with tempfile.TemporaryDirectory() as temporary:
        binary = Path(temporary) / ("probe.exe" if os.name == "nt" else "probe")
        command = ["nvcc", "-O0", "-arch=sm_90", str(cu), "-o", str(binary)]
        metadata["compile_flags"] = ["-O0", "-arch=sm_90"]
        compilation = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (output / "compile.stdout.txt").write_text(compilation.stdout, encoding="utf-8")
        (output / "compile.stderr.txt").write_text(compilation.stderr, encoding="utf-8")
        compilation.check_returncode()
        compiler = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30)
        metadata["compiler_version"] = compiler.stdout
        write_json(output / "metadata.json", metadata)

        def work(g):
            order = [s for s in design["seeds"] for _ in range(RUNS)]
            # Independent, recorded launch orders reduce seed/time confounding
            # across devices. Occurrences 0..7/8/9 define roles within each seed.
            random.Random(args.order_seed + g).shuffle(order)
            write_json(output / f"gpu-{g:03d}.order.json", order)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[g]["uuid"])
            started = time.perf_counter()
            with (output / f"gpu-{g:03d}.stdout.txt").open("w", encoding="utf-8") as stdout, \
                    (output / f"gpu-{g:03d}.stderr.txt").open("w", encoding="utf-8") as stderr:
                subprocess.run([str(binary)] + [str(s) for s in order], env=env,
                               stdout=stdout, stderr=stderr, check=True, timeout=args.timeout)
            elapsed = time.perf_counter() - started
            blocks, headers = parse_batch(
                (output / f"gpu-{g:03d}.stdout.txt").read_text(encoding="utf-8"), order)
            if headers["gpu_uuid"].lower() != devices[g]["uuid"].lower():
                raise ValueError("observed GPU UUID differs from selected GPU")
            grouped = {s: [] for s in design["seeds"]}
            for s, arr in blocks:
                grouped[s].append(arr)
            data = np.stack([np.stack(grouped[s]) for s in design["seeds"]])
            # Save each GPU as soon as it finishes, before any analysis.
            np.savez_compressed(output / f"gpu-{g:03d}.npz",
                                seeds=np.asarray(design["seeds"], dtype=np.uint32), fingerprints=data)
            details = dict(headers=headers, batch_process_seconds=elapsed,
                           timing_scope="whole batch process; not kernel-only or attack latency",
                           npz_sha256=sha256(output / f"gpu-{g:03d}.npz"))
            write_json(output / f"gpu-{g:03d}.json", details)
            print(f"GPU {g}: saved {len(blocks)} fingerprints", flush=True)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(work, g) for g in range(len(devices))]
            for future in futures:
                future.result()
    metadata["status"] = "complete"
    write_json(output / "metadata.json", metadata)


def load_dataset(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("status") != "complete":
        raise ValueError("unsupported or incomplete dataset")
    if meta.get("runs") != RUNS or meta.get("roles") != dict(
            registration=list(range(8)), evaluation=8, calibration=9):
        raise ValueError("unsupported run roles")
    seeds = meta["design"]["seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed IDs")
    A, common_shape = {}, None
    for g in range(len(meta["devices"])):
        file = folder / f"gpu-{g:03d}.npz"
        detail = json.loads((folder / f"gpu-{g:03d}.json").read_text(encoding="utf-8"))
        if sha256(file) != detail["npz_sha256"]:
            raise ValueError("saved array checksum mismatch")
        with np.load(file, allow_pickle=False) as saved:
            x = saved["fingerprints"]
            if saved["seeds"].tolist() != seeds or x.ndim != 4 or x.shape[:2] != (len(seeds), RUNS):
                raise ValueError("invalid saved dimensions or seed IDs")
            if not np.issubdtype(x.dtype, np.integer):
                raise ValueError("raw fingerprints must be integers")
            if common_shape is not None and x.shape != common_shape:
                raise ValueError("incompatible GPU fingerprint dimensions")
            common_shape = x.shape
            for i, s in enumerate(seeds):
                A[(g, s)] = x[i].copy()
    return A, meta


def L1(a, b):
    """Actual L1; conversion to a response happens explicitly before scoring."""
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum())


def response(x, policy):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite prediction")
    if policy == "truncate":
        return np.trunc(x)
    if policy == "round":
        return np.rint(x)
    if policy == "float":
        return x
    raise ValueError("unknown response policy")


def score(x, A, target, seed, G):
    distances = [min(L1(x, r) for r in A[(g, seed)][:8]) for g in range(G)]
    other = min(d for g, d in enumerate(distances) if g != target)
    return dict(target=target, seed=seed, distance=distances[target],
                margin=distances[target] - other,
                # Reject ties instead of silently favoring the first GPU index.
                match=distances[target] < other, tie=distances[target] == other,
                chip_distances=distances)


def summarize(rows, threshold):
    n = len(rows)
    matches = sum(r["match"] for r in rows)
    below = sum(r["distance"] <= threshold for r in rows)
    accepted = sum(r["match"] and r["distance"] <= threshold for r in rows)
    return dict(n=n, matches=matches, below_threshold=below, accepted=accepted,
                match_rate=matches/n, distance_pass_rate=below/n, joint_acceptance=accepted/n,
                median_distance=float(np.median([r["distance"] for r in rows])),
                median_margin=float(np.median([r["margin"] for r in rows])),
                ties=sum(r["tie"] for r in rows))


def analyze(A, meta, policy="truncate"):
    G = len(meta["devices"])
    if G < 2:
        raise ValueError("need >=2 GPUs")
    uniform = meta["design"]["uniform"]
    if len(uniform) < 2 or len(set(uniform)) != len(uniform):
        raise ValueError("need >=2 distinct evaluation seeds")
    calibration = [score(A[(g, s)][CALIBRATION], A, g, s, G)
                   for s in uniform for g in range(G)]
    genuine = [score(A[(g, s)][CHALLENGE], A, g, s, G)
               for s in uniform for g in range(G)]
    thresholds = {str(q): float(np.percentile([r["distance"] for r in calibration], q))
                  for q in (90, 95, 99)}
    means = {(g, s): A[(g, s)][:8].mean(axis=0) for s in uniform for g in range(G)}
    trials = {"genuine": genuine, "calibration": calibration}
    # One fixed decoy per target, declared before inspecting performance.
    # Ensembles are reported separately and use all G-1 other devices.
    for lam in (1.0, 1.5, 2.0):
        trials[f"single_decoy_x{lam}"] = []
        trials[f"ensemble_{G-1}_decoys_x{lam}"] = []
    trials["no_decoy_seed_median"] = []
    for c in uniform:
        training = [s for s in uniform if s != c]
        for t in range(G):
            fixed_decoy = (t + 1) % G
            corrections = {h: np.median(np.stack([means[(t, s)] - means[(h, s)]
                                                  for s in training]), axis=0)
                           for h in range(G) if h != t}
            for lam in (1.0, 1.5, 2.0):
                predictions = {h: A[(h, c)][CHALLENGE].astype(np.float64) + lam * correction
                               for h, correction in corrections.items()}
                single = score(response(predictions[fixed_decoy], policy), A, t, c, G)
                single["decoy"] = fixed_decoy
                trials[f"single_decoy_x{lam}"].append(single)
                ensemble = response(np.median(np.stack(list(predictions.values())), axis=0), policy)
                trials[f"ensemble_{G-1}_decoys_x{lam}"].append(score(ensemble, A, t, c, G))
            baseline = response(np.median(np.stack([means[(t, s)] for s in training]), axis=0), policy)
            trials["no_decoy_seed_median"].append(score(baseline, A, t, c, G))
    for base, group in meta["design"]["residue_groups"].items():
        for kind in ("recorded_run", "mean_of_8"):
            rows = []
            for variant in group[1:]:
                for t in range(G):
                    stored = (A[(t, group[0])][0] if kind == "recorded_run"
                              else A[(t, group[0])][:8].mean(axis=0))
                    row = score(response(stored, policy), A, t, variant, G)
                    row["source_seed"] = group[0]
                    rows.append(row)
            trials[f"replay_base_{base}_{kind}"] = rows
    operating_points = {}
    for q, threshold in thresholds.items():
        operating_points[q] = dict(threshold=threshold,
            results={name: summarize(rows, threshold) for name, rows in trials.items()})
    return dict(schema_version=1, script_version=VERSION, session_id=meta["session_id"],
                response_policy=policy, tie_policy="reject equal closest distances",
                fixed_decoy_policy="(target_index + 1) modulo GPU count",
                calibration="run index 9 only; evaluation uses index 8; 8 registration runs",
                training="leave-one-uniform-seed-out; no residue seeds in correction training",
                thresholds_source="pooled calibration own-chip minimum-of-8 distances",
                scope_notes=["Within-session only; trials share seeds, devices and training data.",
                    "Rates are descriptive, not independent-trial confidence estimates.",
                    "Calibration percentiles are not guaranteed genuine joint-acceptance rates.",
                    "Replay uses thresholds calibrated on uniform seeds, not on residue seeds.",
                    "Only listed residue bases and variants were tested; no full-table inference.",
                    "No claim of end-to-end attack latency or a location-verification bypass."],
                operating_points=operating_points, trials=trials)


def print_report(report):
    print(f"Response conversion: {report['response_policy']}; closest-distance ties rejected")
    print("Thresholds use separate calibration launches; evaluation rates are measured, not imposed.")
    for q, point in report["operating_points"].items():
        print(f"\nCalibration p{q}: threshold {point['threshold']:,.2f}")
        for name, result in point["results"].items():
            print(f"  {name:37s} match {result['matches']:3}/{result['n']:<3} "
                  f"joint {result['accepted']:3}/{result['n']:<3} "
                  f"({result['joint_acceptance']:.1%})")
    for note in report["scope_notes"]:
        print(f"Scope: {note}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cu", nargs="?", help="batched CUDA source; supplying it starts collection")
    parser.add_argument("--analyze", type=Path, help="saved session directory; never invokes GPU tools")
    parser.add_argument("--output", type=Path, help="new output directory (must not already exist)")
    parser.add_argument("--devices", help="comma-separated physical GPU indices or full UUIDs")
    parser.add_argument("--uniform-seeds", type=int, default=16)
    parser.add_argument("--residues", type=int, default=4, help="number of residue bases, 1..64")
    parser.add_argument("--seed", type=int, default=12345, help="seed-generation PRNG seed; recorded")
    parser.add_argument("--order-seed", type=int, default=99)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--response-policy", choices=("truncate", "round", "float"), default="truncate",
                        help="explicit conversion of synthetic responses; default matches old v3")
    args = parser.parse_args(argv)
    if bool(args.cu) == bool(args.analyze):
        parser.error("supply either a CUDA source to collect, or --analyze for offline analysis")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.analyze:
        make_design(args.uniform_seeds, args.residues, args.seed)
        if not Path(args.cu).is_file():
            parser.error("CUDA source does not exist")
    output = args.output or Path("campaign-v4-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    if args.analyze:
        A, meta = load_dataset(args.analyze)
    else:
        print(f"Collecting on selected GPUs; saving raw data to {output.resolve()}", flush=True)
        try:
            collect(Path(args.cu).resolve(), output, args)
        except Exception as error:
            write_json(output / "failure.json", dict(type=type(error).__name__, message=str(error)))
            raise
        A, meta = load_dataset(output)
    report = analyze(A, meta, args.response_policy)
    report["analysis_script_sha256"] = sha256(__file__)
    report["source_dataset"] = str((args.analyze or output).resolve())
    write_json(output / "analysis.json", report)
    print_report(report)
    print(f"\nPer-trial scores and summaries: {(output / 'analysis.json').resolve()}")


if __name__ == "__main__":
    main()
