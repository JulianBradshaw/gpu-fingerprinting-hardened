#!/usr/bin/env python3
"""Eight-card matched collection, with legacy positive control and offline scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import sys
import tarfile
import time

import numpy as np

LABELS = 'ABCDEFGH'
REPEATS = dict(registration=8, calibration=3, evaluation=2)
ARMS = ('legacy', 'hardened')
POOL = Path('data/main-pool.json')
SEAL = Path('results/ensemble-frozen.json')
HERE = Path(__file__).resolve().parent


def collect(action, gap, m):
    labels = dict(zip(LABELS, Path('logs/device-uuids.txt').read_text().splitlines()))
    if len(labels) != 8 or len(set(labels.values())) != 8:
        raise ValueError('Need eight distinct device UUIDs')
    smoke = action == 'smoke'
    pool = Path('data/smoke-pool.json') if smoke else POOL
    if smoke:
        nonces = [secrets.token_hex(32) for _ in range(8)]
    else:
        # Hold the original campaign's 16 uniform legacy seeds fixed.
        seeds = m.legacy.make_design()['uniform']
        nonces = [f'{seed | (secrets.randbits(1) << 31):08x}'+secrets.token_hex(28) for seed in seeds]
    m.write_json(pool, dict(challenges=nonces))
    m.load_pool(pool)
    build = m.sha(m.HARD)
    artifacts = {}
    evaluation_hashes = {}

    def on_card(label, uuid, phase, repeats, phase_index):
        hashes = {}
        order = ARMS if (LABELS.index(label)+phase_index)%2 == 0 else ARMS[::-1]
        for arm in order:
            if arm == 'legacy':
                # This helper preserves the raw legacy stdout, validates the seed order, and writes hashes.
                m.collect_legacy(pool, label, uuid, phase, repeats)
                _, _, files = m.legacy_rows(pool, label, uuid, phase, repeats=repeats)
            else:
                native_phase = 'evaluation' if smoke else phase
                name = f'all-{label}-{phase}'
                command = [sys.executable, '-u', 'code/collect.py', 'capture', '--binary', str(m.HARD),
                           '--device', uuid, '--pool', str(pool), '--phase', native_phase, '--repeats',
                           str(repeats), '--mode', 'all', '--rounds', '128', '--timeout', '120',
                           '--output', f'data/{name}.jsonl']
                with Path(f'logs/{name}.log').open('x') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
                _, _, files = m.load_capture('all', label, uuid, native_phase, pool, nonces, build,
                                             file_phase=phase, repeats=repeats)
            hashes.update(files)
            print(f'[{phase}] GPU {label} / {arm}: {len(nonces)*repeats} responses saved', flush=True)
        return hashes

    phases = {'smoke': 1} if smoke else REPEATS
    started = time.monotonic()
    for i, (phase, repeats) in enumerate(phases.items()):
        if phase == 'evaluation':
            m.write_json(SEAL, dict(frozen_utc=m.utc(), gap_seconds=gap, devices=labels,
                                   training_hashes=artifacts, pool_sha256=m.sha(pool),
                                   legacy_build=m.sha(m.OLD), hardened_build=build))
            print(f'Enrollment and calibration inputs sealed. Waiting {gap} seconds.', flush=True)
            time.sleep(gap)
        print(f'Starting {phase} on eight GPUs (four GPU workers, one probe per GPU).', flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for hashes in executor.map(lambda pair: on_card(*pair, phase, repeats, i), labels.items()):
                (evaluation_hashes if phase == 'evaluation' else artifacts).update(hashes)
    if smoke:
        sm_maps = {tuple(m.read_json(Path(f'data/all-{label}-smoke.jsonl.manifest.json'))['sm_ids']) for label in labels}
        if len(sm_maps) != 1:
            raise ValueError('Hardened SM maps differ across the cohort')
        # 208 main responses versus eight smoke responses per arm/GPU.
        print(f'est_collection_seconds={max(1, round((time.monotonic()-started)*26))}', flush=True)
    else:
        m.write_json(Path('results/ensemble-evaluation-hashes.json'), evaluation_hashes)


def verified_extract(archive, directory):
    pattern = re.compile(r'(data/[A-Za-z0-9_.-]+|results/ensemble-(frozen|evaluation-hashes)\.json|code/probe_(legacy|hardened))')
    with tarfile.open(archive, 'r|gz') as bundle:
        for member in bundle:
            name = PurePosixPath(member.name).as_posix()
            if pattern.fullmatch(name):
                if not member.isfile():
                    raise ValueError(f'Expected regular file: {name}')
                target = directory/name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('xb') as sink:
                    sink.write(source.read())


def load_arm(m, arm, seal, nonces):
    phases = {}
    seen = set()
    frozen = datetime.fromisoformat(seal['frozen_utc'])
    expected_map = None
    for phase, repeats in REPEATS.items():
        groups = []
        for label, uuid in seal['devices'].items():
            if arm == 'legacy':
                rows, start, _ = m.legacy_rows(POOL, label, uuid, phase, repeats=repeats)
                group = rows[:, :, None, :, :]
            else:
                rows, start, _ = m.load_capture('all', label, uuid, phase, POOL, nonces,
                                               seal['hardened_build'], repeats=repeats)
                m.add_ids(seen, rows)
                if expected_map is None:
                    expected_map = rows[0]['sm_ids']
                by_nonce = {nonce: [] for nonce in nonces}
                for row in rows:
                    ranks, mask = m.fp.validate_record(row, rows[0]['profile'], expected_map)
                    if not mask.all():
                        raise ValueError('Expected all-mode full rank matrices')
                    by_nonce[row['challenge']].append(ranks)
                group = np.asarray([by_nonce[nonce] for nonce in nonces])
            if phase == 'evaluation' and (start-frozen).total_seconds() < seal['gap_seconds']:
                raise ValueError(f'Early evaluation capture: {arm}/{label}')
            groups.append(group)
        phases[phase] = np.asarray(groups, dtype=np.int32)
    return phases


def score_batch(refs, inputs, divisor):
    # refs: target x template x coordinate. Equivalent integer L1 on validated all-mode inputs.
    scores = np.empty((len(inputs), inputs.shape[1], len(refs)))
    for i, response in enumerate(inputs):
        for k, launch in enumerate(response):
            distances = np.abs(refs-launch.reshape(-1)).sum(axis=2, dtype=np.int64)
            scores[i,k] = distances.min(axis=1)/divisor
    return scores


def score_arm(arm, phases):
    reg, cal, held = (phases[key] for key in REPEATS)
    devices, challenges = reg.shape[:2]
    divisor = np.prod(reg.shape[-2:]) if arm == 'hardened' else 1
    means = reg.mean(axis=(2, 3))
    scores = {key: [[] for _ in range(devices)] for key in ('calibration','genuine','single','ensemble')}
    if arm == 'legacy':
        scores['ensemble_bounded'] = [[] for _ in range(devices)]
    for c in range(challenges):
        refs = reg[:, c].reshape(devices, -1, int(np.prod(reg.shape[-2:])))
        training = np.arange(challenges) != c
        for target in range(devices):
            scores['calibration'][target].append(score_batch(refs, cal[target,c], divisor))
            scores['genuine'][target].append(score_batch(refs, held[target,c], divisor))
            decoys = [(target+k)%devices for k in range(1, devices)]
            corrected = [held[d,c] + np.median(means[target,training]-means[d,training],axis=0) for d in decoys]
            for key, prediction in [('single',corrected[0]), ('ensemble',np.median(corrected, axis=0))]:
                if arm == 'legacy':
                    values = np.trunc(prediction).astype(np.int32)
                    if key == 'ensemble':
                        bounded = values.clip(0, reg.shape[-2]*reg.shape[-1]-1)
                        scores['ensemble_bounded'][target].append(score_batch(refs, bounded, divisor))
                else:
                    values = np.argsort(np.argsort(prediction, axis=-1, kind='stable'), axis=-1, kind='stable').astype(np.int32)
                scores[key][target].append(score_batch(refs, values, divisor))
        print(f'{arm}: scored challenge {c+1}/{challenges}', flush=True)
    return {key: np.stack([np.concatenate(group) for group in groups]) for key,groups in scores.items()}


def summarize(scores, fp):
    idx = np.arange(len(LABELS))
    n = scores['genuine'].shape[1]
    majority = scores['genuine'].shape[2]//2+1
    calibration = np.sort(scores['calibration'][idx,:,:,idx],axis=-1)[:,:,majority-1]
    thresholds = dict(q95=[fp.conformal_threshold(row,.05) for row in calibration],
                      max=calibration.max(axis=1).tolist(),
                      historical_pooled_p95=[float(np.percentile(calibration,95))]*len(idx))
    joint_scores = {}
    for key, table in scores.items():
        order = np.sort(table, axis=-1)
        unique_win = (table == order[..., :1]) & (order[..., :1] < order[..., 1:2])
        joint_scores[key] = np.sort(np.where(unique_win, table, np.inf), axis=-2)[:,:,majority-1,:]
    result = {}
    for name, limits in thresholds.items():
        accepted = {key: (value<=limits).sum(axis=1) for key,value in joint_scores.items() if key != 'calibration'}
        result[name] = dict(thresholds=limits, genuine=int(accepted['genuine'].trace()),
                            genuine_by_gpu=accepted['genuine'].diagonal().tolist(),
                            wrong_chip=int(accepted['genuine'].sum()-accepted['genuine'].trace()),
                            genuine_n=len(idx)*n, wrong_chip_n=len(idx)*(len(idx)-1)*n,
                            **{key:int(table.trace()) for key,table in accepted.items() if key!='genuine'})
    # Descriptive held-out tradeoff, separate from the frozen operating points.
    scale = np.where(np.asarray(thresholds['q95']) > 0, thresholds['q95'], 1.0)
    observed = np.sort((joint_scores['genuine'][idx,:,idx]/scale[:,None]).ravel())
    tradeoff = {}
    for fraction in (.90,.95):
        factor = float(observed[int(np.ceil(len(observed)*fraction))-1])
        if np.isfinite(factor):
            tradeoff[str(fraction)] = dict(genuine=int((observed<=factor).sum()),
                **{key:int((joint_scores[key][idx,:,idx]/scale[:,None]<=factor).sum())
                   for key in joint_scores if key not in ('genuine','calibration')})
    return dict(frozen=result, heldout_tradeoff=tradeoff,
                nearest_correct=int(np.isfinite(joint_scores['genuine'][idx,:,idx]).sum()))


def analyze(m):
    seal = m.read_json(SEAL)
    if len(set(seal['devices'].values())) != 8 or list(seal['devices']) != list(LABELS):
        raise ValueError('Frozen cohort is incomplete')
    if m.sha(POOL) != seal['pool_sha256'] or m.sha(m.OLD) != seal['legacy_build'] or m.sha(m.HARD) != seal['hardened_build']:
        raise ValueError('Frozen pool or binary changed')
    eval_hashes = m.read_json(Path('results/ensemble-evaluation-hashes.json'))
    for path, digest in (seal['training_hashes'] | eval_hashes).items():
        if not re.fullmatch(r'data/[A-Za-z0-9_.-]+', path) or m.sha(Path(path)) != digest:
            raise ValueError(f'Input hash mismatch: {path}')
    nonces, seeds = m.load_pool(POOL)
    if seeds != m.legacy.make_design()['uniform']:
        raise ValueError('The pool differs from the original 16 uniform legacy seeds')
    results = {}
    for arm in ARMS:
        scores = score_arm(arm, load_arm(m, arm, seal, nonces))
        results[arm] = summarize(scores, m.fp)
        with Path(f'results/{arm}-ensemble-scores.npz').open('xb') as output:
            np.savez_compressed(output, **scores)
    m.write_json(Path('results/ensemble-summary.json'), dict(devices=seal['devices'], gap_seconds=seal['gap_seconds'],
        results=results, notes=['256 held-out requests per arm. Genuine and target-specific spoof counts use the same denominator.',
        'Seven-decoy and single-decoy corrections exclude the tested challenge and use registration on other seeds.',
        'Legacy truncation reproduces the historical scorer; ensemble_bounded separately clips to legacy counter bounds.',
        'Hardened predictions are projected to valid per-round rank permutations. Its native response uses three launches.',
        'Held-out tradeoff is descriptive. Use frozen points for the primary comparison.']))
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke','collect','analyze'))
    parser.add_argument('--gap-seconds', type=int, default=300)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'analyze':
        if not args.archive or not args.output:
            parser.error('analyze requires --archive and --output')
        archive, directory = args.archive.resolve(), args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        verified_extract(archive, directory)
        os.chdir(directory)
        sys.path.insert(0, str(HERE/'gpu_fingerprinting_hardened_v1'))
        sys.path.insert(0, str(HERE.parent/'existing-tool-analysis'))
    else:
        sys.path.insert(0, str(Path('code').resolve()))
    import matched_h200 as m
    if args.stage == 'analyze':
        analyze(m)
    else:
        collect(args.stage, args.gap_seconds, m)


if __name__ == '__main__':
    main()
