#!/usr/bin/env python3
"""Freeze and evaluate two H200 registries using the unchanged hardened scorer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import fingerprint as fp


PHASES = ('registration', 'calibration', 'evaluation')
MODES = ('all', 'subset')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_capture(mode, label, uuid, phase, pool_path, nonces, build, file_phase=None, repeats=8):
    path = Path(f'data/{mode}-{label}-{file_phase or phase}.jsonl')
    manifest_path = Path(str(path) + '.manifest.json')
    stderr_path = Path(str(path) + '.stderr.log')
    manifest = read_json(manifest_path)
    count = len(nonces)*repeats
    required = dict(status='complete', phase=phase, device_uuid=uuid, build_id=build,
                    expected_measurements=count, received_measurements=count,
                    challenge_source_kind='enrollment-pool', challenge_source_sha256=sha(pool_path))
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f'{path}: incomplete capture or inconsistent manifest')
    if manifest['profile']['mode'] != mode or manifest['profile']['rounds'] != 128:
        raise ValueError(f'{path}: unexpected profile')
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    order = [row['challenge'] for row in rows]
    if Counter(order) != Counter({nonce: repeats for nonce in nonces}) or order != manifest['launch_order']:
        raise ValueError(f'{path}: wrong challenge counts or launch order')
    for row in rows:
        if any(row.get(key) != manifest[key] for key in ('collection_id', 'profile', 'sm_ids', 'phase', 'device_uuid', 'build_id')):
            raise ValueError(f'{path}: record disagrees with its manifest')
    files = {str(p): sha(p) for p in (path, manifest_path, stderr_path)}
    return rows, datetime.fromisoformat(manifest['created_utc']), files


def add_ids(seen, rows):
    for row in rows:
        key = row['sample_id']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Missing or reused sample ID')
        seen.add(key)


def freeze(devices, pool, nonces, build, gap):
    artifacts = {}
    seen = set()
    for mode in MODES:
        training = []
        for phase in PHASES[:2]:
            for label, uuid in devices.items():
                rows, _, files = load_capture(mode, label, uuid, phase, pool, nonces, build)
                add_ids(seen, rows)
                training.extend(rows)
                artifacts.update(files)
        path = Path(f'results/{mode}-registry.json')
        write_json(path, fp.enroll(training, metric='raw_l1', alpha=.05))
        artifacts[str(path)] = sha(path)
    write_json(Path('results/frozen.json'), dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
               gap_seconds=gap, devices=devices, pool_sha256=sha(pool), build_id=build,
               artifact_hashes=artifacts, training_sample_ids=sorted(seen)))


def score_row(registry, row):
    """Evaluate threshold-only, nearest-only, and joint decisions independently."""
    table = fp.scores(registry, row)
    decisions = {}
    for i, target in enumerate(registry['device_order']):
        values = table[:, i]
        nearest = values < np.min(np.delete(table, i, axis=1), axis=1)
        within = values <= registry['devices'][target]['threshold']
        decisions[target] = dict(score=float(np.sort(values)[1]),
                                 threshold_pass=bool(within.sum() >= 2),
                                 nearest_win=bool(nearest.sum() >= 2),
                                 joint_pass=bool((nearest & within).sum() >= 2))
    return decisions


def summarize(rows):
    scores = np.asarray([row['score'] for row in rows])
    return dict(n=len(rows), threshold_pass=sum(row['threshold_pass'] for row in rows),
                nearest_wins=sum(row['nearest_win'] for row in rows),
                joint_pass=sum(row['joint_pass'] for row in rows),
                score_min=float(scores.min()), score_median=float(np.median(scores)),
                score_p95=float(np.percentile(scores, 95)), score_max=float(scores.max()))


def evaluate(devices, pool, nonces, build, gap):
    seal = read_json(Path('results/frozen.json'))
    if (seal['devices'], seal['pool_sha256'], seal['build_id'], seal['gap_seconds']) != (devices, sha(pool), build, gap):
        raise ValueError('Run settings differ from frozen calibration')
    for path, expected in seal['artifact_hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Frozen artifact changed: {path}')
    frozen_at = datetime.fromisoformat(seal['frozen_utc'])
    seen = set(seal['training_sample_ids'])
    for mode in MODES:
        registry_path = Path(f'results/{mode}-registry.json')
        registry = read_json(registry_path)
        matrix = {}
        details = []
        artifacts = {}
        for source, uuid in devices.items():
            rows, start, files = load_capture(mode, source, uuid, 'evaluation', pool, nonces, build)
            elapsed = (start - frozen_at).total_seconds()
            if elapsed < gap:
                raise ValueError(f'{mode}/{source}: evaluation started before the planned gap elapsed')
            add_ids(seen, rows)
            artifacts.update(files)
            matrix[source] = dict(seconds_after_freeze=elapsed, targets={})
            for row in rows:
                decisions = score_row(registry, row)
                details.append(dict(source=source, sample_id=row['sample_id'], challenge=row['challenge'],
                                    targets={label: decisions[target_uuid] for label, target_uuid in devices.items()}))
            group = [row for row in details if row['source'] == source]
            for target, target_uuid in devices.items():
                matrix[source]['targets'][target] = dict(
                    **summarize([row['targets'][target] for row in group]),
                    threshold=registry['devices'][target_uuid]['threshold'],
                    by_challenge={nonce: summarize([row['targets'][target] for row in group if row['challenge'] == nonce])
                                  for nonce in nonces})
        write_json(Path(f'results/{mode}-matrix.json'), dict(mode=mode, devices=devices, primary=(mode == 'all'),
                   registry_sha256=sha(registry_path), evaluation_hashes=artifacts, gap_seconds=gap,
                   matrix=matrix, per_burst=details, scope='Two devices on one host, one held-out session'))
        for source, source_result in matrix.items():
            for target, result in source_result['targets'].items():
                print(f"{mode} {source} -> {target}: threshold {result['threshold_pass']}/{result['n']}, "
                      f"nearest {result['nearest_wins']}/{result['n']}, joint {result['joint_pass']}/{result['n']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'evaluate'))
    parser.add_argument('--gap-seconds', type=int, required=True)
    args = parser.parse_args()
    uuids = Path('logs/device-uuids.txt').read_text().splitlines()
    if len(uuids) != 2 or len(set(uuids)) != 2 or any(not uuid.startswith('GPU-') for uuid in uuids):
        raise ValueError('Exactly two distinct GPU UUIDs are required')
    devices = dict(zip(('A', 'B'), uuids))
    pool = Path('data/main-pool.json')
    nonces = read_json(pool)['challenges']
    if len(nonces) != 8 or len(set(nonces)) != 8:
        raise ValueError('Expected eight distinct challenges')
    build = sha(Path('code/probe_hardened'))
    action = freeze if args.stage == 'freeze' else evaluate
    action(devices, pool, nonces, build, args.gap_seconds)


if __name__ == '__main__':
    main()
