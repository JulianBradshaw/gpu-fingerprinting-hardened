#!/usr/bin/env python3
"""Collect legacy batches and compare the original and hardened probes on the same cards."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import campaign_v4 as legacy
import fingerprint as fp
from two_h200_analysis import add_ids, load_capture, read_json, sha, write_json


HARD = Path('code/probe_hardened')
OLD = Path('code/probe_legacy')
SEAL = Path('results/matched-frozen.json')
ARMS = ('legacy', 'hardened')


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_pool(pool):
    nonces = read_json(pool)['challenges']
    seeds = [int(fp.challenge(nonce)[:8], 16) & 0x7fffffff for nonce in nonces]
    if len(set(seeds)) != len(nonces):
        raise ValueError('Challenge pool has a duplicate effective legacy seed')
    return nonces, seeds


def make_pool(path):
    # Four uniform controls and two distinct pairs with one legacy seed bit flipped.
    while True:
        seeds = [secrets.randbelow(1 << 31) for _ in range(6)]
        seeds += [seed ^ 64 for seed in seeds[4:6]]
        if len(set(seeds)) == 8:
            break
    nonces = [f'{seed | (secrets.randbits(1) << 31):08x}' + secrets.token_hex(28) for seed in seeds]
    write_json(path, dict(challenges=nonces, replay_pairs=[[4, 6], [5, 7]]))


def collect_legacy(pool, label, device, phase, repeats):
    _, seeds = load_pool(pool)
    base = Path(f'data/legacy-{label}-{phase}')
    archive = Path(str(base)+'.npz')
    manifest_path = Path(str(base)+'.json')
    if archive.exists() or manifest_path.exists():
        raise ValueError(f'Capture already exists: {base}')
    order = [seed for seed in seeds for _ in range(repeats)]
    shuffle_seed = int.from_bytes(os.urandom(8), 'big')
    random.Random(shuffle_seed).shuffle(order)
    started = utc()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
    start = time.perf_counter()
    with Path(str(base)+'.stdout.log').open('x') as stdout, Path(str(base)+'.stderr.log').open('x') as stderr:
        subprocess.run([str(OLD.resolve()), *map(str, order)], env=env, text=True, stdout=stdout,
                       stderr=stderr, check=True, timeout=300)
    elapsed = time.perf_counter()-start
    if Path(str(base)+'.stderr.log').stat().st_size:
        raise ValueError(f'Legacy probe reported stderr: {base}')
    blocks, headers = legacy.parse_batch(Path(str(base)+'.stdout.log').read_text(), order)
    if headers['gpu_uuid'].lower() != device.lower():
        raise ValueError('Legacy probe UUID differs from selected device')
    grouped = {seed: [] for seed in seeds}
    for seed, block in blocks:
        grouped[seed].append(block)
    with archive.open('xb') as stream:
        np.savez_compressed(stream, seeds=seeds, fingerprints=np.stack([np.stack(grouped[s]) for s in seeds]))
    write_json(manifest_path, dict(status='complete', device_uuid=device, phase=phase, repeats=repeats,
               started_utc=started, completed_utc=utc(), headers=headers, build_id=sha(OLD),
               pool_sha256=sha(pool), archive_sha256=sha(archive), launch_order=order,
               shuffle_seed=shuffle_seed, batch_process_seconds=elapsed))
    print(f'{len(blocks)}/{len(order)} legacy responses saved', flush=True)


def legacy_rows(pool, label, device, phase, repeats=8):
    _, seeds = load_pool(pool)
    base = f'data/legacy-{label}-{phase}'
    archive = Path(base+'.npz')
    meta_path = Path(base+'.json')
    meta = read_json(meta_path)
    required = dict(status='complete', device_uuid=device, phase=phase, repeats=repeats,
                    build_id=sha(OLD), pool_sha256=sha(pool), archive_sha256=sha(archive))
    if any(meta.get(key) != val for key, val in required.items()):
        raise ValueError(f'Mismatched legacy capture: {base}')
    if Counter(meta['launch_order']) != Counter({seed: repeats for seed in seeds}):
        raise ValueError(f'Wrong legacy seed counts: {base}')
    with np.load(archive, allow_pickle=False) as data:
        values = data['fingerprints']
        n_sms = int(meta['headers']['n_sms'])
        if data['seeds'].tolist() != seeds or values.shape != (len(seeds), repeats, n_sms//2, 200) or values.dtype.kind not in 'iu':
            raise ValueError(f'Wrong legacy data shape: {base}')
    return values, datetime.fromisoformat(meta['started_utc']), {str(p): sha(p) for p in (archive, meta_path)}


def hard_rows(pool, label, device, phase):
    nonces, _ = load_pool(pool)
    data_phase = 'evaluation' if phase == 'replay' else phase
    return load_capture('all', label, device, data_phase, pool, nonces, sha(HARD), file_phase=phase)


def legacy_scores(templates, records, indices=None):
    scores = []
    if indices is None:
        indices = range(len(records))
    for i, challenge in zip(indices, records):
        for row in challenge:
            scores.append([[min(legacy.L1(row, template) for template in device[i]) for device in templates]])
    return np.asarray(scores)


def hard_scores(registry, devices, rows):
    order = [registry['device_order'].index(uuid) for uuid in devices.values()]
    return np.stack([fp.scores(registry, row)[:, order] for row in rows])


def cutoffs(tables):
    own = [np.sort(table[:, :, i], axis=1)[:, table.shape[1]//2] for i, table in enumerate(tables)]
    return dict(q95=[fp.conformal_threshold(values, .05) for values in own],
                max=[float(values.max()) for values in own])


def freeze(pool, devices, gap):
    artifacts = {}
    seen = set()
    training = {phase: [] for phase in ('registration', 'calibration')}
    old = {phase: [] for phase in training}
    for phase in training:
        for label, device in devices.items():
            rows, _, files = hard_rows(pool, label, device, phase)
            add_ids(seen, rows)
            training[phase].append(rows)
            artifacts.update(files)
            rows, _, files = legacy_rows(pool, label, device, phase)
            old[phase].append(rows)
            artifacts.update(files)
    registry = fp.enroll([row for phase in training.values() for group in phase for row in group])
    registry_path = Path('results/hardened-registry.json')
    write_json(registry_path, registry)
    artifacts[str(registry_path)] = sha(registry_path)
    thresholds = {
        'legacy': cutoffs([legacy_scores(old['registration'], rows) for rows in old['calibration']]),
        'hardened': cutoffs([hard_scores(registry, devices, rows) for rows in training['calibration']]),
    }
    write_json(SEAL, dict(frozen_utc=utc(), gap_seconds=gap, devices=devices, pool_sha256=sha(pool),
                         artifacts=artifacts, thresholds=thresholds, hardened_training_ids=sorted(seen)))


def decisions(tables, thresholds):
    result = {}
    for i, source in enumerate('AB'):
        table = tables[i]
        majority = table.shape[1]//2+1
        for j, target in enumerate('AB'):
            wins = table[:, :, j] < table[:, :, 1-j]
            within = table[:, :, j] <= thresholds[j]
            result[f'{source}->{target}'] = dict(n=len(table), nearest=int((wins.sum(axis=1)>=majority).sum()),
                threshold_only=int((within.sum(axis=1)>=majority).sum()),
                joint=int(((wins & within).sum(axis=1)>=majority).sum()))
    return result


def evaluate(pool, devices, gap):
    seal = read_json(SEAL)
    if (seal['devices'], seal['pool_sha256'], seal['gap_seconds']) != (devices, sha(pool), gap):
        raise ValueError('Run settings differ from frozen calibration')
    for path, digest in seal['artifacts'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Frozen artifact changed: {path}')
    frozen = datetime.fromisoformat(seal['frozen_utc'])
    registry = read_json(Path('results/hardened-registry.json'))
    templates = [legacy_rows(pool, label, device, 'registration')[0] for label, device in devices.items()]
    nonces, seeds = load_pool(pool)
    pairs = read_json(pool)['replay_pairs']
    if sorted(i for pair in pairs for i in pair) != list(range(4, 8)) or any((seeds[a] ^ seeds[b]) != 64 for a, b in pairs):
        raise ValueError('Unexpected matched replay pairs')
    paired_targets = [(target, source) for a, b in pairs for target, source in ((a, b), (b, a))]
    tables = {arm: {} for arm in ARMS}
    for arm in ARMS:
        tables[arm]['paired_replay'] = []
    start_offsets = {arm: {} for arm in ARMS}
    seen = set(seal['hardened_training_ids'])
    inputs = {}
    for phase in ('replay', 'evaluation'):
        for arm in ARMS:
            tables[arm][phase] = []
            start_offsets[arm][phase] = {}
            for label, device in devices.items():
                if arm == 'legacy':
                    rows, start, files = legacy_rows(pool, label, device, phase)
                    table = legacy_scores(templates, rows)
                    if phase == 'replay':
                        tables[arm]['paired_replay'].append(legacy_scores(templates,
                            [rows[source] for target, source in paired_targets],
                            [target for target, source in paired_targets]))
                else:
                    rows, start, files = hard_rows(pool, label, device, phase)
                    add_ids(seen, rows)
                    table = hard_scores(registry, devices, rows)
                    if phase == 'replay':
                        by_nonce = {nonce: [row for row in rows if row['challenge'] == nonce] for nonce in nonces}
                        relabeled = [dict(row, challenge=nonces[target]) for target, source in paired_targets
                                     for row in by_nonce[nonces[source]]]
                        tables[arm]['paired_replay'].append(hard_scores(registry, devices, relabeled))
                elapsed = (start-frozen).total_seconds()
                if elapsed < (gap if phase == 'evaluation' else 0):
                    raise ValueError(f'{arm}/{label}/{phase} started before its planned window')
                start_offsets[arm][phase][label] = elapsed
                tables[arm][phase].append(table)
                inputs.update(files)
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for policy, thresholds in seal['thresholds'][arm].items():
            heldout = decisions(tables[arm]['evaluation'], thresholds)
            replay = decisions(tables[arm]['replay'], thresholds)
            paired = decisions(tables[arm]['paired_replay'], thresholds)
            results[arm][policy] = dict(thresholds=thresholds, heldout=heldout, replay=replay, paired_replay=paired)
            print(f"{arm}/{policy}: genuine {heldout['A->A']['joint'] + heldout['B->B']['joint']}/128; "
                  f"wrong chip {heldout['A->B']['joint'] + heldout['B->A']['joint']}/128; "
                  f"same-challenge replay {replay['A->A']['joint'] + replay['B->B']['joint']}/128; "
                  f"paired cross-challenge replay {paired['A->A']['joint'] + paired['B->B']['joint']}/64", flush=True)
    write_json(Path('results/matched-summary.json'), dict(
        devices=devices, gap_seconds=gap, start_offsets_seconds=start_offsets, input_hashes=inputs,
        freeze_sha256=sha(SEAL), results=results,
        replay_scope='Independent early responses tested both on the original challenge and relabeled to their paired challenge. Legacy projected seeds differ by bit 6. Both tests score fingerprints offline; protocol freshness is outside this metric.',
        policy_scope='Both cutoffs fitted only on calibration. Use held-out genuine counts to judge comparability.'))
    np.savez_compressed('results/matched-scores.npz',
                        **{f'{arm}_{phase}_{label}': table for arm in ARMS for phase, groups in tables[arm].items()
                           for label, table in zip(devices, groups)})


def timing(pool, devices):
    nonces, seeds = load_pool(pool)
    samples = []
    for label, device in devices.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        for i, (nonce, seed) in enumerate(zip(nonces[:4], seeds[:4])):
            for arm in (ARMS if (i+(label == 'B'))%2 == 0 else ARMS[::-1]):
                command = [str(OLD.resolve()), str(seed)] if arm == 'legacy' else [str(HARD.resolve()), 'all', '128']
                start = time.perf_counter()
                reply = subprocess.run(command, input=(nonce+'\n') if arm == 'hardened' else '',
                                       capture_output=True, text=True, env=env, check=True, timeout=30)
                milliseconds = (time.perf_counter()-start)*1000
                if reply.stderr:
                    raise ValueError(f'{arm} timing attempt reported stderr')
                if arm == 'legacy':
                    _, headers = legacy.parse_batch(reply.stdout, [seed])
                    measured = headers['gpu_uuid']
                else:
                    ready, row = [json.loads(line) for line in reply.stdout.splitlines()]
                    fp.validate_record(row, ready['profile'])
                    if row['challenge'] != nonce:
                        raise ValueError('Timing response challenge mismatch')
                    measured = row['device_uuid']
                if measured.lower() != device.lower():
                    raise ValueError('Timing response device mismatch')
                samples.append(dict(arm=arm, device=label, nonce=nonce, milliseconds=milliseconds))
    write_json(Path('results/matched-timing.json'), dict(samples=samples,
               scope='Local cold process start through exit for one native response; includes context startup and stdout; excludes SSH and verifier scoring.'))
    for arm in ARMS:
        values = [row['milliseconds'] for row in samples if row['arm'] == arm]
        print(f'{arm}: median cold local response {np.median(values):.1f} ms ({len(values)} attempts)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pool', 'legacy-capture', 'freeze', 'evaluate', 'timing'))
    parser.add_argument('--pool', type=Path, default=Path('data/main-pool.json'))
    parser.add_argument('--gap-seconds', type=int, default=600)
    parser.add_argument('--device', choices=('A', 'B'))
    parser.add_argument('--phase', choices=('smoke', 'registration', 'calibration', 'replay', 'evaluation'))
    parser.add_argument('--repeats', type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    if args.stage == 'pool':
        make_pool(args.pool)
        return
    uuids = Path('logs/device-uuids.txt').read_text().splitlines()
    if len(uuids) != 2 or len(set(uuids)) != 2:
        raise ValueError('Need two distinct GPUs')
    devices = dict(zip('AB', uuids))
    if args.stage == 'legacy-capture':
        if not args.device or not args.phase:
            parser.error('legacy capture needs --device and --phase')
        collect_legacy(args.pool, args.device, devices[args.device], args.phase, args.repeats)
    elif args.stage == 'timing':
        timing(args.pool, devices)
    else:
        {'freeze': freeze, 'evaluate': evaluate}[args.stage](args.pool, devices, args.gap_seconds)


if __name__ == '__main__':
    main()
