from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

run = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(run))
os.chdir(run)
import matched_h200 as m

seal = m.read_json(m.SEAL)
pool = Path('data/main-pool.json')
for name, digest in seal['artifacts'].items():
    assert m.sha(Path(name)) == digest
summary = m.read_json(Path('results/matched-summary.json'))
for name, digest in summary['input_hashes'].items():
    assert m.sha(Path(name)) == digest
nonces, seeds = m.load_pool(pool)
mate = {i: j for a, b in m.read_json(pool)['replay_pairs'] for i, j in ((a, b), (b, a))}
devices = seal['devices']
registry = m.read_json(Path('results/hardened-registry.json'))
old_training = [m.legacy_rows(pool, label, uuid, 'registration')[0] for label, uuid in devices.items()]
old_evaluation = [m.legacy_rows(pool, label, uuid, 'evaluation')[0] for label, uuid in devices.items()]


def hard_capture(phase):
    captures = []
    for label, uuid in devices.items():
        rows = m.hard_rows(pool, label, uuid, phase)[0]
        captures.append([[row for row in rows if row['challenge'] == nonce] for nonce in nonces])
    return captures


hard_training = hard_capture('registration')
hard_evaluation = hard_capture('evaluation')
stats = {}
with np.load('results/matched-scores.npz', allow_pickle=False) as scores:
    genuine = {arm: [scores[f'{arm}_evaluation_{label}'] for label in devices] for arm in m.ARMS}

for arm in m.ARMS:
    print(f'Scoring {arm} leave-challenge-out decoys', flush=True)
    if arm == 'legacy':
        means = np.stack([rows.mean(axis=1) for rows in old_training])
    else:
        means = np.asarray([[np.mean([row['fingerprints'] for row in records], axis=(0, 1))
                             for records in card] for card in hard_training])
    attacks = {name: [[], []] for name in ('median_only', 'decoy_x1', 'decoy_x1.5', 'decoy_x2')}
    out_of_range = defaultdict(int)
    for target in range(2):
        decoy = 1-target
        for c in range(8):
            train = [i for i in range(8) if i != c and i != mate.get(c)]
            correction = np.median(means[target, train] - means[decoy, train], axis=0)
            baseline = np.median(means[target, train], axis=0)
            for name in attacks:
                factor = 0 if name == 'median_only' else float(name.split('x')[1])
                if arm == 'legacy':
                    raw = old_evaluation[decoy][c].astype(np.float64)
                    values = np.broadcast_to(baseline, raw.shape) if not factor else raw + factor*correction
                    truncated = np.trunc(values)
                    upper = raw.shape[-2]*raw.shape[-1]-1
                    out_of_range[name] += int(np.any((truncated < 0) | (truncated > upper), axis=(1, 2)).sum())
                    pred = np.clip(truncated, 0, upper)
                    table = m.legacy_scores(old_training, [pred], [c])
                else:
                    rows = hard_evaluation[decoy][c]
                    raw = np.asarray([row['fingerprints'] for row in rows], dtype=np.float64)
                    values = np.broadcast_to(baseline, raw.shape) if not factor else raw + factor*correction
                    # Stable argsort of argsort projects every round back to valid integer ranks.
                    pred = np.argsort(np.argsort(values, axis=-1, kind='stable'), axis=-1, kind='stable')
                    table = m.hard_scores(registry, devices, [dict(row, fingerprints=rank.tolist())
                                                            for row, rank in zip(rows, pred)])
                attacks[name][target].append(table)
    stats[arm] = {}
    for name, matrices in attacks.items():
        stats[arm][name] = dict(policies={})
        if arm == 'legacy':
            stats[arm][name]['clipped_legacy_predictions'] = out_of_range[name]
        for policy, cutoffs in seal['thresholds'][arm].items():
            match = m.decisions(genuine[arm], cutoffs)
            all_attacks = m.decisions([np.concatenate(group) for group in matrices], cutoffs)
            uniform = m.decisions([np.concatenate(group[:4]) for group in matrices], cutoffs)
            by_target = lambda rows: [rows['A->A']['joint'], rows['B->B']['joint']]
            result = dict(genuine=by_target(match), attacks=by_target(all_attacks),
                          uniform_challenge_attacks=by_target(uniform),
                          nearest_attack=[all_attacks['A->A']['nearest'], all_attacks['B->B']['nearest']])
            stats[arm][name]['policies'][policy] = result
            print(arm, name, policy, result, flush=True)
    if arm == 'legacy':
        print('Legacy out-of-range candidates clipped to valid counter bounds:', dict(out_of_range), flush=True)

Path('decoy-comparison.json').write_text(json.dumps(stats, indent=2))
