import json
from pathlib import Path
import sys

import numpy as np


root = Path(sys.argv[1])
mode = sys.argv[2]
sys.path.insert(0, str(root / 'code'))
import fingerprint as fp


def read(path):
    return json.loads((root / path).read_bytes())


seal = read('results/frozen.json')
archived = read(f'{mode}-archived.json')['matrix']
registry = read(f'results/{mode}-registry.json')
devices = list(seal['devices'].values())
labels = list(seal['devices'])
rows = {}
for phase in ('calibration', 'evaluation'):
    for label in labels:
        rows[phase, label] = [
            json.loads(line) for line in (root / f'data/{mode}-{label}-{phase}.jsonl').read_bytes().splitlines()
        ]
        assert len(rows[phase, label]) == 64
indices = [registry['device_order'].index(uuid) for uuid in devices]
sm_ids = registry['sm_ids']
profile = registry['profile']
scaled = []
for uuid in devices:
    for nonce, entry in registry['devices'][uuid]['templates'].items():
        scale = fp.unpack_array(entry['scale'])
        scaled.extend(scale[fp.plan(nonce, profile, sm_ids)] > 1)
print(f'{mode}: {np.mean(scaled):.2%} of active rank bins have learned MAD above the floor', flush=True)

report = {}
for metric in ('raw_l1', 'clipped_mad_l1'):
    registry['metric'] = metric
    registry['scoring_config'] = fp.scoring_config(metric)
    scores = {
        key: np.stack([fp.scores(registry, row)[:, indices] for row in records])
        for key, records in rows.items()
    }
    own = [np.sort(scores['calibration', label][:, :, i], axis=1)[:, 1] for i, label in enumerate(labels)]
    pooled = fp.conformal_threshold(np.concatenate(own), .05)
    cutoffs = {
        'per-chip q95': [fp.conformal_threshold(dist, .05) for dist in own],
        'pooled q95': [pooled] * len(labels),
        'per-chip max': [float(dist.max()) for dist in own],
    }
    if metric == 'raw_l1':
        assert cutoffs['per-chip q95'] == [registry['devices'][uuid]['threshold'] for uuid in devices]
    for policy, thresholds in cutoffs.items():
        results = {}
        for phase in ('calibration', 'evaluation'):
            for source in labels:
                table = scores[phase, source]
                for j, target in enumerate(labels):
                    values = table[:, :, j]
                    wins = values < table[:, :, 1-j]
                    within = values <= thresholds[j]
                    joint = (wins & within).sum(axis=1) >= 2
                    key = f'{phase} {source}->{target}'
                    results[key] = dict(
                        nearest=int((wins.sum(axis=1) >= 2).sum()),
                        threshold_only=int((within.sum(axis=1) >= 2).sum()),
                        joint=int(joint.sum()),
                        threshold=thresholds[j],
                        max_score=float(np.sort(values, axis=1)[:, 1].max()),
                        min_score=float(np.sort(values, axis=1)[:, 1].min()),
                        misses_by_challenge={
                            nonce[:8]: sum(not passed and record['challenge'] == nonce
                                           for record, passed in zip(rows[phase, source], joint))
                            for nonce in registry['challenges']
                        } if source == target else {},
                    )
        if metric == 'raw_l1' and policy == 'per-chip q95':
            for source in labels:
                for target in labels:
                    assert results[f'evaluation {source}->{target}']['joint'] == archived[source]['targets'][target]['joint_pass']
        own_counts = [results[f'evaluation {s}->{s}']['joint'] for s in labels]
        wrong_counts = [results[f'evaluation {s}->{t}']['joint'] for s,t in [('A','B'),('B','A')]]
        wrong_threshold = [results[f'evaluation {s}->{t}']['threshold_only'] for s,t in [('A','B'),('B','A')]]
        print(f'{mode} {metric:15s} {policy:12s} cutoffs {[round(t,6) for t in thresholds]} '
              f'held-out own={own_counts} wrong joint={wrong_counts} wrong threshold={wrong_threshold}', flush=True)
        report[f'{metric} / {policy}'] = results

(root / f'{mode}-calibration-comparison.json').write_text(json.dumps(report, indent=2))
