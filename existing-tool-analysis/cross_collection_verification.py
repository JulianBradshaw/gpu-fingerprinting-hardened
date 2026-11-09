"""Offline genuine verification only; no GPU/network calls or user-code execution."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent  # folder holding campaign_v5_data_*
OUT = Path(__file__).resolve().parent
folders = sorted(ROOT.glob('campaign_v5_data_*'))
if len(folders) != 4:
    raise ValueError('Expected exactly four collection folders')

collections = []
for folder in folders:
    meta = json.loads((folder/'metadata.json').read_text())
    if meta['status'] != 'complete' or meta['roles'] != dict(registration=list(range(8)),evaluation=8,calibration=9):
        raise ValueError('Unexpected collection status or split')
    arrays = {}
    for g, device in enumerate(meta['devices']):
        file = folder/f'gpu-{g:03d}.npz'
        details = json.loads((folder/f'gpu-{g:03d}.json').read_text())
        if hashlib.sha256(file.read_bytes()).hexdigest() != details['npz_sha256']:
            raise ValueError('Checksum mismatch')
        with np.load(file, allow_pickle=False) as saved:
            seeds = saved['seeds'].tolist()
            arr = saved['fingerprints']
            if seeds != meta['design']['seeds'] or arr.shape != (32,10,66,200):
                raise ValueError('Unexpected seed IDs or dimensions')
            # Counter bounds make int32 differences safe; sums use int64.
            if arr.dtype != np.int32 or arr.min() < 0 or arr.max() >= 13200:
                raise ValueError('Unexpected counter values/dtype')
            arrays[device['uuid']] = {s: arr[i].copy() for i,s in enumerate(seeds) if s in meta['design']['uniform']}
    collections.append(dict(folder=folder.name,metadata=meta,arrays=arrays,
                            saved_analysis=json.loads((folder/'analysis.json').read_text())))

uuids = [d['uuid'] for d in collections[0]['metadata']['devices']]
seeds = collections[0]['metadata']['design']['uniform']
for collection in collections:
    if set(collection['arrays']) != set(uuids) or collection['metadata']['design']['uniform'] != seeds:
        raise ValueError('Collections do not share GPU UUIDs and uniform seeds')


def score(response, registry, target):
    # registry: GPUs x registration runs x SMs x rounds; literal minimum-of-eight L1.
    distances = np.abs(registry - response).sum(axis=(2,3),dtype=np.int64).min(axis=1)
    own = int(distances[target]); other = min(int(d) for g,d in enumerate(distances) if g != target)
    return dict(target_uuid=uuids[target],distance=own,margin=own-other,
                match=own<other,tie=own==other,chip_distances=distances.tolist())


def summary(rows, threshold):
    return dict(n=len(rows),matches=sum(r['match'] for r in rows),
                distance_passes=sum(r['distance']<=threshold for r in rows),
                accepted=sum(r['match'] and r['distance']<=threshold for r in rows),
                median_distance=float(np.median([r['distance'] for r in rows])),
                median_margin=float(np.median([r['margin'] for r in rows])))


pairs, source_info = [], []
for src, collection in enumerate(collections):
    registry = {s: np.stack([collection['arrays'][gpu][s][:8] for gpu in uuids]) for s in seeds}
    calibration = [score(collection['arrays'][gpu][s][9],registry[s],g) for s in seeds for g,gpu in enumerate(uuids)]
    thresholds = {str(q): float(np.percentile([r['distance'] for r in calibration],q)) for q in (90,95,99)}
    if any(thresholds[q] != collection['saved_analysis']['operating_points'][q]['threshold'] for q in thresholds):
        raise ValueError('Recomputed calibration differs from saved analysis')
    source_info.append(dict(run=src+1,folder=collection['folder'],session_id=collection['metadata']['session_id'],
                            created_utc=collection['metadata']['created_utc'],thresholds=thresholds))
    for dst, target_collection in enumerate(collections):
        rows=[]
        for s in seeds:
            for g,gpu in enumerate(uuids):
                row=score(target_collection['arrays'][gpu][s][8],registry[s],g)
                row['seed']=s
                rows.append(row)
        at_threshold = {q: summary(rows,T) for q,T in thresholds.items()}
        if src == dst:
            for q in thresholds:
                reference = collection['saved_analysis']['operating_points'][q]['results']['genuine']
                if any(at_threshold[q][k] != reference[k] for k in ('n','matches','accepted')):
                    raise ValueError('Within-collection control does not match saved result')
        delta = (datetime.fromisoformat(target_collection['metadata']['created_utc'])-
                 datetime.fromisoformat(collection['metadata']['created_utc'])).total_seconds()
        pair = dict(registration_run=src+1,verification_run=dst+1,
                    collection_start_gap_seconds=delta,operating_points=at_threshold,trials=rows)
        pairs.append(pair)
        print(f"Register {src+1} -> verify {dst+1}: ID {at_threshold['95']['matches']}/128; "
              f"joint p95 {at_threshold['95']['accepted']}/128; median L1 {at_threshold['95']['median_distance']:,.0f}",flush=True)

payload = dict(method='Genuine responses only; UUID-matched; minimum L1 to eight registration runs; reject ties.',
               threshold_policy='Calibrated exclusively on registration collection run index 9; fixed for all verification collections.',
               evaluation_policy='Uniform seeds only; independent evaluation run index 8 from each verification collection.',
               scope='Four short-timescale collections on the same eight H200 UUIDs; no established power-cycle or environmental intervention.',
               collections=source_info,pairs=pairs)
(OUT/'cross_collection_verification.json').write_text(json.dumps(payload,indent=2)+'\n')

def matrix(q,field):
    lines=['| Registration collection | Verify 1 | Verify 2 | Verify 3 | Verify 4 |',
           '|---|---:|---:|---:|---:|']
    for src in range(1,5):
        cells=[]
        for dst in range(1,5):
            value=next(p for p in pairs if p['registration_run']==src and p['verification_run']==dst)['operating_points'][q]
            n=value[field]
            cells.append(f'{n}/128 ({100*n/128:.1f}%)')
        lines.append('| '+str(src)+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)

report = ['# Cross-collection genuine verification','',
          'All analysis used saved local measurements. No GPU experiments or remote operations were performed.','',
          '## Method','',
          'For each registration collection, use its eight registration measurements per chip/seed and calibrate thresholds solely from its separate calibration measurement (index 9). Verify the separate genuine challenge measurement (index 8) from each collection against that fixed registry. Match chips by UUID and use the 16 uniform seeds common to all four collections. Reject equal-distance identification ties.','',
          'Each comparison contains 128 challenges: eight chips × 16 seeds. Cross-collection thresholds are not recalibrated using the verification collection. The diagonal is a within-collection control and reproduces the existing genuine results at all three thresholds.','',
          '## Collections','',
          '| Run | Start, November 7, 2026 (Mountain time) | p95 threshold |','|---|---|---:|']
for info in source_info:
    when=datetime.fromisoformat(info['created_utc']).astimezone(ZoneInfo('America/Denver'))
    report.append(f"| {info['run']} | {when.strftime('%I:%M:%S %p')} | {info['thresholds']['95']:,.1f} |")
report += ['', '## Joint acceptance at the registration collection’s p95 threshold','',
           'Requires correct nearest-chip identification and distance within the fixed threshold. Rows supply registration/calibration; columns supply verification responses.','',matrix('95','accepted'),'',
           '## Identification without a distance threshold','',matrix('95','matches'),'',
           '## Distance gate alone at p95','',matrix('95','distance_passes'),'',
           '## Joint acceptance at p90','',matrix('90','accepted'),'',
           '## Joint acceptance at p99','',matrix('99','accepted'),'',
           '## Interpretation limits','',
           'Only genuine-chip verification is evaluated here. These results do not establish cross-collection spoofing performance. Threshold differences can make forward and reverse comparisons asymmetric. Repeated challenges share chips, seeds and training measurements, so the rates are descriptive rather than independent-trial confidence estimates.','',
           'Run 1→2 and Run 3→4 remain within their respective rental instance IDs; Run 2→3 crosses rental instance IDs while retaining the same physical GPU UUIDs. Different rental IDs do not establish a power cycle or environmental change. Collection starts span about thirteen minutes; timestamps describe collection starts, not precise per-challenge timing. Comparisons to earlier collections are retrospective checks, not chronological deployment simulations.','',
           'The raw archive checksums and dimensions were validated before scoring. An independent integer-distance implementation reproduced every within-collection genuine control and calibration threshold. Full distances, target margins, seed IDs and results at all three thresholds are in cross_collection_verification.json.','']
(OUT/'cross_collection_verification.md').write_text('\n'.join(report),encoding='utf-8')
