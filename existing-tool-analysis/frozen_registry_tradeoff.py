#!/usr/bin/env python3
"""Offline frozen-registry genuine/spoof tradeoffs. Requires numpy; never uses GPUs/network.

python frozen_registry_tradeoff.py --data-root PATH --output NEW_DIRECTORY

Reads campaign_v5_data_*/ metadata, analysis and npz files. Does not modify input.
Verifier: source collection's eight registration runs, source-only calibration.
Spoof: destination collection's decoy evaluation responses, with corrections learned
either from source registration data or destination registration data, excluding the
challenged seed. Reports fixed single decoy and all-other-GPU ensemble separately.
Threshold sweeps and best-case thresholds are retrospective diagnostics, not held-out
calibration results. All rates are descriptive and trials share devices/training data.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

LAMBDAS = (1.0, 1.5, 2.0)
MODES = ('source_trained', 'destination_trained')


def load_collections(root):
    records = []
    for folder in root.glob('campaign_v5_data_*'):
        if not folder.is_dir():
            continue
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
        if meta['status'] != 'complete' or meta['roles'] != dict(
                registration=list(range(8)), evaluation=8, calibration=9):
            raise ValueError(f'Unexpected status/split: {folder}')
        records.append(dict(folder=folder, meta=meta))
    records.sort(key=lambda r: datetime.fromisoformat(r['meta']['created_utc']))
    if len(records) < 2:
        raise ValueError('Need at least two complete collection folders')
    uuids = [d['uuid'] for d in records[0]['meta']['devices']]
    seeds = records[0]['meta']['design']['uniform']
    if len(set(uuids)) != len(uuids) or len(uuids) < 2 or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError('Need distinct GPUs and seeds')
    common_shape = None
    for record in records:
        meta, folder = record['meta'], record['folder']
        devices = {d['uuid']: g for g,d in enumerate(meta['devices'])}
        if set(devices) != set(uuids) or set(meta['design']['uniform']) != set(seeds):
            raise ValueError('Collections must share GPU UUIDs and uniform seeds')
        by_gpu = []
        for gpu_uuid in uuids:
            index = devices[gpu_uuid]
            file = folder/f'gpu-{index:03d}.npz'
            detail = json.loads((folder/f'gpu-{index:03d}.json').read_text())
            if hashlib.sha256(file.read_bytes()).hexdigest() != detail['npz_sha256']:
                raise ValueError(f'Checksum mismatch: {file}')
            with np.load(file, allow_pickle=False) as archive:
                all_seeds = archive['seeds'].tolist()
                raw = archive['fingerprints']
                if len(set(all_seeds)) != len(all_seeds) or all_seeds != meta['design']['seeds']:
                    raise ValueError('Invalid seed labels')
                if raw.ndim != 4 or raw.shape[0] != len(all_seeds) or raw.shape[1] != 10:
                    raise ValueError('Unexpected raw dimensions')
                if not np.issubdtype(raw.dtype,np.integer) or raw.min() < 0 or raw.max() >= 2**29:
                    raise ValueError('Counter values outside safe scoring range')
                if common_shape is None:
                    common_shape = raw.shape[2:]
                if raw.shape[2:] != common_shape:
                    raise ValueError('Incompatible fingerprint shapes')
                by_gpu.append(np.stack([raw[all_seeds.index(s)] for s in seeds]).astype(np.int32))
        # Uniform seeds x UUID-aligned GPUs x runs x fingerprint elements.
        data = np.stack(by_gpu,axis=1)
        record['data'] = data
        record['means'] = data[:,:,:8].mean(axis=2)
        record['saved'] = json.loads((folder/'analysis.json').read_text())
        if record['saved']['session_id'] != meta['session_id'] or record['saved']['response_policy'] != 'truncate':
            raise ValueError('Unexpected saved analysis or response conversion')
    return records,uuids,seeds


def score(registry, candidate, target):
    candidate = np.asarray(candidate)
    if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) >= 2**29:
        raise ValueError('Prediction outside safe scoring range')
    # This is explicit response conversion, not implicit truncation inside L1.
    submitted = np.trunc(candidate).astype(np.int32)
    distances = np.abs(registry-submitted).sum(axis=(2,3),dtype=np.int64).min(axis=1)
    own = int(distances[target])
    other = min(int(x) for g,x in enumerate(distances) if g != target)
    return dict(distance=own,match=own<other,margin=own-other,tie=own==other)


def corrections_for(means):
    S,G,H,W = means.shape
    result = np.zeros((S,G,G,H,W),dtype=np.float32)
    for c in range(S):
        training = [s for s in range(S) if s != c]
        for t in range(G):
            for h in range(t+1,G):
                correction = np.median(means[training,t]-means[training,h],axis=0)
                converted = correction.astype(np.float32)
                if not np.array_equal(converted.astype(np.float64),correction):
                    raise ValueError('Correction cannot be represented exactly in float32 cache')
                result[c,t,h] = converted
                result[c,h,t] = -converted
    return result


def summary(rows, threshold):
    return dict(n=len(rows),matches=sum(r['match'] for r in rows),
                accepted=sum(r['match'] and r['distance']<=threshold for r in rows),
                distance_passes=sum(r['distance']<=threshold for r in rows))


def sorted_eligible(rows):
    return np.sort(np.asarray([r['distance'] for r in rows if r['match']],dtype=np.int64))


def required_threshold(rows, goal):
    eligible = sorted_eligible(rows)
    required = int(np.ceil(goal*len(rows)))
    if len(eligible) < required:
        return None
    return int(eligible[required-1])


def sweep(pair):
    eligible = {name: sorted_eligible(rows) for name,rows in pair['trials'].items()}
    thresholds = np.unique(np.concatenate([np.asarray([-1],dtype=np.int64)]+list(eligible.values())))
    # Eligible-score breakpoints cover every distinct decision state of <= T.
    return dict(thresholds=thresholds.tolist(),accepted_counts={
        name:np.searchsorted(values,thresholds,side='right').tolist() for name,values in eligible.items()})


def diagnostic_at_goal(pair, goal):
    threshold = required_threshold(pair['trials']['genuine'],goal)
    if threshold is None:
        return dict(goal=goal,attainable=False,reason='Nearest-chip identification limits genuine acceptance')
    results = {name:summary(rows,threshold) for name,rows in pair['trials'].items()}
    return dict(goal=goal,attainable=True,threshold=threshold,results=results)


def models():
    return [f'{mode}/{family}/x{lam}' for mode in MODES
            for family in ('single','ensemble') for lam in LAMBDAS]


def worst_rate(results, prefix=None):
    options = [(value['accepted']/value['n'],name) for name,value in results.items()
               if name!='genuine' and (prefix is None or name.startswith(prefix))]
    rate,name=max(options)
    return dict(rate=rate,model=name)


def analyze(records,uuids,seeds,old_genuine=None):
    G,S=len(uuids),len(seeds)
    for i,record in enumerate(records):
        print(f'Preparing correction cache for collection {i+1}',flush=True)
        record['corrections']=corrections_for(record['means'])
    pairs,collection_info=[],[]
    for src,source in enumerate(records):
        registry = source['data'][:,:,:8]
        calibration = [score(registry[c],source['data'][c,t,9],t) for c in range(S) for t in range(G)]
        thresholds = {str(q):float(np.percentile([r['distance'] for r in calibration],q)) for q in (90,95,99)}
        if any(thresholds[q] != source['saved']['operating_points'][q]['threshold'] for q in thresholds):
            raise ValueError('Calibration differs from reference')
        collection_info.append(dict(run=src+1,folder=source['folder'].name,
            session_id=source['meta']['session_id'],created_utc=source['meta']['created_utc'],thresholds=thresholds))
        for dst,destination in enumerate(records):
            trials={'genuine':[]}
            trials.update({name:[] for name in models()})
            for c,seed in enumerate(seeds):
                for t in range(G):
                    genuine=score(registry[c],destination['data'][c,t,8],t)
                    genuine.update(target_uuid=uuids[t],seed=seed)
                    trials['genuine'].append(genuine)
                    h_fixed=(t+1)%G
                    for mode,training in (('source_trained',source),('destination_trained',destination)):
                        for lam in LAMBDAS:
                            predictions=[]
                            single=None
                            for h in range(G):
                                if h==t:
                                    continue
                                prediction=(destination['data'][c,h,8].astype(np.float64)+
                                            lam*training['corrections'][c,t,h].astype(np.float64))
                                predictions.append(prediction)
                                if h==h_fixed:
                                    single=prediction
                            candidates=dict(single=single,ensemble=np.median(np.stack(predictions),axis=0))
                            for family,prediction in candidates.items():
                                row=score(registry[c],prediction,t)
                                row.update(target_uuid=uuids[t],seed=seed)
                                trials[f'{mode}/{family}/x{lam}'].append(row)
            fixed={q:dict(threshold=T,results={name:summary(rows,T) for name,rows in trials.items()})
                   for q,T in thresholds.items()}
            if src==dst:
                for q,point in fixed.items():
                    for lam in LAMBDAS:
                        for family,oldname in [('single','single_decoy'),('ensemble',f'ensemble_{G-1}_decoys')]:
                            reference=source['saved']['operating_points'][q]['results'][f'{oldname}_x{lam}']
                            actual=point['results'][f'source_trained/{family}/x{lam}']
                            if any(actual[k]!=reference[k] for k in ('n','matches','accepted')):
                                raise ValueError('Within-collection spoof control differs from saved analysis')
                if trials['source_trained/single/x1.0']!=trials['destination_trained/single/x1.0']:
                    raise ValueError('Same-collection training modes differ')
            if old_genuine:
                reference=next(p for p in old_genuine['pairs'] if p['registration_run']==src+1 and p['verification_run']==dst+1)
                for q in thresholds:
                    actual=fixed[q]['results']['genuine']
                    if any(actual[k]!=reference['operating_points'][q][k] for k in ('n','matches','accepted')):
                        raise ValueError('Genuine comparison differs from prior cross-collection analysis')
            pair=dict(registration_run=src+1,verification_run=dst+1,trials=trials,fixed_operating_points=fixed)
            pair['threshold_sweep']=sweep(pair)
            pair['retrospective_goals']=[diagnostic_at_goal(pair,g) for g in (.90,.95,.99)]
            pairs.append(pair)
            point=fixed['95']['results']
            counts=[point['genuine']['accepted'],point['source_trained/single/x1.0']['accepted'],
                    point['source_trained/ensemble/x1.0']['accepted'],
                    point['destination_trained/ensemble/x1.0']['accepted']]
            print(f'Register {src+1} -> verify {dst+1}: genuine/source-single/source-ensemble/refreshed-ensemble {counts} / {S*G}',flush=True)
    # One threshold must work on EVERY chronological later collection, not just a pooled average.
    global_goals=[]
    for src in range(1,len(records)):
        future=[p for p in pairs if p['registration_run']==src and p['verification_run']>src]
        for goal in (.90,.95,.99):
            required=[required_threshold(p['trials']['genuine'],goal) for p in future]
            result=dict(registration_run=src,goal=goal,verification_runs=[p['verification_run'] for p in future])
            if any(x is None for x in required):
                result.update(attainable=False,reason='At least one later collection cannot meet the genuine goal even without a distance gate')
            else:
                threshold=max(required)
                per_run=[]
                for pair in future:
                    values={name:summary(rows,threshold) for name,rows in pair['trials'].items()}
                    per_run.append(dict(verification_run=pair['verification_run'],results=values))
                result.update(attainable=True,threshold=threshold,per_run=per_run,
                    worst_observed_spoof_rate=max(worst_rate(row['results'])['rate'] for row in per_run),
                    worst_source_trained_rate=max(worst_rate(row['results'],'source_trained/') ['rate'] for row in per_run),
                    worst_destination_trained_rate=max(worst_rate(row['results'],'destination_trained/')['rate'] for row in per_run))
            global_goals.append(result)
    return dict(schema_version=1,collections=collection_info,models=models(),pairs=pairs,
        retrospective_one_threshold_all_future_collections=global_goals,
        methods=dict(registry='Source collection runs 0..7, UUID aligned, minimum L1 distance; ties rejected',
          calibration='Source collection run 9 only; fixed p90/p95/p99 applied to every destination',
          genuine='Destination collection run 8, uniform seeds only',
          source_trained='Correction from source registration means over all other uniform seeds; no destination target responses used',
          destination_trained='Correction refreshed using destination registration means over other uniform seeds; optimistic access to target training data under destination conditions',
          decoys='Destination run 8; fixed single (target+1)%G or median ensemble of all G-1 other GPUs',
          response='Explicit truncation toward zero before exact integer L1; lambdas 1,1.5,2',
          sweep='All observed eligible-score breakpoints; retrospective, not independent threshold validation'),
        limits=['Finite observed challenges and specified attack models only; cannot prove security or universal impossibility.',
                'Rates share seeds, chips and training data; no independent-trial confidence claims.',
                'Source-trained corrections exclude the challenge seed; actual verifier registry reuse for training seeds is an evaluation assumption.',
                'Destination-trained mode is not proof that an attacker can obtain same-condition target training data chronologically before each challenge.',
                'Four collections on same eight H200s over about thirteen minutes; no established power-cycle/transport intervention.',
                'No attack latency, network relay, or replay-table study is performed here.'])


def make_report(result):
    N=len(result['collections'])
    lines=['# Frozen-registry joint genuine/spoof tradeoff','',
        'Offline analysis of the four saved datasets. No GPU experiments, rentals, or network operations were performed.','',
        '## Method','',
        'Rows below supply the frozen verifier: eight registration runs per seed/chip and a threshold calibrated only from that collection’s separate calibration run. Columns supply genuine responses and fresh decoy responses from their independent evaluation run. Chips are matched by UUID. Each entry has 128 challenges (eight chips × 16 uniform seeds).','',
        'Source-trained corrections use only the registration collection’s training seeds. Destination-trained corrections use the later collection’s registration samples on other seeds; this models an attacker able to refresh its target/decoy training data under those conditions. Both exclude the challenged seed. The single decoy is fixed in advance; the ensemble uses all seven other GPUs. Synthetic responses are explicitly truncated before scoring.','',
        '## Fixed source-calibrated p95 results','']
    for name,title in [('genuine','Genuine joint acceptance'),
            ('source_trained/single/x1.0','Single decoy ×1, source-trained correction'),
            ('source_trained/ensemble/x1.0','Seven-decoy ensemble ×1, source-trained correction'),
            ('destination_trained/single/x1.0','Single decoy ×1, destination-trained correction'),
            ('destination_trained/ensemble/x1.0','Seven-decoy ensemble ×1, destination-trained correction')]:
        lines += [f'### {title}','',
            '| Register ↓ / Verify → | '+' | '.join(f'Run {i}' for i in range(1,N+1))+' |',
            '|---|'+'---:|'*N]
        for src in range(1,N+1):
            cells=[]
            for dst in range(1,N+1):
                pair=next(p for p in result['pairs'] if p['registration_run']==src and p['verification_run']==dst)
                value=pair['fixed_operating_points']['95']['results'][name]
                cells.append(f"{value['accepted']}/{value['n']} ({100*value['accepted']/value['n']:.1f}%)")
            lines.append('| '+str(src)+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Retrospective bound: one threshold for every later collection','',
        'For each source, find the smallest single threshold that accepts at least the stated percentage of genuine challenges in **each** later collection. Since acceptance is monotone in the threshold, these are the lowest achievable spoof rates at that genuine requirement on these exact samples. Thresholds are selected using evaluation data here: they are optimistic diagnostic bounds, not deployment-calibrated thresholds.','',
        'Worst spoof rate is the maximum observed acceptance across the tested multipliers, single/ensemble models, and later collections. It is not a pooled average or a newly validated attacker-selection strategy.','',
        '| Register | Later runs | Genuine requirement per later run | Smallest threshold | Worst source-trained spoof | Worst destination-trained spoof |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['retrospective_one_threshold_all_future_collections']:
        prefix=f"| {row['registration_run']} | {', '.join(map(str,row['verification_runs']))} | {100*row['goal']:.0f}% |"
        if not row['attainable']:
            lines.append(prefix+' unattainable due to identification errors | — | — |')
        else:
            lines.append(prefix+f" {row['threshold']:,.0f} | {100*row['worst_source_trained_rate']:.1f}% | {100*row['worst_destination_trained_rate']:.1f}% |")
    lines += ['', '## Validation','',
        'All input array checksums were verified. Every within-collection single-decoy/ensemble result at p90/p95/p99 reproduces the saved analysis. The genuine cross-collection results also reproduce the prior independent analysis. All score breakpoints, per-trial distances/margins, fixed-threshold results and retrospective pairwise bounds are retained in results.json.','',
        '## Limits','']+[f'- {x}' for x in result['limits']]
    return '\n'.join(lines)+'\n'


def self_check():
    # Exact observed-sample threshold bound, tie rejection, and L1 conversion.
    rows=[dict(match=True,distance=10),dict(match=True,distance=20),dict(match=False,distance=0)]
    assert required_threshold(rows,.5)==20
    assert required_threshold(rows,1.0) is None
    registry=np.asarray([[[[0]]],[[[2]]]],dtype=np.int32)
    assert score(registry,np.asarray([[1.9]]),0)['tie']
    pair=dict(trials=dict(genuine=rows,attack=[dict(match=True,distance=15)]))
    curve=sweep(pair)
    assert curve['thresholds']==[-1,10,15,20]
    assert curve['accepted_counts']['genuine']==[0,1,1,2]
    assert curve['accepted_counts']['attack']==[0,0,1,1]


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--genuine-reference',type=Path,help='Optional previous cross-collection JSON for extra validation')
    args=parser.parse_args()
    self_check()
    args.output.mkdir(parents=True,exist_ok=False)
    records,uuids,seeds=load_collections(args.data_root)
    old=json.loads(args.genuine_reference.read_text()) if args.genuine_reference else None
    result=analyze(records,uuids,seeds,old)
    result['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(make_report(result),encoding='utf-8')
    print('\nOne-threshold forward-collection bounds:',flush=True)
    for row in result['retrospective_one_threshold_all_future_collections']:
        compact={k:v for k,v in row.items() if k!='per_run'}
        print(json.dumps(compact),flush=True)
    print(f'Saved {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
