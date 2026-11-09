#!/usr/bin/env python3
"""Additional offline analyses. Requires numpy and adjacent frozen_registry_tradeoff.py.

No network, rental, compilation, or GPU operations. Inputs remain read-only.
All 90 role splits evaluate genuine verification; 10 balanced cyclic splits also
evaluate source-trained x1 single/7-decoy spoofs. Default-role ablations and defenses
are evaluated on within-collection controls plus chronological forward comparisons.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np

HELPER=Path(__file__).with_name('frozen_registry_tradeoff.py')
spec=importlib.util.spec_from_file_location('reviewed_frozen_analysis',HELPER)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def counts(own,match,T):
    own=np.asarray(own); match=np.asarray(match)
    return dict(n=int(own.size),matches=int(match.sum()),accepted=int((match & (own<=T)).sum()))


def own_and_match(distances):
    G=distances.shape[-1]
    own=np.diagonal(distances,axis1=1,axis2=2)
    others=np.where(np.eye(G,dtype=bool)[None],np.iinfo(np.int64).max,distances).min(axis=2)
    return own,own<others


def distances(registry,prediction):
    candidate=np.trunc(prediction).astype(np.int32)
    return np.abs(registry-candidate).sum(axis=(2,3),dtype=np.int64).min(axis=1)


def raw_cache(records,cache_path):
    signature=hashlib.sha256(b''.join(hashlib.sha256(r['data'].tobytes()).digest() for r in records)).hexdigest()
    if cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as z:
            if str(z['signature'])==signature:
                print('Loading validated pairwise-distance cache',flush=True)
                return z['distances'].copy()
    N=len(records); S,G,R,H,W=records[0]['data'].shape
    cache=np.empty((N,N,S,G,R,G,R),dtype=np.int64)
    for src in range(N):
        for dst in range(src,N):
            started=time.perf_counter()
            for s in range(S):
                registry=records[src]['data'][s].reshape(G*R,H,W)
                candidates=records[dst]['data'][s].reshape(G*R,H,W)
                block=np.stack([np.abs(registry-x).sum(axis=(1,2),dtype=np.int64) for x in candidates])
                cache[src,dst,s]=block.reshape(G,R,G,R)
            cache[dst,src]=cache[src,dst].transpose(0,3,4,1,2)
            print(f'Raw distance cache {src+1}<->{dst+1}: {time.perf_counter()-started:.1f}s',flush=True)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,signature=np.asarray(signature),distances=cache)
    return cache


def genuine_scores(cache,src,dst,evaluation,registration):
    d=cache[src,dst,:,:,evaluation,:,:][...,registration].min(axis=-1)
    return own_and_match(d)


def all_genuine_splits(cache):
    N=cache.shape[0]; rows=[]
    for evaluation in range(10):
        for calibration in range(10):
            if evaluation==calibration: continue
            registration=[r for r in range(10) if r not in (evaluation,calibration)]
            for src in range(N):
                cal,_=genuine_scores(cache,src,src,calibration,registration)
                T=float(np.percentile(cal,95))
                targets=[]
                for dst in range(N):
                    own,match=genuine_scores(cache,src,dst,evaluation,registration)
                    targets.append(dict(verification_run=dst+1,**counts(own,match,T)))
                rows.append(dict(registration_run=src+1,evaluation=evaluation,calibration=calibration,
                                 registration_indices=registration,threshold=T,targets=targets))
    return dict(method='All 90 ordered calibration/evaluation assignments; remaining eight runs register. Same roles applied across collections. p95 calibration is source-only.',rows=rows)


def attribution(reference,records,uuids,seeds):
    failures_chip=Counter(); failures_seed=Counter(); breakdown=[]; tails=[]
    for pair in reference['pairs']:
        src,dst=pair['registration_run'],pair['verification_run']
        if dst<=src: continue
        T=pair['fixed_operating_points']['95']['threshold']
        for model in ('genuine','source_trained/single/x1.0','source_trained/ensemble/x1.0',
                      'destination_trained/single/x1.0','destination_trained/ensemble/x1.0'):
            bychip=Counter(); byseed=Counter(); n=0; total=0
            for row in pair['trials'][model]:
                event=(not(row['match'] and row['distance']<=T)) if model=='genuine' else (row['match'] and row['distance']<=T)
                total+=1
                if event:
                    n+=1; bychip[row['target_uuid']]+=1; byseed[row['seed']]+=1
                    if model=='genuine': failures_chip[row['target_uuid']]+=1; failures_seed[row['seed']]+=1
            breakdown.append(dict(registration_run=src,verification_run=dst,model=model,event_count=n,n=total,
                                  by_chip=dict(bychip),by_seed=dict(byseed)))
    for index,record in enumerate(records):
        rows=record['saved']['trials']['calibration']
        T=record['saved']['operating_points']['95']['threshold']
        ranked=sorted(rows,key=lambda r:r['distance'],reverse=True)
        top=[dict(chip_uuid=uuids[r['target']],seed=r['seed'],distance=r['distance']) for r in ranked[:10]]
        above=[r for r in rows if r['distance']>T]
        tails.append(dict(run=index+1,threshold=T,above_threshold_n=len(above),
                          tail_by_chip=dict(Counter(uuids[r['target']] for r in above)),
                          tail_by_seed=dict(Counter(r['seed'] for r in above)),top_ten=top))
    return dict(scope='Six chronological forward comparisons. Repeated target challenges scored against multiple registries are repeated events, not independent trials.',
                genuine_failure_by_chip=dict(failures_chip),genuine_failure_by_seed=dict(failures_seed),
                breakdown=breakdown,calibration_tails=tails)


def balanced_spoof_splits(records,cache):
    N=len(records); S,G=records[0]['data'].shape[:2]; results=[]; global_bounds=[]
    for evaluation in range(10):
        calibration=(evaluation+1)%10
        registration=[r for r in range(10) if r not in (evaluation,calibration)]
        for src,source in enumerate(records):
            means=source['data'][:,:,registration].mean(axis=2)
            correction=base.corrections_for(means)
            registry=source['data'][:,:,registration]
            cal,_=genuine_scores(cache,src,src,calibration,registration)
            T=float(np.percentile(cal,95))
            future=[]
            for dst in range(src,N):
                destination=records[dst]
                own,match=genuine_scores(cache,src,dst,evaluation,registration)
                models={'single':[],'ensemble':[]}
                for c in range(S):
                    for t in range(G):
                        other=[h for h in range(G) if h!=t]
                        predicted={h:destination['data'][c,h,evaluation].astype(np.float64)+correction[c,t,h]
                                   for h in other}
                        candidates=dict(single=predicted[(t+1)%G],ensemble=np.median(np.stack(list(predicted.values())),axis=0))
                        for model,p in candidates.items():
                            d=distances(registry[c],p)
                            models[model].append(dict(distance=int(d[t]),match=bool(d[t]<min(d[h] for h in other))))
                genuine=[dict(distance=int(d),match=bool(m)) for d,m in zip(own.ravel(),match.ravel())]
                item=dict(registration_run=src+1,verification_run=dst+1,evaluation=evaluation,
                          calibration=calibration,threshold=T,genuine=base.summary(genuine,T),
                          single=base.summary(models['single'],T),ensemble=base.summary(models['ensemble'],T))
                results.append(item)
                if dst>src: future.append(dict(genuine=genuine,models=models,dst=dst+1))
                if evaluation==8 and calibration==9 and src==dst:
                    for model,old in [('single','single_decoy'),('ensemble',f'ensemble_{G-1}_decoys')]:
                        expected=source['saved']['operating_points']['95']['results'][f'{old}_x1.0']
                        if any(item[model][k]!=expected[k] for k in ('n','matches','accepted')):
                            raise ValueError('Balanced split default control failed')
            if future:
                required=[base.required_threshold(p['genuine'],.95) for p in future]
                bound=dict(registration_run=src+1,evaluation=evaluation,calibration=calibration,
                           genuine_goal=.95,attainable=all(x is not None for x in required))
                if bound['attainable']:
                    needed=max(required)
                    bound.update(threshold=needed,worst_x1_spoof_rate=max(
                        base.summary(rows,needed)['accepted']/len(rows) for p in future for rows in p['models'].values()))
                eligible=[r['distance'] for p in future for rows in p['models'].values() for r in rows if r['match']]
                no_spoof_T=min(eligible)-1 if eligible else max(r['distance'] for p in future for r in p['genuine'])
                bound['zero_observed_spoof_max_min_genuine']=min(base.summary(p['genuine'],no_spoof_T)['accepted']/len(p['genuine']) for p in future)
                global_bounds.append(bound)
            print(f'Balanced split eval={evaluation}, cal={calibration}, registry {src+1} finished',flush=True)
            del correction
    return dict(method='Ten cyclic ordered held-out pairs: eval e, cal (e+1)%10. Each measurement fills each held-out role once. Source-trained x1 single/7-decoy attacks only. Other eight runs register. Not independent replications.',
                results=results,retrospective_bounds=global_bounds)


def rank_model(d,target,threshold,scale=None):
    values=d if scale is None else d/scale
    own=float(values[target]); other=min(float(x) for g,x in enumerate(values) if g!=target)
    return bool(own<other and own<=threshold)


def default_ablation_and_defenses(records,cache):
    N=len(records); S,G=records[0]['data'].shape[:2]
    for i,r in enumerate(records):
        print(f'Default-role correction cache {i+1}',flush=True)
        r['correction']=base.corrections_for(r['means'])
    ablation=[]; defense=[]; defense_details=[]
    for src,source in enumerate(records):
        registry=source['data'][:,:,:8]
        cal_dist=cache[src,src,:,:,9,:,:8].min(axis=-1)
        cal_own,_=own_and_match(cal_dist)
        # Registration-only leave-one-out scores, seven comparison samples each.
        within=cache[src,src,:,:8,:,:8] if False else None
        training_own=[]
        for s in range(S):
            for t in range(G):
                block=cache[src,src,s,t,:8,t,:8].copy()
                np.fill_diagonal(block,np.iinfo(np.int64).max)
                training_own.append((s,t,block.min(axis=1)))
        scales_chip=np.asarray([np.median(np.concatenate([x for s,t,x in training_own if t==g])) for g in range(G)])
        scales_seed=np.asarray([np.median(np.concatenate([x for s,t,x in training_own if s==c])) for c in range(S)])
        scales_chip=np.maximum(scales_chip,1); scales_seed=np.maximum(scales_seed,1)
        rules={}
        for q in (90,95,99):
            rules[str(q)]=dict(pooled=float(np.percentile(cal_own,q)),
                chip_threshold=np.percentile(cal_own,q,axis=0),
                chip_normalized=float(np.percentile(cal_own/scales_chip[None,:],q)),
                seed_normalized=float(np.percentile(cal_own/scales_seed[:,None],q)))
        defense_details.append(dict(registration_run=src+1,chip_scales=scales_chip.tolist(),seed_scales=scales_seed.tolist(),
            thresholds={q:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in rule.items()} for q,rule in rules.items()}))
        for dst in range(src,N):
            destination=records[dst]
            # Raw nearest-chip vectors are kept transiently, not serialized thousands of times.
            vectors={'genuine':[]}
            for mode in base.MODES:
                for family in ('single','ensemble'):
                    for lam in base.LAMBDAS: vectors[f'{mode}/{family}/x{lam}']=[]
            configs={(mode,k,offset):[] for mode in base.MODES for k in (1,2,4,7)
                     for offset in (range(7) if k<7 else range(1))}
            for c in range(S):
                for t in range(G):
                    true_dist=cache[src,dst,c,t,8,:,:8].min(axis=-1)
                    vectors['genuine'].append((c,t,true_dist))
                    others=[(t+offset)%G for offset in range(1,G)]
                    for mode,training in [('source_trained',source),('destination_trained',destination)]:
                        predictions={h:destination['data'][c,h,8].astype(np.float64)+training['correction'][c,t,h]
                                     for h in others}
                        for lam in base.LAMBDAS:
                            scaled={h:destination['data'][c,h,8].astype(np.float64)+lam*training['correction'][c,t,h]
                                    for h in others}
                            for family,p in [('single',scaled[(t+1)%G]),('ensemble',np.median(np.stack(list(scaled.values())),axis=0))]:
                                d=distances(registry[c],p)
                                vectors[f'{mode}/{family}/x{lam}'].append((c,t,d))
                        for k in (1,2,4,7):
                            for offset in (range(7) if k<7 else range(1)):
                                selected=[others[(offset+j)%7] for j in range(k)]
                                prediction=np.median(np.stack([predictions[h] for h in selected]),axis=0)
                                d=distances(registry[c],prediction)
                                configs[(mode,k,offset)].append((c,t,d))
            for q,rule in rules.items():
                for rule_name in ('pooled','chip_threshold','chip_normalized','seed_normalized'):
                    evaluated={}
                    for model,rows in vectors.items():
                        accepted=0; matched=0
                        for c,t,d in rows:
                            scale=scales_chip if rule_name=='chip_normalized' else None
                            scores=d/scale if scale is not None else d
                            matched+=int(scores[t]<min(scores[g] for g in range(G) if g!=t))
                            if rule_name=='chip_threshold': T=rule['chip_threshold'][t]
                            elif rule_name=='seed_normalized': T=rule['seed_normalized']*scales_seed[c]
                            else: T=rule[rule_name]
                            accepted+=int(rank_model(d,t,T,scale))
                        evaluated[model]=dict(n=len(rows),accepted=accepted,matches=matched)
                    defense.append(dict(registration_run=src+1,verification_run=dst+1,percentile=int(q),rule=rule_name,results=evaluated))
            T=rules['95']['pooled']
            for (mode,k,offset),rows in configs.items():
                accepted=sum(rank_model(d,t,T) for c,t,d in rows)
                ablation.append(dict(registration_run=src+1,verification_run=dst+1,mode=mode,decoys=k,
                                     assignment_offset=offset,n=len(rows),accepted=accepted))
            # Check x1 fixed cyclic assignment and seven-decoy results against raw-scoring baseline.
            for mode in base.MODES:
                for k,family in [(1,'single'),(7,'ensemble')]:
                    a=next(r for r in ablation if r['registration_run']==src+1 and r['verification_run']==dst+1 and r['mode']==mode and r['decoys']==k and r['assignment_offset']==0)
                    b=next(r for r in defense if r['registration_run']==src+1 and r['verification_run']==dst+1 and r['rule']=='pooled' and r['percentile']==95)
                    if a['accepted']!=b['results'][f'{mode}/{family}/x1.0']['accepted']:
                        raise ValueError('Ablation/defense baseline mismatch')
            if src==dst:
                ref=source['saved']['operating_points']['95']['results']
                b=next(r for r in defense if r['registration_run']==src+1 and r['verification_run']==dst+1 and r['rule']=='pooled' and r['percentile']==95)
                for lam in base.LAMBDAS:
                    for family,old in [('single','single_decoy'),('ensemble','ensemble_7_decoys')]:
                        if b['results'][f'source_trained/{family}/x{lam}']['accepted'] != ref[f'{old}_x{lam}']['accepted']:
                            raise ValueError('Default-role attack control mismatch')
            print(f'Ablation and defenses {src+1}->{dst+1} finished',flush=True)
    return (dict(method='Default split, x1 correction, fixed pooled source p95. Seven predeclared cyclic assignments for 1/2/4 decoys; one all-seven ensemble. All seven individual decoy choices covered; multi-decoy subsets are not exhaustive. No best-subset selection on test data.',rows=ablation),
            dict(method='Default split. Thresholds from source calibration only at p90/p95/p99; scales from source registration leave-one-out scores only. Evaluate all baseline attack families and multipliers. Chip-specific thresholds use only 16 calibration scores per chip.',
                 details=defense_details,rows=defense))


def span(values):
    return dict(min=float(np.min(values)),median=float(np.median(values)),max=float(np.max(values)))


def report(output,split,attribution_data,balanced,ablation,defenses):
    lines=['# Additional offline analyses','',
        'Four saved collections on the same eight H200s. No rentals, GPU experiments or remote operations were performed. All raw archives were checksum-validated.','',
        '## 1. Sample-split sensitivity','',
        'Genuine verification uses all 90 ordered calibration/evaluation assignments; the other eight measurements register. The 10 cyclic splits additionally evaluate source-trained x1 single-decoy and seven-decoy ensemble spoofs. These splits overlap heavily and are sensitivity checks, not independent replications.','',
        '| Registry collection | Original p95 | p95 across all 90 splits: min / median / max |',
        '|---|---:|---:|']
    for src in range(1,5):
        vals=[r['threshold'] for r in split['rows'] if r['registration_run']==src]
        original=next(r['threshold'] for r in split['rows'] if r['registration_run']==src and r['evaluation']==8 and r['calibration']==9)
        lines.append(f'| {src} | {original:,.0f} | {min(vals):,.0f} / {np.median(vals):,.0f} / {max(vals):,.0f} |')
    lines += ['', '| Registry | Balanced splits with attainable 95% genuine goal in every later collection | Minimum / median / maximum worst x1 spoof rate at that goal |',
              '|---|---:|---:|']
    for src in range(1,4):
        bounds=[r for r in balanced['retrospective_bounds'] if r['registration_run']==src]
        rates=[r['worst_x1_spoof_rate'] for r in bounds if r['attainable']]
        text='—' if not rates else ' / '.join(f'{100*x:.1f}%' for x in (min(rates),np.median(rates),max(rates)))
        lines.append(f'| {src} | {len(rates)}/{len(bounds)} | {text} |')
    lines += ['', 'The 95% bounds are retrospective choices using later genuine scores; they are not held-out calibrations. Rotating roles changes registry samples as well as calibration/evaluation samples, so this does not isolate physical drift from sampling effects.','',
        '## 2. Failure concentration','',
        'Genuine rejection events below pool the six chronological forward comparisons at their original source p95 thresholds. A later response can be rejected by multiple source registries; event counts are not independent challenges.','',
        '| Chip (UUID suffix) | Genuine rejection events | Share of rejection events |','|---|---:|---:|']
    total=sum(attribution_data['genuine_failure_by_chip'].values())
    for gpu,n in sorted(attribution_data['genuine_failure_by_chip'].items(),key=lambda p:-p[1]):
        lines.append(f'| {gpu[-12:]} | {n} | {100*n/total:.1f}% |')
    lines += ['', 'Top seeds by genuine rejection events: '+', '.join(f'{seed}: {n}' for seed,n in sorted(attribution_data['genuine_failure_by_seed'].items(),key=lambda p:-p[1])[:5])+'.','',
        'The JSON includes per-pair, per-chip and per-seed genuine rejection and spoof-acceptance breakdowns, plus each collection’s largest calibration scores. These identify concentration, not a proven hardware cause.','',
        '## 3. Decoy-count ablation','',
        'Ranges are across seven predetermined assignments, not confidence intervals. Each assignment stays fixed across targets/seeds by cyclic GPU order. Values use the frozen pooled source p95 and x1 correction. The seven-decoy ensemble has one assignment.','',
        '| Register → later verify | Training | 1 decoy | 2 decoys | 4 decoys | 7 decoys |',
        '|---|---|---:|---:|---:|---:|']
    for src,dst in itertools.combinations(range(1,5),2):
        for mode in base.MODES:
            cells=[]
            for k in (1,2,4,7):
                rates=[r['accepted']/r['n'] for r in ablation['rows'] if r['registration_run']==src and r['verification_run']==dst and r['mode']==mode and r['decoys']==k]
                cells.append(f'{100*min(rates):.1f}–{100*max(rates):.1f}%' if len(rates)>1 else f'{100*rates[0]:.1f}%')
            lines.append(f'| {src}→{dst} | {mode} | '+' | '.join(cells)+' |')
    lines += ['', '## 4. Source-trained defense experiments','',
        'Four rules: original pooled threshold; per-chip calibration threshold with raw nearest-chip matching; chip-normalized matching and threshold; and seed-normalized threshold with raw matching. Chip/seed scales are median leave-one-out distances from source registration samples, floored at one. All cutoffs use only source calibration. No defense is tuned on later evaluation outcomes.','',
        'The table uses p95. Spoof rate is the highest observed across the tested fixed-single and seven-decoy families, both training modes and multipliers 1/1.5/2. It is not a pooled average or a deployable best-attack selection policy.','',
        '| Rule | Minimum / median / maximum genuine acceptance across six forward pairs | Minimum / median / maximum worst spoof acceptance |','|---|---:|---:|']
    for rule in ('pooled','chip_threshold','chip_normalized','seed_normalized'):
        rows=[r for r in defenses['rows'] if r['percentile']==95 and r['rule']==rule and r['verification_run']>r['registration_run']]
        gen=[r['results']['genuine']['accepted']/r['results']['genuine']['n'] for r in rows]
        spoof=[max(v['accepted']/v['n'] for k,v in r['results'].items() if k!='genuine') for r in rows]
        fmt=lambda x:' / '.join(f'{100*y:.1f}%' for y in (min(x),np.median(x),max(x)))
        lines.append(f'| {rule} | {fmt(gen)} | {fmt(spoof)} |')
    lines += ['', 'Paired defense results at all three source-calibrated percentiles are in defense_experiments.json. The ranges above do not imply that minimum genuine and maximum spoof values occur in the same comparison. Per-chip p95 is particularly uncertain because it uses 16 calibration scores. These candidate defenses are exploratory and evaluated against the existing attacks, not adaptively redesigned attacks.','',
        '## Scope','',
        'All analyses share the same four short-timescale collections, chips and seeds. They do not establish transport, power-cycle or long-term stability. Split and assignment ranges are descriptive, not confidence intervals. The balanced-split spoof study uses x1 source-trained attacks only; the defense study includes the original three multipliers and both training modes.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    records,uuids,seeds=base.load_collections(args.data_root)
    if len(records)!=4 or len(uuids)!=8: raise ValueError('This report expects four collections on eight chips')
    cache=raw_cache(records,args.cache)
    reference=json.loads(args.reference.read_text())
    def phase(name,fn):
        file=args.output/name
        if args.resume and file.exists(): return json.loads(file.read_text())
        value=fn(); save(file,value); return value
    split=phase('all_90_genuine_splits.json',lambda:all_genuine_splits(cache))
    attribution_data=phase('failure_attribution.json',lambda:attribution(reference,records,uuids,seeds))
    print('All 90 genuine splits and attribution saved',flush=True)
    balanced=phase('balanced_split_spoofs.json',lambda:balanced_spoof_splits(records,cache))
    if args.resume and (args.output/'decoy_count_ablation.json').exists() and (args.output/'defense_experiments.json').exists():
        ablation=json.loads((args.output/'decoy_count_ablation.json').read_text())
        defenses=json.loads((args.output/'defense_experiments.json').read_text())
    else:
        ablation,defenses=default_ablation_and_defenses(records,cache)
        save(args.output/'decoy_count_ablation.json',ablation)
        save(args.output/'defense_experiments.json',defenses)
    report(args.output,split,attribution_data,balanced,ablation,defenses)
    save(args.output/'provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),session_ids=[r['meta']['session_id'] for r in records],
        models_and_controls='Checksummed raw arrays; default split controls; overlapping splits, no independent-replication claims.'))
    print(f'Completed: {args.output}',flush=True)


if __name__=='__main__': main()
