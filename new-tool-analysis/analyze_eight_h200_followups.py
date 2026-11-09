from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np

work = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
project = Path(__file__).resolve().parents[1]  # repository root
sys.path[:0] = [str(project/'new-tool-analysis'), str(project/'existing-tool-analysis'),
                str(project/'new-tool-analysis/gpu_fingerprinting_hardened_v1')]
import eight_h200_ensemble as e
import matched_h200 as m

os.chdir(work)
seal = m.read_json(e.SEAL)
eval_hashes = m.read_json(Path('results/ensemble-evaluation-hashes.json'))
for path, expected in (seal['training_hashes'] | eval_hashes).items():
    assert m.sha(Path(path)) == expected, path
nonces, _ = m.load_pool(e.POOL)
phases = e.load_arm(m, 'hardened', seal, nonces)
reg, cal, held = (phases[k].astype(np.int16) for k in e.REPEATS)
del phases
labels = list(seal['devices'])
target = labels.index('C')
idx = np.arange(len(labels))
means = reg.mean(axis=(2,3))
coordinates = int(np.prod(reg.shape[-2:]))
with np.load('results/hardened-ensemble-scores.npz', allow_pickle=False) as npz:
    saved = dict(npz)
rule = m.read_json(Path('results/ensemble-summary.json'))['results']['hardened']['frozen']
upper = np.asarray(rule['q95']['thresholds'])
pooled = np.asarray(rule['historical_pooled_p95']['thresholds'])
floor_q95 = np.sort(saved['calibration'][idx,:,:,idx],axis=-1)[:,:,1].min(axis=1)


def project_ranks(array):
    return np.argsort(np.argsort(array, axis=-1, kind='stable'),axis=-1,kind='stable').astype(np.int16)


def corrected(g, c):
    train = np.arange(len(nonces)) != c
    decoys = [i for i in idx if i != g]
    return decoys, np.stack([held[i,c]+np.median(means[g,train]-means[i,train],axis=0) for i in decoys])


def variability(ranks):
    differences = np.stack([np.abs(ranks[...,a,:,:]-ranks[...,b,:,:]).mean(axis=(-2,-1))
                            for a,b in ((0,1),(0,2),(1,2))],axis=-1)
    return np.median(differences,axis=-1)


def accept(table, limits):
    ordered = np.sort(table,axis=-1)
    winner = (table==ordered[...,:1]) & (ordered[...,:1]<ordered[...,1:2])
    return (winner & (table<=limits)).sum(axis=-2) >= 2


def describe(values):
    return {key:float(np.percentile(values,p)) for key,p in [('min',0),('median',50),('p95',95),('max',100)]}


def template_features(ranks, templates):
    residuals = np.abs(ranks[:,:,None,:,:]-templates[None,None,:,:,:]).mean(axis=(-2,-1))
    return np.median(np.median(residuals,axis=2),axis=1)


print('Computing genuine and projected ensemble burst features across all eight GPUs.',flush=True)
features = {}
variations = {'calibration':variability(cal).reshape(8,-1), 'genuine':variability(held).reshape(8,-1)}
attacks = []
for g in idx:
    attacks.append(np.stack([project_ranks(np.median(corrected(g,c)[1],axis=0)) for c in range(len(nonces))]))
attacks = np.asarray(attacks)
variations['ensemble'] = variability(attacks).reshape(8,-1)
base_pass = accept(saved['ensemble'],upper)[target,:,target]
for kind, ranks in [('genuine',held[target]),('ensemble',attacks[target])]:
    typical = np.concatenate([template_features(ranks[c],reg[target,c].reshape(-1,*reg.shape[-2:]))
                              for c in range(len(nonces))])
    own = saved[kind][target,:,:,target]
    next_best = np.min(np.delete(saved[kind][target],target,axis=-1),axis=-1)
    margin = np.median(next_best-own,axis=-1)
    selection = base_pass if kind=='ensemble' else np.ones(32,dtype=bool)
    features[kind] = dict(n=int(selection.sum()), typical_template_distance=describe(typical[selection]),
        intra_burst_distance=describe(variations[kind][target,selection]), nearest_margin=describe(margin[selection]),
        minimum_template_score=describe(np.sort(own,axis=1)[selection,1]))

print('Testing fixed calibration-only lower gates, using the original held-out captures.',flush=True)
variability_floor = variations['calibration'].min(axis=1)
variant_report = {}
for name in ('q95_baseline','q95_with_distance_floor','q95_with_variability_floor'):
    outcomes = {}
    for kind in ('genuine','ensemble','single'):
        table = saved[kind]
        passed = accept(table,upper)
        if name=='q95_with_distance_floor':
            passed &= np.sort(table,axis=2)[:,:,1,:] >= floor_q95
        if name=='q95_with_variability_floor':
            if kind=='single':
                # The original single is the first cyclic decoy after projection.
                bursts = []
                for g in idx:
                    bursts.append(np.stack([project_ranks(corrected(g,c)[1][0]) for c in range(len(nonces))]))
                variation = variability(np.asarray(bursts)).reshape(8,-1)
            else:
                variation = variations[kind]
            passed &= variation[:,:,None] >= variability_floor
        diagonal = passed[idx,:,idx]
        outcomes[kind] = dict(accepted=int(diagonal.sum()), by_gpu=diagonal.sum(axis=1).tolist(),
                              by_challenge=diagonal.reshape(8,16,2).sum(axis=(0,2)).tolist())
        if kind=='genuine':
            outcomes['wrong_chip'] = int(passed.sum()-diagonal.sum())
    variant_report[name] = outcomes
    print(name, 'genuine', outcomes['genuine']['accepted'], 'ensemble',outcomes['ensemble']['accepted'],
          'wrong chip',outcomes['wrong_chip'],flush=True)

# Only inspect rivals for launches that already clear at least one cutoff.
def target_passes(refs, prediction, target_index, thresholds):
    flat = prediction.reshape(prediction.shape[0],3,-1)
    accepted = np.zeros((len(thresholds),len(prediction)),dtype=bool)
    own_scores = np.empty(flat.shape[:2],dtype=float)
    for i, response in enumerate(flat):
        own = np.abs(refs[target_index]-response[:,None,:]).sum(axis=-1,dtype=np.int32).min(axis=-1)
        own_scores[i] = own/coordinates
        near = np.zeros(3,dtype=bool)
        for j in np.flatnonzero(own_scores[i] <= max(thresholds)):
            # Equal nearest distance is a rejection, as in the production scorer.
            near[j] = all(np.abs(refs[rival]-response[j]).sum(axis=-1,dtype=np.int32).min()>own[j]
                          for rival in idx if rival!=target_index)
        for p, threshold in enumerate(thresholds):
            accepted[p,i] = ((own_scores[i]<=threshold)&near).sum()>=2
    return accepted,own_scores

decoys = [g for g in idx if g!=target]
subsets = [subset for n in (1,2,4,6,7) for subset in combinations(decoys,n)]
subset_indices = [tuple(decoys.index(g) for g in subset) for subset in subsets]
limits = [upper[target],pooled[target]]
subset_counts = np.zeros((len(subsets),len(limits),len(nonces)),dtype=int)
subset_floor_counts = np.zeros((len(subsets),2,len(nonces)),dtype=int)
print(f'Enumerating {len(subsets)} decoy subsets for target C.',flush=True)
for c in range(len(nonces)):
    refs = reg[:,c].reshape(8,-1,coordinates)
    _, predictions = corrected(target,c)
    for k, subset in enumerate(subset_indices):
        ranks = project_ranks(np.median(predictions[list(subset)],axis=0))
        passed, own = target_passes(refs,ranks,target,limits)
        subset_counts[k,:,c] = passed.sum(axis=1)
        subset_floor_counts[k,0,c] = (passed[0] & (np.sort(own,axis=1)[:,1] >= floor_q95[target])).sum()
        subset_floor_counts[k,1,c] = (passed[0] & (variability(ranks) >= variability_floor[target])).sum()
        if len(subset)==7:
            assert np.array_equal(own,saved['ensemble'][target,2*c:2*c+2,:,target])
            for p, threshold in enumerate((upper,pooled)):
                expected=accept(saved['ensemble'][target,2*c:2*c+2],threshold)[:,target]
                assert np.array_equal(passed[p],expected)
    if (c+1)%4==0:
        print(f'Subset sweep completed {c+1}/{len(nonces)} challenges.',flush=True)
subset_report = {}
for k, subset in enumerate(subsets):
    key = ''.join(labels[g] for g in subset)
    subset_report[key] = dict(size=len(subset),q95=int(subset_counts[k,0].sum()),
        pooled_p95=int(subset_counts[k,1].sum()),q95_by_challenge=subset_counts[k,0].tolist(),
        q95_with_distance_floor=int(subset_floor_counts[k,0].sum()),
        q95_with_variability_floor=int(subset_floor_counts[k,1].sum()))

report = dict(target='C', features=features, thresholds=dict(q95=upper.tolist(),
    minimum_calibration_distance=floor_q95.tolist(),minimum_calibration_variability=variability_floor.tolist()),
    alternative_scorers=variant_report, subsets=subset_report,
    notes=['Feature comparisons condition on the 19 successful seven-decoy responses for C.',
           'Both extra gates are fitted on genuine calibration only and preserve the original unique-nearest and upper threshold checks.',
           'Exploratory evaluation on the original held-out split, which informed the hypothesis.'])
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Features:',json.dumps(features),flush=True)
for size in (1,2,4,6,7):
    scores = [(key,row['q95']) for key,row in subset_report.items() if row['size']==size]
    print(size,'subsets:',len(scores),'median:',np.median([count for _,count in scores]),
          'min/max:',min(scores,key=lambda x:x[1]),max(scores,key=lambda x:x[1]),flush=True)
    for gate in ('q95_with_distance_floor','q95_with_variability_floor'):
        print(gate, 'worst', max(((key,row[gate]) for key,row in subset_report.items() if row['size']==size), key=lambda x:x[1]),flush=True)
print('Six-decoy leave-one-out:',{''.join(set(labels[g] for g in decoys)-set(key)): row['q95']
                                   for key,row in subset_report.items() if row['size']==6},flush=True)
