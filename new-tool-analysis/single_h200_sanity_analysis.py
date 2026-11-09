#!/usr/bin/env python3
"""CPU-only, single-device engineering/distance-consistency analysis.

This is NOT nearest-chip identification, authentication, or spoof evaluation.
It deliberately does not bypass the enrollment tool's two-device requirement.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    if not a.size or not np.all(np.isfinite(a)): raise ValueError('missing/non-finite statistics')
    return dict(n=int(a.size),minimum=float(a.min()),p5=float(np.percentile(a,5)),
                median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))


def analyze(args):
    spec=importlib.util.spec_from_file_location('sanity_fingerprint',args.code/'fingerprint.py')
    fp=importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
    pool=json.loads(args.pool.read_text())
    nonces=pool['challenges']
    if len(nonces)!=8 or len(set(nonces))!=8: raise ValueError('main plan requires eight distinct challenges')
    for nonce in nonces: fp.challenge(nonce)
    records={}; manifests={}; hashes={}; sample_ids=set(); build_ids=set(); collection_ids=set()
    profile=None; smids=None
    for phase,path in [('registration',args.registration),('calibration',args.calibration),('evaluation',args.evaluation)]:
        sidecar=path.with_suffix(path.suffix+'.manifest.json')
        stderr=path.with_suffix(path.suffix+'.stderr.log')
        manifest=json.loads(sidecar.read_text())
        expected=len(nonces)*args.repeats
        if manifest.get('status')!='complete' or manifest.get('phase')!=phase or manifest.get('device_uuid')!=args.device:
            raise ValueError(f'{phase}: incomplete or mismatched capture manifest')
        if manifest.get('expected_measurements')!=expected or manifest.get('received_measurements')!=expected:
            raise ValueError(f'{phase}: capture counts differ from plan')
        if manifest.get('challenge_source_kind')!='enrollment-pool' or manifest.get('challenge_source_sha256')!=sha(args.pool):
            raise ValueError(f'{phase}: challenge-pool binding mismatch')
        if manifest['collection_id'] in collection_ids: raise ValueError('phases must be separate captures')
        collection_ids.add(manifest['collection_id'])
        phase_rows=[]; order=[]
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if len(line)>32*1024*1024: raise ValueError('input line too large')
                record=json.loads(line)
                if profile is None:
                    profile=record['profile']; smids=record['sm_ids']
                    if profile['mode']!=args.mode or profile['rounds']!=args.rounds:
                        raise ValueError('wrong mode or round count')
                raw,mask=fp.validate_record(record,profile,smids)
                if record.get('phase')!=phase or record.get('device_uuid')!=args.device or record.get('collection_id')!=manifest['collection_id']:
                    raise ValueError('record phase/device/capture binding mismatch')
                sid=record.get('sample_id')
                if not isinstance(sid,str) or not sid or sid in sample_ids: raise ValueError('missing/duplicate sample identity')
                sample_ids.add(sid)
                if record.get('build_id')!=manifest.get('build_id'): raise ValueError('record build binding mismatch')
                build_ids.add(record['build_id'])
                nonce=record['challenge']; order.append(nonce)
                host=record.get('host_measurement_ms')
                if type(host) not in (float,int) or not math.isfinite(host) or host<0: raise ValueError('bad host measurement time')
                phase_rows.append(dict(raw=raw,mask=mask,challenge=nonce,kernel_ms=record['kernel_ms'],host_ms=host,
                    telemetry_available=[bool(record.get(name,{}).get('available',False)) for name in ('telemetry_before','telemetry_after')]))
        if len(phase_rows)!=expected or Counter(order)!=Counter({n:args.repeats for n in nonces}):
            raise ValueError(f'{phase}: per-challenge counts differ from plan')
        if order!=manifest.get('launch_order'): raise ValueError('saved record order differs from launch manifest')
        if manifest.get('profile')!=profile or manifest.get('sm_ids')!=smids: raise ValueError('manifest profile/SM map mismatch')
        manifests[phase]=manifest; records[phase]=phase_rows
        hashes[phase]=dict(data=sha(path),manifest=sha(sidecar),stderr=sha(stderr),stderr_bytes=stderr.stat().st_size)
    if len(build_ids)!=1: raise ValueError('mixed binaries')
    intervals=[]
    for phase in ('registration','calibration','evaluation'):
        m=manifests[phase]
        start=datetime.fromisoformat(m['created_utc']); stop=datetime.fromisoformat(m['completed_utc'])
        if stop<start: raise ValueError('capture timestamps go backwards')
        intervals.append((start,stop))
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])): raise ValueError('phases overlap or are out of order')

    templates={n:np.concatenate([r['raw'] for r in records['registration'] if r['challenge']==n],axis=0) for n in nonces}
    def burst_scores(row,reference):
        return [fp.metric_scores(x,reference,row['mask'],'raw_l1',np.ones(row['mask'].shape)) for x in row['raw']]
    cal=[sorted(burst_scores(r,templates[r['challenge']]))[1] for r in records['calibration']]
    threshold=fp.conformal_threshold(cal,.05)
    evaluation=[]
    for r in records['evaluation']:
        scores=burst_scores(r,templates[r['challenge']])
        evaluation.append(dict(challenge=r['challenge'],scores=scores,burst_score=sorted(scores)[1],
                               distance_consistent=sum(s<=threshold for s in scores)>=2))
    n=len(evaluation); accepted=sum(r['distance_consistent'] for r in evaluation)
    minimum=math.ceil(.90*n)
    by_seed={nonce:dict(n=sum(r['challenge']==nonce for r in evaluation),
                        distance_consistent=sum(r['challenge']==nonce and r['distance_consistent'] for r in evaluation)) for nonce in nonces}
    within=[]; between=[]; loo=[]
    for nonce in nonces:
        rows=[r for r in records['registration'] if r['challenge']==nonce]
        mask=rows[0]['mask']
        for row in rows:
            within.extend(float(np.abs(a.astype(np.int64)-b).mean()) for a,b in
                [(row['raw'][i][mask],row['raw'][j][mask]) for i,j in itertools.combinations(range(3),2)])
        for a,b in itertools.combinations(rows,2):
            between.extend(float(np.abs(x[mask].astype(np.int64)-y[mask]).mean()) for x in a['raw'] for y in b['raw'])
        for index,row in enumerate(rows):
            other=np.concatenate([r['raw'] for j,r in enumerate(rows) if j!=index],axis=0)
            loo.append(sorted(burst_scores(row,other))[1])
    timing={}
    for phase,rows in records.items():
        m=manifests[phase]
        elapsed=(datetime.fromisoformat(m['completed_utc'])-datetime.fromisoformat(m['created_utc'])).total_seconds()
        timing[phase]=dict(kernel_ms=stats([v for r in rows for v in r['kernel_ms']]),
            kernel_ms_by_burst_position=[stats([r['kernel_ms'][b] for r in rows]) for b in range(3)],
            host_measurement_ms=stats([r['host_ms'] for r in rows]),capture_wall_seconds=elapsed,
            telemetry_snapshots_available=sum(v for r in rows for v in r['telemetry_available']),
            telemetry_snapshots_expected=2*len(rows))
    return dict(scope='one-device engineering and own-distance consistency only',
        engineering_checks_pass=True,identity_tested=False,spoof_resistance_tested=False,location_tested=False,
        profile=profile,device_uuid=args.device,physical_sm_ids=smids,build_id=next(iter(build_ids)),
        pool_sha256=sha(args.pool),input_hashes=hashes,phase_counts={k:len(v) for k,v in records.items()},
        scoring=f'minimum raw mean-active L1 to {3*args.repeats} registration kernels; second score of three kernels per burst',
        calibration_burst_score=stats(cal),threshold=threshold,raw_empirical_calibration_p95=float(np.percentile(cal,95)),
        corrected_quantile_rank=math.ceil((len(cal)+1)*.95),evaluation_burst_score=stats([r['burst_score'] for r in evaluation]),
        evaluation_distance_consistent=accepted,evaluation_n=n,evaluation_distance_consistency=accepted/n,
        by_challenge=by_seed,triage_minimum_count=minimum,triage_pass=accepted>=minimum,
        within_registration_burst_distance=stats(within),between_registration_burst_distance=stats(between),
        registration_leave_one_burst_out_score=stats(loo),timing=timing,
        limits=['Triage floor is an engineering choice, not an estimated authentication error bound.',
                'Bursts share a device, seeds, context and conditions; no independence or confidence claim.',
                'Repeatability and challenge variation do not establish physical-chip specificity.',
                'No nearest-chip matching is performed; no dummy second GPU is introduced.',
                'Client-reported times and telemetry do not verify location or attestation.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',type=Path,required=True); p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True); p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True); p.add_argument('--mode',choices=('all','subset'),required=True)
    p.add_argument('--device',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=128); p.add_argument('--repeats',type=int,default=8)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output exists')
    if args.repeats<2: raise ValueError('need at least two registration bursts per challenge')
    result=analyze(args)
    with args.output.open('x',encoding='utf-8') as out: json.dump(result,out,indent=2,allow_nan=False); out.write('\n')
    print(json.dumps({k:result[k] for k in ('engineering_checks_pass','evaluation_distance_consistent','evaluation_n','threshold','triage_pass','identity_tested')},indent=2))
    if not result['triage_pass']: raise SystemExit(2)


if __name__=='__main__': main()
