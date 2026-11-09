"""Research-only schema, challenge expansion, enrollment and matching.

No function claims hardware attestation or location verification. NumPy required.
The public challenge is used as the ChaCha20 block key; there is no secret PRF key.
"""
from functools import lru_cache
import base64
import hashlib
import json
import math
import re
import struct
import zlib
import numpy as np

FORMAT='gpu-fp-hardened-v1'
LOCAL_DOMAIN=0x47505532
GLOBAL_DOMAIN=0x47505531
MASK=0xffffffff


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pack_array(value,dtype):
    value=np.asarray(value,dtype=dtype)
    return dict(dtype=value.dtype.str,shape=list(value.shape),
                data=base64.b64encode(zlib.compress(value.tobytes())).decode('ascii'))


@lru_cache(maxsize=32)
def _unpack(dtype,shape,data):
    if dtype not in ('<i4','<f8') or not shape or any(type(x) is not int or x<1 for x in shape):
        raise ValueError('invalid packed array description')
    size=math.prod(shape)*np.dtype(dtype).itemsize
    if size>256*1024*1024: raise ValueError('packed array too large')
    dec=zlib.decompressobj()
    raw=dec.decompress(base64.b64decode(data,validate=True),size+1)
    if len(raw)!=size or not dec.eof or dec.unused_data or dec.unconsumed_tail:
        raise ValueError('invalid compressed array size')
    value=np.frombuffer(raw,dtype=dtype).reshape(shape)
    return value


def unpack_array(packed):
    if not isinstance(packed,dict) or set(packed)!= {'dtype','shape','data'}:
        raise ValueError('invalid packed array')
    return _unpack(packed['dtype'],tuple(packed['shape']),packed['data'])


def challenge(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('challenge must be 256 bits in canonical lowercase hexadecimal')
    return value


def chacha_block(key,counter,nonce):
    if len(key)!=32 or len(nonce)!=3 or not 0<=counter<=MASK:
        raise ValueError('invalid ChaCha block parameters')
    initial=list(struct.unpack('<4I',b'expand 32-byte k'))+list(struct.unpack('<8I',key))+[counter]+list(nonce)
    if any(type(x) is not int or not 0<=x<=MASK for x in initial):
        raise ValueError('invalid word')
    x=initial.copy()
    def rotate(v,n): return ((v<<n)|(v>>(32-n)))&MASK
    def quarter(a,b,c,d):
        x[a]=(x[a]+x[b])&MASK; x[d]=rotate(x[d]^x[a],16)
        x[c]=(x[c]+x[d])&MASK; x[b]=rotate(x[b]^x[c],12)
        x[a]=(x[a]+x[b])&MASK; x[d]=rotate(x[d]^x[a],8)
        x[c]=(x[c]+x[d])&MASK; x[b]=rotate(x[b]^x[c],7)
    for _ in range(10):
        for a,b,c,d in ((0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15),
                        (0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)):
            quarter(a,b,c,d)
    return [(a+b)&MASK for a,b in zip(x,initial)]


def validate_profile(profile):
    expected=dict(format=FORMAT,prng='chacha20-public-challenge',mode='subset',rounds=128,
                  warmup_rounds=32,burst=3,threads=64,banks=16,delay_base=2048,delay_jitter=2048)
    if not isinstance(profile,dict) or set(profile)!=set(expected):
        raise ValueError('unknown or incomplete probe profile')
    for name,value in expected.items():
        if name not in ('mode','rounds') and (type(profile[name]) is not type(value) or profile[name]!=value):
            raise ValueError(f'unsupported profile field {name}')
    if profile['mode'] not in ('all','subset') or type(profile['rounds']) is not int or not 16<=profile['rounds']<=512:
        raise ValueError('unsupported mode/round count')
    return profile


@lru_cache(maxsize=64)
def _plan(nonce,profile_json,smids):
    profile=validate_profile(json.loads(profile_json)); key=bytes.fromhex(challenge(nonce))
    active=np.zeros((profile['rounds'],len(smids)),dtype=bool)
    for output_round in range(profile['rounds']):
        r=output_round+profile['warmup_rounds']
        if profile['mode']=='all':
            active[output_round,:]=True
        else:
            global_words=chacha_block(key,0,(MASK,r,GLOBAL_DOMAIN))
            k=4+global_words[0]%5
            ordering=sorted(range(len(smids)),key=lambda i:(chacha_block(key,0,(smids[i],r,LOCAL_DOMAIN))[2],smids[i]))
            active[output_round,ordering[:k]]=True
    active.flags.writeable=False
    return active


def plan(nonce,profile,smids):
    validate_profile(profile)
    if not isinstance(smids,(list,tuple)) or not 8<=len(smids)<=1024:
        raise ValueError('SM map length outside supported range')
    if any(type(x) is not int or not 0<=x<MASK for x in smids) or list(smids)!=sorted(set(smids)):
        raise ValueError('SM map must be sorted, unique physical IDs')
    return _plan(challenge(nonce),canonical(profile),tuple(smids))


def validate_record(record,expected_profile=None,expected_smids=None):
    if not isinstance(record,dict) or record.get('type')!='measurement':
        raise ValueError('not a measurement')
    profile=validate_profile(record.get('profile'))
    if expected_profile is not None and profile!=expected_profile:
        raise ValueError('profile differs from enrollment')
    smids=record.get('sm_ids')
    if expected_smids is not None and smids!=expected_smids:
        raise ValueError('SM map differs from enrollment cohort')
    mask=plan(record.get('challenge'),profile,smids)
    raw=np.asarray(record.get('fingerprints'))
    if raw.shape!=(3,profile['rounds'],len(smids)) or raw.dtype.kind not in 'iu':
        raise ValueError('fingerprints must be three integer matrices of enrolled shape')
    if raw.min() < -1 or raw.max()>=len(smids):
        raise ValueError('rank outside physical range')
    for burst in raw:
        if np.any(burst[~mask]!=-1):
            raise ValueError('inactive SMs must have sentinel -1')
        for r,active in enumerate(mask):
            if not np.array_equal(np.sort(burst[r,active]),np.arange(int(active.sum()))):
                raise ValueError('active ranks are not a permutation for this challenge')
    times=record.get('kernel_ms')
    if not isinstance(times,list) or len(times)!=3 or any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in times):
        raise ValueError('invalid timing metadata')
    # Timing is caller-reported and is NEVER evidence of location/attestation.
    return raw.astype(np.int32),mask


def metric_scores(candidate,templates,mask,metric,scale):
    residual=np.abs(np.asarray(templates,dtype=np.float64)-candidate)
    if metric=='clipped_mad_l1':
        residual=np.minimum(residual/np.asarray(scale),6.0)
    elif metric!='raw_l1':
        raise ValueError('unknown metric')
    return float(residual[:,mask].mean(axis=1).min())


def scoring_config(metric):
    if metric not in ('raw_l1','clipped_mad_l1'): raise ValueError('unknown metric')
    return dict(version=1,metric=metric,residual_cap=6.0,scale_floor=1.0,
                aggregation='minimum-template mean-active-residual',required_votes=2,burst=3)


def scores(registry,record):
    if registry.get('format')!=FORMAT or registry.get('scoring_config')!=scoring_config(registry.get('metric')):
        raise ValueError('unsupported registry/scoring configuration')
    raw,mask=validate_record(record,registry['profile'],registry['sm_ids'])
    nonce=record['challenge']
    if nonce not in registry['challenges']:
        raise ValueError('unenrolled challenge')
    ids=registry['device_order']; table=np.empty((3,len(ids)))
    for j,gpu in enumerate(ids):
        entry=registry['devices'][gpu]['templates'][nonce]
        templates=unpack_array(entry['values']); scale=unpack_array(entry['scale'])
        if templates.ndim!=3 or templates.shape[1:]!=mask.shape or scale.shape!=mask.shape or np.any(scale<=0) or not np.all(np.isfinite(scale)):
            raise ValueError('invalid enrolled template/scale dimensions or values')
        for b in range(3):
            table[b,j]=metric_scores(raw[b],templates,mask,registry['metric'],scale)
    return table


def decision(registry,record,target):
    if target not in registry['devices']:
        raise ValueError('unenrolled target')
    table=scores(registry,record); t=registry['device_order'].index(target)
    threshold=registry['devices'][target]['threshold']
    other=np.min(np.delete(table,t,axis=1),axis=1)
    correct=table[:,t]<other
    votes=correct & (table[:,t]<=threshold)
    return dict(accepted=bool(votes.sum()>=2),votes=int(votes.sum()),required_votes=2,
                own_scores=table[:,t].tolist(),threshold=threshold,
                unique_target_wins=int(correct.sum()),
                attestation_verified=False,location_verified=False,
                scope='experimental fingerprint match only')


def conformal_threshold(values,alpha):
    if not 0<alpha<.5:
        raise ValueError('alpha outside supported range')
    values=sorted(float(v) for v in values)
    index=math.ceil((len(values)+1)*(1-alpha))
    if index>len(values):
        raise ValueError('too few calibration samples for a finite requested quantile')
    return values[index-1]


def enroll(records,metric='raw_l1',alpha=.05,min_registration=8,min_calibration=32):
    if min_registration<8 or min_calibration<32:
        raise ValueError('minimums are 8 registration bursts/challenge and 32 calibration bursts/device')
    records=list(records)
    if not records: raise ValueError('no records')
    profile=validate_profile(records[0]['profile']); smids=records[0]['sm_ids']
    build_id=records[0].get('build_id')
    if not isinstance(build_id,str) or re.fullmatch('[0-9a-f]{64}',build_id) is None:
        raise ValueError('missing binary build hash from trusted collection')
    grouped={}; sample_ids=set(); evaluation=[]
    for record in records:
        validate_record(record,profile,smids)
        if record.get('build_id')!=build_id: raise ValueError('mixed measurement builds')
        sid=record.get('sample_id'); gpu=record.get('device_uuid'); phase=record.get('phase')
        if not isinstance(sid,str) or not sid or sid in sample_ids:
            raise ValueError('missing or duplicate sample identity; phases must be disjoint')
        if not isinstance(gpu,str) or not gpu or phase not in ('registration','calibration','evaluation'):
            raise ValueError('missing device/phase metadata')
        sample_ids.add(sid)
        if phase=='evaluation':
            evaluation.append(record); continue
        grouped.setdefault(gpu,{}).setdefault(record['challenge'],{}).setdefault(phase,[]).append(record)
    ids=sorted(grouped)
    if len(ids)<2: raise ValueError('at least two compatible GPUs required for nearest-chip evaluation')
    nonces=sorted(grouped[ids[0]])
    if len(nonces)<2: raise ValueError('at least two enrolled challenges required')
    if any(r['device_uuid'] not in ids or r['challenge'] not in nonces for r in evaluation):
        raise ValueError('evaluation labels reference an unenrolled chip/challenge')
    registry=dict(format=FORMAT,profile=profile,sm_ids=smids,metric=metric,scoring_config=scoring_config(metric),alpha=alpha,build_id=build_id,
                  device_order=ids,challenges=nonces,devices={},
                  claim='research enrollment; no attestation, location guarantee or proven spoof resistance')
    calibration={}
    for gpu in ids:
        if sorted(grouped[gpu])!=nonces: raise ValueError('all chips need the same enrolled challenge pool')
        templates={}; calibration[gpu]=[]
        for nonce in nonces:
            phases=grouped[gpu][nonce]
            registration=phases.get('registration',[]); cal=phases.get('calibration',[])
            if len(registration)<min_registration or not cal:
                raise ValueError('missing registration/calibration coverage for a chip/challenge')
            values=np.concatenate([np.asarray(r['fingerprints'],dtype=np.int32) for r in registration],axis=0)
            median=np.median(values,axis=0)
            mad=np.maximum(np.median(np.abs(values-median),axis=0),1.0)
            templates[nonce]=dict(values=pack_array(values,'<i4'),scale=pack_array(mad,'<f8'),registration_bursts=len(registration))
            calibration[gpu].extend(cal)
        if len(calibration[gpu])<min_calibration: raise ValueError('insufficient calibration bursts')
        registry['devices'][gpu]=dict(templates=templates,threshold=0.0)
    for gpu in ids:
        t=ids.index(gpu)
        # Calibrate the actual 2-of-3 own-distance statistic, not a single launch.
        own_second=[float(np.sort(scores(registry,r)[:,t])[1]) for r in calibration[gpu]]
        registry['devices'][gpu]['threshold']=conformal_threshold(own_second,alpha)
    statistics={}
    for gpu in ids:
        cal=calibration[gpu]
        evaluate=[r for r in evaluation if r['device_uuid']==gpu]
        cal_rate=sum(decision(registry,r,gpu)['accepted'] for r in cal)/len(cal)
        eval_rate=sum(decision(registry,r,gpu)['accepted'] for r in evaluate)/len(evaluate) if evaluate else None
        covered={r['challenge'] for r in evaluate}==set(nonces)
        statistics[gpu]=dict(calibration_bursts=len(cal),
            calibration_joint_acceptance=cal_rate,
            evaluation_bursts=len(evaluate),evaluation_joint_acceptance=(
                eval_rate),evaluation_covers_challenges=covered)
        registry['devices'][gpu]['ready_for_research_verification']=(
            cal_rate>=1-alpha and len(evaluate)>=32 and covered and eval_rate>=1-alpha)
    registry['enrollment_report']=statistics
    registry['calibration_note']='Finite-sample quantile of burst statistic; coverage assumptions require exchangeability. No adversarial false-accept guarantee.'
    return registry
