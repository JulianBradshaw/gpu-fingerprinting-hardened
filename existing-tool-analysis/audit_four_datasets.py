import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

root = Path(__file__).resolve().parent  # folder holding campaign_v5_data_*
folders = sorted(root.glob('campaign_v5_data_*'))
datasets = []
for folder in folders:
    meta = json.loads((folder/'metadata.json').read_text())
    point = json.loads((folder/'analysis.json').read_text())
    checks, shapes, hashes, ids = [], [], [], []
    for g, dev in enumerate(meta['devices']):
        file = folder/f'gpu-{g:03d}.npz'
        detail = json.loads((folder/f'gpu-{g:03d}.json').read_text())
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        checks.append(digest == detail['npz_sha256'])
        hashes.append(digest)
        ids.append(dev['uuid'])
        with np.load(file, allow_pickle=False) as saved:
            arr = saved['fingerprints']
            shapes.append(list(arr.shape))
            checks.append(saved['seeds'].tolist() == meta['design']['seeds'])
            checks.append(arr.shape[:2] == (32, 10))
            checks.append(np.issubdtype(arr.dtype,np.integer))
    p95 = point['operating_points']['95']
    summary = {}
    for name in ('genuine','single_decoy_x1.0','ensemble_7_decoys_x1.0'):
        rows = point['trials'][name]
        actual = sum(r['match'] and r['distance'] <= p95['threshold'] for r in rows)
        checks.append(actual == p95['results'][name]['accepted'])
        summary[name] = [int(actual),len(rows)]
    for kind in ('recorded_run','mean_of_8'):
        rows = [r for b in range(4) for r in point['trials'][f'replay_base_{b}_{kind}']]
        summary['replay_'+kind] = [sum(r['match'] and r['distance'] <= p95['threshold'] for r in rows),len(rows)]
    when = datetime.fromisoformat(meta['created_utc']).astimezone(ZoneInfo('America/Denver'))
    info = dict(folder=folder.name,session_id=meta['session_id'],created_local=when.isoformat(),
                status=meta['status'],uuids=ids,shape=shapes,archive_hashes=hashes,
                checks_pass=all(checks),seed_design=meta['design'],source_sha256=meta['source_sha256'],
                script_sha256=meta['script_sha256'],p95_threshold=p95['threshold'],results=summary,
                analysis_session_matches=point['session_id']==meta['session_id'])
    datasets.append(info)
    print(json.dumps({k:v for k,v in info.items() if k not in ('uuids','archive_hashes','seed_design','shape')}))

comparison = []
for i,a in enumerate(datasets):
    for j,b in enumerate(datasets[:i]):
        comparison.append(dict(runs=[j+1,i+1],shared_gpu_uuids=len(set(a['uuids'])&set(b['uuids'])),
            same_gpu_order=a['uuids']==b['uuids'],same_design=a['seed_design']==b['seed_design'],
            identical_archives=sum(x==y for x,y in zip(a['archive_hashes'],b['archive_hashes']))))
print('COMPARISONS',json.dumps(comparison))
old_meta = root/'campaign_v4_data_retry'/'metadata.json'  # earlier incomplete v4 attempt; not included in this repository
if old_meta.exists():
    old = json.loads(old_meta.read_text())
    print('Shared GPUs with earlier incomplete v4:',[len(set(x['uuids'])&{d['uuid'] for d in old['devices']}) for x in datasets])
output = Path(__file__).resolve().parent / 'four_run_data_audit.json'
output.write_text(json.dumps(dict(datasets=datasets,comparisons=comparison),indent=2)+'\n')
