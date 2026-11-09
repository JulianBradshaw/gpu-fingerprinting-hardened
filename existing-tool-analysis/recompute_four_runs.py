"""Offline reanalysis using the reviewed local analysis implementation only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent  # folder holding campaign_v5_data_*
implementation = Path(__file__).resolve().parent / 'campaign_v4.py'
spec = importlib.util.spec_from_file_location('reviewed_offline_analysis', implementation)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
records = []
for folder in sorted(root.glob('campaign_v5_data_*')):
    saved = json.loads((folder/'analysis.json').read_text())
    A, meta = c.load_dataset(folder)
    calculated = c.analyze(A, meta, saved['response_policy'])
    equal = calculated['operating_points'] == saved['operating_points'] and calculated['trials'] == saved['trials']
    hashes = {str(g): hashlib.sha256(np.stack([A[(g,s)] for s in meta['design']['seeds']]).tobytes()).hexdigest()
              for g in range(len(meta['devices']))}
    record = dict(folder=folder.name,full_reanalysis_matches=equal,raw_array_hashes=hashes,
                  shape=list(next(iter(A.values())).shape))
    records.append(record)
    print(folder.name, 'all per-trial scores and operating points match:', equal, flush=True)
    del A, calculated
comparisons = []
for i,a in enumerate(records):
    for j,b in enumerate(records[:i]):
        comparisons.append(dict(runs=[j+1,i+1],identical_gpu_array_sets=sum(
            a['raw_array_hashes'][g]==b['raw_array_hashes'][g] for g in a['raw_array_hashes'])))
print('Raw array comparisons:',json.dumps(comparisons),flush=True)
(Path(__file__).resolve().parent / 'four_run_reanalysis.json').write_text(
    json.dumps(dict(records=records,comparisons=comparisons),indent=2)+'\n')
