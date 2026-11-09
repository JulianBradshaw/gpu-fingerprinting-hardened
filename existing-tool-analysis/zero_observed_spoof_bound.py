import json
from pathlib import Path

folder=Path(__file__).resolve().parent/'frozen_registry_tradeoff'
data=json.loads((folder/'results.json').read_text())
output=[]
for src in range(1,len(data['collections'])):
    future=[p for p in data['pairs'] if p['registration_run']==src and p['verification_run']>src]
    for prefix in ('source_trained/','destination_trained/',''):
        distances=[r['distance'] for p in future for name,rows in p['trials'].items()
                   if name!='genuine' and name.startswith(prefix) for r in rows if r['match']]
        if not distances:
            raise ValueError('No eligible attack scores; handle this case separately')
        # Scores are integer L1 distances. This is the largest decision threshold
        # with zero accepted tested attacks in every later collection.
        threshold=min(distances)-1
        accepts=[dict(verification_run=p['verification_run'],
                      genuine_accepted=sum(r['match'] and r['distance']<=threshold for r in p['trials']['genuine']),
                      n=len(p['trials']['genuine'])) for p in future]
        output.append(dict(registration_run=src,attack_scope=prefix or 'all_tested_models',
                           threshold=threshold,per_run=accepts,
                           minimum_genuine_acceptance=min(r['genuine_accepted']/r['n'] for r in accepts)))
(folder/'zero_observed_spoof_bound.json').write_text(json.dumps(output,indent=2)+'\n')
for row in output:
    print(row['registration_run'],row['attack_scope'],row['threshold'],row['minimum_genuine_acceptance'])
lines=['','## Retrospective zero-observed-spoof bound','',
       'For completeness, the largest threshold rejecting every tested spoof in every chronological later collection gives the following minimum genuine acceptance across those later collections. These are observed-sample bounds, not guarantees of zero future spoof probability.','',
       '| Register | Source-trained models only | Destination-trained models only | Both sets of models |',
       '|---|---:|---:|---:|']
for src in range(1,len(data['collections'])):
    values=[r for r in output if r['registration_run']==src]
    lines.append('| '+str(src)+' | '+' | '.join(f"{100*r['minimum_genuine_acceptance']:.1f}%" for r in values)+' |')
with (folder/'report.md').open('a',encoding='utf-8') as report:
    report.write('\n'.join(lines)+'\n')
