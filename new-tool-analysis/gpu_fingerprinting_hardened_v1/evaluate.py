#!/usr/bin/env python3
"""Offline held-out genuine/attack evaluation. Does not issue or consume live challenges.

Ground-truth targets/labels in local files are analysis labels, NOT identity evidence.
Invalid synthetic predictions are counted separately so schema rejection is not
mistaken for proof of resistance to an attacker that constructs valid permutations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import fingerprint as fp
from manage import read,write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('output exists')
    registry=read(args.registry); cases={}; errors=Counter()
    for file in args.input:
        with file.open(encoding='utf-8') as stream:
            for line in stream:
                if len(line)>32*1024*1024: raise ValueError('input line too large')
                record=json.loads(line)
                target=record.get('claimed_target',record.get('device_uuid'))
                label=record.get('case_label',record.get('phase','unlabeled'))
                if not isinstance(label,str) or not isinstance(target,str): raise ValueError('missing analysis labels')
                key=label+' / '+target
                case=cases.setdefault(key,dict(n=0,structurally_valid=0,accepted=0,structural_rejections=0))
                case['n']+=1
                try:
                    result=fp.decision(registry,record,target)
                    case['structurally_valid']+=1; case['accepted']+=int(result['accepted'])
                except (ValueError,TypeError,KeyError) as error:
                    case['structural_rejections']+=1; errors[str(error)]+=1
    write_new(args.output,dict(scope='offline diagnostic; no live freshness, attestation, location or future-security guarantee',
        registry_id=fp.digest(registry),cases=cases,structural_errors=dict(errors),
        note='Attack claims need valid-format adaptive attacks, new seeds, held-out conditions and adequate sample sizes.'))
    print(json.dumps(cases,indent=2))


if __name__=='__main__': main()
