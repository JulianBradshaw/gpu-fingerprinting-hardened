#!/usr/bin/env python3
"""Local enrollment, challenge issuance and one-attempt verification. No network service."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import sqlite3
import time
import fingerprint as fp


def read(path,max_bytes=512*1024*1024):
    if path.stat().st_size>max_bytes: raise ValueError('input exceeds local size limit')
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path,value):
    with path.open('x',encoding='utf-8') as out:
        out.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


@contextmanager
def connect(path):
    db=sqlite3.connect(path,timeout=30)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, nonce TEXT UNIQUE NOT NULL, registry TEXT NOT NULL, target TEXT NOT NULL, issued REAL NOT NULL, ttl INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)')
        db.commit()
        with db: yield db
    finally:
        db.close()


def issue(registry,state,target,ttl=120):
    if target not in registry['devices'] or type(ttl) is not int or not 1<=ttl<=3600:
        raise ValueError('invalid target/TTL')
    if not registry['devices'][target].get('ready_for_research_verification',False):
        raise ValueError('target lacks passing held-out genuine validation; enrollment is not eligible')
    with connect(state) as db:
        db.execute('BEGIN IMMEDIATE')
        used={row[0] for row in db.execute('SELECT nonce FROM tickets')}
        available=[c for c in registry['challenges'] if c not in used]
        if not available: raise ValueError('enrolled unused challenge pool exhausted; no automatic reuse')
        nonce=secrets.choice(available); rid=secrets.token_hex(16); issued=time.time()
        reg_id=fp.digest(registry)
        db.execute('INSERT INTO tickets(id,nonce,registry,target,issued,ttl) VALUES(?,?,?,?,?,?)',
                   (rid,nonce,reg_id,target,issued,ttl))
    return dict(request_id=rid,challenge=nonce,registry_id=reg_id,target=target,profile=registry['profile'],
                freshness_ttl_seconds=ttl,scope='freshness expiry, NOT a location/RTT deadline')


def verify(registry,state,request,response,require_attestation=False):
    # Requiring an unavailable attestation provider must never silently downgrade.
    if require_attestation:
        raise NotImplementedError('hardware attestation integration is not implemented')
    with connect(state) as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT nonce,registry,target,issued,ttl,consumed FROM tickets WHERE id=?',
                       (request.get('request_id'),)).fetchone()
        if row is None or row[5]: raise ValueError('unknown or already consumed request')
        db.execute('UPDATE tickets SET consumed=1 WHERE id=?',(request['request_id'],))
    # The first attempt burns the ticket even if the response is malformed.
    nonce,reg_id,target,issued,ttl,_=row
    if reg_id!=fp.digest(registry) or request.get('registry_id')!=reg_id or request.get('target')!=target or request.get('challenge')!=nonce or request.get('profile')!=registry['profile']:
        raise ValueError('request is not bound to this registry/target/challenge')
    age=time.time()-issued
    if age<0 or age>ttl: raise ValueError('request expired or server clock moved backwards')
    if response.get('challenge')!=nonce: raise ValueError('response to a different challenge')
    if not registry['devices'][target].get('ready_for_research_verification',False):
        raise ValueError('target enrollment is not eligible')
    # Claimed device_uuid and client timing are not trusted as identity evidence.
    return fp.decision(registry,response,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    e=sub.add_parser('enroll'); e.add_argument('--input',type=Path,nargs='+',required=True)
    e.add_argument('--output',type=Path,required=True); e.add_argument('--metric',choices=('raw_l1','clipped_mad_l1'),default='raw_l1')
    e.add_argument('--alpha',type=float,default=.05)
    i=sub.add_parser('issue'); i.add_argument('--registry',type=Path,required=True); i.add_argument('--state',type=Path,required=True)
    i.add_argument('--target',required=True); i.add_argument('--output',type=Path,required=True); i.add_argument('--ttl',type=int,default=120)
    v=sub.add_parser('verify'); v.add_argument('--registry',type=Path,required=True); v.add_argument('--state',type=Path,required=True)
    v.add_argument('--request',type=Path,required=True); v.add_argument('--response',type=Path,required=True)
    v.add_argument('--require-attestation',action='store_true')
    args=parser.parse_args()
    if args.command=='enroll':
        records=[]
        for file in args.input:
            with file.open(encoding='utf-8') as stream:
                for line in stream:
                    if len(line)>32*1024*1024: raise ValueError('measurement line too large')
                    records.append(json.loads(line))
        registry=fp.enroll(records,args.metric,args.alpha)
        write_new(args.output,registry)
        print(json.dumps(registry['enrollment_report'],indent=2))
    elif args.command=='issue':
        if args.output.exists(): raise ValueError('request output already exists')
        write_new(args.output,issue(read(args.registry),args.state,args.target,args.ttl))
    else:
        result=verify(read(args.registry),args.state,read(args.request,1024*1024),read(args.response,32*1024*1024),args.require_attestation)
        print(json.dumps(result,indent=2,allow_nan=False))
        if not result['accepted']: raise SystemExit(2)


if __name__=='__main__': main()
