#!/usr/bin/env python3
"""Generate challenges or collect local hardware responses. No cloud provisioning.

'pool' is CPU-only. 'capture' explicitly starts the compiled CUDA probe.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import secrets
import subprocess
import threading
import uuid
import fingerprint as fp


def utc(): return datetime.now(timezone.utc).isoformat()


def telemetry(device):
    try:
        result=subprocess.run(['nvidia-smi','-i',device,'-q','-x'],capture_output=True,text=True,timeout=10)
        return dict(captured_utc=utc(),available=result.returncode==0,raw_xml=result.stdout if result.returncode==0 else None,
                    error=result.stderr[-2000:] if result.returncode else None,
                    trust='local observational metadata; not hardware attestation')
    except (OSError,subprocess.SubprocessError) as error:
        return dict(captured_utc=utc(),available=False,error=str(error))


def capture(args):
    request_path=getattr(args,'request',None)
    source_path=request_path or args.pool
    request=json.loads(request_path.read_text()) if request_path else None
    if request is not None:
        if args.repeats!=1 or args.phase!='evaluation': raise ValueError('issued requests require evaluation phase and repeats=1')
        fp.validate_profile(request['profile'])
        args.mode=request['profile']['mode']; args.rounds=request['profile']['rounds']
        pool=dict(challenges=[request['challenge']])
    else:
        pool=json.loads(args.pool.read_text())
    nonces=pool['challenges']
    if not nonces or len(nonces)!=len(set(nonces)): raise ValueError('empty or duplicate challenge pool')
    for nonce in nonces: fp.challenge(nonce)
    if args.repeats<1 or args.timeout<=0: raise ValueError('invalid count/timeout')
    if not args.device.startswith('GPU-'): raise ValueError('select a full GPU UUID, not a remapped ordinal')
    binary=args.binary.resolve()
    build_id=hashlib.sha256(binary.read_bytes()).hexdigest()
    order=[nonce for nonce in nonces for _ in range(args.repeats)]
    shuffle_seed=secrets.randbits(64)
    random.Random(shuffle_seed).shuffle(order)
    manifest_path=args.output.with_suffix(args.output.suffix+'.manifest.json')
    stderr_path=args.output.with_suffix(args.output.suffix+'.stderr.log')
    if any(p.exists() for p in (args.output,manifest_path,stderr_path)): raise ValueError('output/sidecar already exists')
    manifest=dict(format=fp.FORMAT,collection_id=uuid.uuid4().hex,created_utc=utc(),status='collecting',
                  build_id=build_id,device_uuid=args.device,phase=args.phase,shuffle_seed=shuffle_seed,
                  launch_order=order,expected_measurements=len(order),received_measurements=0,
                  challenge_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  challenge_source_kind='issued-request' if request is not None else 'enrollment-pool')
    def save_manifest():
        temporary=manifest_path.with_suffix(manifest_path.suffix+'.tmp')
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(manifest_path)
    save_manifest()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.device)
    with args.output.open('x',encoding='utf-8') as output, stderr_path.open('x',encoding='utf-8') as stderr:
        try:
            process=subprocess.Popen([str(binary),args.mode,str(args.rounds)],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except OSError as error:
            manifest.update(status='failed',error=str(error)); save_manifest(); raise
        replies=queue.Queue()
        def reader():
            try:
                while True:
                    line=process.stdout.readline(32*1024*1024+1)
                    if not line: replies.put(RuntimeError('probe exited or closed output')); return
                    if len(line)>32*1024*1024: raise ValueError('probe response too large')
                    replies.put(json.loads(line))
            except Exception as error: replies.put(error)
        thread=threading.Thread(target=reader,daemon=True); thread.start()
        def receive():
            try: value=replies.get(timeout=args.timeout)
            except queue.Empty: raise TimeoutError('probe response timeout; partial file preserved')
            if isinstance(value,Exception): raise value
            return value
        try:
            ready=receive()
            if ready.get('type')!='ready' or ready.get('device_uuid','').lower()!=args.device.lower():
                raise ValueError('probe/device startup mismatch')
            fp.validate_profile(ready['profile'])
            if ready['profile']['mode']!=args.mode or ready['profile']['rounds']!=args.rounds:
                raise ValueError('probe profile differs from requested capture')
            manifest['profile']=ready['profile']; save_manifest()
            for index,nonce in enumerate(order):
                before=telemetry(args.device)
                process.stdin.write(nonce+'\n'); process.stdin.flush()
                record=receive()
                if record.get('challenge')!=nonce or record.get('device_uuid')!=ready['device_uuid']:
                    raise ValueError('response challenge/device mismatch')
                fp.validate_record(record,ready['profile'],manifest.get('sm_ids'))
                manifest['sm_ids']=record['sm_ids']
                record.update(sample_id=uuid.uuid4().hex,collection_id=manifest['collection_id'],phase=args.phase,
                    build_id=build_id,captured_utc=utc(),telemetry_before=before,telemetry_after=telemetry(args.device))
                if request is not None: record['request_id']=request['request_id']
                output.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                output.flush(); os.fsync(output.fileno())
                manifest['received_measurements']=index+1; save_manifest()
                print(f'{index+1}/{len(order)} saved',flush=True)
            process.stdin.close(); process.wait(timeout=10)
            if process.returncode!=0: raise RuntimeError(f'probe exited with {process.returncode}')
            manifest.update(status='complete',completed_utc=utc()); save_manifest()
        except BaseException as error:
            manifest.update(status='failed',error=str(error)); save_manifest(); raise
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            if not process.stdin.closed: process.stdin.close()
            thread.join(timeout=1)
            process.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('pool'); g.add_argument('--count',type=int,default=32); g.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('capture'); c.add_argument('--binary',type=Path,required=True); c.add_argument('--device',required=True)
    source=c.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool',type=Path); source.add_argument('--request',type=Path)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--phase',choices=('registration','calibration','evaluation'),required=True)
    c.add_argument('--mode',choices=('all','subset'),default='subset'); c.add_argument('--rounds',type=int,default=128)
    c.add_argument('--repeats',type=int,default=8); c.add_argument('--timeout',type=float,default=120)
    args=p.parse_args()
    if args.command=='pool':
        if not 2<=args.count<=100000: raise ValueError('count outside supported range')
        values=set()
        while len(values)<args.count: values.add(secrets.token_hex(32))
        with args.output.open('x',encoding='utf-8') as out:
            json.dump(dict(format=fp.FORMAT,challenges=sorted(values),note='Keep unenrolled/unissued challenge pool private.'),out,indent=2)
            out.write('\n')
    else:
        if not 16<=args.rounds<=512: raise ValueError('rounds outside supported range')
        capture(args)


if __name__=='__main__': main()
