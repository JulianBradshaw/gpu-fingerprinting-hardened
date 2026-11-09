"""CPU-only tests. Synthetic ranks test logic, never fingerprint security."""
import copy
import json
import os
from pathlib import Path
import random
import queue
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import zlib
import base64
import numpy as np
import fingerprint as fp
import manage
import collect

PROFILE=dict(format=fp.FORMAT,prng='chacha20-public-challenge',mode='all',rounds=16,
             warmup_rounds=32,burst=3,threads=64,banks=16,delay_base=2048,delay_jitter=2048)
NONCES=['01'*32,'02'*32]
VECTOR='10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e' \
       'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e'
WORK=Path(os.environ.get('FP_TEST_WORK',tempfile.gettempdir()))/('gpu-fp-test-'+uuid.uuid4().hex)
WORK.mkdir(parents=True,exist_ok=False)


def record(gpu,nonce,phase='evaluation',i=0):
    rank=list(range(8)) if gpu=='A' else list(reversed(range(8)))
    return dict(type='measurement',device_uuid=gpu,challenge=nonce,profile=copy.deepcopy(PROFILE),
        sm_ids=list(range(8)),kernel_ms=[1.0,1.0,1.0],fingerprints=[[rank.copy() for _ in range(16)] for _ in range(3)],
        phase=phase,sample_id=f'{gpu}-{nonce}-{phase}-{i}',build_id='0'*64)


def dataset():
    return [record(gpu,nonce,phase,i) for gpu in ('A','B') for nonce in NONCES
            for phase,n in (('registration',8),('calibration',16),('evaluation',16)) for i in range(n)]


class CryptoTests(unittest.TestCase):
    def test_rfc8439_python_vector(self):
        words=fp.chacha_block(bytes(range(32)),1,(0x09000000,0x4a000000,0))
        self.assertEqual(struct.pack('<16I',*words).hex(),VECTOR)

    def test_cpp_core_vector_and_python_agreement(self):
        compiler=shutil.which('g++') or shutil.which('clang++')
        if compiler is None: self.skipTest('no C++ compiler; CUDA still not invoked')
        binary=WORK/('crypto_selftest.exe' if os.name=='nt' else 'crypto_selftest')
        subprocess.run([compiler,'-std=c++11','-O2',str(Path(__file__).with_name('crypto_selftest.cpp')),'-o',str(binary)],check=True,capture_output=True)
        subprocess.run([str(binary)],check=True,capture_output=True)
        rng=random.Random(123)
        for _ in range(8):
            key=bytes(rng.randrange(256) for _ in range(32)); counter=rng.randrange(2**32)
            nonce=tuple(rng.randrange(2**32) for _ in range(3))
            output=subprocess.run([str(binary),'--block',key.hex(),str(counter),*[str(x) for x in nonce]],check=True,capture_output=True,text=True).stdout.strip()
            self.assertEqual(output,struct.pack('<16I',*fp.chacha_block(key,counter,nonce)).hex())

    def test_every_challenge_bit_affects_sampled_lane_and_delay_trajectories(self):
        def trajectory(key):
            words=[fp.chacha_block(key,0,(sm,r,fp.LOCAL_DOMAIN)) for sm in range(8) for r in range(4)]
            return [w[0]%64 for w in words],[w[1]%2049 for w in words]
        lanes,delays=trajectory(bytes(32))
        for bit in range(256):
            key=bytearray(32); key[bit//8]=1<<(bit%8)
            new_lanes,new_delays=trajectory(bytes(key))
            self.assertNotEqual(new_lanes,lanes)
            self.assertNotEqual(new_delays,delays)


class SchemaTests(unittest.TestCase):
    def test_reject_malformed_challenge(self):
        for value in ('1','g'*64,'A'*64,True):
            with self.assertRaises(ValueError): fp.challenge(value)

    def test_rank_and_type_constraints(self):
        valid=record('A',NONCES[0]); fp.validate_record(valid)
        for bad in ('duplicate','fraction','boolean'):
            r=copy.deepcopy(valid)
            if bad=='duplicate': r['fingerprints'][0][0][1]=0
            elif bad=='fraction': r['fingerprints'][0][0][1]=1.5
            else: r['fingerprints']=[[[True]*8 for _ in range(16)] for _ in range(3)]
            with self.assertRaises(ValueError): fp.validate_record(r)

    def test_profile_and_sm_map_binding(self):
        r=record('A',NONCES[0]); r['sm_ids'][0]=1
        with self.assertRaises(ValueError): fp.validate_record(r)
        r=record('A',NONCES[0]); r['profile']['delay_base']=1
        with self.assertRaises(ValueError): fp.validate_record(r)

    def test_subset_mask_validation(self):
        p=copy.deepcopy(PROFILE); p['mode']='subset'
        mask=fp.plan(NONCES[0],p,list(range(8)))
        ranks=np.full((3,16,8),-1,dtype=np.int32)
        for b in range(3):
            for r in range(16): ranks[b,r,mask[r]]=np.arange(mask[r].sum())
        item=record('A',NONCES[0]); item['profile']=p; item['fingerprints']=ranks.tolist()
        fp.validate_record(item)
        r,c=np.argwhere(~mask)[0]; item['fingerprints'][0][r][c]=0
        with self.assertRaises(ValueError): fp.validate_record(item)

    def test_packed_arrays_size_limits(self):
        packed=fp.pack_array(np.arange(100,dtype=np.int32),'<i4')
        np.testing.assert_array_equal(fp.unpack_array(packed),np.arange(100))
        packed['shape']=[1]
        with self.assertRaises(ValueError): fp.unpack_array(packed)


class EnrollmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.registry=fp.enroll(dataset())

    def test_positive_matching_and_readiness(self):
        self.assertTrue(self.registry['devices']['A']['ready_for_research_verification'])
        result=fp.decision(self.registry,record('A',NONCES[0]),'A')
        self.assertTrue(result['accepted']); self.assertFalse(result['location_verified'])

    def test_two_of_three_votes(self):
        r=record('A',NONCES[0]); other=record('B',NONCES[0])
        r['fingerprints'][0]=other['fingerprints'][0]
        self.assertTrue(fp.decision(self.registry,r,'A')['accepted'])
        r['fingerprints'][1]=other['fingerprints'][1]
        self.assertFalse(fp.decision(self.registry,r,'A')['accepted'])

    def test_duplicate_samples_rejected(self):
        rows=dataset(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError): fp.enroll(rows)

    def test_evaluation_never_changes_thresholds_or_templates(self):
        rows=dataset()
        for r in rows:
            if r['phase']=='evaluation' and r['device_uuid']=='A': r['fingerprints']=record('B',r['challenge'])['fingerprints']
        changed=fp.enroll(rows)
        for gpu in ('A','B'):
            self.assertEqual(changed['devices'][gpu]['threshold'],self.registry['devices'][gpu]['threshold'])
            self.assertEqual(changed['devices'][gpu]['templates'],self.registry['devices'][gpu]['templates'])
        self.assertFalse(changed['devices']['A']['ready_for_research_verification'])
        with self.assertRaises(ValueError): manage.issue(changed,WORK/'bad-validation.sqlite','A')

    def test_clipped_metric_and_tie_rejection(self):
        robust=fp.enroll(dataset(),'clipped_mad_l1')
        self.assertTrue(fp.decision(robust,record('A',NONCES[0]),'A')['accepted'])
        tied=copy.deepcopy(robust)
        tied['devices']['B']['templates']=copy.deepcopy(tied['devices']['A']['templates'])
        self.assertFalse(fp.decision(tied,record('A',NONCES[0]),'A')['accepted'])

    def test_scoring_configuration_binding(self):
        modified=copy.deepcopy(self.registry)
        modified['scoring_config']['residual_cap']=3.0
        with self.assertRaises(ValueError): fp.decision(modified,record('A',NONCES[0]),'A')

    def test_quantile_requires_finite_sample_support(self):
        with self.assertRaises(ValueError): fp.conformal_threshold([0,1],.01)
        self.assertEqual(fp.conformal_threshold(list(range(32)),.05),31)


class FreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.registry=fp.enroll(dataset())
    def state(self): return WORK/(uuid.uuid4().hex+'.sqlite')

    def test_single_use_and_ignoring_claimed_uuid(self):
        db=self.state(); request=manage.issue(self.registry,db,'A')
        response=record('A',request['challenge']); response['device_uuid']='untrusted-label'
        self.assertTrue(manage.verify(self.registry,db,request,response)['accepted'])
        with self.assertRaises(ValueError): manage.verify(self.registry,db,request,response)

    def test_cross_challenge_attempt_burns_ticket(self):
        db=self.state(); request=manage.issue(self.registry,db,'A')
        with self.assertRaises(ValueError): manage.verify(self.registry,db,request,record('A','03'*32))
        with self.assertRaises(ValueError): manage.verify(self.registry,db,request,record('A',request['challenge']))

    def test_pool_exhaustion(self):
        db=self.state(); a=manage.issue(self.registry,db,'A'); b=manage.issue(self.registry,db,'B')
        self.assertNotEqual(a['challenge'],b['challenge'])
        with self.assertRaises(ValueError): manage.issue(self.registry,db,'A')

    def test_concurrent_issuance_never_reuses_a_nonce(self):
        db=self.state()
        with ThreadPoolExecutor(max_workers=2) as executor:
            requests=list(executor.map(lambda target:manage.issue(self.registry,db,target),('A','B')))
        self.assertEqual(len({r['challenge'] for r in requests}),2)

    def test_expiry_registry_binding_and_attestation_fail_closed(self):
        db=self.state(); request=manage.issue(self.registry,db,'A')
        with manage.connect(db) as connection: connection.execute('UPDATE tickets SET issued=0')
        with self.assertRaises(ValueError): manage.verify(self.registry,db,request,record('A',request['challenge']))
        db=self.state(); request=manage.issue(self.registry,db,'A'); request['target']='B'
        with self.assertRaises(ValueError): manage.verify(self.registry,db,request,record('A',request['challenge']))
        with self.assertRaises(NotImplementedError): manage.verify(self.registry,self.state(),{}, {},True)


class FakeProbe:
    """In-memory pipe protocol; no child process or CUDA operation."""
    invalid=False
    def __init__(self,*args,**kwargs):
        self.returncode=None; self.messages=queue.Queue()
        self.messages.put(json.dumps(dict(type='ready',device_uuid='GPU-test',profile=PROFILE))+'\n')
        owner=self
        class Input:
            closed=False
            def write(self,text):
                nonce='03'*32 if owner.invalid else text.strip()
                r=record('A',nonce); r['device_uuid']='GPU-test'
                owner.messages.put(json.dumps(r)+'\n')
            def flush(self): pass
            def close(self): self.closed=True; owner.returncode=0; owner.messages.put('')
        class Output:
            def readline(self,limit): return owner.messages.get()
            def close(self): pass
        self.stdin=Input(); self.stdout=Output()
    def poll(self): return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None: self.returncode=0
        return self.returncode
    def terminate(self): self.returncode=-15; self.messages.put('')
    def kill(self): self.terminate()


class CollectorTests(unittest.TestCase):
    def arguments(self):
        folder=WORK/uuid.uuid4().hex; folder.mkdir()
        binary=folder/'fake-probe'; binary.write_bytes(b'not an executable; mocked')
        pool=folder/'pool.json'; pool.write_text(json.dumps(dict(challenges=NONCES)))
        return SimpleNamespace(pool=pool,binary=binary,device='GPU-test',mode='all',rounds=16,
            repeats=1,timeout=1,phase='calibration',output=folder/'capture.jsonl')

    def test_persistent_capture_and_complete_manifest(self):
        args=self.arguments()
        with patch.object(collect.subprocess,'Popen',FakeProbe),patch.object(collect,'telemetry',return_value=dict(available=True)):
            collect.capture(args)
        rows=[json.loads(line) for line in args.output.read_text().splitlines()]
        self.assertEqual({r['challenge'] for r in rows},set(NONCES))
        self.assertTrue(all(r['phase']=='calibration' for r in rows))
        manifest=json.loads(args.output.with_suffix('.jsonl.manifest.json').read_text())
        self.assertEqual(manifest['status'],'complete'); self.assertEqual(manifest['received_measurements'],2)

    def test_mismatched_response_preserves_failed_manifest(self):
        class BadProbe(FakeProbe): invalid=True
        args=self.arguments()
        with patch.object(collect.subprocess,'Popen',BadProbe),patch.object(collect,'telemetry',return_value=dict(available=True)):
            with self.assertRaises(ValueError): collect.capture(args)
        manifest=json.loads(args.output.with_suffix('.jsonl.manifest.json').read_text())
        self.assertEqual(manifest['status'],'failed'); self.assertEqual(manifest['received_measurements'],0)

    def test_issued_request_capture_and_single_use_verification(self):
        args=self.arguments(); args.phase='evaluation'; args.pool=None
        registry=fp.enroll(dataset()); state=WORK/(uuid.uuid4().hex+'.sqlite')
        request=manage.issue(registry,state,'A')
        args.request=args.output.parent/'request.json'
        args.request.write_text(json.dumps(request))
        with patch.object(collect.subprocess,'Popen',FakeProbe),patch.object(collect,'telemetry',return_value=dict(available=True)):
            collect.capture(args)
        response=json.loads(args.output.read_text())
        self.assertEqual(response['request_id'],request['request_id'])
        self.assertTrue(manage.verify(registry,state,request,response)['accepted'])
        with self.assertRaises(ValueError): manage.verify(registry,state,request,response)


if __name__=='__main__': unittest.main(verbosity=2)
