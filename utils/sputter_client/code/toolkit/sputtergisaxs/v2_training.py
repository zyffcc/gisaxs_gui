'Source-bound candidate bank adapter; no final bank or trained model.'
import json
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from .stage5_partial import sha, selected_array
from .v2_dataset import array_header, read_split
from .v2_input import INPUT_KEYS, HOST_CLASSES, prepare
from .v2_scene import SCHEMA, DIM, SCENES, active_coordinates, template_compatibility, decode
from .v2_observation import COUNT_CONTRACT, TEMPLATE_CONTRACT

BANK_SCHEMA = 'effective_v2_candidate_bank_2'
SHAPES = dict(image=(2,32,70), pre=(2,32,70), features=(40,), mandatory=(5,),
              optional=(3,), known=(4,), z=(DIM,), active=(DIM,), coarse=(),
              template=(), geometry=(2,), public_N=(8,70), public_visible_peak=())
FORWARD_PINS=tuple('sputtergisaxs/'+n+'.py' for n in (
    'effective_host','effective_measurement','effective_cuda','cell_response',
    'formfactor','surface_graded','surface_separable','multilayer','optics','forward_v06',
    'forward_v06_gpu','bessel_cuda','forward','tool_qinput','stage5_measurement',
    'stage5_input','stage5_gpu','tool_prior','prediction','resolvable','features','geometry'))+('assets/optics_chantler.npz',)
PINS = ('sputtergisaxs/v2_scene.py', 'sputtergisaxs/v2_input.py',
        'sputtergisaxs/v2_observation.py',
        'sputtergisaxs/v2_runtime.py',
        'sputtergisaxs/v2_ppc.py', 'sputtergisaxs/v2_predictive.py',
        'sputtergisaxs/v2_evaluation.py',
        'sputtergisaxs/v2_dataset.py', 'sputtergisaxs/stage5_partial.py',
        'sputtergisaxs/v2_posterior.py', 'sputtergisaxs/v2_training.py',
        'scripts/v2_train.py') + FORWARD_PINS


def source_sha(path):
    'Match make_dist LF semantics on Windows/Linux; array bytes stay exact.'
    p=Path(path);raw=p.read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n',b'\n') if p.suffix=='.py' else raw).hexdigest()


def _bound(base, item):
    p = (base/item['path']).resolve()
    if sha(p) != item['sha256']:
        raise ValueError('Bound input SHA changed: '+item['path'])
    return p


class CandidateBank:
    'Verify compressed bytes, headers and ALL identities without TEST science.'
    def __init__(self, manifest, package, *, allow_fixture=False):
        self.path=Path(manifest).resolve();self.base=self.path.parent
        self.manifest=json.loads(self.path.read_text(encoding='utf8'));m=self.manifest
        if not isinstance(allow_fixture,bool) or not isinstance(m.get('fixture',False),bool):
            raise ValueError('Explicit Boolean interface-fixture mode required')
        if allow_fixture and m.get('fixture') is not True:
            raise ValueError('Validation bypass requires an explicitly marked interface fixture')
        if m.get('schema')!=BANK_SCHEMA or m.get('physics_schema')!=SCHEMA or m.get('status')!='completed':
            raise ValueError('Complete candidate v2 bank required, not a diagnostic/legacy bank')
        if m.get('count_support_contract')!=COUNT_CONTRACT or m.get('template_contract')!=TEMPLATE_CONTRACT:
            raise ValueError('Explicit pixel-frame and dimensionless-template contracts required')
        if m.get('fixture',False) and not allow_fixture:
            raise ValueError('Synthetic interface fixture is forbidden for formal training')
        pins=m['source_pins'];package=Path(package)
        if not set(PINS).issubset(pins) or not m['generator_entry'] in pins:
            raise ValueError('Complete generator/decoder/measurement source binding required')
        for k,digest in pins.items():
            p=(package/k).resolve()
            if package.resolve() not in p.parents or source_sha(p)!=digest:
                raise ValueError('Generator/input/physics source differs: '+k)
        split_path=_bound(self.base,m['real_split']);split=json.loads(split_path.read_text(encoding='utf8'))
        policy=split['synthetic_split']
        if m['synthetic_split']!=policy or policy['TRAIN_family_remainders']!=list(range(8)) or policy['CAL_family_remainders']!=[8] or policy['TEST_family_remainders']!=[9]:
            raise ValueError('Frozen family partition/seed declaration differs')
        hp=_bound(self.base,m['host_metadata']);ha=_bound(self.base,m['host_array'])
        hosts=json.loads(hp.read_text(encoding='utf8'))
        if hosts['split_sha256']!=m['real_split']['sha256'] or hosts['array_sha256']!=sha(ha):
            raise ValueError('TRAIN host library binding differs')
        if (hosts.get('template_contract')!=TEMPLATE_CONTRACT or
            hosts.get('normalization_method')!='q80_positive_fallback_1' or
            hosts.get('normalization_source_sha256')!=pins['sputtergisaxs/v2_observation.py']):
            raise ValueError('Host metadata must bind its own unit transform')
        original_hp=_bound(self.base,hosts['original_metadata'])
        original_ha=_bound(self.base,hosts['original_array'])
        original_hosts=json.loads(original_hp.read_text(encoding='utf8'))
        if (original_hosts['array_sha256']!=sha(original_ha) or
            original_hosts.get('split_sha256')!=m['real_split']['sha256']):
            raise ValueError('Original TRAIN template source changed')
        self.templates=hosts['rows'];keys=tuple(t['key'] for t in self.templates)
        if set(keys)!={t['key'] for t in original_hosts['rows']} or any(
            not np.isfinite(t.get('normalization_scale',np.nan)) or t.get('normalization_scale',0)<=0
            for t in self.templates):
            raise ValueError('Complete original template keys and positive normalizations required')
        train={s['scan'] for s in split['scans'] if s['split']=='TRAIN'}
        original_rows={t['key']:t for t in original_hosts['rows']}
        if len(original_rows)!=len(original_hosts['rows']) or any(
            t.get('split')!='TRAIN' or t.get('scan') not in train for t in original_rows.values()):
            raise ValueError('Original template source is not unique TRAIN-only metadata')
        if any(any(t.get(k)!=v for k,v in original_rows[t['key']].items()
                   if k!='normalization_scale') for t in self.templates):
            raise ValueError('Template unit transform changed original identity/geometry/metadata')
        if len(set(keys))!=len(keys) or any(t['split']!='TRAIN' or t['scan'] not in train for t in self.templates):
            raise ValueError('Host source is not unique TRAIN-only metadata')
        self.keys=keys
        identities=m['family_ids'];siblings=m['siblings']
        if (not identities or any(isinstance(f,bool) or not isinstance(f,int) or f<0 for f in identities)
            or len(set(identities))!=len(identities) or isinstance(siblings,bool) or not isinstance(siblings,int)
            or not 1<=siblings<=8 or max(identities)>(np.iinfo(np.int64).max-8)//siblings):
            raise ValueError('Original integer family/sibling identity required')
        if len(identities)<30000 and not allow_fixture:
            raise ValueError('Formal C4 requires at least 30000 original families')
        if not allow_fixture:
            gate=json.loads(_bound(self.base,m['c3_gate']).read_text(encoding='utf8'))
            if (gate.get('schema')!='effective_v2_forward_training_gate_1' or
                gate.get('status')!='closed_pass_with_declared_limits' or
                gate.get('numerical_passed') is not True or
                gate.get('real_same_system_bilateral_windows_tested',0)<10 or gate.get('new_domain_families',0)<3000 or
                gate.get('common_signal_ba_decrease',0)<.05 or
                gate.get('source_pins')!={k:pins[k] for k in FORWARD_PINS}):
                raise ValueError('Source-bound C3 findings required before formal C4 training')
            if gate.get('forward_choice') not in ('native','certified_fast','certified_sharp_fast_native_graded'):
                raise ValueError('Declared certified/native forward routing required')
            if gate['forward_choice']!='native':
                certificate=json.loads(_bound(self.base,m['fast_certificate']).read_text(encoding='utf8'))
                strict=(certificate.get('status')=='completed' and certificate.get('passed') is True and
                        certificate.get('completed_points')==200 and certificate.get('passed_points')==200)
                early=(certificate.get('schema')=='effective_v2_early_weak_certificate_1' and
                       certificate.get('status')=='certified_early' and certificate.get('authorized_pass') is True and
                       certificate.get('completed_points',0)>=50 and certificate.get('exception_fraction',1)<=.05)
                if (not(strict or early) or
                    certificate.get('TEST_read') is not False or
                    any(certificate.get('source_dependencies',{}).get(k)!=pins[k] for k in certificate.get('source_dependencies',{})) or
                    not {'sputtergisaxs/effective_host.py','sputtergisaxs/effective_measurement.py','sputtergisaxs/effective_cuda.py'}.issubset(certificate.get('source_dependencies',{}))):
                    raise ValueError('Actual same-source authorized fast certificate required')
                if 'zero transition' in certificate.get('scope','') and gate['forward_choice']!='certified_sharp_fast_native_graded':
                    raise ValueError('Sharp-only certificate cannot certify the graded candidate bank')
                if gate['forward_choice']=='certified_sharp_fast_native_graded' and m.get('forward_policy')!='certified sharp fast; native graded, no inherited graded fast certificate':
                    raise ValueError('Graded scenes must retain original native routing')
        families=set(identities)
        expected={f*siblings+s for f in identities for s in range(siblings)}
        seen=set();paths=set();self.shards=[];counts=dict(TRAIN=0,CAL=0,TEST=0)
        for item in m['shards']:
            path=_bound(self.base,item)
            if path in paths:raise ValueError('Repeated shard')
            paths.add(path)
            with zipfile.ZipFile(path) as ar:
                for key in ('family_id','index'):
                    sh,dt=array_header(ar,key)
                    if len(sh)!=1 or dt.kind not in 'iu':raise ValueError('Integer identities required')
                fid=selected_array(ar,'family_id');index=selected_array(ar,'index');n=len(fid)
                if n!=len(index) or n!=item['rows'] or any(int(f) not in families for f in fid):
                    raise ValueError('Shard original identity/count differs')
                if np.any(index//siblings!=fid) or len(set(map(int,index)))!=n or seen.intersection(map(int,index)):
                    raise ValueError('Duplicate or inconsistent original row')
                seen.update(map(int,index))
                for key,shape in SHAPES.items():
                    sh,dt=array_header(ar,key)
                    if sh!=(n,)+shape:raise ValueError('Bank scientific shape differs: '+key)
                    if (key=='z' and dt!=np.dtype('float64') or key in ('known','active','public_visible_peak') and dt.kind!='b'
                        or key in ('coarse','template') and dt.kind not in 'iu'):
                        raise ValueError('Joint labels/flags/class dtype differs: '+key)
                for k in counts:
                    counts[k]+=int(np.sum(fid%10<8 if k=='TRAIN' else fid%10==(8 if k=='CAL' else 9)))
            self.shards.append((path,item['sha256']))
        if seen!=expected or any(v==0 for v in counts.values()):
            raise ValueError('Original bank incomplete or split empty; no dropped/failed rows allowed')
        self.receipt=dict(manifest_sha256=sha(self.path),original_families=len(identities),
                          original_rows=len(seen),split_rows=counts,source_pins=pins,
                          count_support_contract=COUNT_CONTRACT,template_contract=TEMPLATE_CONTRACT,
                          fixture=bool(m.get('fixture',False)),validation_mode='interface_fixture' if allow_fixture else 'formal_source_bound_candidate',
                          TEST_scientific_rows_decoded=False,structural_release=False)

    def iterate(self,split,*,epoch=0):
        if split not in ('TRAIN','CAL'):raise ValueError('Training adapter never decodes TEST science')
        order=np.arange(len(self.shards))
        rng=np.random.default_rng(410620261012+epoch)
        if split=='TRAIN':rng.shuffle(order)
        parents={}
        for j in order:
            path,digest=self.shards[j];a,_=read_split(path,digest,split,SHAPES)
            order_rows=rng.permutation(len(a['index'])) if split=='TRAIN' else np.arange(len(a['index']))
            for i in order_rows:
                row={k:v[i] for k,v in a.items()}
                if not np.isfinite(row['public_N']).all() or np.any(row['public_N']<0):
                    raise ValueError('Original public pixel-frame support required')
                if np.any(row['public_N']!=np.rint(row['public_N'])) or np.any(row['features'][33:37]!=0):
                    raise ValueError('Pixel-frame support and shape-only reference contract violated')
                coarse=int(row['coarse']);template=int(row['template'])
                if not 0<=coarse<len(SCENES) or not 0<=template<=len(self.templates):
                    raise ValueError('Original scene/template label outside bound catalog')
                active=active_coordinates(coarse,template)
                if not np.array_equal(active,row['active']) or not np.isfinite(row['z'][active]).all():
                    raise ValueError('Original active joint coordinates differ')
                E,ai=row['geometry'];mt=row['mandatory'];metal='Au' if mt[2]==1 else 'Ag'
                if mt[2] not in (0,1) or not np.allclose(mt[:2],[(E-14000)/6000,(ai-.425)/.275],rtol=0,atol=2e-6):
                    raise ValueError('Original mandatory geometry differs')
                decoded=decode(row['z'],coarse,template,metal=metal,energy_ev=E,alpha_i_deg=ai,
                               template_keys=self.keys,template_metadata=self.templates)
                parent=(coarse,tuple(row['z'][:6]),metal)
                f=int(row['family_id'])
                if f in parents and parents[f]!=parent:
                    raise ValueError('Original family changed shape/structural parent across siblings')
                parents[f]=parent

                row['diameter_nm']=decoded['diameter_nm']
                yield row


def optional_missing(row,epoch):
    'Hide available input only; stable row/epoch streams shared by four modes.'
    x={k:np.array(row[k],copy=True) for k in INPUT_KEYS}
    rng=np.random.default_rng(np.random.SeedSequence([410620261013,int(row['index']),int(epoch)]))
    u=rng.random();keep=np.zeros(4,bool) if u<.5 else np.ones(4,bool) if u<.75 else rng.random(4)<.5
    x['known'] &= keep
    return x


def batch_inputs(rows,mode,epoch,device,*,templates):
    'Joint labels kept exact; encoder-only hiding never changes photon arrays.'
    import torch
    inputs=[optional_missing(r,epoch) for r in rows]
    batch={k:np.stack([r[k] for r in inputs]) for k in INPUT_KEYS}
    encoded=prepare(batch,mode)
    tensors={k:torch.as_tensor(encoded[k],device=device) for k in ('vector','image') if encoded[k] is not None}
    legal=[]
    for r,x in zip(rows,inputs):
        cat=HOST_CLASSES[int(x['optional'][0])] if x['known'][0] else None
        legal.append(template_compatibility(*r['geometry'],templates,host_category=cat))
    target={k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device) for k in ('z','active','coarse','template')}
    return tensors,target,torch.as_tensor(np.stack(legal),device=device)
