'Source-bound TRAIN/CAL reader for the new exploratory mainline bank.'
import itertools
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .scenario_mainline_chart import (SCHEMA,COORDINATES,DIM,CATALOG,
    SCENARIO_NAMES,active_coordinates,template_compatibility)
from .scenario_training import source_sha,shapes
from .stage5_partial import sha,selected_array
from .v2_dataset import array_header,read_split
from .v2_input import INPUT_KEYS,HOST_CLASSES,prepare
from .v2_training import optional_missing

BANK_SCHEMA='exploratory_mainline_combined_bank_1'
BANK_SCHEMAS=frozenset((BANK_SCHEMA,'mainline_combined_production_bank_1'))
FIELDS=INPUT_KEYS+('z','active','coarse','template','scenario','geometry','public_visible_peak','generation_failed')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def generating_source_sha(path):
    'Exact original v2_scenario_real_ppc generator pin, including binary assets.'
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def bound(base,item):
    path=(base/item['path']).resolve()
    if base not in path.parents or sha(path)!=item['sha256']:
        raise ValueError('Bound mainline asset changed: '+item['path'])
    return path


class MainlineBank:
    'Complete original new bank; indices include failures, iterate omits only them.'
    def __init__(self,bank,*,producer_package=None,delta_bank=None):
        path=Path(bank).resolve();self.base=path if path.is_dir() else path.parent
        self.package=Path(producer_package).resolve() if producer_package is not None else Path(__file__).resolve().parents[1]
        self.plan=read(self.base/'plan.json');run=read(self.base/'run.json');plan=self.plan
        schema=plan.get('schema')
        if (schema not in BANK_SCHEMAS or run.get('schema')!=schema
                or plan.get('status')!='frozen' or run.get('status')!='completed'
                or plan.get('fixture') is not False or run.get('fixture') is not False
                or plan.get('TEST_rows')!=0 or run.get('TEST_rows')!=0
                or plan.get('real_TEST_decoded') is not False or run.get('TEST_read') is not False
                or run['plan_sha256']!=sha(self.base/'plan.json') or plan.get('physics_schema')!=SCHEMA):
            raise ValueError('Complete actual non-TEST new mainline bank required')
        self.coordinates=tuple(plan['coordinates']);self.catalog=tuple(plan['catalog']);self.dim=len(self.coordinates)
        if (self.coordinates!=COORDINATES or self.catalog!=CATALOG or self.dim!=DIM
                or plan.get('continuous_dim')!=DIM or run.get('continuous_dim')!=DIM):
            raise ValueError('Explicit new mainline chart/catalog differs')
        self.pins=dict(plan['source_pins'])
        for relative,digest in self.pins.items():
            original=(self.package/relative).resolve()
            if self.package not in original.parents or generating_source_sha(original)!=digest:
                raise ValueError('Original mainline generating source changed: '+relative)


        executing=Path(__file__).resolve().parents[1]
        for relative in ('sputtergisaxs/scenario_mainline_chart.py','sputtergisaxs/v2_input.py'):
            if relative not in self.pins or source_sha(executing/relative)!=self.pins[relative]:
                raise ValueError('Executing mainline decoder/input differs from the generating bank: '+relative)
        self.bindings={name:bound(self.base,item) for name,item in plan['bindings'].items()}
        split=read(self.bindings['split.json']);train={s['scan'] for s in split['scans'] if s['split']=='TRAIN'}
        hosts=read(self.bindings['host_shapes.json']);self.templates=tuple(hosts['rows']);self.keys=tuple(t['key'] for t in self.templates)
        if (len(set(self.keys))!=len(self.keys) or any(t.get('split')!='TRAIN' or t['scan'] not in train for t in self.templates)):
            raise ValueError('Unique TRAIN-only original host templates required')
        policy=plan['synthetic_split']
        if (policy['TRAIN_family_remainders']!=list(range(8)) or policy['CAL_family_remainders']!=[8]
                or policy['TEST_family_remainders']!=[9]):
            raise ValueError('New declared immutable TRAIN/CAL remainder split differs')
        with np.load(self.bindings['assignments.npz'],allow_pickle=False) as a:
            self.families=np.array(a['family_id'],copy=True);scenes=np.array(a['scenario'],copy=True)
        if (len(self.families)!=plan['families'] or np.any(self.families%10==9)
                or len(set(map(int,self.families)))!=len(self.families) or np.any((scenes<0)|(scenes>=4))):
            raise ValueError('Original new mainline family identities/count differ')
        self.scenario_by_family=dict(zip(map(int,self.families),map(int,scenes)))
        self.shapes=shapes(DIM);self.shapes['generation_failed']=()
        atomic_shape=self.shapes['mean_rate']
        self.shapes.update({k:atomic_shape for k in ('atomic_Y','public_atomic_Y','P','public_P')})
        self.files=[];seen=[];self.failed_by_family={};self.location={}
        for item in run['shards']:
            file=bound(self.base,item);record_path=file.parent/'run.json';record=read(record_path)
            if (record.get('schema')!=schema or record.get('status')!='completed'
                    or record['identity']['plan_sha256']!=run['plan_sha256'] or record['identity']['source_pins']!=self.pins
                    or record['checkpoint_sha256']!=item['sha256'] or record['rows']!=item['rows']
                    or sha(record_path)!=item['receipt_sha256']):
                raise ValueError('Original complete mainline shard/source/receipt differs')
            with zipfile.ZipFile(file) as archive:
                ids=selected_array(archive,'family_id');index=selected_array(archive,'index')
                failures=selected_array(archive,'generation_failed')
                if (ids.dtype.kind not in 'iu' or failures.dtype.kind!='b' or not np.array_equal(index,ids)
                        or ids.tolist()!=record['identity']['family_ids'] or len(ids)!=item['rows']
                        or len(record['receipts'])!=len(ids)):
                    raise ValueError('Original mainline row identity/failure flag differs')
                for name,expected in self.shapes.items():
                    shape,dtype=array_header(archive,name)
                    if shape!=(len(ids),)+expected or dtype.hasobject:
                        raise ValueError('Original mainline scientific shape differs: '+name)
                    if name in ('atomic_Y','public_atomic_Y','P','public_P') and dtype.kind not in 'iu':
                        raise ValueError('Original saved mainline counts/support must be integer')
                for fid,failed,receipt in zip(ids,failures,record['receipts']):
                    fid=int(fid)
                    if fid in self.failed_by_family or receipt['id']!=fid or receipt['scenario']!=SCENARIO_NAMES[self.scenario_by_family[fid]]:
                        raise ValueError('Original mainline generation receipt/order differs')
                    if bool(failed)!=(receipt['status']=='failed_original_family_no_redraw'):
                        raise ValueError('Generation failure flag disagrees with its original receipt')
                    self.failed_by_family[fid]=bool(failed)
                    self.location[fid]=file
                seen.extend(map(int,ids))
            self.files.append((file,item['sha256']))
        if seen!=self.families.tolist():raise ValueError('All original mainline family slots required without duplicates')
        self.files=tuple(self.files)
        self.split_rows={s:len(self.indices(s)) for s in ('TRAIN','CAL')}
        self.successful_split_rows={s:sum(not self.failed_by_family[int(fid)] for fid in self.indices(s)) for s in ('TRAIN','CAL')}
        self.scenario_split_rows={s:{name:sum(self.scenario_by_family[int(fid)]==i for fid in self.indices(s))
            for i,name in enumerate(SCENARIO_NAMES)} for s in ('TRAIN','CAL')}
        self.successful_scenario_split_rows={s:{name:sum(self.scenario_by_family[int(fid)]==i and not self.failed_by_family[int(fid)] for fid in self.indices(s))
            for i,name in enumerate(SCENARIO_NAMES)} for s in ('TRAIN','CAL')}
        self.receipt=dict(schema=schema,bank_run_sha256=sha(self.base/'run.json'),plan_sha256=run['plan_sha256'],
            original_families=len(self.families),split_rows=self.split_rows,successful_split_rows=self.successful_split_rows,
            scenario_split_rows=self.scenario_split_rows,successful_scenario_split_rows=self.successful_scenario_split_rows,
            generation_failed_family_ids=[int(fid) for fid in self.families if self.failed_by_family[int(fid)]],
            source_pins=self.pins,physics_schema=SCHEMA,continuous_dim=DIM,coarse_classes=len(CATALOG),
            template_classes=len(self.templates)+1,fixture=False,TEST_scientific_rows_decoded=False,
            generation_failure_policy='retain original denominator; no zero-input fabricated likelihood examples',
            exploratory=True,production_selected=False,structural_release=False)
        self.delta=None
        if delta_bank is not None:
            from .scenario_mainline_delta import DeltaBank
            self.delta=DeltaBank(delta_bank,expected_bank_plan_sha256=run['plan_sha256'])
            if self.delta.receipt['family_ids']!=self.families.tolist():raise ValueError('Delta branch must retain every original bank family slot')
        self.receipt['optional_signed_delta_bank']=self.delta.receipt if self.delta is not None else None

    def indices(self,split):
        if split not in ('TRAIN','CAL'):raise ValueError('Mainline training reader only opens TRAIN or CAL')
        return self.families[self.families%10<8 if split=='TRAIN' else self.families%10==8].copy()

    def iterate(self,split,*,epoch=0):
        if split not in ('TRAIN','CAL'):raise ValueError('Mainline reader never decodes TEST')
        rng=np.random.default_rng(np.random.SeedSequence([672520,int(epoch)]))
        order=rng.permutation(len(self.files)) if split=='TRAIN' else np.arange(len(self.files))
        for j in order:
            file,digest=self.files[int(j)];arrays,_=read_split(file,digest,split,{k:self.shapes[k] for k in FIELDS})
            rows=rng.permutation(len(arrays['family_id'])) if split=='TRAIN' else np.arange(len(arrays['family_id']))
            for i in rows:
                fid=int(arrays['family_id'][i])
                if bool(arrays['generation_failed'][i])!=self.failed_by_family[fid]:raise ValueError('Original failure flag changed during read')
                if self.failed_by_family[fid]:continue
                row={k:v[i] for k,v in arrays.items()};c=int(row['coarse']);t=int(row['template']);s=int(row['scenario'])
                if not 0<=c<len(CATALOG) or CATALOG[c]['scenario_index']!=s or self.scenario_by_family[fid]!=s or not 0<=t<=len(self.templates):
                    raise ValueError('Original successful mainline class/scene identity differs')
                if not np.array_equal(row['active'],active_coordinates(c,t)) or not np.isfinite(row['z'][row['active']]).all():
                    raise ValueError('Original actual51D active coordinates differ')
                if (row['mandatory'][2] not in (0,1) or not np.isfinite(row['geometry']).all()
                        or not np.allclose(row['mandatory'][:2],[(row['geometry'][0]-14000)/6000,(row['geometry'][1]-.425)/.275],rtol=0,atol=2e-6)):
                    raise ValueError('Original mainline mandatory geometry differs')
                yield row

    def iter_batches(self,split,*,batch_size=128,mode='features',epoch=0,device='cpu'):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:raise ValueError('Positive batch size required')
        rows=self.iterate(split,epoch=epoch if split=='TRAIN' else 0)
        while True:
            block=list(itertools.islice(rows,batch_size))
            if not block:return
            yield batch_inputs(block,mode,epoch if split=='TRAIN' else 0,device,templates=self.templates,delta_bank=self.delta)


def batch_inputs(rows,mode,epoch,device,*,templates,delta_bank=None):
    import torch
    if not rows:raise ValueError('Nonempty original successful mainline batch required')
    inputs=[optional_missing(row,epoch) for row in rows]
    encoded=prepare({k:np.stack([r[k] for r in inputs]) for k in INPUT_KEYS},mode)
    encoded={k:torch.as_tensor(encoded[k],device=device) for k in ('vector','image') if encoded[k] is not None}
    if delta_bank is not None:
        branch=delta_bank.for_families([int(r['family_id']) for r in rows],enabled=np.asarray([r['known'][2] for r in inputs],bool))
        encoded['delta']=torch.as_tensor(branch['vector'],device=device)
    legal=[]
    for row,visible in zip(rows,inputs):
        category=HOST_CLASSES[int(visible['optional'][0])] if visible['known'][0] else None
        legal.append(template_compatibility(*row['geometry'],templates,host_category=category))
    target={k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device) for k in ('z','active','coarse','template','scenario')}
    return encoded,target,torch.as_tensor(np.stack(legal),device=device)
