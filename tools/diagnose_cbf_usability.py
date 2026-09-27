"""Controlled observation scale/noise comparisons using unchanged V5 weights."""
import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import WorkflowEngine,prepare_sides,write_side
from src.gimap.features.fitting.application.workflow_v5 import validate_options
import numpy as np

def main():
    v=json.loads((ROOT/'validation/center_symmetry/VERIFIED.json').read_text())
    request=json.loads((Path(v['fit_output'])/'request.json').read_text())
    out=ROOT/'validation/usability_20260921';out.mkdir(exist_ok=True)
    engine=WorkflowEngine();records=[]
    # These are declared sensitivity tests, not measured uncertainties.
    for divisor in (1,100,1000):
        options=validate_options({**request['options'],'absolute_noise':1.0})
        sides=prepare_sides(request['q'],request['intensity'],None,options)
        for item in sides:
            item['normalizer']/=divisor
            start=time.perf_counter()
            result,artifact,raw=engine.fit_side(item,options,lambda *a:None,lambda:False)
            dest=out/f'floor1_norm_div{divisor}'/item['side']
            rows=write_side(dest,item,result,artifact,raw)
            mask=raw['mask'][0]>0;y=item['observed'];positive=y>0
            pred=raw['curves'][0][:,mask]*item['normalizer']
            errors=np.sqrt(np.mean(np.log(np.maximum(pred[:,positive],1e-30)/y[positive])**2,axis=1))
            record=dict(divisor=divisor,side=item['side'],normalizer=item['normalizer'],seconds=time.perf_counter()-start,rank1_log=rows[0]['best_log_rmse'],best_any_log=float(errors.min()),best_any_stage=str(raw['candidate_stage'][errors.argmin()]),rank1_combo=rows[0]['combination'])
            records.append(record);print(json.dumps(record),flush=True)
            (out/'noise_scale_results.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
