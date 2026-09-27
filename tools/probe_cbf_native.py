"""Diagnostic: can the same particle form family fit CBF with free instrument amplitudes?"""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,nnls
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'modules/Fitting_1D_Model/Workflow_v5/conditional_v2'))
from numpy_forward64 import component
from matplotlib.figure import Figure

def main():
 v=json.loads((ROOT/'validation/center_symmetry/VERIFIED.json').read_text());base=ROOT/'validation/usability_20260921/native_counts';out=ROOT/'validation/usability_20260921/native_probe';out.mkdir(exist_ok=True);records=[];fig=Figure(figsize=(12,4),tight_layout=True)
 for i,side in enumerate(('positive','negative'),1):
  a=dict(np.load(base/side/'display.npz'));q=a['reference_q'];y=a['observed'];sg=np.hypot(.1*abs(y),1);pos=y>0;roi=(q>.15)&(q<2)&(y>20);best=None
  for typ in (1,3):
   def model(z):
    R,sr,eta,sd,rs,nu=z;R=np.exp(R);D=np.exp(np.log(max(3,2*R*1.001))+eta*(np.log(500)-np.log(max(3,2*R*1.001))))
    part=component(q,typ,R,sr,10,.2,D,sd,True);shape=1/(1+(q/rs)**nu);basis=np.stack([part,np.ones_like(q),shape],1);des=basis/sg[:,None];sc=np.maximum(np.linalg.norm(des,axis=0),1e-30);co=nnls(des/sc,y/sg)[0]/sc
    return basis@co,co,dict(type=typ,R=R,sigma_R=sr,D=D,sigma_D=sd,sigma_res=rs,nu_res=nu)
   z=[np.log(1.5),.2,(np.log(4.5)-np.log(3.003))/(np.log(500)-np.log(3.003)),.12,.02,2.5]
   start=time.perf_counter();opt=least_squares(lambda z:(model(z)[0]-y)/sg,z,bounds=([0,.02,0,.05,.001,1],[np.log(100),.9,1,.9,.1,20]),loss='soft_l1',max_nfev=150,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7)
   pred,co,params=model(opt.x);err=float(np.sqrt(np.mean(np.log(np.maximum(pred[pos],1e-30)/y[pos])**2)));peak=float(np.sqrt(np.mean(np.log(pred[roi]/y[roi])**2)))
   row=dict(side=side,params=params,amplitudes=co.tolist(),logrmse=err,peak_logrmse=peak,seconds=time.perf_counter()-start,nfev=opt.nfev);records.append(row);print(json.dumps(row),flush=True)
   if best is None or peak<best[0]:best=(peak,pred)
  ax=fig.add_subplot(1,2,i);ax.plot(q[pos],y[pos],'.',ms=3,label='measured');ax.plot(q,best[1],label='diagnostic physical fit');ax.set_yscale('log');ax.set_ylim(.5,max(y)*2);ax.set_title(side);ax.legend()
  (out/'physical_probe.json').write_text(json.dumps(records,indent=2))
 fig.savefig(out/'physical_probe.png',dpi=140)
if __name__=='__main__':main()
