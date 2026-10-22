"""Required-input-only GISAXS API. Unknown optional information stays unknown.

 analyze_frame(path, geometry_dict, alpha_i_deg=.4, metal='Ag')
 .poni paths use pyFAI rays, including detector orientation/rotations.
 Manual beam centres are zero-based pixel-centre indices, row downwards.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time,json,hashlib
import numpy as np
from .features import AF_BANDS,Q_EDGES,V_EDGES,fit_peak,ridge

@dataclass
class FrameResult:
    row: dict
    profiles: dict

STRUCTURE_FIELDS=('D_nm','diameter_nm','height_nm',
    'D_lo68_nm','D_hi68_nm','D_lo95_nm','D_hi95_nm',
    'diameter_lo68_nm','diameter_hi68_nm','diameter_lo95_nm','diameter_hi95_nm',
    'height_lo68_nm','height_hi68_nm','height_lo95_nm','height_hi95_nm',
    'size_sigma_log','size_sigma_log_lo68','size_sigma_log_hi68','size_sigma_log_lo95','size_sigma_log_hi95')

def calibrated_quantities(receipt):
    """Calibration is independent per quantity; height is interval-only."""
    good=set()
    if not isinstance(receipt,dict) or receipt.get('schema') not in ('v1.1_quantity','v1_aggregate'):return good
    if receipt.get('numerical_gates_pass') is not True:return good
    try:
        if receipt.get('required_inputs_only_pass') is not True or not 0<=float(receipt['max_cpu_seconds'])<=10:return good
    except (KeyError,ValueError,TypeError):return good
    calibrations=receipt.get('quantity_calibration',{})
    if not isinstance(calibrations,dict):return good
    for name,c in calibrations.items():
        try:
            if name not in ('D','diameter','height','size_sigma_log'):continue
            cov=[float(c['coverage68_unknown']),float(c['coverage95_unknown'])]
            if not np.isfinite(cov).all() or abs(cov[0]-.68)>.05+1e-12 or abs(cov[1]-.95)>.05+1e-12:continue
            if name in ('D','diameter'):
                r2=float(c['R2_unknown_nm'])
                if not np.isfinite(r2) or r2<.9:continue
            good.add(name)
        except (KeyError,ValueError,TypeError):continue
    return good

def bind_quantity_calibration(row,allowed):
    """Missing adapter gates withhold that quantity; no legacy global fallback."""
    supplied=row.get('quantity_gates',{})
    if not isinstance(supplied,dict):supplied={}
    gates={}
    for name in ('D','diameter','height','size_sigma_log'):
        g=supplied.get(name,{})
        gates[name]=dict(g) if isinstance(g,dict) else {}
        gates[name]['calibrated']=name in allowed
        if name not in allowed:gates[name]['reason']='simulation_calibration_failed'
    row['quantity_gates']=gates
    return row

def release_passes(receipt):
    """Current v1 release requires every declared aggregate acceptance gate.

    Historical v1.1 per-quantity calibration remains readable for research,
    but cannot activate the public v1 model hook.
    """
    if not isinstance(receipt,dict) or receipt.get('schema')!='v1_aggregate':return False
    if not {'D','diameter','height'}.issubset(calibrated_quantities(receipt)):return False
    for key in ('PPC_supported_windows','failure_windows','layer_comparison_windows'):
        count=receipt.get(key)
        if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or count<=0:return False
    for key,limit in (('PPC_supported_pass_rate',.70),('failure_unreliable_recall',.80),('known_unknown_agreement',.80)):
        try:
            value=float(receipt[key])
            if not np.isfinite(value) or not limit<=value<=1:return False
        except (KeyError,TypeError,ValueError):return False
    return True

def input_scope_reason(energy,alpha_i,metal,thickness,scope):
    """Known input domain facts are checked independently of network OOD."""
    if str(metal).strip().lower() not in ('au','ag'):return 'unsupported_metal_Au_Ag_only'
    try:
        for value,key in ((energy,'energy_ev'),(alpha_i,'alpha_i_deg')):
            lo,hi=map(float,scope[key]);value=float(value)
            if not np.isfinite(value) or not lo<=value<=hi:return key+'_outside_trained_range'
        if thickness is not None:
            value=float(thickness);lo,hi=map(float,scope['interlayer_nm'])
            if not np.isfinite(value) or not (lo<=value<=hi or value==0 and scope.get('no_interlayer_supported') is True):return 'provided_interlayer_thickness_outside_trained_range'
    except (KeyError,ValueError,TypeError):return 'model_scope_missing_or_invalid'
    return None

def enforce_reliability(row,*,acceptance_receipt=None,research_quantity_mode=False):
    """Public v1 is all-or-features; legacy research gates require opt-in."""
    # Keep the required output columns. No authentic four-stage model is
    # currently calibrated; shape class must never be substituted for stage.
    row['growth_stage']=None;row['stage_probabilities']=None
    row['growth_stage_status']='withheld';row['growth_stage_reason']='no_validated_connectivity_stage_model'
    if not research_quantity_mode and not release_passes(acceptance_receipt):
        for key in STRUCTURE_FIELDS:row[key]=None
        for quantity in ('D','diameter','height','size_sigma_log'):
            row[quantity+'_status']='withheld';row[quantity+'_reason']='v1_aggregate_not_validated'
        if row.get('reliability')!='out_of_domain':row['reliability']='unreliable'
        return row
    gates=row.get('quantity_gates')
    if isinstance(gates,dict):
        gates={name:dict(gates.get(name,{})) if isinstance(gates.get(name,{}),dict) else {} for name in ('D','diameter','height','size_sigma_log')};row['quantity_gates']=gates
        scope=row.get('reliability')=='out_of_domain' or row.get('OOD_flag') is True
        d=gates.get('D',{})
        def within(v,lim):
            try:return np.isfinite(float(v)) and 0<=float(v)<=lim
            except (TypeError,ValueError):return False
        featuregate=(not scope and row.get('feature_status')=='bilateral' and within(d.get('q_relative_error'),.05) and within(d.get('variant_D_relative_spread'),.15))
        okD=featuregate and d.get('calibrated') is True
        dd=gates.get('diameter',{});okdiam=(featuregate and dd.get('calibrated') is True and within(dd.get('width_relative_error'),.10) and within(dd.get('highq_ratio_relative_error'),.15))
        ss=gates.get('size_sigma_log',{});okspread=(featuregate and ss.get('calibrated') is True and within(ss.get('width_relative_error'),.10) and within(ss.get('highq_ratio_relative_error'),.15))
        okheight=(featuregate and gates.get('height',{}).get('calibrated') is True)
        flags={'D':okD,'diameter':okdiam,'height':okheight,'size_sigma_log':okspread}
        for quantity,passed in flags.items():
            if not passed:continue
            suffix='' if quantity=='size_sigma_log' else '_nm'
            try:
                bounds=[float(row[quantity+'_lo95'+suffix]),float(row[quantity+'_lo68'+suffix]),float(row[quantity+'_hi68'+suffix]),float(row[quantity+'_hi95'+suffix])]
                valid=np.isfinite(bounds).all() and min(bounds)>0 and np.all(np.diff(bounds)>=0)
                if quantity!='height':
                    point=float(row[quantity+suffix]);valid=valid and np.isfinite(point) and bounds[1]<=point<=bounds[2]
            except (KeyError,TypeError,ValueError):valid=False
            if not valid:flags[quantity]=False;gates[quantity]['reason']='invalid_or_missing_interval_payload'
        for quantity,passed in flags.items():
            row[quantity+'_status']='released' if passed else 'withheld'
            row[quantity+'_reason']='passed_quantity_gates' if passed else gates.get(quantity,{}).get('reason') or ('OOD_or_out_of_domain' if scope else 'no_visible_bilateral_feature' if row.get('feature_status')!='bilateral' else 'simulation_calibration_failed' if gates.get(quantity,{}).get('calibrated') is not True else 'posterior_feature_or_variant_gate_failed')
            if not passed:
                for key in STRUCTURE_FIELDS:
                    if key==quantity+'_nm' or key.startswith(quantity+'_lo') or key.startswith(quantity+'_hi') or quantity=='size_sigma_log' and key.startswith(quantity):row[key]=None
        row['height_nm']=None
        if not scope:row['reliability']='partially_reliable' if any(flags.values()) else 'unreliable'
        return row
    for key in STRUCTURE_FIELDS:row[key]=None
    row['height_nm']=None
    return row

def validated_model(model):
    """Require bound validation and weight artifacts before a public hook runs."""
    receipt=getattr(model,'acceptance_receipt',None)
    if not release_passes(receipt):return False
    try:
        scope=receipt['scope']
        for key,limits in (('energy_ev',(8000,14000)),('alpha_i_deg',(.15,.7)),('interlayer_nm',(.5,100))):
            lo,hi=map(float,scope[key])
            if not np.isfinite([lo,hi]).all() or not limits[0]<=lo<hi<=limits[1]:return False
        weights=Path(model.weights_path);validation=Path(model.validation_path)
        raw=validation.read_bytes();doc=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=receipt['validation_sha256']:return False
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=receipt['weights_sha256']:return False
        checked=doc['acceptance_receipt']
        return (release_passes(checked) and checked.get('weights_sha256')==receipt['weights_sha256']
                and all(checked.get(k)==v for k,v in receipt.items() if k!='validation_sha256'))
    except (OSError,AttributeError,KeyError,TypeError,ValueError):return False

def summarize_profiles(Y,N,VY,VN,peak_window=None):
    """Shared descriptor rule for one frame or exact summed window profiles."""
    q=(Q_EDGES[:-1]+Q_EDGES[1:])/2
    kwargs={} if peak_window is None else {'lo':peak_window[0],'hi':peak_window[1]}
    fits=[fit_peak(q,y,n,**kwargs) for y,n in zip(Y,N)];bands=[]
    for band in range(4):
        pair=fits[2*band:2*band+2];good=[f for f in pair if f['status']=='resolved_empirical_feature'];qs=[f['q_star'] for f in good]
        agree=len(good)==2 and np.ptp(qs)/np.mean(qs)<=.10
        bands.append({'AF_band':band,'af_range_deg':list(AF_BANDS[band]),'support':'bilateral' if agree else 'bilateral_conflict' if len(good)==2 else 'one_side' if len(good)==1 else 'unresolved','q_star':float(np.mean(qs)) if agree or len(good)==1 else None,'FWHM':float(np.mean([f['fwhm'] for f in good])) if agree or len(good)==1 else None,'fit_se_q':float(max(f['q_star_fit_se'] for f in good)) if agree or len(good)==1 else None,'candidate_scores':[f.get('delta_AIC') for f in pair]})
    supported=[b for b in bands if b['support']=='bilateral'];supported=supported or [b for b in bands if b['support']=='one_side']
    return fits,bands,supported[0] if supported else None,ridge(VY,VN)

def read_image(path,frame_index=0):
    path=Path(path);ext=path.suffix.lower()
    if ext=='.cbf':
        from .cbfio import read_cbf
        return read_cbf(path)
    if ext not in ('.tif','.tiff','.edf'):raise ValueError('Supported 2D image formats: CBF, TIFF, EDF')
    import fabio
    obj=fabio.open(str(path));obj=obj.getframe(frame_index) if frame_index else obj
    image=np.asarray(obj.data,dtype=float);image[(image<0)|(~np.isfinite(image))]=np.nan
    hdr={}
    for key in ('Exposure_time','exposure_s','ExposureTime','Exposure'):
        if key in obj.header:
            try:hdr['exposure_s']=float(str(obj.header[key]).split()[0])
            except ValueError:pass
    return image,hdr

class FeatureMap:
    def __init__(self,geometry,shape,alpha_i_deg,sample_roll_deg=0.,crop=None,exclude_rings=()):
        self.shape=tuple(shape);self.alpha_i_deg=float(alpha_i_deg);self.exclude_rings=exclude_rings
        if len(shape)!=2:raise ValueError('A frame must be two dimensional')
        if crop is None:crop=(0,shape[0],0,shape[1])
        r0,r1,c0,c1=map(int,crop)
        if not (0<=r0<r1<=shape[0] and 0<=c0<c1<=shape[1]):raise ValueError('Crop exceeds image bounds')
        self.crop=tuple(crop);rr,cc=np.mgrid[r0:r1,c0:c1]
        if isinstance(geometry,(str,Path)):
            import pyFAI
            g=pyFAI.load(str(geometry));p3,p1,p2=g.calc_pos_zyx(d1=rr.astype(float),d2=cc.astype(float))
            # pyFAI p3 is beam, p1 follows downwards detector row, p2 column.
            u=np.stack([p3,p2,-p1]);wl=float(g.wavelength)*1e9
            if not np.isfinite(wl) or wl<=0:raise ValueError('.poni must provide wavelength')
            self.geometry_meta={'source':'pyFAI.poni','energy_ev':1239.841984/wl,'sdd_mm':float(g.dist)*1000,'rotations_rad':[float(g.rot1),float(g.rot2),float(g.rot3)],'pixel_mm':[float(g.detector.pixel1)*1000,float(g.detector.pixel2)*1000]}
        else:
            g=dict(geometry);energy=float(g['energy_ev']);L=float(g['sdd_mm']);pixels=g['pixel_mm'];pr,pc=(float(pixels),float(pixels)) if np.isscalar(pixels) else map(float,pixels)
            if min(energy,L,pr,pc)<=0:raise ValueError('Energy/SDD/pixel dimensions must be positive')
            wl=1239.841984/energy;u=np.stack([np.full(rr.shape,L),(cc-float(g['db_col']))*pc,(float(g['db_row'])-rr)*pr]);self.geometry_meta={**g,'source':'manual','pixel_mm':[pr,pc]}
            sample_roll_deg=float(g.get('sample_roll_deg',sample_roll_deg))
        u/=np.linalg.norm(u,axis=0);ai=np.radians(alpha_i_deg);roll=np.radians(sample_roll_deg)
        ns=np.array([-np.sin(ai),np.cos(ai)*np.sin(roll),np.cos(ai)*np.cos(roll)]);xs=np.array([np.cos(ai),np.sin(ai)*np.sin(roll),np.sin(ai)*np.cos(roll)]);ys=np.cross(ns,xs);k=2*np.pi/wl
        af=np.degrees(np.arcsin(np.clip(np.einsum('i,ijk->jk',ns,u),-1,1)));qvec=k*(u-np.array([1.,0.,0.])[:,None,None]);qy=np.einsum('i,ijk->jk',ys,qvec);qt=np.linalg.norm(qvec,axis=0);q=abs(qy)
        ok=(q>.05)
        for centre,half in exclude_rings:ok&=(abs(qt-centre)>half)
        qi=np.searchsorted(Q_EDGES,q,side='right')-1;nq=len(Q_EDGES)-1;self.h=[]
        for lo,hi in AF_BANDS:
            for positive in (False,True):
                keep=ok&(af>=lo)&(af<hi)&((qy>0)==positive)&(qi>=0)&(qi<nq)
                self.h.append((np.ravel_multi_index((rr[keep],cc[keep]),shape),qi[keep]))
        vi=np.searchsorted(V_EDGES,af,side='right')-1;keep=ok&(q>=.15)&(q<1.2)&(vi>=0)&(vi<len(V_EDGES)-1)
        self.v=np.ravel_multi_index((rr[keep],cc[keep]),shape),vi[keep]
        self.geometry_meta.update(alpha_i_deg=float(alpha_i_deg),sample_roll_deg=float(sample_roll_deg),shape=list(shape),crop=list(crop),exclude_rings_nm_inv=list(exclude_rings))
        # Canonical model tensor mapping is separate from empirical features;
        # the original .025 q / .01-degree feature descriptors are unchanged.
        from .tool_qinput import Q_EDGES as QE,AF_EDGES as AE,GRID_SHAPE
        qi2=np.searchsorted(QE,q,side='right')-1;ai2=np.searchsorted(AE,af,side='right')-1;n,m=GRID_SHAPE;sel=(qi2>=0)&(qi2<m)&(ai2>=0)&(ai2<n)
        self.tensor_indices=np.ravel_multi_index((rr[sel],cc[sel]),shape);self.tensor_bins=(ai2[sel]*m+qi2[sel]).astype(np.int32)

    def project_tensor(self,image,exposure=1.,mask=None):
        from .tool_qinput import GRID_SHAPE,tensor
        im=np.asarray(image,float)
        if im.shape!=self.shape:raise ValueError('Image differs from cached geometry')
        valid=np.isfinite(im)&(im>=0)
        if mask is not None:valid&=np.asarray(mask,bool)
        ix=self.tensor_indices;good=valid.ravel()[ix];bins=self.tensor_bins[good];Y=np.bincount(bins,weights=im.ravel()[ix[good]],minlength=np.prod(GRID_SHAPE)).reshape(GRID_SHAPE);N=np.bincount(bins,minlength=np.prod(GRID_SHAPE)).reshape(GRID_SHAPE)*exposure
        rate=np.divide(Y,N,out=np.zeros_like(Y),where=N>0);t,s=tensor(rate,N>0)
        return {'tensor':t,'Y':Y,'N':N,'scale':s}

    def extract(self,image,exposure,mask=None):
        image=np.asarray(image,float)
        if image.shape!=self.shape:raise ValueError('Image shape differs from geometry map')
        valid=np.isfinite(image)&(image>=0)
        if mask is not None:
            if np.shape(mask)!=self.shape:raise ValueError('Explicit mask shape differs from image')
            valid&=np.asarray(mask,bool)
        flat=image.ravel();valid=valid.ravel();Y=[];N=[]
        for idx,bin in self.h:
            good=valid[idx];Y.append(np.bincount(bin[good],weights=flat[idx[good]],minlength=len(Q_EDGES)-1));N.append(np.bincount(bin[good],minlength=len(Q_EDGES)-1)*exposure)
        idx,bin=self.v;good=valid[idx];VY=np.bincount(bin[good],weights=flat[idx[good]],minlength=len(V_EDGES)-1);VN=np.bincount(bin[good],minlength=len(V_EDGES)-1)*exposure
        return np.array(Y),np.array(N),VY,VN

def analyze_frame(image,geometry,alpha_i_deg,metal,*,layers=None,interlayer_thickness_nm=None,preframe=None,qcm_rate=None,exposure_s=None,pre_exposure_s=None,mask=None,sample_roll_deg=0.,crop=None,exclude_rings=(),feature_map=None,frame_index=0,model=None):
    start=time.perf_counter();cpu_start=time.process_time();hdr={};source='<array>'
    if isinstance(image,(str,Path)):source=str(image);image,hdr=read_image(image,frame_index)
    image=np.asarray(image,float)
    if exposure_s is None:exposure_s=hdr.get('exposure_s')
    if exposure_s is not None and (not np.isfinite(exposure_s) or exposure_s<=0):exposure_s=None
    fm=feature_map or FeatureMap(geometry,image.shape,alpha_i_deg,sample_roll_deg,crop,exclude_rings);effective=1. if exposure_s is None else float(exposure_s)
    if not np.isclose(fm.alpha_i_deg,alpha_i_deg,atol=1e-12,rtol=0):raise ValueError('Cached feature map uses a different incidence angle')
    Y,N,VY,VN=fm.extract(image,effective,mask);q=(Q_EDGES[:-1]+Q_EDGES[1:])/2
    from .resolvable import PEAK_WINDOW,coverage,in_range,visible_for_regression
    fits,bands,selected,vr=summarize_profiles(Y,N,VY,VN,PEAK_WINDOW)
    qm=selected['q_star'] if selected else None;fw=selected['FWHM'] if selected else None
    row={'image':source,'frame_index':frame_index,'metal':str(metal),'mode':'features_only','D_corr_nm':float(2*np.pi/qm) if qm else None,'width_over_qstar':fw/qm if qm else None,'q_star_nm_inv':qm,'FWHM_nm_inv':fw,'Yoneda_deg':vr.get('alpha_f_observed_ridge'),'feature_status':selected['support'] if selected else 'unresolved','feature_AF_band':selected['AF_band']+1 if selected else None,'Yoneda_status':vr['ridge_status'],'exposure_s':exposure_s,'intensity_unit':'counts_per_valid_pixel' if exposure_s is None else 'counts_per_valid_pixel_second','D_nm':None,'diameter_nm':None,'height_nm':None,'D_lo68_nm':None,'D_hi68_nm':None,'D_lo95_nm':None,'D_hi95_nm':None,'diameter_lo68_nm':None,'diameter_hi68_nm':None,'diameter_lo95_nm':None,'diameter_hi95_nm':None,'height_lo68_nm':None,'height_hi68_nm':None,'height_lo95_nm':None,'height_hi95_nm':None,'growth_stage':None,'stage_probabilities':None,'reliability':'unreliable','reason':'model_not_available','OOD_score':None,'optional_layers_known':layers is not None,'optional_thickness_known':interlayer_thickness_nm is not None,'optional_preframe_known':preframe is not None,'optional_qcm_known':qcm_rate is not None,'cpu_seconds':None,'geometry':fm.geometry_meta,'bands':bands}
    # Conditional fit SE is distinct from model validation and systematics.
    sq=selected['fit_se_q'] if selected else None
    row.update(q_star_fit_se_nm_inv=sq,
               D_corr_conditional_fit_se_nm=2*np.pi*sq/qm**2 if qm and sq is not None else None,
               feature_uncertainty='Residual-scaled fit covariance; excludes geometry/background/model systematics',
               Yoneda_resolution_deg=vr.get('ridge_resolution_deg'))
    for quantity in ('D','diameter','height','size_sigma_log'):
        row.update({quantity+'_status':'withheld',quantity+'_reason':'model_not_available' if selected and selected['support']=='bilateral' else 'no_visible_bilateral_feature'})
    cov=coverage(N)
    row.update(D_search_min_nm=cov['D_search_min_nm'],D_search_max_nm=cov['D_search_max_nm'],
               D_search_intervals_nm=cov['D_corr_intervals_nm'],D_search_definition=cov['definition'],
               visible_for_D_regression=visible_for_regression(selected,cov),
               feature_in_search_range=bool(in_range(row['D_corr_nm'],cov['bands'][selected['AF_band']]['D_corr_intervals_nm'])) if selected and row['D_corr_nm'] is not None else None)
    profiles={'q':q,'Y':Y,'N':N,'VY':VY,'VN':VN,'fits':fits,'ridge':vr,'coverage':cov,'image_thumbnail':image[::max(1,image.shape[0]//256),::max(1,image.shape[1]//256)]}
    from .reference_changes import reference_changes,no_peak_reason
    row.update(no_peak_reason=no_peak_reason(fits) if not selected else 'only_one_side_feature' if selected['support']!='bilateral' else None,
               reference_change_status='not_provided',reference_highq_ratio_range=None,reference_lowq_excess_range=None,
               reference_length_range_nm=None,reference_view_summaries=None)
    if not selected:row['mode']='no_peak'
    model_reference=None
    if preframe is not None:
        try:
            if isinstance(preframe,(str,Path)):ref,rhdr=read_image(preframe)
            else:ref=np.asarray(preframe,float);rhdr={}
            if ref.shape!=image.shape:raise ValueError('Reference image shape differs; common geometry not possible')
            rexp=pre_exposure_s if pre_exposure_s is not None else rhdr.get('exposure_s')
            if rexp is not None and (not np.isfinite(rexp) or rexp<=0):rexp=None
            if (rexp is None)!=(exposure_s is None):raise ValueError('Current/reference exposure units differ; supply missing exposure')
            common=np.isfinite(image)&(image>=0)&np.isfinite(ref)&(ref>=0)
            if mask is not None:common&=np.asarray(mask,bool)
            cy,cn,_,_=fm.extract(image,effective,common);ry,rn,_,_=fm.extract(ref,1. if rexp is None else float(rexp),common)
            change=reference_changes(q,cy,cn,ry,rn);profiles['reference_change']=change
            model_reference=(ref,1. if rexp is None else float(rexp),common)
            row.update(reference_change_status='exposure_normalized_observed_change' if rexp is not None else 'counts_ratio_unknown_relative_exposure',
                       reference_highq_ratio_range=change['highq_ratio_view_range'],reference_length_range_nm=change['length_view_range_nm'],
                       reference_lowq_excess_range=[min(s['lowq_signed_excess_integral'] for s in change['summaries'] if s['lowq_signed_excess_integral'] is not None),max(s['lowq_signed_excess_integral'] for s in change['summaries'] if s['lowq_signed_excess_integral'] is not None)] if any(s['lowq_signed_excess_integral'] is not None for s in change['summaries']) else None,
                       reference_view_summaries=change['summaries'])
        except (OSError,ValueError,TypeError) as exc:row.update(reference_change_status='incomparable_reference',reference_change_reason=str(exc))
    row['metal']=str(metal)
    if str(metal).strip().lower() not in ('au','ag'):row.update(reliability='out_of_domain',reason='unsupported_metal_Au_Ag_only')
    if model is not None:
        if validated_model(model):
            scope=model.acceptance_receipt['scope'];why=input_scope_reason(fm.geometry_meta['energy_ev'],alpha_i_deg,metal,interlayer_thickness_nm,scope)
            row['model_scope']=scope;row['model_domain_note']='Prediction conditional on model assumptions; unknown layers are not verified in domain'
            if why:row.update(reliability='out_of_domain',reason=why)
            else:
                observed={k:row[k] for k in ('D_corr_nm','width_over_qstar','q_star_nm_inv','FWHM_nm_inv','Yoneda_deg','feature_status','feature_AF_band','Yoneda_status')}
                canonical=fm.project_tensor(image,effective,mask);pre_tensor=None
                if model_reference is not None:
                    ref,rexp,common=model_reference;rc=fm.project_tensor(ref,rexp,common)
                    rate=np.divide(rc['Y'],rc['N'],out=np.zeros_like(rc['Y']),where=rc['N']>0)
                    pre_tensor=np.stack([np.log1p(rate/canonical['scale'])*(rc['N']>0),(rc['N']>0).astype(float)]).astype(np.float32)
                profiles['model_input']={'image':canonical['tensor'],'pre':pre_tensor,'scale':canonical['scale'],
                    'mandatory':np.array([(fm.geometry_meta['energy_ev']-14000)/6000,(alpha_i_deg-.425)/.275,float(str(metal).strip().lower()=='au')],np.float32),
                    'reference_known':pre_tensor is not None}
                model.apply(row,profiles,layers=layers,thickness=interlayer_thickness_nm,preframe=preframe,qcm_rate=qcm_rate)
                row.update(observed)
                allowed=calibrated_quantities(model.acceptance_receipt);bind_quantity_calibration(row,allowed)
        else:row['reason']='model_not_validated'
    # Model hooks cannot override domain facts supplied by the caller.
    if str(metal).strip().lower() not in ('au','ag'):row.update(reliability='out_of_domain',reason='unsupported_metal_Au_Ag_only')
    enforce_reliability(row,acceptance_receipt=model.acceptance_receipt if model is not None and validated_model(model) else None)
    row['cpu_seconds']=time.process_time()-cpu_start;row['wall_seconds']=time.perf_counter()-start
    return FrameResult(row,profiles)
