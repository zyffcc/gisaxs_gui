'Readonly CBF intake and conditional automatic centre for polymer_dev_v0.'
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json,hashlib,re,time
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter,label
from .cbfio import read_cbf_bytes
from .stage5_input import AtomicMap
from .neural_refine_report import read,write,sha


READERS=int(__import__('os').environ.get('IAM_READERS','0'))
RAW_CACHE_N=0
_READ_GATE=None
_RAW_CACHE={}
READ_STATS=dict(seconds=0.,bytes=0,files=0)


def _init_reader(gate):
    global _READ_GATE,RAW_CACHE_N
    _READ_GATE=gate;RAW_CACHE_N=0


def _read_raw(path):
    key=str(path)
    if key in _RAW_CACHE:return _RAW_CACHE[key]
    t0=time.perf_counter()
    if _READ_GATE is not None:
        with _READ_GATE:raw=Path(path).read_bytes()
    else:raw=Path(path).read_bytes()
    READ_STATS['seconds']+=time.perf_counter()-t0;READ_STATS['bytes']+=len(raw);READ_STATS['files']+=1
    if RAW_CACHE_N>0:
        _RAW_CACHE[key]=raw
        while len(_RAW_CACHE)>RAW_CACHE_N:_RAW_CACHE.pop(next(iter(_RAW_CACHE)))
    return raw


def frames(path):
    path=Path(path)
    files=[path] if path.is_file() else sorted(path.glob('*.cbf'))
    if not files:raise ValueError('No CBF frames in supplied path')
    return files


class DamagedFrame(ValueError):
    'A frame file that cannot be decoded - truncated or damaged on disk (2026-11-20: one truncated CBF of a scan'


def load_frame(path):

    for attempt in range(6):
        try:
            return _load_frame_once(path)
        except PermissionError:
            if attempt==5:raise
            time.sleep(1.+2.*attempt)


def _load_frame_once(path):
    raw=_read_raw(path);header=raw[:16384].decode('latin1')
    cutoff=re.search(r'Count_cutoff\s+([\d.eE+-]+)',header)
    try:image,meta=read_cbf_bytes(raw,count_limit=float('inf'))
    except (ValueError,IndexError,KeyError,EOFError) as error:
        raise DamagedFrame(f'{Path(path).name}: {type(error).__name__}: {error}'[:300]) from error
    limit=float(cutoff.group(1)) if cutoff else None
    removed=int(np.sum(image>=limit)) if limit is not None else 0
    if limit is not None:image[image>=limit]=np.nan
    if not meta.get('exposure_s',0)>0:raise ValueError('Positive header exposure required')
    meta.update(count_cutoff=limit,cutoff_removed_pixels=removed,header_sha256=hashlib.sha256(header.encode('latin1')).hexdigest(),
                raw_sha256=hashlib.sha256(raw).hexdigest())
    return image,meta


def auto_center(files,sdd_mm,alpha_i_deg,pixel_mm=.172):
    ids=np.unique(np.linspace(0,len(files)-1,min(9,len(files)),dtype=int));points=[];candidates=[]
    for i in ids:
        try:im,meta=load_frame(files[i])
        except DamagedFrame:continue
        ny,nx=im.shape
        sm=gaussian_filter(np.where(np.isfinite(im),im,0.),1.)


        region=np.zeros(im.shape,bool);region[int(.15*ny):int(.9*ny),int(.2*nx):int(.8*nx)]=True
        peak=(sm==maximum_filter(sm,size=31))&region
        yy,xx=np.nonzero(peak);order=np.argsort(sm[yy,xx])[::-1][:8]
        cand=[dict(row=int(yy[k]),col=int(xx[k]),smoothed_counts=float(sm[yy[k],xx[k]])) for k in order]
        if not cand or cand[0]['smoothed_counts']<=0:raise ValueError('No supported specular-centre candidate; supply beam centre')
        c=cand[0];row,col=c['row'],c['col'];r0=max(0,row-5);c0=max(0,col-5)
        sub=np.maximum(np.nan_to_num(im[r0:row+6,c0:col+6],nan=0.)-np.nanmedian(im[r0:row+6,c0:col+6]),0.)
        ry,cx=np.indices(sub.shape)
        if sub.sum()>0:row=float((sub*(ry+r0)).sum()/sub.sum());col=float((sub*(cx+c0)).sum()/sub.sum())
        points.append([col,row]);candidates.append(dict(frame=files[i].name,peaks=cand))
    if not points:raise ValueError('No readable frame for the automatic centre; supply beam centre')
    points=np.asarray(points);centre=np.median(points,axis=0);mad=np.median(abs(points-centre),axis=0)
    col=float(centre[0]);separation=sdd_mm*np.tan(2*np.radians(alpha_i_deg))/pixel_mm


    profiles=[]
    for i in ids:
        try:im,_=load_frame(files[int(i)])
        except DamagedFrame:continue
        strip=im[:,max(0,int(round(col))-3):int(round(col))+4]
        valid=np.isfinite(strip).sum(axis=1)>strip.shape[1]/2
        sums=np.nansum(strip,axis=1);count=np.isfinite(strip).sum(axis=1)
        prof=np.divide(sums,count,out=np.full(len(sums),np.nan),where=valid)
        if np.nanmax(prof)<20:continue
        value=np.log1p(prof);value-=np.nanquantile(value,.95);profiles.append(value)
    if not profiles:raise ValueError('No illuminated frames for automatic beamstop geometry')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);profile=np.nanmedian(profiles,axis=0)
    valid=np.isfinite(profile);filled=np.interp(np.arange(len(profile)),np.flatnonzero(valid),profile[valid])
    contrast=maximum_filter(filled,size=101)-filled
    holes=(contrast>2.)&valid;holes[:int(.35*len(holes))]=False;holes[int(.9*len(holes)):]=False
    labels,n=label(holes);intervals=[]
    for group in range(1,n+1):
        positions=np.flatnonzero(labels==group)
        if 8<=len(positions)<=90:intervals.append([int(positions[0]),int(positions[-1])])
    pairs=[(abs((b[0]+b[1]-a[0]-a[1])/2-separation),a,b) for a in intervals for b in intervals if b[0]>a[1]]
    if not pairs or min(pairs)[0]>20:raise ValueError('No two stable beamstop shadows match2alpha_i; supply centre')
    discrepancy,upper,lower=min(pairs);row=float(np.mean(upper));lower_row=float(np.mean(lower))
    dbrow=float((lower_row+row+separation)/2)


    for i in sorted(ids,key=lambda i:abs(int(i)-int(ids[len(ids)//2]))):
        try:image,_=load_frame(files[int(i)]);break
        except DamagedFrame:continue
    else:raise ValueError('No readable frame for the automatic centre; supply beam centre')
    ny,nx=image.shape
    ys=np.arange(max(0,int(row)-250),min(ny,int(row)+180),3);offset=np.arange(20,min(350,int(col)-2,nx-int(col)-2),3)
    left=image[np.ix_(ys,np.rint(col-offset).astype(int))];right=image[np.ix_(ys,np.rint(col+offset).astype(int))]
    ok=np.isfinite(left)&np.isfinite(right)
    symmetry=float(np.corrcoef(np.log1p(left[ok]),np.log1p(right[ok]))[0,1]) if ok.sum()>20 else 0.
    return dict(db_col=float(col),db_row=float(dbrow),sample_roll_deg=0.,
        centre_method='Persistent rod column, paired specular/direct beamstop shadows at supplied2alpha_i separation; horizontal Yoneda/diffuse symmetry crosscheck',
        specular_col=float(col),specular_row=float(row),frame_centres_col_row=points.tolist(),used_frame_indices=ids.tolist(),
        median_absolute_deviation_pixels=mad.tolist(),working_uncertainty_pixels=[max(2.,3*float(mad[0])),max(5.,float(discrepancy))],
        shadow_intervals=intervals,selected_specular_shadow=upper,selected_direct_shadow=lower,mirror_separation_discrepancy_pixels=float(discrepancy),
        log_intensity_left_right_correlation=symmetry,competing_maxima=candidates,
        geometry_status='Conditional beamstop/rod estimate; beamstop placement, detector tilt and incidence not independently calibrated',
        roll_status='0deg working assumption; not inherited',
        beamstop_status='Central absent/obscured direct beam not used as a measured bright point; negative/gap pixels excluded')


def extract_chunk(task):
    files,geometry,maskpath,out,index,comparison=task;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/f'chunk_{index:05d}.npz';meta=dest.with_suffix('.json')
    if meta.exists():
        record=read(meta)
        if dest.exists() and sha(dest)==record['array_sha256'] and record['files']==files and record['geometry']==geometry:return record
        if record['geometry']!=geometry:raise ValueError('Existing intake chunk has another geometry')
        meta.unlink();dest.unlink(missing_ok=True)
    stats0=dict(READ_STATS);start=time.perf_counter()
    skipped=[];first=None
    for i0,file in enumerate(files):
        try:first,h_first=load_frame(file);break
        except DamagedFrame as error:skipped.append(dict(frame=Path(file).name,reason=str(error)))
    if first is None:raise ValueError(f'No frame of intake chunk {index} can be read: '+'; '.join(s['reason'] for s in skipped))
    shape=first.shape


    crop=[max(0,int(round(geometry['db_row']-939.))),min(shape[0],int(round(geometry['db_row']+31.))),max(0,int(round(geometry['db_col']-556.3))),min(shape[1],int(round(geometry['db_col']+553.7)))]
    mapping=AtomicMap(geometry,shape,geometry['alpha_i_deg'],crop=crop,exclude_rings=((2.1,.13),))
    second=AtomicMap(dict(geometry,energy_ev=12000.),shape,geometry['alpha_i_deg'],crop=crop,exclude_rings=((2.1,.13),)) if comparison else None
    mask=np.load(maskpath).astype(bool) if maskpath else None
    ys=[];ps=[];alt_y=[];alt_p=[];rows=[]
    for i,file in enumerate(files):
        if i<i0:continue
        if i==i0:image,h=first.copy(),dict(h_first)
        else:
            try:image,h=load_frame(file)
            except DamagedFrame as error:skipped.append(dict(frame=Path(file).name,reason=str(error)));continue
        Y,E=mapping.project(image,h['exposure_s'],mask)
        P=E/h['exposure_s'];assert np.allclose(P,np.rint(P)) and np.max(P)<65535 and np.max(Y)<2**32
        ys.append(Y.astype(np.uint32));ps.append(np.rint(P).astype(np.uint16))
        if second:
            Y2,E2=second.project(image,h['exposure_s'],mask);alt_y.append(Y2.astype(np.uint32));alt_p.append(np.rint(E2/h['exposure_s']).astype(np.uint16))
        rows.append(dict(frame=Path(file).name,frame_id=int(Path(file).stem.rsplit('_',1)[-1]),timestamp=h.get('timestamp'),exposure_s=h['exposure_s'],
            count_cutoff=h['count_cutoff'],cutoff_removed_pixels=h['cutoff_removed_pixels'],header_sha256=h['header_sha256'],raw_sha256=h['raw_sha256'],
            selected_counts=int(Y.sum()),valid_pixel_count=int(P.sum())))
    arrays=dict(Y=np.asarray(ys),P=np.asarray(ps))
    if second:arrays.update(Y_12keV=np.asarray(alt_y),P_12keV=np.asarray(alt_p))
    np.savez_compressed(dest,**arrays)
    record=dict(files=files,geometry=geometry,crop=crop,mask=maskpath or 'Native finite/nonnegative/header-cutoff pixels; no transferred mask',
        rows=rows,skipped=skipped,array_file=dest.name,array_sha256=sha(dest),wall_seconds=time.perf_counter()-start,raw_saved=False,
        read_seconds=READ_STATS['seconds']-stats0['seconds'],read_bytes=READ_STATS['bytes']-stats0['bytes'])
    write(meta,record);print('INTAKE_CHUNK',out.name,index,len(files),round(record['wall_seconds'],1),flush=True);return record


def prepare(input_path,out,*,energy_ev,sdd_mm,alpha_i_deg,metal,beam_col=None,beam_row=None,sample_roll_deg=0.,
            max_frames=None,maskpath=None,workers=8,compare_12keV=False,system_note='Polymer',indices=None):
    'indices (2026-11-17): the frame numbers to read (sorted; default all, or max_frames spread over the scan). A'
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    out=Path(out);files=frames(input_path);all_count=len(files)
    if indices is not None:selected=np.unique(np.asarray(indices,int))
    else:selected=np.unique(np.linspace(0,len(files)-1,min(len(files),max_frames),dtype=int)) if max_frames else np.arange(len(files))
    blocks=selected//128
    files=[files[i] for i in selected]
    if (beam_col is None)!=(beam_row is None):raise ValueError('Provide both beam coordinates or neither')
    centre=auto_center(files,sdd_mm,alpha_i_deg) if beam_col is None else dict(db_col=float(beam_col),db_row=float(beam_row),sample_roll_deg=sample_roll_deg,
        geometry_status='User-supplied/shared-beamtime centre')
    geometry=dict(energy_ev=energy_ev,sdd_mm=sdd_mm,pixel_mm=.172,alpha_i_deg=alpha_i_deg,
        db_col=centre['db_col'],db_row=centre['db_row'],sample_roll_deg=centre['sample_roll_deg'])
    out.mkdir(parents=True,exist_ok=True);write(out/'geometry.json',centre|dict(geometry=geometry))
    tasks=[([str(p) for p,b in zip(files,blocks) if b==k],geometry,str(maskpath) if maskpath else None,str(out),int(k),compare_12keV) for k in np.unique(blocks)]
    for stale in out.glob('chunk_*.json'):
        if int(stale.stem.split('_')[1]) not in set(int(k) for k in np.unique(blocks)):
            stale.unlink();stale.with_suffix('.npz').unlink(missing_ok=True)
    records=[]
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_init_reader,initargs=(ctx.BoundedSemaphore(READERS) if READERS>0 else None,)) as pool:
        for future in as_completed([pool.submit(extract_chunk,t) for t in tasks]):records.append(future.result())
    records.sort(key=lambda r:r['array_file']);rows=[r for chunk in records for r in chunk['rows']]
    skipped=[x for chunk in records for x in chunk.get('skipped',[])]
    if len(rows)+len(skipped)!=len(files):raise ValueError('Frame denominator changed')
    if not rows:raise ValueError('No frame of the selection can be read')
    run=dict(status='completed',input_source=str(input_path),received_frames=all_count,selected_frames=len(files),selection_indices=selected.tolist(),
        skipped_frames=skipped,
        geometry=geometry,centre=centre,metal=metal,model_metal='Ag' if metal=='Cu' else metal,system_note=system_note,
        Cu_density_ratio=.89 if metal=='Cu' else 'Not applicable',Cu_scope='Cu以Ag模型近似' if metal=='Cu' else 'Native metal label',
        compare_12keV=compare_12keV,chunks=[dict(file=r['array_file'],sha256=r['array_sha256'],metadata_sha256=sha(out/Path(r['array_file']).with_suffix('.json'))) for r in records],
        first_timestamp=rows[0]['timestamp'],raw_readonly=True,raw_saved=False,exposure_normalization='P integer pixels; E=P*each header exposure exactly once',
        validation_scope='聚合物开发版v0; application only, not newTEST or validation',source_sha256=sha(__file__))
    write(out/'run.json',run);print('INTAKE_COMPLETE',out.name,len(rows),'/',all_count,flush=True);return run
