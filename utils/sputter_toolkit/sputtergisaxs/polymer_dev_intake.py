"""Readonly CBF intake and conditional automatic centre for polymer_dev_v0.

Existing atomic geometry kernel is unchanged. Only derived histograms leave
the raw-data host. Geometry inference is conditional on a specular maximum
and the supplied incidence angle, not an independent beam calibration.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json,hashlib,re,time
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter,label
from .cbfio import read_cbf,CBFDamagedError
from .stage5_input import AtomicMap
from .neural_refine_report import read,write,sha

# Detector support window (pixels) relative to the direct-beam centre: rows
# from 939 above to 31 below the beam, columns from 556.3 left to 553.7 right.
SUPPORT_ROWS_FROM_BEAM=(-939.,31.)
SUPPORT_COLS_FROM_BEAM=(-556.3,553.7)


def frames(path):
    path=Path(path)
    files=[path] if path.is_file() else sorted(path.glob('*.cbf'))
    if not files:raise ValueError('No CBF frames in supplied path')
    return files


def load_frame(path):
    with Path(path).open('rb') as stream:header=stream.read(16384).decode('latin1')
    cutoff=re.search(r'Count_cutoff\s+([\d.eE+-]+)',header)
    image,meta=read_cbf(path,count_limit=float('inf'))
    limit=float(cutoff.group(1)) if cutoff else None
    removed=int(np.sum(image>=limit)) if limit is not None else 0
    if limit is not None:image[image>=limit]=np.nan
    if not meta.get('exposure_s',0)>0:raise ValueError('Positive header exposure required')
    meta.update(count_cutoff=limit,cutoff_removed_pixels=removed,header_sha256=hashlib.sha256(header.encode('latin1')).hexdigest())
    return image,meta


def damaged_record(path,exc):
    """Explicit per-frame exclusion; the raw file is only read, never repaired."""
    path=Path(path)
    with path.open('rb') as stream:stamp=re.search(r'# (\d{4}-\d\d-\d\dT[\d:.]+)',stream.read(16384).decode('latin1'))
    return dict(frame=path.name,frame_id=int(path.stem.rsplit('_',1)[-1]),header_timestamp=stamp.group(1) if stamp else None,
        raw_sha256=sha(path),reason=exc.reason,detail=exc.detail,raw_unchanged=True)


def readable_ids(files,ids):
    """Sampled frame indices with a complete declared payload (geometry sampling only)."""
    from .cbfio import payload_check
    keep=[]
    for i in ids:
        try:payload_check(files[int(i)]);keep.append(int(i))
        except CBFDamagedError:pass
    if not keep:raise ValueError('No readable CBF among sampled frames; supply beam centre')
    return np.asarray(keep)


def auto_center(files,sdd_mm,alpha_i_deg,pixel_mm=.172):
    ids=readable_ids(files,np.unique(np.linspace(0,len(files)-1,min(9,len(files)),dtype=int)));points=[];candidates=[]
    for i in ids:
        im,meta=load_frame(files[i]);ny,nx=im.shape
        sm=gaussian_filter(np.where(np.isfinite(im),im,0.),1.)
        # Isolated intense reflection, avoiding panel boundaries and outer
        # edges. Save competing maxima rather than treating the maximum as
        # a known direct-beam spot.
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
    points=np.asarray(points);centre=np.median(points,axis=0);mad=np.median(abs(points-centre),axis=0)
    col=float(centre[0]);separation=sdd_mm*np.tan(2*np.radians(alpha_i_deg))/pixel_mm
    # The brightest rod maximum can be a moving Yoneda feature. Locate the
    # two persistent beamstop shadows on that rod instead, and require their
    # separation to agree with the supplied mirror-angle displacement.
    profiles=[]
    for i in ids:
        im,_=load_frame(files[int(i)]);strip=im[:,max(0,int(round(col))-3):int(round(col))+4]
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
    # Independent horizontal reflection symmetry around the inferred column;
    # includes the Yoneda/diffuse band, excludes the narrow specular core.
    image,_=load_frame(files[int(ids[len(ids)//2])]);ny,nx=image.shape
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
        roll_status='0deg working assumption; no sample roll inferred',
        beamstop_status='Central absent/obscured direct beam not used as a measured bright point; negative/gap pixels excluded')


def extract_chunk(task):
    files,geometry,maskpath,out,index,comparison=task;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/f'chunk_{index:05d}.npz';meta=dest.with_suffix('.json')
    if meta.exists():
        record=read(meta)
        if sha(dest)!=record['array_sha256'] or record['files']!=files or record['geometry']!=geometry:raise ValueError('Existing intake chunk differs')
        return record
    mask=np.load(maskpath).astype(bool) if maskpath else None
    ys=[];ps=[];alt_y=[];alt_p=[];rows=[];excluded=[];mapping=second=crop=None;start=time.perf_counter()
    for file in files:
        try:image,h=load_frame(file)
        except CBFDamagedError as exc:
            excluded.append(damaged_record(file,exc));print('INTAKE_EXCLUDED',Path(file).name,exc.reason,flush=True);continue
        if mapping is None:
            shape=image.shape
            # Fixed detector support window placed relative to the beam centre.
            r0,r1=SUPPORT_ROWS_FROM_BEAM;c0,c1=SUPPORT_COLS_FROM_BEAM
            row,col=geometry['db_row'],geometry['db_col']
            crop=[max(0,int(round(row+r0))),min(shape[0],int(round(row+r1))),max(0,int(round(col+c0))),min(shape[1],int(round(col+c1)))]
            mapping=AtomicMap(geometry,shape,geometry['alpha_i_deg'],crop=crop,exclude_rings=((2.1,.13),))
            second=AtomicMap(dict(geometry,energy_ev=12000.),shape,geometry['alpha_i_deg'],crop=crop,exclude_rings=((2.1,.13),)) if comparison else None
        Y,E=mapping.project(image,h['exposure_s'],mask)
        P=E/h['exposure_s'];assert np.allclose(P,np.rint(P)) and np.max(P)<65535 and np.max(Y)<2**32
        ys.append(Y.astype(np.uint32));ps.append(np.rint(P).astype(np.uint16))
        if second:
            Y2,E2=second.project(image,h['exposure_s'],mask);alt_y.append(Y2.astype(np.uint32));alt_p.append(np.rint(E2/h['exposure_s']).astype(np.uint16))
        rows.append(dict(frame=Path(file).name,frame_id=int(Path(file).stem.rsplit('_',1)[-1]),timestamp=h.get('timestamp'),exposure_s=h['exposure_s'],
            count_cutoff=h['count_cutoff'],cutoff_removed_pixels=h['cutoff_removed_pixels'],header_sha256=h['header_sha256'],raw_sha256=sha(file),
            selected_counts=int(Y.sum()),valid_pixel_count=int(P.sum())))
    if len(rows)+len(excluded)!=len(files):raise ValueError('Chunk frame denominator changed')
    arrays=dict(Y=np.asarray(ys),P=np.asarray(ps))
    if comparison and mapping is not None:arrays.update(Y_12keV=np.asarray(alt_y),P_12keV=np.asarray(alt_p))
    np.savez_compressed(dest,**arrays)
    record=dict(files=files,geometry=geometry,crop=crop,mask=maskpath or 'Native finite/nonnegative/header-cutoff pixels; no detector mask supplied',
        rows=rows,excluded=excluded,array_file=dest.name,array_sha256=sha(dest),wall_seconds=time.perf_counter()-start,raw_saved=False)
    write(meta,record);print('INTAKE_CHUNK',out.name,index,len(files),round(record['wall_seconds'],1),flush=True);return record


def prepare(input_path,out,*,energy_ev,sdd_mm,alpha_i_deg,metal,beam_col=None,beam_row=None,sample_roll_deg=0.,
            max_frames=None,maskpath=None,workers=8,compare_12keV=False,system_note='Polymer'):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    out=Path(out);files=frames(input_path);all_count=len(files)
    selected=np.unique(np.linspace(0,len(files)-1,min(len(files),max_frames),dtype=int)) if max_frames else np.arange(len(files))
    files=[files[i] for i in selected]
    if (beam_col is None)!=(beam_row is None):raise ValueError('Provide both beam coordinates or neither')
    centre=auto_center(files,sdd_mm,alpha_i_deg) if beam_col is None else dict(db_col=float(beam_col),db_row=float(beam_row),sample_roll_deg=sample_roll_deg,
        geometry_status='User-supplied beam centre')
    geometry=dict(energy_ev=energy_ev,sdd_mm=sdd_mm,pixel_mm=.172,alpha_i_deg=alpha_i_deg,
        db_col=centre['db_col'],db_row=centre['db_row'],sample_roll_deg=centre['sample_roll_deg'])
    out.mkdir(parents=True,exist_ok=True);write(out/'geometry.json',centre|dict(geometry=geometry))
    tasks=[([str(p) for p in files[i:i+128]],geometry,str(maskpath) if maskpath else None,str(out),i//128,compare_12keV) for i in range(0,len(files),128)]
    records=[];print('INTAKE_PLAN',len(tasks),len(files),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(extract_chunk,t) for t in tasks]):records.append(future.result())
    records.sort(key=lambda r:r['array_file']);rows=[r for chunk in records for r in chunk['rows']]
    excluded=[e for chunk in records for e in chunk.get('excluded',[])]
    if len(rows)+len(excluded)!=len(files):raise ValueError('Frame denominator changed')
    if not rows:raise ValueError('No readable CBF frames among the selected frames')
    write(out/'excluded_frames.json',dict(received_frames=all_count,selected_frames=len(files),excluded=excluded,raw_readonly=True))
    run=dict(status='completed',input_source=str(input_path),received_frames=all_count,selected_frames=len(files),selection_indices=selected.tolist(),
        analysed_frames=len(rows),excluded_frames=excluded,
        frame_denominator=f'{len(rows)} analysed = {len(files)} selected - {len(excluded)} damaged (truncated/undecodable; listed in excluded_frames, no interpolation)',
        geometry=geometry,centre=centre,metal=metal,model_metal='Ag' if metal=='Cu' else metal,system_note=system_note,
        Cu_density_ratio=.89 if metal=='Cu' else 'Not applicable',Cu_scope='Cu以Ag模型近似' if metal=='Cu' else 'Native metal label',
        compare_12keV=compare_12keV,chunks=[dict(file=r['array_file'],sha256=r['array_sha256'],metadata_sha256=sha(out/Path(r['array_file']).with_suffix('.json'))) for r in records],
        first_timestamp=rows[0]['timestamp'],raw_readonly=True,raw_saved=False,exposure_normalization='P integer pixels; E=P*each header exposure exactly once',
        validation_scope='聚合物开发版v0; application only, not newTEST or validation',source_sha256=sha(__file__))
    write(out/'run.json',run);print('INTAKE_COMPLETE',out.name,len(rows),'/',all_count,'excluded',len(excluded),flush=True);return run
