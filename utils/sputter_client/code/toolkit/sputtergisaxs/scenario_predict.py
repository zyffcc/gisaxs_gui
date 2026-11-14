'Detector-file adapter for the actual four-scenario research runtime.'
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .prediction import FrameResult, STRUCTURE_FIELDS, analyze_frame
from .detector_image_input import read_detector_image
from .substrate_context import layer_context

SCENES = ('S1','S2','S3','S4')
QUANTITIES = ('D_nm','diameter_nm','height_nm','gap_nm','coverage_fraction',
              'depth_span_nm','exposed_span_nm','burial_span_nm')
DEFAULT_PUBLIC_RINGS = ((2.1,.13),)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((a.dtype,tuple(a.shape))).encode('ascii'))
    digest.update(a.tobytes()); return digest.hexdigest()


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return _plain(value.tolist())
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _optional_number(value,name,notes,*,zero=False):
    if value is None:return None
    try:
        if isinstance(value,(bool,np.bool_)):raise ValueError()
        v = float(value)
        if not np.isfinite(v) or (v < 0 if zero else v <= 0):raise ValueError()
        return v
    except (ValueError,TypeError,OverflowError):
        notes.append(name+'_invalid_kept_unknown'); return None


def _host(category,context,notes):
    if category is not None:
        value = str(category).strip()
        if value in ('Si','oxide_particles','polymer'):return value
        if value.lower() == 'si':return 'Si'
        if value.lower() == 'unknown':return None
        notes.append('unrecognised_host_category_kept_unknown'); return None
    if context['is_polymer_label']:return 'polymer'
    label = str(context.get('host_label') or '').strip().lower()
    if label in ('si','bare si','裸si'):return 'Si'
    if label in ('tio2','tio₂','sio2','sio₂'):
        notes.append('oxide_material_does_not_prove_particulate_architecture; host_class_kept_unknown')
    return None


def _applicable(host,scene):
    return not (host == 'Si' and scene in ('S2','S3'))


def _empty_scenarios(host,reason):
    out = {}
    for scene in SCENES:
        allowed = _applicable(host,scene)
        out[scene] = dict(applicable=allowed,
            not_applicable_reason='known bare Si excludes S2/S3' if not allowed else None,
            status='not_applicable' if not allowed else 'not_evaluated',
            consistency='not_evaluated', missing_reason=reason,
            conditional_quantities={q:dict(median=None,interval68=None,interval95=None,
                calibrated=False,structural_release=False,reason=reason) for q in QUANTITIES},
            public_quantities={q:None for q in ('D_nm','diameter_nm','height_nm','gap_nm','coverage_fraction')},
            original_draws=0, decode_failed_draws=0,
            height_scope='step-column depth support, not an island height' if scene=='S3'
                         else 'island quantities not applicable' if scene=='S4' else 'conditional cap/hemi height',
            geometry_export_available=False,
            geometry_export_missing_reason='no v2 source-bound calibrated/qualified geometry export',
            calibrated=False,OOD_qualified=False,outside_range_status='unknown',structural_release=False)
    return out


def _load(image,frame_index):
    if (isinstance(frame_index,(bool,np.bool_)) or not isinstance(frame_index,(int,np.integer))
            or frame_index<0):
        raise ValueError('Frame index must be a nonnegative integer')
    if isinstance(image,(str,Path)):
        path = Path(image)
        if path.suffix.lower()=='.cbf' and frame_index!=0:
            raise ValueError('The single-frame CBF reader supports only frame_index=0')
        data,header = read_detector_image(path,frame_index)
        return np.asarray(data,float),header,dict(path=str(path),sha256=_sha(path),
            frame_index=int(frame_index),format=path.suffix.lower(),
            source='unchanged CBF read_image' if path.suffix.lower()=='.cbf' else 'public unit-aware Fabio intake',
            exposure_metadata=header.get('exposure_metadata'))
    data = np.asarray(image,float)
    return data,{},dict(path=None,array_sha256=_array_sha(data),frame_index=int(frame_index),
                       format='array',source='supplied 2D detector array')


def detector_valid_mask(geometry,mask=None,*,shape=None):
    'Intersect caller-valid pixels with PONI detector-invalid pixels.'
    target=None if shape is None else tuple(shape)
    valid=None
    if mask is not None:
        valid=np.asarray(mask,bool)
        if valid.ndim!=2 or target is not None and valid.shape!=target:
            raise ValueError('Explicit valid mask shape differs from image')
        valid=valid.copy()
    meta=dict(caller_valid_mask_provided=mask is not None,
        caller_valid_mask_sha256=None if valid is None else _array_sha(valid),
        intrinsic_detector_mask_provided=False,intrinsic_masked_pixels=0,
        intrinsic_mask_source='not_inferred_from_manual_geometry',
        zero_count_pixels_are_valid=True)
    if isinstance(geometry,(str,Path)):
        import pyFAI
        g=pyFAI.load(str(geometry));bad=g.detector.mask
        meta.update(intrinsic_mask_source='pyFAI.poni_detector',detector_name=str(g.detector.name))
        if bad is not None:
            bad=np.asarray(bad,bool)
            if (bad.ndim!=2 or target is not None and bad.shape!=target
                    or valid is not None and bad.shape!=valid.shape):
                raise ValueError('PONI intrinsic detector mask differs from image/valid mask shape')
            valid=~bad if valid is None else valid&~bad
            meta.update(intrinsic_detector_mask_provided=True,
                intrinsic_invalid_mask_sha256=_array_sha(bad),intrinsic_masked_pixels=int(bad.sum()))
    meta.update(effective_valid_mask_sha256=None if valid is None else _array_sha(valid),
        effective_masked_pixels=0 if valid is None else int((~valid).sum()),
        mask_shape=None if valid is None else list(valid.shape))
    return valid,meta


def analyze_feature_frame(image,geometry,alpha_i_deg,metal,*,preframe=None,
        exposure_s=None,pre_exposure_s=None,mask=None,frame_index=0,**kwargs):
    'Current feature-only input adapter; frozen projection remains unchanged.'
    start=time.perf_counter();cpu=time.process_time();notes=[]
    data,header,source=_load(image,frame_index)
    if data.ndim!=2:raise ValueError('A detector frame must be two dimensional')
    mask,validity=detector_valid_mask(geometry,mask,shape=data.shape)
    exp=_optional_number(exposure_s if exposure_s is not None else header.get('exposure_s'),
                         'exposure_s',notes)
    reference=None;reference_source=None;reference_error=None;rexp=None
    if preframe is not None:
        try:
            reference,rheader,reference_source=_load(preframe,0)
            rexp=_optional_number(pre_exposure_s if pre_exposure_s is not None else rheader.get('exposure_s'),
                                  'pre_exposure_s',notes)
        except (OSError,ValueError,TypeError) as error:
            reference_error=str(error)
            notes.append('preframe_unavailable_kept_unknown: '+reference_error)
    result=analyze_frame(data,geometry,alpha_i_deg,metal,preframe=reference,
        exposure_s=exp,pre_exposure_s=rexp,mask=mask,frame_index=frame_index,**kwargs)
    result.row.update(image=source['path'] or '<array>',frame_index=int(frame_index),
        detector_validity=validity,input_source=source,reference_source=reference_source,
        exposure_source='explicit_user_seconds' if exposure_s is not None else 'file_header' if exp is not None else 'unknown',
        reference_exposure_source='explicit_user_seconds' if pre_exposure_s is not None else 'file_header' if rexp is not None else 'unknown',
        input_notes=notes,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start)
    if reference_error is not None:
        result.row.update(optional_preframe_known=False,reference_change_status='incomparable_reference',
                          reference_change_reason=reference_error)
    return result


def _photon_semantics(data,source,explicit,notes):
    if explicit is not None:
        if not isinstance(explicit,(bool,np.bool_)):
            notes.append('invalid_photon_statistics_flag_kept_unknown'); return False,'unknown'
        return bool(explicit),'explicit caller photon-count semantics'


    good = np.isfinite(data)&(data>=0)
    if source['format']=='.cbf' and good.any() and np.all(data[good]==np.rint(data[good])):
        notes.append('CBF_photon_statistics_assume_raw_Pilatus_counts; corrected_CBF_requires_explicit_false')
        return True,'project raw Pilatus CBF reader, supported integer-count assumption'
    notes.append('detector_units_not_established_photons; feature_output_retained_no_Poisson_release')
    return False,'unknown photon statistics; file extension/integer dtype is not proof'


def _metadata(image,geometry,ai,metal,layers,host,thickness,pre,qcm,sem,notes):
    if isinstance(geometry,(str,Path)):
        geometry_source = dict(path=str(geometry),sha256=_sha(geometry),format='pyFAI.poni')
    else:
        values = _plain(dict(geometry))
        geometry_source = dict(values=values,sha256=hashlib.sha256(json.dumps(values,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest(),format='manual')
    return dict(schema='four_scenario_detector_adapter_1',input=image,geometry=geometry_source,
        alpha_i_deg=float(ai),metal=metal,optional_inputs=dict(declared_layers=_plain(layers),
            host_category=host,interlayer_thickness_nm=thickness,preframe_provided=pre is not None,
            QCM_rate_nm_s=qcm,SEM_statistics=_plain(sem),SEM_used_for_posterior=False),
        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        notes=list(notes),raw_image_saved=False,TEST_opened=False,structural_release=False)


def _base_row(source,metal,geometry,exposure,host,thickness,pre,qcm,notes):
    row = dict(image=source['path'] or '<array>',frame_index=source['frame_index'],metal=metal,
        mode='features_only',geometry=geometry,exposure_s=exposure,
        intensity_unit='detector_units_per_valid_pixel_frame',support_unit='pixel_frames',
        optional_layers_known=False,optional_host_category_known=host is not None,
        optional_thickness_known=thickness is not None,optional_preframe_known=pre is not None,
        optional_qcm_known=qcm is not None,optional_notes=list(notes),
        growth_stage=None,stage_probabilities=None,growth_stage_status='withheld',
        growth_stage_reason='scenario hypothesis is not validated connectivity stage',
        OOD_score=None,OOD_flag=None,OOD_status='not_qualified',reliability='unreliable',
        reason='model_not_available',calibrated=False,structural_release=False,
        model_status='not_supplied',model_failure_reason=None,runtime_receipt=None,
        original_posterior_seed=None,original_replica_seed=None,
        original_draws_per_applicable_scenario=0,
        PPC_status='not_requested_no_weights',PPC_failure_reason=None,
        reference_source=None,reference_exposure_s=None,reference_change_reason=None,
        reference_photon_statistics_source=None,
        posterior_cpu_seconds=None,posterior_wall_seconds=None,
        PPC_cpu_seconds=None,PPC_wall_seconds=None,
        posterior_images_averaged=False,nuisance_resampled=False,
        scenarios=_empty_scenarios(host,'model_not_available'),
        reference_scenario_probabilities={s:None for s in SCENES},
        scenario_probabilities_are_reference_only=True,scenarios_pruned=False,
        SEM_comparison_status='not_provided',geometry_export_available=False)
    row.update({q:None for q in STRUCTURE_FIELDS})
    return row


def _attach_empirical(row,public,image):
    from .stage5_measurement import Q
    from .resolvable import in_range,visible_for_regression
    from .scenario_observation_summary import empirical_summary
    summary = empirical_summary(public)
    observed = summary['observed']
    hy,hn,vy,vn,picture,scale = (summary[k] for k in ('hy','hn','vy','vn','picture','scale'))
    fits,bands,selected,ridge = (summary[k] for k in ('fits','bands','selected','ridge'))
    values = observed['values']; q = values['q_star']; fw = values['FWHM']; band = observed['AF_band']
    cov = summary['coverage']


    selected_q = selected.get('q_star') if selected else None
    selected_fw = selected.get('FWHM') if selected else None
    row.update(D_corr_nm=float(2*np.pi/selected_q) if selected_q is not None and selected_q>0 else None,
        width_over_qstar=float(selected_fw/selected_q) if selected_q is not None and selected_q>0 and selected_fw is not None else None,
        q_star_nm_inv=selected_q,FWHM_nm_inv=selected_fw,Yoneda_deg=values['Yoneda'],
        feature_status=selected['support'] if selected else 'unresolved',
        feature_AF_band=selected['AF_band']+1 if selected else None,
        Yoneda_status=ridge['ridge_status'],bands=bands,
        q_star_fit_se_nm_inv=selected.get('fit_se_q') if selected else None,
        feature_uncertainty='empirical residual-scaled fit SE; independent conditional Poisson SE separately recorded',
        feature_stat_se=observed['stat_se'],feature_stat_se_source=observed['stat_se_source'],
        feature_missing_reasons=dict(D_corr=None if selected_q is not None else 'no_resolved_lateral_feature',
            width_rel=None if selected_q is not None and selected_fw is not None else 'no_resolved_peak_width',
            Yoneda=observed['feature_missing_reasons']['Yoneda']),
        model_feature_missing_reasons=observed['feature_missing_reasons'],
        D_corr_conditional_fit_se_nm=float(2*np.pi*selected['fit_se_q']/selected_q**2)
            if selected_q is not None and selected_q>0 and selected.get('fit_se_q') is not None else None,
        model_bilateral_q_star_nm_inv=q,model_bilateral_FWHM_nm_inv=fw,
        no_peak_reason=None if selected and selected['support']=='bilateral' else
            'only_one_side_feature' if selected else 'no_resolved_lateral_feature',
        D_search_min_nm=cov['D_search_min_nm'],D_search_max_nm=cov['D_search_max_nm'],
        D_search_intervals_nm=cov['D_corr_intervals_nm'],D_search_definition=cov['definition'],
        visible_for_D_regression=visible_for_regression(selected,cov),
        feature_in_search_range=bool(in_range(2*np.pi/selected_q,cov['bands'][selected['AF_band']]['D_corr_intervals_nm']))
            if selected_q is not None and selected_q>0 else None,
        photon_count_statistics_known=bool(public.photon_counts_known),
        reference_change_status='not_provided')
    profiles = dict(q=Q.copy(),Y=hy,N=hn,VY=vy,VN=vn,fits=fits,ridge=ridge,coverage=cov,
        image_thumbnail=image[::max(1,image.shape[0]//256),::max(1,image.shape[1]//256)],
        input_intensity_unit=row['intensity_unit'],public_count_frame=public,
        observed_ppc_features=observed,canonical_image=picture,canonical_scale=scale)
    return profiles


def _research_summary(sampled,ppc=None):
    from .scenario_credibility import D_semantics
    output = {}
    for scene in SCENES:
        one = sampled['scenarios'][scene]; records = one['draws']
        prediction = ppc.get('scenarios',{}).get(scene,{}) if ppc is not None else {}
        check = prediction.get('PPC')
        measurable = check is not None and any(f['applicable'] for f in check['features'].values())
        agreed = (prediction['fastline_qwidth_pass'] if 'fastline_qwidth_pass' in prediction
                  else check['all_measurable_feature_checks_pass'] if check is not None else False)
        consistency = ('not_applicable' if not one['applicable'] else
            'no_measurable_feature_for_consistency' if check is not None and not measurable else
            'consistent_features_but_structure_unqualified' if check is not None and agreed
            else 'inconsistent_features' if check is not None else 'not_evaluated')
        definitions = next((r['conditional_labels'] for r in records if not r['decode_failed']),{})
        semantics = D_semantics(scene, records)
        output[scene] = dict(applicable=one['applicable'],not_applicable_reason=one['not_applicable_reason'],
            status=prediction.get('status','conditional_posterior_uncalibrated') if one['applicable'] else 'not_applicable',
            consistency=consistency,
            conditional_quantities=_plain(one['conditional_quantities']),
            public_quantities=_plain(one['public_quantities']),
            original_draws=one['original_draws'],
            decode_failed_draws=sum(bool(r['decode_failed']) for r in records),
            quantity_definitions=dict({k:_plain(v) for k,v in definitions.items()
                                  if k.endswith('_definition') or k.endswith('_missing_reason')},
                                  D_definition=semantics['D_definition'],
                                  raw_D_definitions=semantics['raw_D_definitions']),
            D_semantics=semantics,
            PPC=_plain(check),PPC_forward_failed_draws=check['forward_failed_draws'] if check is not None else None,
            PPC_gate_features=prediction.get('fastline_feature_gates','all_measurable_features'),
            PPC_warning_features=_plain(prediction.get('warning_features',{})),
            PPC_peak_width_highq_pass=prediction.get('fastline_qwidth_highq_pass'),
            missing_reason=one['missing_reason'],
            height_scope='effective column depth support, not an island height' if scene=='S3'
                        else 'island sizes not applicable' if scene=='S4' else 'conditional cap/hemi island height',
            merged_phase_assumed=scene=='S4',connected_topology_verified=False,
            outside_range_status='unknown',calibrated=False,OOD_qualified=False,
            geometry_export_available=False,
            geometry_export_missing_reason='no v2 source-bound calibrated/qualified geometry export',
            structural_release=False)
    return output


def analyze_scenarios(image,geometry,alpha_i_deg,metal,*,runtime=None,checkpoint=None,
        host_category=None,layers=None,interlayer_thickness_nm=None,preframe=None,
        qcm_rate=None,exposure_s=None,pre_exposure_s=None,mask=None,sample_roll_deg=0.,
        crop=None,exclude_rings=(),public_exclude_rings=DEFAULT_PUBLIC_RINGS,
        frame_index=0,photon_counts_known=None,pre_photon_counts_known=None,
        draws=128,seed=673120261018,device='cpu',run_ppc=False,bank_plan=None,
        feature_certificate=None,ppc_backend='cpu',ppc_seed=673220261018,sem=None,bundle=None,
        training_package=None,producer_package=None):
    'Return one CSV-ready row and plot/original-draw data; no release bypass.'
    start=time.perf_counter();cpu=time.process_time(); notes=[]
    if (isinstance(alpha_i_deg,(bool,np.bool_)) or not np.isscalar(alpha_i_deg)
            or not np.isfinite(float(alpha_i_deg)) or not 0<float(alpha_i_deg)<90):
        raise ValueError('Finite positive incidence angle in degrees is required')
    if sum(v is not None for v in (runtime,checkpoint,bundle))>1:
        raise ValueError('Provide one existing runtime, selected checkpoint or portable bundle')
    if (isinstance(draws,(bool,np.bool_)) or not isinstance(draws,(int,np.integer)) or draws<2
            or isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0):
        raise ValueError('At least two original posterior draws and a nonnegative seed required')
    value=str(metal).strip().lower(); element={'au':'Au','ag':'Ag'}.get(value,str(metal))
    data,header,source = _load(image,frame_index)
    if data.ndim!=2:raise ValueError('A detector frame must be two dimensional')
    mask,detector_validity=detector_valid_mask(geometry,mask,shape=data.shape)
    exposure = _optional_number(exposure_s if exposure_s is not None else header.get('exposure_s'),
                                'exposure_s',notes)
    thickness = _optional_number(interlayer_thickness_nm,'interlayer_thickness_nm',notes,zero=True)
    qcm = _optional_number(qcm_rate,'QCM_rate',notes)
    layer_hint = layers
    backing_only = False
    if isinstance(layers,dict):
        top=layers.get('upper',layers.get('top'))
        middle=layers.get('middle',layers.get('interlayer'))
        bottom=layers.get('lower',layers.get('bottom'))
        backing_only=middle is None and top is None and bottom is not None
        layer_hint=[v for v in (top,middle,bottom) if v is not None]

        layer_hint='/'.join(map(str,layer_hint)) if not backing_only else None
    elif isinstance(layers,(list,tuple)):
        layer_hint='/'.join(map(str,layers))
    context = layer_context(layer_hint,thickness_nm=thickness)
    if backing_only:
        context['backing_label']=str(bottom)
        notes.append('only_backing_material_provided; interlayer_host_kept_unknown')
    category = _host(host_category,context,notes)
    known,semantics = _photon_semantics(data,source,photon_counts_known,notes)
    common_rings = tuple(tuple(map(float,r)) for r in exclude_rings)
    public_rings = tuple(dict.fromkeys(common_rings+tuple(tuple(map(float,r)) for r in public_exclude_rings)))
    if any(len(r)!=2 or not np.isfinite(r).all() or r[1]<=0 for r in public_rings):
        raise ValueError('Declared ring masks require finite centre/positive half-width')
    try:
        from .stage5_input import AtomicMap
        from .v2_observation import CountFrame,project_frame
    except ModuleNotFoundError as error:
        if not str(error.name).startswith('torch'):raise


        result = analyze_feature_frame(data,geometry,alpha_i_deg,element,layers=layers,
            interlayer_thickness_nm=thickness,preframe=preframe,qcm_rate=qcm,
            exposure_s=exposure,pre_exposure_s=pre_exposure_s,mask=mask,
            sample_roll_deg=sample_roll_deg,crop=crop,exclude_rings=public_rings)
        notes.append('v2_native_input_dependency_missing; unchanged legacy feature projection used')
        result.row.update(image=source['path'] or '<array>',frame_index=int(frame_index),
            input_source=source,
            exposure_source='explicit_user_seconds' if exposure_s is not None else 'file_header' if exposure is not None else 'unknown',
            detector_validity=detector_validity,
            scenarios=_empty_scenarios(category,'native v2 input requires optional torch dependency'),
            reference_scenario_probabilities={s:None for s in SCENES},scenarios_pruned=False,
            scenario_probabilities_are_reference_only=True,calibrated=False,structural_release=False,
            model_status='unavailable_native_input_dependency',optional_notes=notes,
            projection='legacy FeatureMap fallback; not a v2 network tensor',
            provenance=_metadata(source,geometry,alpha_i_deg,element,layers,category,thickness,preframe,qcm,sem,notes))
        result.row['cpu_seconds']=time.process_time()-cpu;result.row['wall_seconds']=time.perf_counter()-start
        result.row['provenance']['detector_validity']=detector_validity
        return result
    fullmap=AtomicMap(geometry,data.shape,alpha_i_deg,sample_roll_deg=sample_roll_deg,
                     crop=crop,exclude_rings=common_rings)
    pubmap=AtomicMap(geometry,data.shape,alpha_i_deg,sample_roll_deg=sample_roll_deg,
                    crop=crop,exclude_rings=public_rings)
    full=project_frame(fullmap,data,mask=mask,photon_counts_known=known,exposure_s=exposure)
    public=project_frame(pubmap,data,mask=mask,photon_counts_known=known,exposure_s=exposure)
    row=_base_row(source,element,fullmap.meta,exposure,category,thickness,preframe,qcm,notes)
    row['detector_validity']=detector_validity
    profiles=_attach_empirical(row,public,data)
    profiles.update(full_count_frame=full,substrate_context=context)
    row.update(optional_layers_known=bool(context['declared_stack']),photon_statistics_source=semantics,
               projection='original two-sided native q/AF pixel-frame histogram',
               full_excluded_rings_nm_inv=_plain(common_rings),public_excluded_rings_nm_inv=_plain(public_rings),
               full_valid_pixel_frames=int(full.P.sum()),public_valid_pixel_frames=int(public.P.sum()))
    reference = None
    current_common = reference_common = None
    if preframe is not None:
        try:
            ref,rheader,rsource=_load(preframe,0)
            if ref.shape!=data.shape:raise ValueError('Reference shape differs from declared common geometry')
            rexp=_optional_number(pre_exposure_s if pre_exposure_s is not None else rheader.get('exposure_s'),
                                   'pre_exposure_s',notes)
            rknown,rsemantics=_photon_semantics(ref,rsource,pre_photon_counts_known,notes)
            reference=project_frame(fullmap,ref,mask=mask,photon_counts_known=rknown,exposure_s=rexp)
            profiles['reference_count_frame']=reference
            common = np.isfinite(data)&(data>=0)&np.isfinite(ref)&(ref>=0)
            if mask is not None:
                common &= np.asarray(mask,bool)
            current_common = project_frame(pubmap,data,mask=common,photon_counts_known=known,exposure_s=exposure)
            reference_common = project_frame(pubmap,ref,mask=common,photon_counts_known=rknown,exposure_s=rexp)
            from .scenario_reference_change import reference_change
            changes = reference_change(current_common,reference_common,
                geometry_identity=fullmap.meta,current_source=source,
                reference_source=rsource,common_pixel_support_verified=True)
            profiles['reference_state_change'] = changes
            profiles['reference_change'] = changes['raw_change']
            row.update(reference_change_status='signed_observed_change_and_fixed_highq_shape_subtraction; relative fluence unknown',
                       reference_photon_statistics_source=rsemantics,reference_source=rsource,
                       reference_exposure_s=rexp,optional_preframe_known=True)
        except Exception as error:
            notes.append('preframe_unavailable_kept_unknown: '+str(error));row.update(optional_preframe_known=False,
                reference_change_status='incomparable_reference',reference_change_reason=str(error))
    row['SEM_comparison_status'] = 'retained_metadata_not_a_trained_constraint' if sem is not None else 'not_provided'
    row['SEM_statistics'] = _plain(sem);row['SEM_statistics_used_by_model']=False
    row['model_status']='not_supplied'
    sampled=None;ppc=None;ppc_cpu=0.;ppc_wall=0.;portable=None
    if element not in ('Au','Ag'):
        row.update(reliability='out_of_domain',reason='unsupported_metal_Au_Ag_only',model_status='unsupported_metal')
    elif checkpoint is not None or runtime is not None or bundle is not None:
        try:
            from .scenario_runtime import ScenarioRuntime
            from .scenario_minimal_runtime import MinimalScenarioRuntime
            from .scenario_mainline_posterior import MainlineRuntime, TRAINING_SCHEMA
            from .scenario_mainline_ppc import MainlinePPCRuntime
            from .scenario_minimal51 import Minimal51Runtime, TRAINING_SCHEMA as MINIMAL51_SCHEMA
            from .scenario_mainline_interaction_posterior import TRAINING_SCHEMA as INTERACTION_SCHEMA
            from .scenario_inverse_ppc import InteractionPPCRuntime
            from .scenario_qpreserve_runtime import load_runtime
            if bundle is not None:
                from .scenario_bundle import ScenarioBundle
                portable=bundle if isinstance(bundle,ScenarioBundle) else ScenarioBundle(bundle,device=device)
                runtime=portable.runtime
            if runtime is None:
                import torch
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                schema=saved['declaration']['schema'];del saved
                if schema==MINIMAL51_SCHEMA:
                    runtime=Minimal51Runtime(checkpoint,device=device,package_root=training_package)
                elif schema==TRAINING_SCHEMA:
                    runtime=MainlinePPCRuntime(checkpoint,device=device,package_root=training_package)
                elif schema==INTERACTION_SCHEMA:
                    runtime=InteractionPPCRuntime(checkpoint,device=device,package_root=training_package)
                elif training_package is not None and producer_package is not None:
                    if schema=='four_scenario_minimal_training_1':
                        runtime=MinimalScenarioRuntime(checkpoint,training_package=training_package,
                            producer_package=producer_package,device=device)
                    elif schema=='four_scenario_peak_highq_training_1':
                        from .scenario_peak_highq_runtime import PeakHighqScenarioRuntime
                        runtime=PeakHighqScenarioRuntime(checkpoint,training_package=training_package,
                            producer_package=producer_package,device=device)
                    elif schema=='four_scenario_peak_geometry_training_1':
                        from .scenario_peak_geometry_runtime import PeakGeometryScenarioRuntime
                        runtime=PeakGeometryScenarioRuntime(checkpoint,training_package=training_package,
                            producer_package=producer_package,device=device)
                    else:runtime=load_runtime(checkpoint,device=device)
                else:runtime=load_runtime(checkpoint,device=device)
            if not isinstance(runtime,(ScenarioRuntime,MinimalScenarioRuntime,MainlineRuntime)):
                raise TypeError('Actual source-bound ScenarioRuntime required')
            begun=time.perf_counter();bcpu=time.process_time()
            provider=portable if portable is not None else runtime
            options=dict(host_category=category,n=int(draws),seed=int(seed))
            if not isinstance(runtime,MinimalScenarioRuntime):
                options.update(pre=reference,thickness_nm=thickness,qcm_rate=qcm)
                if (isinstance(runtime,MainlineRuntime) and not isinstance(runtime,Minimal51Runtime)
                        and current_common is not None and reference_common is not None):
                    from .scenario_mainline_delta import paired_delta
                    signed_delta=paired_delta(current_common,reference_common,
                        geometry_identity=fullmap.meta,current_source=source,
                        reference_source=row['reference_source'],common_pixel_support_verified=True)
                    options['delta']=signed_delta
                    profiles['mainline_signed_delta']=signed_delta
            elif reference is not None or thickness is not None or qcm is not None:
                notes.append('peak_feature_model_has_no_preframe_thickness_QCM_inputs; retained_as_metadata')
            sampled=provider.sample(full,public,fullmap.meta['energy_ev'],alpha_i_deg,element,**options)
            row.update(mode='four_scenario_research',model_status='source_bound_conditional_posterior_uncalibrated',
                reason='simulation_calibration_OOD_range_PPC_variant_release_not_qualified',
                posterior_cpu_seconds=time.process_time()-bcpu,posterior_wall_seconds=time.perf_counter()-begun,
                reference_scenario_probabilities=_plain(sampled['reference_scenario_probabilities']),
                runtime_receipt=_plain(runtime.receipt),original_posterior_seed=int(seed),
                original_draws_per_applicable_scenario=int(draws))
            profiles['scenario_samples']=sampled
            profiles['scenario_runtime']=runtime


            automatic_ppc=bool(portable is not None and portable.qualification.get('complete_original_TEST_evidence'))
            if run_ppc or automatic_ppc:
                if portable is None and (bank_plan is None or
                        (not isinstance(runtime,MainlineRuntime) and feature_certificate is None)):
                    row['PPC_status']='not_evaluated_missing_original_bank_plan_or_certificate'
                else:
                    begun=time.perf_counter();bcpu=time.process_time()
                    try:
                        if portable is not None:
                            ppc=portable.posterior_predictive(sampled,full,public,
                                backend=ppc_backend,seed=int(ppc_seed))
                        else:
                            ppc=runtime.posterior_predictive(sampled,full,public,bank_plan=bank_plan,
                                feature_certificate=feature_certificate,backend=ppc_backend,seed=int(ppc_seed))
                        row['PPC_status']=ppc['status'];profiles['scenario_ppc']=ppc
                    except (OSError,ValueError,TypeError,RuntimeError,ImportError,MemoryError) as error:
                        row.update(PPC_status='failed_original_PPC_call',PPC_failure_reason=type(error).__name__+': '+str(error))
                    ppc_cpu=time.process_time()-bcpu;ppc_wall=time.perf_counter()-begun
                    row.update(PPC_cpu_seconds=ppc_cpu,PPC_wall_seconds=ppc_wall,
                               original_replica_seed=int(ppc_seed))
            else:row['PPC_status']='not_requested'
            row['scenarios']=_research_summary(sampled,ppc)
            if portable is not None:
                assessment=portable.assess(full,public,sampled,ppc,pre=reference,
                    host_category=category,thickness_nm=thickness,qcm_rate=qcm)
                release=assessment['release']
                profiles['bundle_assessment']=assessment
                row.update(bundle_receipt=_plain(assessment['bundle_receipt']),
                    quantity_qualification_status=assessment['qualification']['status'],
                    credibility=_plain(assessment['credibility']),
                    OOD_score=release['OOD_score'],OOD_flag=release['OOD_flag'],
                    reliability=release['reliability'],structural_release=release['structural_release'],
                    calibrated=release['calibrated'],
                    calibrated_quantities=release['calibrated_quantities'],
                    calibrated_features=release['calibrated_features'],
                    calibration_scope=release['calibration_scope'],
                    conditional_feature_release=release['conditional_feature_release'],
                    model_status='qualified_scenario_quantities' if release['structural_release']
                        else 'qualified_conditional_scenario_features' if release['conditional_feature_release']
                        else 'conditional_posterior_pending_quantity_qualification',
                    reason='qualified_quantities_are_scenario_specific' if release['structural_release']
                        else 'qualified_S4_mean_features_do_not_certify_connectivity' if release['conditional_feature_release']
                        else 'independent_acceptance_or_frame_quantity_gates_not_passed')
                for scene,one in release['scenarios'].items():
                    row['scenarios'][scene].update(_plain(one))
            if sem is not None:


                try:
                    from .scenario_sem import compare_sem
                    comparison=compare_sem(sampled,sem)
                    row['SEM_comparison']=_plain(comparison)
                    row['SEM_comparison_status']=comparison['status']
                    row['SEM_reference_ranking']=comparison['ranked_scenarios']
                    profiles['SEM_comparison']=comparison
                except Exception as error:
                    row.update(SEM_comparison_status='comparison_unavailable_original_posterior_retained',
                        SEM_comparison_failure_reason=type(error).__name__+': '+str(error))
        except Exception as error:
            row.update(model_status='unavailable_or_binding_failed',reason='source_bound_model_not_available',
                model_failure_reason=type(error).__name__+': '+str(error),
                scenarios=_empty_scenarios(category,'actual source-bound model unavailable'))
    else:row['PPC_status']='not_requested_no_weights'


    for key in STRUCTURE_FIELDS:row[key]=None
    row.setdefault('calibrated',False);row.setdefault('structural_release',False)
    row.update(optional_notes=notes,
        cpu_seconds=time.process_time()-cpu-ppc_cpu,wall_seconds=time.perf_counter()-start-ppc_wall,
        provenance=_metadata(source,geometry,alpha_i_deg,element,layers,category,thickness,preframe,qcm,sem,notes))
    row['provenance']['detector_validity']=detector_validity
    return FrameResult(_plain(row),profiles)
