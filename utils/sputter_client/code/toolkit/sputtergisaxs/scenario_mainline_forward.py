'Combined exploratory mainline forward from one decoded51D joint record.'
from dataclasses import replace

import numpy as np

from .effective_host import build_context
from .effective_measurement import compose
from .forward_v06 import SSCAParams
from .optics import nd
from .scenario_chart import SCENARIO_NAMES
from .scenario_coverage_variants import (_WideSpreadParams, _CoverageOperator,
    _host, _cell_means, _two_population_intensity)
from .scenario_forward import QUICK, context as original_context, harddisk_intensity, expected_scale
from .scenario_mainline_chart import SCHEMA, catalog_entry
from .scenario_template import TemplateGrowth, compose_template
from .scenario_ts import labels as ts_labels
from .v2_observation import CountFrame


def context(row,recipe=QUICK):
    if row.get('schema')!=SCHEMA:raise ValueError('Actual mainline chart record required')
    c=catalog_entry(row['mainline_coarse'])
    if row['scenario']!=c['scenario'] or row['modifiers']!=c['modifiers']:
        raise ValueError('Decoded mainline coarse/physical law differs')
    if row['scenario'] in ('S3','S4'):
        return original_context(row,recipe)
    particle=_WideSpreadParams(**row['particle'])
    if row['decoupling']=='DA':particle=replace(particle,kappa=0.)
    return build_context(_host(row),particle,row['metal'],row['E'],row['ai'],
        nbins=recipe['marks'],ndepth=recipe['depth'],reference_mesh='compact')


def _add_clusters(main,row,recipe):
    params=row['small_clusters'];radius=params['small_base_radius_nm'];aspect=params['cluster_aspect']
    particle=SSCAParams(D_nm=1000.,base_radius_nm=radius,height_nm=radius*aspect,
        sigma_log=params['cluster_sigma_log'],omega_rel=.2,kappa=0.,shape=row['particle']['shape'])
    initial=build_context(main['host'],particle,row['metal'],row['E'],row['ai'],
        nbins=recipe['marks'],ndepth=recipe['depth'],reference_mesh='compact')
    maximum_area=initial['max_phi']/initial['n'];remaining=.999-main['max_phi']
    if maximum_area<=0 or remaining<=0:raise ValueError('No remaining physical small-cluster occupation budget')
    density=params['remaining_occupation_fraction']*remaining/maximum_area
    particle=replace(particle,D_nm=float(np.sqrt(2/(np.sqrt(3)*density))))
    small=build_context(main['host'],particle,row['metal'],row['E'],row['ai'],
        nbins=recipe['marks'],ndepth=recipe['depth'],reference_mesh='compact')
    edges=np.unique(np.r_[main['edges'],small['edges']]);widths=np.diff(edges);middle=(edges[:-1]+edges[1:])/2
    phi=_cell_means(main,edges)+_cell_means(small,edges)
    if not np.isfinite(phi).all() or np.any(phi<0) or np.any(phi>1):raise ValueError('Combined metal fraction outside[0,1]')
    base=np.zeros(len(widths),complex)
    base[(middle<0)&(middle>-main['host'].thickness_nm)]=main['host'].decrement
    base[middle<-main['host'].thickness_nm]=nd('SiO2',energy_ev=row['E'])
    post=base+(nd(row['metal'],energy_ev=row['E'])-base)*phi
    sigma=np.zeros(len(widths)+2)
    for z,rough in ((0.,main['host'].roughness_nm),(-main['host'].thickness_nm,main['host'].buried_roughness_nm)):
        where=np.flatnonzero(edges[::-1]==z)
        if len(where):sigma[int(where[0])+1]=max(sigma[int(where[0])+1],rough)
    common=dict(edges=edges,phi=phi,base=base,
        pre_nds=[0j]+base[::-1].tolist()+[nd('Si',energy_ev=row['E'])],
        post_nds=[0j]+post[::-1].tolist()+[nd('Si',energy_ev=row['E'])],
        thick=widths[::-1].tolist(),sigma=sigma.tolist(),mass_nm=float(phi@widths))
    main.update(common);small.update(common);cache={};main['_field_cache']=small['_field_cache']=cache
    main['_small_cluster_context']=small
    return dict(main_number_density_nm2=float(main['n']),small_number_density_nm2=float(small['n']),
        small_median_base_radius_nm=radius,small_median_maximum_footprint_diameter_nm=particle.diameter(),
        combined_mean_metal_mass_nm=float(phi@widths),maximum_cell_phi=float(phi.max()),
        common_mean_field=True,population_fluctuation_cross='assumed zero',
        main_D_density_unchanged=True,total_metal_mass_held_fixed=False,geometry_overlap_verified=False)


def mean_forward(row,P,hostarr,*,recipe=QUICK,backend='cpu',disk=None):
    'Rates/pre/receipt from the exact actual joint record, never sequential rates.'
    c=context(row,recipe);fn=None;small=None
    if 'a' in row['modifiers']:
        small=_add_clusters(c,row,recipe)
    if row.get('structure_factor')=='harddisk' and row['scenario']!='S4':
        if disk is None:raise ValueError('Bound original hard-disk table required')
        c['_harddisk_table']=disk;c['_harddisk_eta']=row['eta'];fn=harddisk_intensity
    if small is not None:fn=_two_population_intensity
    op=_CoverageOperator(c,scenario='S1' if row['scenario']=='S2' else row['scenario'],
        reference=recipe['reference'],qscale=row['qscale'],af_shift=row['af_shift'],
        psf_af_bins=row['psf_af'],psf_q_bins=row['psf_q'],backend=backend,intensity_fn=fn)
    parts=op.evaluate(db_xi_nm=row['db_xi'],power_exponent=row['power_exponent'],scatter_depth_nm=row['scatter_depth'])
    ref=max(float(np.quantile(parts['metal'],.8)),1e-30);empirical=np.zeros_like(P,dtype=float);pre=None;missing=1.
    key=row.get('host_template_key')
    if key:
        shape=hostarr[key+'__shape'];valid=hostarr[key+'__valid']
        empirical=np.where(valid,shape,parts['db'])*ref*row['host_fraction']
        pre=CountFrame(shape*row['target_scale']*P*valid,P*valid,False,None);missing=float((~valid).mean())
    lowq=op.lowq_component(row['lowq'])*ref*row['lowq']['lowq_gain'] if 'b' in row['modifiers'] else 0.
    low_host=lowq if 'b' in row['modifiers'] and row['lowq']['lowq_origin']=='host' else 0.
    low_instrument=lowq if 'b' in row['modifiers'] and row['lowq']['lowq_origin']=='instrument' else 0.
    background=ref*(row['flat_fraction']*parts['flat']+row['ring_fraction']*parts['ring'])+low_instrument
    if row['scenario']=='S2':
        host=empirical+low_host+ref*(row['db_fraction']*parts['db']+row['power_fraction']*parts['power'])
        mixed=compose_template(parts['metal'],host,loading=row['loading'],growth=TemplateGrowth(**row['growth']),
            cross_coefficient=row['gamma'],transmission=parts['transport'],transmitted_fraction=row['transmitted_fraction'],
            instrumental_rate=background,host_source=key or 'parametric_host')
        rate=mixed['rate'];composition=mixed['metadata']
    else:
        rate=compose(parts,empirical+low_host,db_gain=ref*row['db_fraction'],power_gain=ref*row['power_fraction'],
            flat_gain=ref*row['flat_fraction'],ring_gain=ref*row['ring_fraction'],gamma=row['gamma'],
            transmitted_fraction=row['transmitted_fraction'])+low_instrument
        composition=dict(host_source=key,cross='effective intensity correlation')
    scale=expected_scale(rate,P)
    if not np.isfinite(scale) or scale<=0:raise ValueError('No positive candidate input scale')
    rate=np.broadcast_to(rate*row['target_scale']/scale,P.shape)
    if not np.isfinite(rate).all() or np.any(rate<0):raise ValueError('Nonfinite/negative mainline mean')
    if row['scenario']=='S4':labels=ts_labels(c)
    elif row['scenario']=='S3':labels=c['labels']
    else:labels=dict(D_nm=row['D_nm'],lateral_diameter_nm=row['diameter_nm'],island_height_nm=row['height_nm'],
        definition='main population conditional primitive geometry, not SEM nearest neighbour',structural_release=False)
    return rate,pre,dict(status='computed_exploratory_mainline',mainline_coarse=row['mainline_coarse'],
        modifiers=row['modifiers'],composition_order=['d','c','a','b'],small_population=small,
        actual_joint_dim=51,old_chart_label_unchanged=False,operator=op.meta,recipe=recipe,labels=labels,
        expected_scale=scale,metal_reference=ref,host_missing_fraction=missing,composition=composition,
        mean_rate_support_applied=False,empirical_template_reblurred=False,
        numerical_certificate=False,calibrated_model=False,structural_release=False)
