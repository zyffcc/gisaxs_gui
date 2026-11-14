'Signed two-state changes and fixed-highq conditional reference subtraction.'
import hashlib
from pathlib import Path

import numpy as np

from .reference_changes import reference_changes, fit_excess_length
from .v2_observation import CountFrame
from .stage5_measurement import Q, counts_profiles
from .features import AF_BANDS, Q_EDGES


SCHEMA = 'signed_same_geometry_reference_change_1'
HIGHQ_INTERVAL = (1.3, 1.7)
PLATFORM_RELATIVE_SLOPE_LIMIT = .30
MINIMUM_PLATFORM_Q_BINS = 6
SOURCE_FILES = ('sputtergisaxs/scenario_reference_change.py',
    'sputtergisaxs/reference_changes.py', 'sputtergisaxs/v2_observation.py',
    'sputtergisaxs/stage5_measurement.py', 'sputtergisaxs/features.py')


def _source_pins():
    root = Path(__file__).resolve().parent.parent
    return {name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name in SOURCE_FILES}


def _array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.dtype,tuple(a.shape))).encode('ascii'))
    h.update(a.tobytes()); return h.hexdigest()


def _frame_identity(frame):
    return dict(Y_sha256=_array_sha(frame.Y), P_sha256=_array_sha(frame.P),
        exposure_s=float(frame.exposure_s) if frame.exposure_s is not None else None,
        photon_counts_known=bool(frame.photon_counts_known))


def _unsupported_length(reason):
    return dict(xi_nm=None,xi_lo95_nm=None,xi_hi95_nm=None,status=reason,
        interpretation='Conditional Debye-Bueche change length; not pore/particle size or D')


def _unknown_variance(raw):
    "Do not present the old helper's formal count variance for unknown units."
    raw['difference_se'] = None; raw['ratio_se'] = None
    raw['conditional_error'] = 'Photon-count semantics unknown; no Poisson variance or statistical length interval'
    raw['length_view_range_nm'] = None
    for view in raw['summaries']:
        view['highq_ratio_conditional_se'] = None
        view['lowq_integral_conditional_se'] = None
        view['length'] = _unsupported_length('unknown_photon_variance')
    for view in raw['side_consistency']:
        view['ratio_side_conditional_z'] = None


def _conditional_profiles(raw, hy, hn, rhy, rhn, c, shape_condition, variance_known):
    if c is None:
        return dict(status='not_computable_highq_scale', applied=False,
            shape_condition_met=False, difference=None, ratio=None,
            variance_fixed_c=None, difference_se_fixed_c=None, summaries=[])
    valid = (hn>0)&(rhn>0)
    cur = np.divide(hy,hn,out=np.full_like(hy,np.nan),where=valid)
    ref = np.divide(rhy,rhn,out=np.full_like(rhy,np.nan),where=valid)
    difference = cur-c*ref
    ratio = np.divide(cur,c*ref,out=np.full_like(cur,np.nan),where=valid&(c*ref>0))
    vcur = np.divide(hy,hn*hn,out=np.full_like(hy,np.nan),where=valid)
    vref = np.divide(rhy,rhn*rhn,out=np.full_like(rhy,np.nan),where=valid)
    variance = vcur+c*c*vref if variance_known else None
    dq = np.diff(Q_EDGES); summaries = []
    for i in range(len(hy)):
        low=(Q<.3)&valid[i]
        length = (fit_excess_length(Q,difference[i],variance[i])
            if shape_condition and variance_known else _unsupported_length(
                'highq_shape_condition_failed' if not shape_condition else 'unknown_photon_variance'))
        values = difference[i,valid[i]]
        summaries.append(dict(view=i,AF_band=i//2,AF_bounds_deg=list(AF_BANDS[i//2]),
            side='-' if i%2==0 else '+',
            supported_q_bins=int(valid[i].sum()),negative_difference_q_bins=int((values<0).sum()),
            negative_difference_fraction=float((values<0).mean()) if len(values) else None,
            lowq_signed_excess_integral=float(np.sum(difference[i,low]*dq[low])) if low.any() else None,
            lowq_integral_conditional_se_fixed_c=float(np.sqrt(np.sum(variance[i,low]*dq[low]**2)))
                if variance_known and low.any() else None,
            length=length,length_interval_conditions_on_fitted_c=True))
    lengths=[s['length']['xi_nm'] for s in summaries if s['length']['xi_nm'] is not None]
    return dict(status='conditional_reference_subtraction' if shape_condition else
            'shape_condition_failed_diagnostic_subtraction_only',
        applied=bool(shape_condition),shape_condition_met=bool(shape_condition),
        difference=difference,ratio=ratio,valid=valid,
        variance_fixed_c=variance,difference_se_fixed_c=np.sqrt(variance) if variance_known else None,
        negative_difference_clipped=False,summaries=summaries,
        length_view_range_nm=[min(lengths),max(lengths)] if lengths else None,
        length_interpretation='conditional scattering-change length only; not a metal, pore or host particle size',
        conditional_variance_definition='Var(rcur)+c^2 Var(rref), given fitted scalar c',
        c_estimation_and_shared_reference_covariance_not_in_DB_interval=True,
        signed_difference_is_CountFrame=False,Poisson_feature_fitter_used=False)


def reference_change(current, reference, *, geometry_identity=None,
                     current_source=None, reference_source=None,
                     common_pixel_support_verified=False):
    'Analyze same-geometry paired frames without modifying either frame.'
    base = dict(schema=SCHEMA,source_pins=_source_pins(),
        geometry_identity=geometry_identity,current_source=current_source,reference_source=reference_source,
        common_pixel_support_verified=bool(common_pixel_support_verified),
        caller_must_use_same_geometry=True,prestate_is_known_zero_metal=False,
        highq_interval_nm_inv=list(HIGHQ_INTERVAL),highq_interval_adapted=False,
        platform_relative_slope_limit=PLATFORM_RELATIVE_SLOPE_LIMIT,
        minimum_platform_q_bins=MINIMUM_PLATFORM_Q_BINS,
        all_AF_both_sides_one_global_scale=True,
        signed_difference_is_CountFrame=False,raw_counts_unchanged=True,
        TEST_read=False,absolute_metal_amount_inferred=False)
    if reference is None:
        return dict(base,status='not_provided',reason='No reference frame supplied; primary analysis continues',
                    raw_change=None,conditional_subtraction=None,global_scale=None)
    if not isinstance(current,CountFrame) or not isinstance(reference,CountFrame):
        raise TypeError('Two original same-geometry CountFrames required')
    if not isinstance(common_pixel_support_verified,(bool,np.bool_)):
        raise ValueError('Common detector-pixel support verification must be explicit boolean')
    known_exposure=current.exposure_s is not None and reference.exposure_s is not None
    tc=float(current.exposure_s) if known_exposure else 1.
    tr=float(reference.exposure_s) if known_exposure else 1.
    common=(current.P>0)&(reference.P>0)
    y=np.where(common,current.Y,0.); yr=np.where(common,reference.Y,0.)
    n=np.where(common,current.P*tc,0.); nr=np.where(common,reference.P*tr,0.)
    rc=np.divide(y,n,out=np.zeros_like(y),where=common)
    rr=np.divide(yr,nr,out=np.zeros_like(yr),where=common)
    variance_known=bool(current.photon_counts_known and reference.photon_counts_known)
    high=(Q>=HIGHQ_INTERVAL[0])&(Q<HIGHQ_INTERVAL[1])
    high_atomic=common&high[None,None,:]
    numerator=float(y[high_atomic].sum())
    expected=n*rr; denominator=float(expected[high_atomic].sum())
    c=numerator/denominator if denominator>0 else None
    actual_q=y.sum(axis=(0,1)); reference_expected_q=expected.sum(axis=(0,1))
    global_ratio=np.divide(actual_q,reference_expected_q,out=np.full(len(Q),np.nan),
        where=reference_expected_q>0)
    usable=high&np.isfinite(global_ratio)
    slope=None
    if usable.sum()>=MINIMUM_PLATFORM_Q_BINS and c is not None and c>0:
        slope=float(abs(np.polyfit(Q[usable],global_ratio[usable],1)[0])
                    *(HIGHQ_INTERVAL[1]-HIGHQ_INTERVAL[0])/c)
    shape_condition=bool(slope is not None and slope<=PLATFORM_RELATIVE_SLOPE_LIMIT)
    c_variance=None; variance_numerator=None; variance_denominator=None
    if variance_known and denominator>0:
        weights=np.divide(n,nr,out=np.zeros_like(n),where=common)
        variance_numerator=float(y[high_atomic].sum())
        variance_denominator=float((weights*weights*yr)[high_atomic].sum())
        c_variance=variance_numerator/denominator**2+numerator**2*variance_denominator/denominator**4
    hy,hn,_,_,_,_=counts_profiles(y,n)
    rhy,rhn,_,_,_,_=counts_profiles(yr,nr)
    raw=reference_changes(Q,hy,hn,rhy,rhn)
    if not variance_known:_unknown_variance(raw)
    raw.update(intensity_unit='detector_units_per_pixel_second' if known_exposure else 'detector_units_per_pixel_frame',
        actual_exposures_normalized=known_exposure,relative_exposure_or_dose_unknown=not known_exposure,
        photon_variance_known=variance_known,
        interpretation='Signed observed reference change; flux/attenuation/motion/host response are not separated from metal scattering')
    conditional=_conditional_profiles(raw,hy,hn,rhy,rhn,c,shape_condition,variance_known)
    cur_var=np.divide(y,n*n,out=np.zeros_like(y),where=common) if variance_known else None
    ref_var=np.divide(yr,nr*nr,out=np.zeros_like(yr),where=common) if variance_known else None
    profile_ref_var=np.divide(rhy,rhn*rhn,out=np.full_like(rhy,np.nan),where=(hn>0)&(rhn>0)) if variance_known else None
    return dict(base,status='paired_signed_change_complete',
        current_frame_identity=_frame_identity(current),reference_frame_identity=_frame_identity(reference),
        common_native_support_sha256=_array_sha(common),common_native_supported_cells=int(common.sum()),
        current_supported_native_cells=int((current.P>0).sum()),reference_supported_native_cells=int((reference.P>0).sum()),
        pixel_frame_multiplicities_equal_on_common=bool(np.array_equal(current.P[common],reference.P[common])),
        common_support_scope='actual detector pixel intersection supplied by caller' if common_pixel_support_verified else
            'native angular cell intersection only; actual pixel identity not established in CountFrame',
        actual_exposures_normalized=known_exposure,current_exposure_s=current.exposure_s,
        reference_exposure_s=reference.exposure_s,partial_exposure_not_used=not known_exposure,
        intensity_unit=raw['intensity_unit'],relative_exposure_or_dose_unknown=not known_exposure,
        I0_attenuation_capture_unknown=True,raw_change=raw,conditional_subtraction=conditional,
        global_scale=dict(c=c,shape_condition_met=shape_condition,
            formula='sum(Ycur/t_cur)/sum(Pcur*rref)' if known_exposure else 'sum(Ycur)/sum(Pcur*rref)',
            rref_definition='Yref/(Pref*t_ref)' if known_exposure else 'Yref/Pref',
            estimate_source='fixed highq region, all common native AF cells and both sides',
            original_photon_numerator=numerator,reference_expected_photon_denominator=denominator,
            count_support_applications=1,highq_supported_native_cells=int(high_atomic.sum()),
            highq_comparable_ratio_q_bins=int(usable.sum()),q_nm_inv=Q.copy(),
            current_to_reference_expected_ratio=global_ratio,relative_slope=slope,
            c_conditional_variance=c_variance,c_conditional_se=float(np.sqrt(c_variance)) if c_variance is not None else None,
            numerator_conditional_variance=variance_numerator,denominator_conditional_variance=variance_denominator,
            uncertainty_scope='independent Poisson delta method, excludes flux/systematics' if variance_known else 'unknown photon variance',
            reason=None if shape_condition else 'nonflat_or_insufficient_fixed_highq_ratio; range remains fixed'),
        covariance=dict(photon_count_semantics_known=variance_known,
            current_reference_independent_counts_assumed=True,
            shared_reference_windows_not_independent=True,
            shared_reference_identity=_frame_identity(reference),
            shared_reference_profile_variance=profile_ref_var,
            shared_reference_covariance_given_fixed_scales='Cov(delta_t(q),delta_u(q))=c_t*c_u*Var(rref(q)) for independent current counts',
            c_estimation_common_lowq_covariance='Var(c)*rref(q_i)*rref(q_j); a rank-one contribution, not included in conditional DB interval',
            c_conditional_variance=c_variance,
            c_current_covariance_formula='Cov(c,rcur_j)=Ycur_j/(T*Ncur_j) on highq support,0 elsewhere',
            c_reference_covariance_formula='Cov(c,rref_j)=-c*Ncur_j*Var(rref_j)/T on highq support,0 elsewhere',
            covariance_formula_symbols='N=P*actual exposure when both known, otherwise P; T=sum_highq(Ncur*rref)',
            native_current_rate_variance_sha256=_array_sha(cur_var) if cur_var is not None else None,
            native_reference_rate_variance_sha256=_array_sha(ref_var) if ref_var is not None else None),
        reference_known_zero_metal=False,metal_only_scattering_extracted=False,
        differences_clipped=False,conditional_length_is_structure_size=False)
