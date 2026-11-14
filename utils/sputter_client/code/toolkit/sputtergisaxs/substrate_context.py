'Optional material/architecture intake, separate from physical inference.'
from __future__ import annotations
import re


POLYMERS = {'ps', 'pmma', 'pet', 'psbma', 'polymer'}
KNOWN_OPTICS = {'Si', 'SiO2', 'TiO2', 'PS', 'PMMA'}
ARCHITECTURES = ('continuous_layer', 'particle_layer', 'porous_matrix',
                 'lateral_domains', 'no_interlayer')


def layer_context(layers=None, *, morphology=None, thickness_nm=None):
    'Return candidates/unknowns; never infer an unprovided substrate.'
    tokens = re.split(r'\s*[/|>]\s*', str(layers).strip()) if layers else []
    tokens = [t for t in tokens if t]

    host_tokens = tokens[1:] if tokens and tokens[0].lower() in ('au', 'ag') else tokens
    host = host_tokens[0] if host_tokens else None
    backing = host_tokens[-1] if len(host_tokens) >= 2 else None
    polymer = host is not None and host.lower() in POLYMERS
    notes = []
    if host is None:
        candidates = list(ARCHITECTURES)
    elif polymer:
        candidates = ['continuous_layer', 'particle_layer', 'porous_matrix', 'lateral_domains']
        notes.append('polymer_particle_morphology_is_possible; never_convert_host_to_TiO2')
    elif host.lower() == 'tio2':
        candidates = ['continuous_layer', 'particle_layer', 'porous_matrix']
    else:
        candidates = list(ARCHITECTURES)
    if morphology in ARCHITECTURES:
        candidates = [morphology]
    elif morphology is not None:
        notes.append('unrecognised_morphology_kept_unknown')
    try:
        import math
        thickness = float(thickness_nm) if thickness_nm is not None else None
        if thickness is not None and (not math.isfinite(thickness) or thickness < 0):
            raise ValueError()
    except (TypeError, ValueError):
        thickness = None
        notes.append('invalid_optional_thickness_kept_unknown')
    known_optics = host in KNOWN_OPTICS
    if polymer and not known_optics:
        notes.append('chemical_optics_not_in_current_table; no_PS_PMMA_substitution')
    return {'schema': 'optional-substrate-context-1', 'declared_stack': tokens,
            'underlying_labels': host_tokens[1:], 'host_label': host,
            'backing_label': backing, 'is_polymer_label': polymer,
            'architecture_candidates': candidates,
            'architecture_known': morphology in ARCHITECTURES,
            'interlayer_thickness_nm': thickness,
            'host_optics_table_available': known_optics,
            'notes': notes, 'structure_validated': False,
            'metal_locations_to_test': ['surface', 'embedded_or_interface_crossing']}


def analyze_with_layer_context(image, geometry, alpha_i_deg, metal, *,
                               layers=None, layer_morphology=None,
                               interlayer_thickness_nm=None, **kwargs):
    'Development wrapper: same empirical features plus optional context.'
    from .prediction import analyze_frame
    result = analyze_frame(image, geometry, alpha_i_deg, metal, layers=layers,
                           interlayer_thickness_nm=interlayer_thickness_nm, **kwargs)
    context = layer_context(layers, morphology=layer_morphology,
                            thickness_nm=interlayer_thickness_nm)
    result.profiles['substrate_context'] = context
    result.row['layer_architecture_candidates'] = context['architecture_candidates']
    result.row['layer_architecture_known'] = context['architecture_known']
    result.row['layer_backing_label'] = context['backing_label']
    result.row['layer_optics_table_available'] = context['host_optics_table_available']
    return result
