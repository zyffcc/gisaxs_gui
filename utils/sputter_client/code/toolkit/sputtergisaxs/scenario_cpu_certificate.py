'Consume actual TRAIN equivalence evidence for optional CPU caching.'
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .scenario_cached_cpu import source_receipt
from .scenario_forward import QUICK
from .v2_ppc import GATES


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def cpu_certificate(path):
    'Validate sealed original comparisons; report runtime applicability.'
    path = Path(path).resolve(); digest = _sha(path)
    cert = json.loads(path.read_text(encoding='utf8'))
    gates = {k:v/3 for k,v in GATES.items()}
    if (cert.get('schema') != 'four_scenario_exact_cached_CPU_certificate_1'
            or cert.get('passed') is not True or cert.get('fixture') is not False
            or cert.get('TEST_read') is not False or cert.get('numerical_only') is not True
            or cert.get('structural_release') is not False
            or cert.get('original_quick') != QUICK or cert.get('rate_gate') != 1e-10
            or cert.get('canonical_feature_gates') != gates):
        raise ValueError('Actual unchanged TRAIN numerical equivalence certificate required')
    if cert.get('source_pins') != source_receipt()['source_pins']:
        raise ValueError('Certified cached or original forward implementation changed')
    root = Path(__file__).parent
    expected_features = ('v2_ppc.py','v2_observation.py','stage5_measurement.py','prediction.py','features.py')
    if cert.get('feature_source_pins') != {n:_source_sha(root/n) for n in expected_features}:
        raise ValueError('Certified feature implementation changed')
    windows = cert.get('original_windows', [])
    if (not 10 <= len(windows) <= 20 or len(set(windows)) != len(windows)
            or cert.get('point_counts_per_scenario') != {S:len(windows) for S in ('S1','S2','S3','S4')}):
        raise ValueError('Original actual TRAIN point scope differs')
    points = cert.get('points', [])
    if len(points) != 4*len(windows):
        raise ValueError('Original complete per-scenario TRAIN comparisons required')
    seen = set()
    for point in points:
        key = (point.get('window'), point.get('scenario'))
        if key in seen or key[0] not in windows or key[1] not in ('S1','S2','S3','S4'):
            raise ValueError('Original comparison identity duplicated or substituted')
        seen.add(key)
        error = point.get('max_rate_error_over_scale')
        if (point.get('split') != 'TRAIN' or point.get('passed') is not True
                or point.get('draw_index') != 0 or point.get('original_draws') != 128
                or not isinstance(error,(int,float)) or isinstance(error,bool)
                or not np.isfinite(error) or not 0 <= error <= cert['rate_gate']
                or point.get('support_agreed') is not True):
            raise ValueError('Failed or replaced original TRAIN comparison')
        features = point.get('feature_errors', {})
        if set(features) != set(gates):
            raise ValueError('All original feature comparisons required')
        for name, value in features.items():
            if value.get('status') == 'originally_missing_NA':
                continue
            relative = value.get('relative')
            if (value.get('status') != 'computed' or not isinstance(relative,(int,float))
                    or isinstance(relative,bool) or not np.isfinite(relative)
                    or not 0 <= relative <= gates[name]):
                raise ValueError('Failed or unknown original feature comparison')
    enabled = cert.get('enabled_scenarios', [])
    if (not enabled or len(set(enabled)) != len(enabled)
            or any(S not in ('S1','S2','S3') for S in enabled)
            or cert.get('S4_original_forward_retained') is not True):
        raise ValueError('Explicit certified accelerated scenarios required')
    current = dict(system=platform.system(),numpy=np.__version__,scipy=scipy.__version__)
    same = cert.get('environment') == current
    if _sha(path) != digest:
        raise ValueError('Certificate changed while being read')
    return dict(schema=cert['schema'],certificate_sha256=digest,
        applicable_in_current_environment=same,enabled_scenarios=enabled if same else [],
        reason=None if same else 'original_CPU_used_in_untested_numerical_environment',
        certified_environment=cert.get('environment'),current_environment=current,
        source_pins=cert['source_pins'],precision_or_nodes_changed=False,
        posterior_draws_changed=False,structural_release=False)
