'Public detector-file intake without changing frozen producer readers.'
from collections.abc import Mapping
from contextlib import ExitStack
from numbers import Integral, Real
from pathlib import Path
import math
import re


EXPOSURE_KEYS = ('exposure_s', 'Exposure_time', 'ExposureTime', 'Exposure')
_NUMBER = re.compile(r'^\s*([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))\s*(.*?)\s*$', re.IGNORECASE)
_DESCRIPTION_LINE = re.compile(r'^\s*(exposure_s|Exposure_time|ExposureTime|Exposure)\s*=\s*(.*?)\s*$')
_UNITS = {
    's': 1., 'sec': 1., 'secs': 1., 'second': 1., 'seconds': 1., '秒': 1.,
    'ms': 1e-3, 'msec': 1e-3, 'msecs': 1e-3, 'millisecond': 1e-3, 'milliseconds': 1e-3, '毫秒': 1e-3,
    'us': 1e-6, 'usec': 1e-6, 'usecs': 1e-6, 'microsecond': 1e-6, 'microseconds': 1e-6, '微秒': 1e-6,
    'ns': 1e-9, 'nsec': 1e-9, 'nsecs': 1e-9, 'nanosecond': 1e-9, 'nanoseconds': 1e-9, '纳秒': 1e-9,
}


def _text(value):
    if isinstance(value, bytes):
        try:
            return value.decode('utf8')
        except UnicodeDecodeError:
            return value.decode('latin1')
    return str(value)


def _parse_value(value):
    result = dict(raw_value=_text(value), status='unknown', reason=None,
                  seconds=None, unit=None, unitless_seconds_assumed=False)
    if isinstance(value, bool):
        result['reason'] = 'boolean_is_not_exposure'
        return result
    if isinstance(value, Real):
        try:
            number = float(value)
        except (ValueError, OverflowError):
            result['reason'] = 'nonfinite_exposure'
            return result
        unit = ''
    else:
        match = _NUMBER.fullmatch(_text(value))
        if match is None:
            result['reason'] = 'not_a_number_with_optional_time_unit'
            return result
        number = float(match.group(1)); unit = match.group(2).strip()
    if not math.isfinite(number):
        result['reason'] = 'nonfinite_exposure'
        return result
    if number <= 0:
        result['reason'] = 'nonpositive_exposure'
        return result


    normalized = unit.casefold().replace('\u00b5', 'u').replace('\u03bc', 'u')
    if not normalized:
        multiplier = 1.; result['unitless_seconds_assumed'] = True
    elif normalized in _UNITS:
        multiplier = _UNITS[normalized]
    else:
        result.update(unit=unit, reason='unknown_exposure_unit')
        return result
    seconds = number*multiplier
    if not math.isfinite(seconds) or seconds <= 0:
        result.update(unit=unit, reason='nonfinite_or_nonpositive_seconds_after_conversion')
        return result
    result.update(status='valid', seconds=float(seconds), unit=unit or 's (unitless legacy convention)',
                  unit_multiplier_to_seconds=multiplier,
                  reason='unitless_value_assumed_seconds' if not unit else 'explicit_time_unit')
    return result


def _entries(header):
    'Read only the four exposure fields, never execute or parse arbitrary text.'
    entries = []
    for priority, key in enumerate(EXPOSURE_KEYS):
        if key in header:
            entries.append((priority, 0, key, 'header.'+key, header[key]))
        info = header.get('info')
        if isinstance(info, Mapping) and key in info:
            entries.append((priority, 1, key, 'header.info.'+key, info[key]))
    for alias_index, alias in enumerate(('imageDescription', 'ImageDescription')):
        description = header.get(alias)
        if not isinstance(description, (str, bytes)):
            continue
        for line_number, line in enumerate(_text(description).splitlines(), 1):
            match = _DESCRIPTION_LINE.fullmatch(line)
            if match is not None:
                key, value = match.groups()
                entries.append((EXPOSURE_KEYS.index(key), 2+alias_index, key,
                                'header.%s:line%d.%s' % (alias, line_number, key), value))
    return sorted(entries, key=lambda row: (row[0], row[1]))


def parse_exposure_header(header):
    'Return a compatible header and explicit exposure provenance.'
    metadata = dict(schema='public_detector_exposure_header_1', known=False,
        key_priority=list(EXPOSURE_KEYS), source_priority=['direct header', 'FabIO info', 'known key=value description line'],
        selected_key=None, selected_source=None, seconds=None, unitless_seconds_assumed=False,
        reason='missing_exposure_header', candidates=[], conflicts=[], conflict_detected=False,
        numeric_agreement_relative_tolerance=1e-12, numeric_agreement_absolute_tolerance=0.)
    result = dict(exposure_metadata=metadata)
    if header is None:
        return result
    if not isinstance(header, Mapping):
        metadata['reason'] = 'header_is_not_a_mapping'
        return result
    for _, _, key, source, raw in _entries(header):
        value = _parse_value(raw)
        value.update(key=key, source=source, selected=False)
        metadata['candidates'].append(value)
    valid = [value for value in metadata['candidates'] if value['status'] == 'valid']
    if not valid:
        if metadata['candidates']:
            metadata['reason'] = 'no_valid_exposure_value'
        return result
    selected = valid[0]; selected['selected'] = True
    metadata.update(known=True, selected_key=selected['key'], selected_source=selected['source'],
        seconds=selected['seconds'], unitless_seconds_assumed=selected['unitless_seconds_assumed'], reason=selected['reason'])
    result['exposure_s'] = selected['seconds']
    for other in valid[1:]:
        if not math.isclose(other['seconds'], selected['seconds'], rel_tol=1e-12, abs_tol=0.):
            metadata['conflicts'].append(dict(selected_key=selected['key'], selected_source=selected['source'],
                selected_seconds=selected['seconds'], ignored_key=other['key'], ignored_source=other['source'],
                ignored_seconds=other['seconds'], reason='lower_priority_valid_value_differs'))
    metadata['conflict_detected'] = bool(metadata['conflicts'])
    return result


def read_detector_image(path, frame_index=0):
    'Read one frame; preserve CBF pixels/headers and label unknown exposure.'
    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral) or frame_index < 0:
        raise ValueError('Frame index must be a nonnegative integer')
    frame_index = int(frame_index); path = Path(path); extension = path.suffix.lower()
    if extension == '.cbf':
        if frame_index != 0:
            raise ValueError('The single-frame CBF reader supports only frame_index=0')
        from .prediction import read_image
        return read_image(path, frame_index)
    if extension not in ('.tif', '.tiff', '.edf'):
        raise ValueError('Supported 2D image formats: CBF, TIFF, EDF')
    import numpy as np
    import fabio
    with ExitStack() as stack:
        opened = fabio.open(str(path))
        close = getattr(opened, 'close', None)
        if callable(close):
            stack.callback(close)
        selected = opened.getframe(frame_index) if frame_index else opened
        if selected is not opened:
            close = getattr(selected, 'close', None)
            if callable(close):
                stack.callback(close)
        image = np.array(selected.data, dtype=float, copy=True)
        image[(image < 0) | ~np.isfinite(image)] = np.nan
        header = parse_exposure_header(selected.header)
    header['exposure_metadata']['file_format'] = extension
    header['exposure_metadata']['frame_index'] = frame_index
    return image, header
