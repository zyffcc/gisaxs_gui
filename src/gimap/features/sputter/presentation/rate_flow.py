"""The deposition rate of the input bar: its unit, the question about an unusual rate and the nominal thickness reached
(a mixin of ``SputterPage``).

A quartz balance (QCM) shows Å/s, so the bar takes the rate in Å/s unless nm/s is chosen beside it; the request always
carries nm/s. The unit is remembered with the rate, per scan folder and as the last used one; a folder or project an
older page remembered has its rate in nm/s (it had no choice). Before Analyze runs, a rate at the sample (a QCM reading
times the package's factor) outside what sputtering gives asks once whether the unit is right. The scan's line under the
bar adds the nominal thickness the rate reaches at the last frame (rate at the sample × the time from the onset to the
last frame, or to the end of the deposition when its duration is given).
"""

from __future__ import annotations

from typing import Optional

from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.question_box import ACCEPT, REJECT, ask_choice

from ..application import QCM_FACTOR
from .client_texts import ANALYZE_ANYWAY, NOMINAL_AT_END, RATE_UNITS, UNUSUAL_RATE, UNUSUAL_RATE_TITLE, shown_number
from .texts import text

DEFAULT_UNIT = RATE_UNITS[0][1]
UNIT_SCALE = {key: scale for _shown, key, scale in RATE_UNITS}
OLD_UNIT = "nm/s"
"""The unit of a rate an older page remembered (it took nm/s only)."""


def stored_unit(values, rate=None) -> str:
    """The unit kept with ``values`` (a folder's or a project's); without one, nm/s when a rate is kept (an older
    page's), else the default."""
    unit = values.get("unit") if isinstance(values, dict) else None
    if unit in UNIT_SCALE:
        return unit
    return OLD_UNIT if rate not in (None, "") else DEFAULT_UNIT


def nm_per_s(value: Optional[float], unit: str) -> Optional[float]:
    """``value`` in ``unit`` as nm/s (12 significant digits: 2.2 Å/s is 0.22 nm/s, not 0.22000000000000003)."""
    if value is None:
        return None
    return float(f"{value * UNIT_SCALE.get(unit, 1.0):.12g}")


class RateMixin:
    """Needs the bar's ``sample_rate``, ``rate_unit``, ``qcm_check``, ``onset``, ``duration``, ``scan_info`` and the
    geometry mixin's scan line (``_summary_text``, ``_show_scan_summary``)."""

    def _connect_rate(self) -> None:
        self.rate_unit.activated.connect(lambda _index: self._client_typed_now())
        self.rate_unit.currentIndexChanged.connect(lambda _index: self._show_scan_summary())
        self.qcm_check.toggled.connect(lambda _on: self._show_scan_summary())
        for field in (self.sample_rate, self.onset, self.duration):
            field.textChanged.connect(lambda _text: self._show_scan_summary())

    def _rate_nm_s(self) -> Optional[float]:
        try:
            value = float(self.sample_rate.text().strip())
        except ValueError:
            return None
        return nm_per_s(value, self.rate_unit.currentData())

    def _confirm_rate(self, request) -> bool:
        """Ask whether the unit of an unusual rate is right: True to analyse anyway."""
        rate = request.rate_at_sample()
        answer = ask_choice(self, tr(UNUSUAL_RATE_TITLE),
                            text(UNUSUAL_RATE, nm=shown_number(rate), angstrom=shown_number(rate * 10.0)),
                            ((tr(ANALYZE_ANYWAY), ACCEPT), (tr("Cancel"), REJECT)), default=1, escape=1)
        return answer == 0

    # -- the nominal thickness at the last frame -----------------------------------------------------------------
    def _scan_seconds(self) -> Optional[float]:
        """From the first frame to the last of the scan in the picker (frames × period), when the header says it."""
        info = self.scan_info
        if info is None or info.problem or not self.header_ready():
            return None
        period = info.period_s or info.exposure_s
        return info.frames * period if info.frames and period else None

    def _nominal_text(self) -> str:
        rate, seconds = self._rate_nm_s(), self._scan_seconds()
        if rate is None or rate <= 0 or seconds is None:
            return ""
        rate *= QCM_FACTOR if self.qcm_check.isChecked() else 1.0
        values = []
        for field in (self.onset, self.duration):
            try:
                values.append(float(field.text().strip()))
            except ValueError:
                values.append(None)
        start, duration = values[0] or 0.0, values[1]
        stop = seconds if duration is None or duration <= 0 else min(seconds, start + duration)
        return text(NOMINAL_AT_END, thickness=shown_number(rate * max(0.0, stop - start)), rate=shown_number(rate))

    def _summary_text(self) -> str:
        words = super()._summary_text()
        if self._summary is not None or not words:
            return words
        nominal = self._nominal_text()
        return f"{words} · {nominal}" if nominal else words


__all__ = ["DEFAULT_UNIT", "OLD_UNIT", "RateMixin", "UNIT_SCALE", "nm_per_s", "stored_unit"]
