"""Details ▸ Growth Process: the growth process of the in-situ result shown (a mixin of ``SputterPage``).

From the analysis package (``ClientResult.process``; a newer package only), on the solutions shown by default:

- **Stages**: islands, coalescence, continuous film, after the deposition — each with its time range and metal amount.
- **Growth laws**: when the coverage reaches 0.5, when the film forms, D = D₁ h^p and R = R₁ h^q before the film,
  the distance once the film has formed, the coarsening after the deposition stopped …, each with the range of the
  same metal on the same substrate over the scans the package was built from and where this scan lies in it.
- **Components**: per window the share of each physical term of the model (grains, film, substrate …) in the
  counts, for the solution shown (it follows the choices of the main screen).

The metal amount h is the equivalent thickness of the metal (nm): it does not depend on the X-ray wavelength. An older
result, or a series of static samples, says in one line why there is nothing here.
"""

from __future__ import annotations

import html

from src.gimap.app.presentation.i18n import tr

from ..application import COMPONENT_NAMES, STAGE_NAMES
from .client_texts import shown_number
from .components_view import percent
from .texts import text

PROCESS_TITLE_STAGES = "Stages of the growth"
PROCESS_TITLE_LAWS = "Growth laws"
PROCESS_TITLE_COMPONENTS = "Physical terms of the model"
PROCESS_SOURCE = "From the solutions: {source}"
PROCESS_SOURCES = ("joint fit of the whole scan", "smooth path through the windows' solutions")
COLUMN_STAGE, COLUMN_STAGE_TIME, COLUMN_AMOUNT = "Stage", "Time (s)", "Metal amount (nm)"
COLUMN_QUANTITY, COLUMN_THIS, COLUMN_SYSTEM, COLUMN_POSITION = "Quantity", "This scan", "{system}, {scans} scans", "Here"
SYSTEM_NONE = "Same metal and substrate"
POSITION_WORDS = {"within": "within the range", "below": "below the range", "above": "above the range"}
RANGE_CELL = "{median} ({low} – {high})"
PROCESS_NOTE = ("Metal amounts are equivalent thicknesses of the metal (nm), independent of the X-ray wavelength. The "
                "range is that of the same metal on the same substrate over the scans the analysis package was "
                "built from (development reference: no independent characterisation).")
NO_PROCESS = "This result has no analysis of the growth process: analyze the scan again with the current analysis package."
NO_PROCESS_STATIC = "The analysis of the growth process is made for in-situ scans."
COLUMN_WINDOW = "Time (s)"
TEMPLATES = (PROCESS_TITLE_STAGES, PROCESS_TITLE_LAWS, PROCESS_TITLE_COMPONENTS, PROCESS_SOURCE, *PROCESS_SOURCES,
             COLUMN_STAGE, COLUMN_STAGE_TIME, COLUMN_AMOUNT, COLUMN_QUANTITY, COLUMN_THIS, COLUMN_SYSTEM,
             COLUMN_POSITION, SYSTEM_NONE, *POSITION_WORDS.values(), RANGE_CELL, PROCESS_NOTE, NO_PROCESS,
             NO_PROCESS_STATIC)
TABLE = "<table cellspacing='0' cellpadding='3' style='margin-bottom:10px'>"


def _row(cells, header: bool = False) -> str:
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag} align='left'>{cell}</{tag}>" for cell in cells) + "</tr>"


def _span(low, high) -> str:
    if low is None:
        return "–"
    return shown_number(low) if high is None or abs(high - low) < 1e-9 else f"{shown_number(low)} – {shown_number(high)}"


def process_html(result, choices) -> str:
    """The Process tab of ``result`` with the solutions chosen in ``choices``."""
    process = result.process
    if process is None:
        return html.escape(tr(NO_PROCESS_STATIC if result.static else NO_PROCESS))
    source = tr(process.source) if process.source in PROCESS_SOURCES else process.source
    parts = [f"<p>{html.escape(text(PROCESS_SOURCE, source=source))}</p>"]
    if process.stages:
        rows = [_row([tr(COLUMN_STAGE), tr(COLUMN_STAGE_TIME), tr(COLUMN_AMOUNT)], header=True)]
        for stage in process.stages:
            rows.append(_row([html.escape(tr(STAGE_NAMES.get(stage.stage, stage.stage))), _span(stage.t_start, stage.t_end),
                              _span(stage.h_start, stage.h_end)]))
        parts.append(f"<h3>{html.escape(tr(PROCESS_TITLE_STAGES))}</h3>{TABLE}{''.join(rows)}</table>")
    shown = [(key, name, unit, value) for key, name, unit, value in process.shown() if key != "h_max"]
    if shown:
        system = text(COLUMN_SYSTEM, system=process.system, scans=process.system_scans) if process.system else tr(SYSTEM_NONE)
        rows = [_row([tr(COLUMN_QUANTITY), tr(COLUMN_THIS), html.escape(system), tr(COLUMN_POSITION)], header=True)]
        for key, name, unit, value in shown:
            found = process.ranges.get(key)
            cell = text(RANGE_CELL, median=shown_number(found.median), low=shown_number(found.low),
                        high=shown_number(found.high)) if found is not None else "–"
            where = tr(POSITION_WORDS[process.positions[key]]) if key in process.positions else ""
            label = html.escape(tr(name)) + (f" ({unit})" if unit else "")
            rows.append(_row([label, shown_number(value), html.escape(cell), html.escape(where)]))
        parts.append(f"<h3>{html.escape(tr(PROCESS_TITLE_LAWS))}</h3>{TABLE}{''.join(rows)}</table>")
    if result.components:
        keys = [key for key, _name in result.components]
        rows = [_row([tr(COLUMN_WINDOW)] + [html.escape(tr(COMPONENT_NAMES.get(key, key))) for key in keys], header=True)]
        for window in result.windows:
            entry = window.chosen(choices)
            if entry is None:
                continue
            rows.append(_row([shown_number(window.time_s)] + [percent(entry.components[key]) if key in entry.components
                                                                else "–" for key in keys]))
        parts.append(f"<h3>{html.escape(tr(PROCESS_TITLE_COMPONENTS))}</h3>{TABLE}{''.join(rows)}</table>")
    parts.append(f"<p><small>{html.escape(tr(PROCESS_NOTE))}</small></p>")
    return "".join(parts)


class ProcessTabMixin:
    """Needs ``process_view`` and ``tab_process`` of ``views/sputter_page_view.py``, ``client_result`` and
    ``client_choices``."""

    def show_process_tab(self) -> None:
        result = self.client_result
        self.tab_process.setCurrentIndex(0 if result is None else 1)
        if result is not None:
            self.process_view.setHtml(process_html(result, self.client_choices))


__all__ = ["ProcessTabMixin", "TEMPLATES", "process_html"]
