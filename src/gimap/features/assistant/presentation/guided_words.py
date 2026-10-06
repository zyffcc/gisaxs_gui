"""The sentences a run writes, in the interface language of the Results tab and the answer form.

The procedure (domain and application layers) writes its reasons in English, with the numbers in them: the
report, the command line and the AI read them so. Here every known form of such a sentence is an English
template (``SENTENCES``); a sentence of that form is shown as the template's translation filled with the same
values, so numbers, units, file and model names stay exactly as written. Values named in ``NESTED`` are
sentences themselves and are translated the same way. A sentence of no known form stays as it is.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, current_language, tr, trf

SENTENCES = (
    # A ring's orientation (domain/orientation.py): the texture, the notes and why there is no f.
    "{texture} — but the strongest measured |χ| ({chi}°) borders an unmeasured range, so the true maximum may lie "
    "inside it",
    "no single preferred orientation: maxima of similar height at |χ| ≈ {a}° and {b}°",
    "weak or no preferred orientation (f = {f}): the ring is strongest at |χ| ≈ {chi}° but nearly as intense elsewhere",
    "tilted: maximum at χ ≈ ±{chi}°",
    "The radial background under the ring ({background} counts/pixel) was subtracted.",
    "|χ| {where} is shadowed: the intensity there is below {fraction} of the diffuse background at this q (a shadow, "
    "absorber or insensitive detector area), so it counts as unmeasured.",
    "|χ| < {angle}° is not measured (missing wedge); f then leaves out the orientations closest to the surface normal "
    "and is biased low.",
    "A random (isotropic) ring measured over the same |χ| would give f = {f}: compare f with that, not with 0.",
    "Only {n} measured χ bins in this ring.",
    "The ring is not above the radial background: at most {snr}σ in any 1° χ bin.",
    "Only {covered} of the orientation range Herman's f weighs (sin χ) is measured at this q (|χ| {where}); f and the "
    "orientation distribution need most of it, above all near the sample plane. Within the measured part the ring is "
    "strongest at |χ| ≈ {chi}°.",
    # The halves of a GISAXS cut (domain/gisaxs_cut.py) and the curve that was fitted.
    "Only the {where} half is on the detector.",
    "The {side} half reaches only |qy| = {reach} Å⁻¹ (the {other} half {other_reach} Å⁻¹), so the {used} half is used.",
    "The {side} half has points in only {coverage} of its |qy| range (detector gaps or the beam-stop shadow; the "
    "{other} half: {other_coverage}), so the {used} half is used.",
    "The halves differ by {percent} % (median) up to |qy| = {q} Å⁻¹ even after the symmetry correction (a shadow, an "
    "absorber or real in-plane anisotropy): both are kept and the {better} half, the better covered one, is fitted.",
    "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both "
    "exist, which halves the noise{extend}.",
    "agree within {percent} %",
    "; beyond {q} Å⁻¹ the longer {side} half continues alone to {reach} Å⁻¹",
    "mean of both halves up to |qy| = {q} Å⁻¹",
    "then the {side} half alone to {q} Å⁻¹",
    "the {side} half",
    "{group} neighbouring points merged ({count} points for the fit)",
    # The points a GISAXS run asks a person to look at (application/gisaxs_procedure.py).
    "{n} solutions of {families} particle families fit within {percent} of the best χ²: the curve alone does not "
    "decide the model.",
    "Solutions whose D agrees with the observed spacing: {solutions}. Choose the particle shape you expect and refine "
    "that model in Fitting; compare the fitted curves.",
    "The symmetry axis is {shift} px from the calibrated centre: check the calibration or the sample alignment.",
    # Why a run could not go on (application/pipeline.py, pipeline_progress.py).
    "No calibration file and no image of a standard GIMaP can fit was found ({n} folders searched).",
    "GIMaP reduces this frame as {mode} and could not switch to GISAXS.",
    "GIMaP reduces this frame as {mode} and could not switch to GIWAXS.",
    "{name} has no pixel size in its header, so it cannot be calibrated.",
    "{name} and the frame give no energy.",
    "The frame changed during the run: it started on {start}, Analyze now shows {now}.",
    # How good a calibration is and which standard an image shows (domain/calibration_quality.py).
    "good: {lines} lines of the standard land within {error} of their q on average",
    "usable: {lines} lines within {error} of their q; peak positions good to about that",
    "doubtful: the standard's lines are {error} off on average; try another image or standard",
    "none fits well: the best, {standard}, still puts its lines {error} off their q; the image may show another "
    "standard, or the energy or pixel size is wrong",
    "clear: {standard} — {lines} lines within {error} of their q",
    "ambiguous: {standard} ({error}) and {second} ({second_error}) place their lines about equally well; check a log "
    "or ask the user",
    "none fits well (best {standard}: {rings} rings, rms {rms} px): the image may show another standard, the energy "
    "or pixel size may be wrong, or the rings are too weak",
    "clear: {standard} ({rings} rings, rms {rms} px) beats {second} ({second_rings} rings, rms {second_rms} px)",
    "clear: {standard} ({rings} rings, rms {rms} px)",
    "clear: {standard}, and its distance agrees with the expected {distance} mm",
    "probably {second}: its distance agrees with the expected {distance} mm, although {standard} matches as many rings",
    "ambiguous: {standard} and {second} fit about equally well; check the distance in the header or a log, or ask "
    "the user",
    # The decisions about the geometry (application/pipeline.py); file names as they are.
    "calibrated from {name}",
    "could not use {name}",
    "rejected {name} ({standard})",
    "rejected {name}",
    "{name} ({standard}) failed",
    "skipped {name}",
    "from {name}",
    "it is a {kind} file, not a calibration",
    "{shape} pixels, the frame has {frame}: another detector",
    "made for {pixel} µm pixels, the frame has {frame_pixel} µm: another detector",
    "made for {shape} pixels, the frame has {frame}",
    # Calibrations, calibrants and the pixel size the person gave (application/pipeline_geometry.py).
    "the notes name {n}: {names}",
    "compared ({name})",
    "the notes name {noted}, the file name {from_name}",
    "given; the image header says {header} µm, the given value is used",
    "did not keep the instrument profile '{name}'",
    "made for {pixel} µm pixels, the given pixel size is {given} µm",
    "The calibration given was not used ({reason}); the geometry comes from {origin} instead.",
    "The calibration given was not used ({reason}), and no other calibration gave a good geometry.",
    "GIMaP's Auto detection could not classify this frame ({kind}), so the GIWAXS procedure ran.",
)
"""English templates of the sentences a run writes; ``{name}`` is a value written as it is (or ``NESTED``).
The fixed sentences (no values) are plain keys of the table. The form with the most fixed text is tried first,
so a short form never takes a sentence of a longer one ("clear: … beats …")."""
NESTED = frozenset({"texture", "agree", "extend", "now"})
"""Values that are sentences themselves (a texture, whether the halves agree, how far one goes on, "no frame")."""
_FIELD = re.compile(r"\{(\w+)\}")


@lru_cache(maxsize=1)
def _ordered() -> tuple[str, ...]:
    return tuple(sorted(SENTENCES, key=lambda template: -len(_FIELD.sub("", template))))


@lru_cache(maxsize=None)
def _pattern(template: str) -> re.Pattern:
    pieces = _FIELD.split(template)  # text, name, text, name, …, text
    return re.compile("".join(re.escape(piece) if index % 2 == 0 else f"(?P<{piece}>.*?)"
                              for index, piece in enumerate(pieces)), re.DOTALL)


def words(text) -> str:
    """A sentence of a run in the interface language: its own translation, else that of its form
    (``SENTENCES``) with the values as written; else the sentence as it is."""
    text = str(text or "")
    if not text or current_language() == DEFAULT_LANGUAGE:
        return text
    exact = tr(text)
    if exact != text:
        return exact
    for template in _ordered():
        found = _pattern(template).fullmatch(text)
        if found is not None:
            values = {name: words(value) if name in NESTED else value for name, value in found.groupdict().items()}
            return trf(template, **values)
    return text


def parts(text, separator: str = "; ") -> str:
    """A line of parts joined by ``separator`` (the curve that was fitted), each part in the interface language."""
    pieces = [words(piece) for piece in str(text or "").split(separator)]
    joiner = "；" if current_language() == "zh" and separator.strip() == ";" else separator
    return joiner.join(pieces)


__all__ = ["NESTED", "SENTENCES", "parts", "words"]
