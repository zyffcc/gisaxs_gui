"""What a folder of curve files holds: the kinds of curves in it. One kind is one series.

Analyze writes several curves of every frame into one folder — I(q), the sectors, I(χ), every cut region —
and a Batch Export also one table per curve with a column per frame. Read together they are no series: the
frames of one curve are. ``FolderReader.kinds`` tells the kinds of a folder apart (the adapter knows how
Analyze names its files); a folder of curve files from elsewhere is one kind, as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

SEVERAL_KINDS = "{folder} holds {n} kinds of curves ({kinds}): choose one of them."
NO_SUCH_KIND = "{folder} holds no curves “{kind}” ({kinds})."
KIND_IN_NAME = "{name} · {kind}"
"""How a series of one kind among several is named: the sample (or batch), then the curve."""
FIT_INPUT_KIND = "fit_input"
"""The kind of the curves Analyze wrote for Fitting (``<frame>_fit_input.dat``); its ``title`` is the curve's own."""
OTHER_KIND = "other"
"""The kind of the curve files in a folder of Analyze's exports that are none of its exports (no ``title``).
What these two kinds are is said in words by the page, in the interface language."""
NO_TABLE = "no table of frames: a header row and at least three rows of x and one column per frame"
NOT_NUMBERS = "not a table of numbers ({error})"
"""Why a batch's table of frames cannot be read (``CurveReader.read_table``; ``{error}``: Python's reason)."""


@dataclass(frozen=True)
class CurveKind:
    """The frames of one curve in a folder."""

    key: str
    """What Analyze calls the curve (``radial``, ``region1_chi`` …, as its file names end); ``""``: the curve
    files of the folder as they are (no kinds to tell apart)."""
    title: str = ""
    """The curve in words, as its files or the batch's README say it (“In-plane sector I(q)”)."""
    files: tuple = ()
    """One file per frame, in counting order (empty when only ``table`` holds the frames)."""
    table: Optional[Path] = None
    """The batch's table of this curve: x in the first column, then one column per frame."""
    frames: int = 0


@dataclass(frozen=True)
class FolderContent:
    name: str
    """What a series from this folder is called: the batch or the sample, not “curves” or “fit_input”."""
    kinds: tuple = ()


class FolderReader(Protocol):
    def content(self, folder: Path, pattern: str = "*") -> FolderContent:
        """The kinds of curves in ``folder`` (its curve files matching ``pattern``)."""


class SeveralKinds(ValueError):
    """A folder holds more than one kind of curve and none was chosen; ``kinds`` are those to choose from."""

    def __init__(self, folder: str, kinds: Sequence[CurveKind]):
        self.folder, self.kinds = str(folder), tuple(kinds)
        super().__init__(SEVERAL_KINDS.format(folder=self.folder, n=len(self.kinds), kinds=kind_keys(self.kinds)))


def kind_keys(kinds: Sequence[CurveKind]) -> str:
    return ", ".join(kind.key for kind in kinds)


def chosen_kind(folder: Path, kinds: Sequence[CurveKind], key: Optional[str]) -> CurveKind:
    """The kind ``key`` of ``kinds``; the only one when there is one; ``SeveralKinds`` when one must be chosen."""
    if key:
        for kind in kinds:
            if kind.key == key:
                return kind
        raise ValueError(NO_SUCH_KIND.format(folder=Path(folder).name, kind=key, kinds=kind_keys(kinds)))
    if len(kinds) == 1:
        return kinds[0]
    raise SeveralKinds(Path(folder).name, kinds)


__all__ = ["CurveKind", "FIT_INPUT_KIND", "FolderContent", "FolderReader", "KIND_IN_NAME", "NOT_NUMBERS", "NO_SUCH_KIND",
           "NO_TABLE", "OTHER_KIND", "SEVERAL_KINDS", "SeveralKinds", "chosen_kind", "kind_keys"]
