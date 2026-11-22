"""The ordered edits of the user mask. Include restores a pixel to detector validity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


RECTANGLE, POLYGON = "rectangle", "polygon"
SHAPES = (RECTANGLE, POLYGON, "circle", "annulus", "ellipse", "brush", "eraser", "q_annulus")
RULES = ("above", "below", "saturated", "negative", "grow", "shrink", "invert", "clear")


def _sources(paths) -> tuple[str, ...]:
    if isinstance(paths, str):
        raise ValueError("Mask sources must be a list of paths.")
    return tuple(str(path) for path in paths)


@dataclass(frozen=True)
class MaskShape:
    kind: str
    points: tuple[tuple[float, float], ...]
    """Canonical detector coordinates, or a single (inner, outer) pair in Å⁻¹ for q_annulus."""
    mode: str = "exclude"
    radius: float = 1.0
    """Brush or eraser radius in detector pixels."""
    sources: tuple[str, ...] = ()
    """JSON mask files from which this edit was imported, oldest first."""

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.points)
        if self.kind not in SHAPES:
            raise ValueError(f"Unknown mask shape {self.kind!r}.")
        if self.mode not in ("exclude", "include"):
            raise ValueError("Choose whether to include or exclude pixels.")
        if not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("Mask coordinates must be finite.")
        if self.kind == RECTANGLE and len(points) != 2:
            raise ValueError("A rectangle mask needs two corners.")
        if self.kind == POLYGON and len(points) < 3:
            raise ValueError("A polygon mask needs at least three vertices.")
        if self.kind == "ellipse" and (len(points) != 2 or any(a == b for a, b in zip(*points))):
            raise ValueError("An ellipse needs two opposite corners with positive width and height.")
        if self.kind == "circle" and (len(points) != 2 or points[0] == points[1]):
            raise ValueError("A circle needs a centre and a point on its rim.")
        if self.kind == "annulus" and (len(points) != 3 or
                math.dist(points[0], points[1]) >= math.dist(points[0], points[2])):
            raise ValueError("Choose the centre, inner rim and outer rim in that order.")
        if self.kind == "q_annulus" and (len(points) != 1 or not 0 <= points[0][0] < points[0][1]):
            raise ValueError("The q limits must satisfy 0 ≤ inner q < outer q.")
        if self.kind in ("brush", "eraser") and not points:
            raise ValueError("A stroke needs at least one point.")
        radius = float(self.radius) if self.kind in ("brush", "eraser") else 1.0
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("The brush radius must be finite and positive.")
        if self.kind == "eraser":
            object.__setattr__(self, "mode", "include")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "sources", _sources(self.sources))

    def describe(self) -> str:
        if self.kind == "q_annulus":
            low, high = self.points[0]
            return f"Beam-centred annulus: {low:g}–{high:g} Å⁻¹"
        xs, ys = zip(*self.points)
        what = f"Polygon ({len(self.points)} points)" if self.kind == POLYGON else self.kind.capitalize()
        return f"{what}: x {min(xs):.0f}–{max(xs):.0f}, y {min(ys):.0f}–{max(ys):.0f}"


@dataclass(frozen=True)
class MaskRule:
    kind: str
    value: float | None = None
    relative: bool = False
    """For above/below, multiply the median of the remaining valid pixels by value."""
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in RULES:
            raise ValueError(f"Unknown mask rule {self.kind!r}.")
        if not isinstance(self.relative, bool) or self.relative and self.kind not in ("above", "below"):
            raise ValueError("Only above and below thresholds can use the median.")
        if self.kind in ("above", "below", "saturated", "grow", "shrink"):
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError("Enter a finite mask threshold or distance.")
            value = float(self.value)
            if self.kind in ("grow", "shrink") and (not value.is_integer() or value < 0):
                raise ValueError("Mask growth and shrinkage need a non-negative integer radius.")
            object.__setattr__(self, "value", value)
        object.__setattr__(self, "sources", _sources(self.sources))


@dataclass(frozen=True)
class MaskImage:
    path: str
    sources: tuple[str, ...] = ()
    kind: str = field(default="image", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("A mask image needs a file path.")
        object.__setattr__(self, "sources", _sources(self.sources))


MaskEdit = MaskShape | MaskRule | MaskImage


def mask_edit_record(edit: MaskEdit) -> dict:
    """Compact JSON form shared by settings, mask recipes and the reduction record."""
    record = {"kind": edit.kind}
    if isinstance(edit, MaskShape):
        record["points"] = [list(point) for point in edit.points]
        if edit.mode != "exclude":
            record["mode"] = edit.mode
        if edit.kind in ("brush", "eraser"):
            record["radius"] = edit.radius
    elif isinstance(edit, MaskImage):
        record["path"] = edit.path
    else:
        if edit.value is not None:
            record["value"] = edit.value
        if edit.relative:
            record["relative"] = True
    if edit.sources:
        record["sources"] = list(edit.sources)
    return record


def mask_edit_from_record(record) -> MaskEdit:
    kind = record["kind"]
    sources = _sources(record.get("sources", ()))
    if kind == "image":
        return MaskImage(record["path"], sources)
    if kind in RULES:
        return MaskRule(kind, record.get("value"), record.get("relative", False), sources)
    return MaskShape(kind, record["points"], record.get("mode", "exclude"), record.get("radius", 1.0), sources)
