"""Qt-free access to application-owned Fitting calculation commands."""

from __future__ import annotations


class FittingScientificViewModel:
    def __init__(self, *, curve, ai, refinement, model=None):
        self.curve = curve
        self.ai = ai
        self.refinement = refinement
        self.model = model
