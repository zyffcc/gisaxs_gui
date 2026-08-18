"""Qt-free access to application-owned Fitting calculation commands."""

from __future__ import annotations


class FittingScientificViewModel:
    def __init__(
        self, *, image, cut, curve, ai, refinement, insitu_cut, model=None,
        q_space=None,
    ):
        self.image = image
        self.cut = cut
        self.curve = curve
        self.ai = ai
        self.refinement = refinement
        self.insitu_cut = insitu_cut
        self.model = model
        self.q_space = q_space
