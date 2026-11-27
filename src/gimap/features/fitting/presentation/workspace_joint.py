"""Fitting ▸ Fit Several Curves Together… and Analyze ▸ Send Both Cuts to Fitting: one window per workspace
(``single/joint_fit_dialog.py``), kept while the workspace lives, its model the one in Single analysis."""

from __future__ import annotations

from typing import Optional, Sequence


class JointFitMixin:
    """Needs ``fit_page``, ``view_model`` and ``show_context`` (the workspace)."""

    _joint = None

    def joint_fit_dialog(self):
        if self._joint is None:
            from .single.joint_fit_dialog import JointFitDialog

            self._joint = JointFitDialog(self.fit_page, self.view_model, self.fit_page)
            self.fit_page.add_disposer(self._joint.dispose)
        return self._joint

    def open_joint_fit(self, paths: Sequence = (), geometries: Optional[Sequence] = None) -> bool:
        """The window; with ``paths``, those curves (each with its ``geometries`` entry: kind, αi, αf, λ …) in place
        of the ones listed. Returns whether every curve was added."""
        self.show_context("single")
        dialog = self.joint_fit_dialog()
        added = True
        if paths:
            dialog.specs.clear()
            dialog.result = None
            for index, path in enumerate(paths):
                geometry = (geometries or [None] * len(paths))[index]
                added = dialog.add_file(path, geometry) and added
            substrate = self.fit_page.session.model.substrate
            if substrate is not None:
                dialog.set_substrate(substrate.material, substrate.delta, substrate.beta)
        dialog.show()
        return added

    _angles = None

    def open_angle_rank(self):
        """Fitting ▸ Rank Incidence Angles…: the window, with the fit in Single analysis (``single/angle_rank_dialog.py``)."""
        self.show_context("single")
        if self._angles is None:
            from .single.angle_rank_dialog import AngleRankDialog

            self._angles = AngleRankDialog(self.fit_page, self.fit_page)
            self.fit_page.add_disposer(self._angles.dispose)
        self._angles.show()
        return self._angles

    def dispose_joint(self) -> None:
        if self._joint is not None:
            self._joint.dispose()


__all__ = ["JointFitMixin"]
