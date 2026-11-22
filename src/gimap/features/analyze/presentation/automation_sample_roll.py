"""Apply a frame measurement to the same Analyze controls as a manual edit."""

from __future__ import annotations

from functools import partial

from ..application.sample_roll import measure_frame_roll


class SampleRollAutomationMixin:
    @staticmethod
    def sample_roll_status(analysis):
        return {"sample_roll_override": getattr(analysis.request, "sample_roll_deg", None),
                "sample_roll_measurement": analysis.sample_roll_measurement}

    def prepare_sample_roll(self):
        model = self._page.view_model
        analysis = model.state.analysis
        return analysis, partial(measure_frame_roll, analysis, model.frames)

    def set_sample_roll(self, degrees, measurement, original, done):
        page = self._page
        model = page.view_model
        if original is not None:
            if model.state.analysis is not original or model.request() != original.request:
                done(False, "The frame or settings changed while measuring sample roll.")
                return
            measurement = dict(measurement, origin="automatic", path=str(original.path),
                               frame_index=original.frame_index,
                               summed_frames=[[str(path), index] for path, index in original.summed_frames])
        model.set_sample_roll(degrees, measurement)
        page._sync_sample_roll_widgets()
        page._remember()
        self.reanalyse(done)
