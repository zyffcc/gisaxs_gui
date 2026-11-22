"""Apply a frame measurement to the same Analyze controls as a manual edit."""

from __future__ import annotations

from functools import partial

from ..application.incidence_record import frame_incidence_record, incidence_choice, nominal_incidence
from ..application.sample_roll import measure_frame_angles, measure_frame_roll


class SampleRollAutomationMixin:
    @staticmethod
    def sample_roll_status(analysis):
        return {"sample_roll_override": getattr(analysis.request, "sample_roll_deg", None),
                "sample_roll_measurement": analysis.sample_roll_measurement,
                "incidence_measurement": analysis.incidence_measurement,
                "incidence_locked": getattr(analysis.request, "incidence_locked", False)}

    def prepare_frame_angles(self, roll_policy, roll_deg, critical_deg, yoneda_material=None):
        model = self._page.view_model
        frame = model.state.analysis
        fixed_roll = frame.geometry.sample_roll_deg if roll_policy == "keep" else roll_deg
        source = "run option"
        if critical_deg is None and yoneda_material is None:
            critical_deg = getattr(frame.request, "yoneda_critical_deg", None)
            yoneda_material = getattr(frame.request, "yoneda_material", None)
            source = ("Yoneda critical angle" if critical_deg is not None else
                      "material optical constants" if yoneda_material else "configured film critical angle")
        # An unresolved automatic ψ drops the previous frame override. Measure incidence
        # with the roll that will actually remain after that reduction.
        profile = frame.resolution.profile
        fallback = profile.geometry.sample_roll_deg if profile else 0.
        measure = partial(measure_frame_angles, frame, model.frames, roll_deg=fixed_roll,
                          critical_angle_deg=critical_deg, fallback_roll_deg=fallback, yoneda_material=yoneda_material)
        return frame, measure, source

    def set_frame_angles(self, frame, roll_policy, roll_deg, roll, incidence, reference, source, critical_source,
                         locked, done):
        page = self._page
        model = page.view_model
        if model.state.analysis is not frame or model.request() != frame.request:
            done(False, "The frame or settings changed during the angle measurement.")
            return
        if source is None:
            reference, source = nominal_incidence(frame)
        estimates = incidence.as_dict()
        if estimates.get("critical_angle_deg") or estimates.get("yoneda_material"):
            estimates["critical_angle_source"] = critical_source
        applied, decision = incidence_choice(reference, estimates, locked=locked)
        record = frame_incidence_record(frame, estimates, origin="automatic", reference_deg=reference,
                                        reference_source=source, applied_deg=applied, decision=decision)
        if roll_policy == "measure":
            measured_roll = dict(roll.as_dict(), origin="automatic", path=record["path"],
                                 frame_index=record["frame_index"], summed_frames=record["summed_frames"])
            model.set_sample_roll(roll.deg if roll.significant else None, measured_roll)
        elif roll_policy == "set":
            model.set_sample_roll(roll_deg)
        model.set_incidence_measurement(record)
        page._sync_sample_roll_widgets()
        page._remember()
        self.reanalyse(done)

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
