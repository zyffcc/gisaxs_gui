"""Fitting's step rail: Curve → Model → Fit → Results, with the state of each and a click to go there."""

from __future__ import annotations

import numpy as np

from src.gimap.features.fitting.presentation.fit_steps import components_text, drawn_solution


def test_components_are_counted() -> None:
    assert components_text(["sphere", "sphere", "sphere"]) == "Sphere ×3"
    assert components_text(["sphere", "cylinder", "cylinder"]) == "Sphere + Cylinder ×2"
    assert components_text(["vertical cylinder"]) == "Vertical Cylinder"


def test_a_drawn_workflow_solution_names_the_model() -> None:
    class Binding:
        fitting = {"meta": {"source": "native_v5", "candidate": {"combination": "random_cylinder"}}}

    assert drawn_solution(Binding()) == "random cylinder"  # not the component choosers, which it did not change
    Binding.fitting = {"meta": {"source": "manual"}}
    assert drawn_solution(Binding()) == "" and drawn_solution(None) == ""
