from __future__ import annotations

from src.gimap.app.presentation import parameter_validation


def test_parameter_validation_presentation_reports_success(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parameter_validation.QMessageBox,
        "information",
        lambda parent, title, message: calls.append((parent, title, message)),
    )

    parameter_validation.show_parameter_validation(None, (("Fitting", True, ""),))

    assert calls == [(None, "Parameter Validation", "All parameters are valid!")]


def test_parameter_validation_presentation_aggregates_only_failures(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parameter_validation.QMessageBox,
        "warning",
        lambda parent, title, message: calls.append((parent, title, message)),
    )

    parameter_validation.show_parameter_validation(
        None,
        (
            ("Fitting", False, "missing curve"),
            ("Prediction", True, ""),
            ("Trainset", False, "invalid ROI"),
        ),
    )

    assert calls == [
        (
            None,
            "Parameter Validation Failed",
            "The following parameters have issues:\n\n"
            "Fitting: missing curve\nTrainset: invalid ROI",
        )
    ]
