"""Qt presentation for cross-workspace parameter validation results."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5.QtWidgets import QMessageBox, QWidget


ValidationResult = tuple[str, bool, str]


def show_parameter_validation(
    parent: QWidget | None,
    results: Sequence[ValidationResult],
) -> None:
    invalid = [(name, message) for name, is_valid, message in results if not is_valid]
    if not invalid:
        QMessageBox.information(parent, "Parameter Validation", "All parameters are valid!")
        return

    error_messages = "\n".join(f"{name}: {message}" for name, message in invalid)
    QMessageBox.warning(
        parent,
        "Parameter Validation Failed",
        "The following parameters have issues:\n\n" + error_messages,
    )


__all__ = ["ValidationResult", "show_parameter_validation"]
