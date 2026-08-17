"""Small standalone gallery for visually checking shared components。"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .components import (
    AdvancedSection,
    EmptyState,
    ErrorBanner,
    FilePicker,
    JobStatus,
    ParameterSection,
    PlotPanel,
    ResultTable,
)
from .styles import apply_design_system


class DesignSystemShowcase(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GIMaP UI Design System")
        self.resize(900, 900)
        root = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)

        input_section = ParameterSection(
            "Input",
            "Shared sections establish the same reading order in every workspace.",
            content,
        )
        input_section.add_widget(FilePicker(placeholder="Choose example input…"))
        layout.addWidget(input_section)

        advanced = AdvancedSection(
            "Advanced parameters",
            "Low-frequency values remain available without competing with the main path.",
            content,
        )
        form = QFormLayout()
        form.addRow("Optional value", QLineEdit("Default"))
        form.addRow(QCheckBox("Enable expert behavior"))
        advanced.add_layout(form)
        layout.addWidget(advanced)

        plot = PlotPanel(
            "Preview",
            "Feature code supplies the actual canvas.",
            content,
        )
        plot.setMinimumHeight(220)
        layout.addWidget(plot)

        table = ResultTable(("Name", "State", "Value"), content)
        table.set_rows((("scan_001", "Succeeded", "1.24"), ("scan_002", "Queued", "—")))
        results = ParameterSection("Results", parent=content)
        results.add_widget(table)
        layout.addWidget(results)

        job = JobStatus(content)
        job.set_state("running", "Processing scan_002", progress=0.46)
        layout.addWidget(job)
        layout.addWidget(
            ErrorBanner(
                "One input needs attention",
                "The file can remain in the list while its path is corrected.",
                content,
                level="warning",
                show_details=True,
            )
        )
        layout.addWidget(
            EmptyState(
                "No saved exports",
                "Run the workspace before exporting results.",
                content,
                action_text="Choose input",
            )
        )
        layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll)
        apply_design_system(self)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DesignSystemShowcase()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
