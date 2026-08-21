"""Workflow-oriented control rail for the WAXS workspace."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QTabWidget, QVBoxLayout, QWidget


def _scroll_step(section: QWidget, object_name: str) -> QScrollArea:
    content = QWidget()
    content.setObjectName(f"{object_name}Content")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    section.setParent(content)
    layout.addWidget(section)
    layout.addStretch(1)

    scroll = QScrollArea()
    scroll.setObjectName(object_name)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


def install_waxs_workflow(page) -> None:
    """Replace one long control column with three explicit workflow tabs."""
    old_scroll = page.waxsControlsScrollArea
    workflow_tabs = QTabWidget()
    workflow_tabs.setObjectName("waxsWorkflowTabs")
    workflow_tabs.setMinimumWidth(400)
    workflow_tabs.setMaximumWidth(560)

    primary_scroll = _scroll_step(page.waxs_configure_section, "waxsControlsScrollArea")
    advanced_scroll = _scroll_step(page.waxs_advanced_section, "waxsAdvancedScrollArea")
    batch_scroll = _scroll_step(page.waxs_run_section, "waxsBatchScrollArea")
    workflow_tabs.addTab(primary_scroll, "1  Cut + integrate")
    workflow_tabs.addTab(advanced_scroll, "2  Advanced")
    workflow_tabs.addTab(batch_scroll, "3  Batch")

    page.waxs_advanced_section.set_expanded(True)
    page.waxsAdvancedToggle.hide()
    page.waxsAdvancedDescription.hide()
    page.waxsConfigureTitle.setText("Cut and integrate")
    page.waxsConfigureDescription.setText(
        "Define a reciprocal-space or geometric cut, then calculate a 1D curve."
    )
    page.waxsRunTitle.setText("Batch processing")
    page.waxsRunDescription.setText(
        "Apply the current cut and integration settings to a folder of detector files."
    )
    page.waxsInputTitle.setText("Load data")

    page.open_button.setText("Open detector file...")
    page.open_button.setProperty("waxsPrimaryAction", True)
    page.integrate_button.setText("Calculate 1D curve")
    page.integrate_button.setProperty("waxsPrimaryAction", True)
    page.batch_start_button.setText("Start batch processing")
    page.batch_start_button.setProperty("waxsPrimaryAction", True)

    page.waxsContentSplitter.replaceWidget(1, workflow_tabs)
    workflow_tabs.show()
    old_scroll.setParent(None)
    old_scroll.deleteLater()
    page.waxsControlsScrollArea = primary_scroll
    page.controls_scroll = primary_scroll
    page.waxs_workflow_tabs = workflow_tabs
    page.waxs_advanced_scroll = advanced_scroll
    page.waxs_batch_scroll = batch_scroll
    page.waxsContentSplitter.setStretchFactor(0, 1)
    page.waxsContentSplitter.setStretchFactor(1, 0)
    page.waxsContentSplitter.setSizes([1100, 460])


__all__ = ["install_waxs_workflow"]
