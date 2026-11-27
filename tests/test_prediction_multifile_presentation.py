import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.features.prediction.presentation.multifile_results import (
    DistributionHeatmapWindow,
    ExportDialog,
    MultiFilePredictResultsWidget,
    ParameterTrendWindow,
    PredictResult,
    PredictStatus,
)
from src.gimap.features.prediction.presentation.views import (
    DistributionHeatmapDialogView,
    ExportDialogView,
    MultiFileResultsWidgetView,
    ParameterTrendDialogView,
)


_TEST_APP = None


def _app() -> QApplication:
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def test_predict_result_remains_a_typed_dataclass_after_module_split() -> None:
    result = PredictResult(
        file_path="/tmp/scan_001.cbf",
        file_name="scan_001.cbf",
        status=PredictStatus.COMPLETED,
        start_time=datetime.now(),
        processing_time=1.25,
    )

    assert result.duration_str == "1.25s"
    assert result.status is PredictStatus.COMPLETED


def test_multifile_result_widgets_use_feature_python_views() -> None:
    app = _app()
    results = MultiFilePredictResultsWidget()
    export = ExportDialog(total_count=3, selected_count=0, current_count=2)
    heatmap = DistributionHeatmapWindow(results)
    trend = ParameterTrendWindow(results)

    assert isinstance(results, MultiFileResultsWidgetView)
    assert isinstance(export, ExportDialogView)
    assert isinstance(heatmap, DistributionHeatmapDialogView)
    assert isinstance(trend, ParameterTrendDialogView)
    assert results.status_filter.count() == 6
    assert results.sort_combo.count() == 4
    assert export.range_group.checkedId() == 0
    assert export.selected_radio.isEnabled() is False
    assert heatmap.plot_host.layout() is heatmap.plot_host_layout
    assert trend.plot_host.layout() is trend.plot_host_layout

    for widget in (heatmap, trend, export, results):
        widget.close()
    app.processEvents()


def test_the_counts_of_a_live_batch_are_kept_and_open_figures_follow_at_most_once_a_second() -> None:
    """The line above the table counts the rows by state as they are added and change state (kept by the table:
    counting all the rows at every change took 15 ms of the window at 50 000 rows); an open trend is drawn again
    once for the changes of a second, not at each."""
    app = _app()
    results = MultiFilePredictResultsWidget()
    drawn = []
    results.refreshParameterTrend = lambda: drawn.append("trend")
    results._windows_timer.timeout.disconnect()
    results._windows_timer.timeout.connect(lambda: results.refreshParameterTrend())
    try:
        for k in range(6):
            results.addPredictResult(f"/tmp/spray_{k:04d}.edf")
        for k, state in ((0, PredictStatus.RUNNING), (0, PredictStatus.COMPLETED), (1, PredictStatus.RUNNING),
                         (1, PredictStatus.FAILED), (2, PredictStatus.RUNNING)):
            results.updatePredictResult(k, status=state)
        counted = {state: sum(1 for result in results.table_model.results if result.status == state)
                   for state in PredictStatus}
        kept = {state: n for state, n in results.table_model.counts().items() if n}
        assert kept == {state: n for state, n in counted.items() if n}
        assert results.stats_label.text() == "Total: 6, Completed: 1, Failed: 1, Running: 1"
        assert drawn == [] and results._windows_timer.isActive()
        results._windows_timer.timeout.emit()
        assert drawn == ["trend"]
        results.clearResults()
        assert results.table_model.counts() == {} and results.stats_label.text() == "Ready"
    finally:
        results.close()
        app.processEvents()
