import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from controllers.multifile_predict_results import (
    MultiFilePredictResultsWidget as LegacyMultiFilePredictResultsWidget,
)
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


def test_legacy_multifile_results_path_reexports_prediction_owner() -> None:
    assert LegacyMultiFilePredictResultsWidget is MultiFilePredictResultsWidget


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
