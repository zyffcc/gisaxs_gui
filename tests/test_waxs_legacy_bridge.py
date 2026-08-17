from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_window_composes_waxs_view_model():
    source = (ROOT / "ui/components/main_window_components.py").read_text(
        encoding="utf-8"
    )

    assert "create_waxs_view_model(self.ui.app_context)" in source
    assert "CalibrationWaxsImageRepository" not in source


def test_waxs_qt_workers_are_view_model_bridges():
    source = (ROOT / "ui/waxs_page.py").read_text(encoding="utf-8")
    batch = source[source.index("class BatchWorker") : source.index("class ScatteringImageViewer")]

    assert "view_model.run_batch" in batch
    assert "view_model.cancel_batch" in batch
    assert "glob.glob" not in batch
    assert "load_image_matrix" not in batch
    assert "export_curve_csv" not in batch
    assert "from src.gimap.features.waxs.domain" not in source


def test_legacy_standalone_waxs_window_was_not_rewritten():
    source = (ROOT / "WAXS/WAXS.py").read_text(encoding="utf-8")

    assert "class MainWindow" in source
    assert len(source.splitlines()) > 3000
