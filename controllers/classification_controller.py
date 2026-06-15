"""
Classification 控制器

职责：
- Import 区域（类别/路径/规则 + 列表填充 + 数据读取）
- 预览（右侧 QGraphicsView）
- 降维（PCA/t-SNE/UMAP）
- 分类（KNN/SVM，保存与加载模型）
- 日志（底部文本框）
"""

import os
import io
import fnmatch
import time
import re
import sys
import traceback
from functools import partial
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtCore import QThreadPool, QRunnable, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QMessageBox, QInputDialog, QMenu,
    QTableWidgetItem, QPushButton, QGraphicsScene
)
from core.global_params import global_params  # 全局参数管理器，用于持久化到 user_parameters.json
from ui.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
    move_window_to_cursor_screen,
)
from utils.path_utils import normalize_path


@dataclass
class Sample:
    file_path: str
    file_name: str
    data_type: str  # "1D" or "2D"
    category: str
    raw_data: Optional[np.ndarray] = None
    preprocessed_data: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    predicted_label: Optional[str] = None
    preprocessing_desc: str = ""


TABLE_COL_LABEL = 0
TABLE_COL_TYPE = 1
TABLE_COL_FILES = 2
TABLE_COL_LOADED = 3
TABLE_COL_SHAPE = 4
TABLE_COL_STATUS = 5
TABLE_COL_PREDICTION = 6
TABLE_COL_CONFIDENCE = 7
TABLE_COL_PREVIEW = 8
TABLE_HEADERS = [
    'Label',
    'Type',
    'Files',
    'Loaded',
    'Shape',
    'Status',
    'Prediction',
    'Confidence',
    'Preview',
]


class ClassificationController(QObject):
    """Classification控制器：实现 Import 列表、缓存、路径/规则联动等逻辑"""

    # 基础信号（保留，以便主控制器监听）
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    classification_completed = pyqtSignal(dict)

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        self.main_window = parent.parent if hasattr(parent, 'parent') else None

        # Import 缓存：name -> { 'path': str, 'rule': str }
        self.import_cache = {}
        self.current_item_name = None

        # 便捷属性（示例：Particle 1）会动态设置：self.particle1, self.particle1_rule

        # 初始化标志
        self._initialized = False
        self._in_item_changed = False
        self._rename_old_name = None
        self._in_table_item_changed = False

        # 数据与状态
        self.samples: List[Sample] = []
        self._row_to_index: Dict[int, int] = {}
        self._row_to_category: Dict[int, str] = {}
        self._path_to_index: Dict[str, int] = {}
        self._feature_crop_shape_2d: Optional[Tuple[int, int]] = None
        self._loaded_model = None
        self._last_embedding: Optional[np.ndarray] = None
        # Display panel state
        self._image_auto_scale: bool = True
        self._image_log_scale: bool = False
        self._image_vmin: Optional[float] = None
        self._image_vmax: Optional[float] = None
        self._curve_log_y: bool = False
        self._category_show_index: Dict[str, int] = {}
        self._image_cmap_name: str = 'jet'
        self._external_window = None
        self._dr_window = None
        self._last_preview_index: Optional[int] = None
        self._classification_workflow_ready = False
        self._classification_status_cards = {}
        self._classification_status_card_frames = {}
        self._classification_status_card_titles = {}
        self._classification_responsive_mode = None
        self._classification_responsive_refs = {}
        self._classification_active_tasks = []

    # ---------------------------- 初始化与连接 ----------------------------
    def initialize(self):
        if self._initialized:
            return

        self._install_classification_exception_hook()
        self._setup_connections()
        try:
            self._setup_workflow_layout()
        except Exception as e:
            self.log(f"[UI] Classification layout refactor skipped: {e}")
        # Build classification panel with 1D/2D display controls
        try:
            self._setup_classification_panel()
        except Exception as e:
            self.log(f"[UI] Preview controls setup failed: {e}")
        # Prepare table headers/columns
        try:
            self._ensure_table_headers()
        except Exception as e:
            self.log(f"[UI] Classification table setup failed: {e}")
        # 优先恢复缓存，再初始化UI，避免默认项覆盖
        try:
            params = global_params.get_module_parameters('classification')
            cache = params.get('import_cache') if isinstance(params, dict) else None
            if isinstance(cache, dict) and cache:
                self.import_cache = {k: {'path': v.get('path', ''), 'rule': v.get('rule', '*')} for k, v in cache.items()}
        except Exception as e:
            print(f"[classification] restore import_cache failed: {e}")
        self._initialize_ui()
        # 初始化后，若各类别已配置路径，则自动扫描一次以填充 Status（m/n）
        try:
            self._refresh_status_for_all_categories()
        except Exception as e:
            self.log(f"[Import] Initial category scan failed: {e}")
        self._initialized = True
        # 初始化时确保降维控件状态与当前方法一致（隐藏/显示 nNeighborsWidget 等）
        try:
            dim_method = getattr(self.ui, 'DimensionalityReductionMethodCombox', None)
            if dim_method is not None:
                self._on_dim_method_changed(dim_method.currentText())
        except Exception as e:
            self.log(f"[UI] Dimensionality controls setup failed: {e}")
        self._update_dataset_status_cards()

    def _install_classification_exception_hook(self):
        if getattr(self, '_classification_exception_hook_installed', False):
            return
        self._classification_exception_hook_installed = True
        self._previous_exception_hook = sys.excepthook

        def _hook(exc_type, exc_value, exc_tb):
            text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self._write_classification_crash_log(text)
            try:
                print(text)
            except Exception:
                pass
            try:
                self.log(f"[Crash] {exc_type.__name__}: {exc_value}")
            except Exception:
                pass
            previous = getattr(self, '_previous_exception_hook', None)
            if previous is not None and previous is not _hook:
                try:
                    previous(exc_type, exc_value, exc_tb)
                except Exception:
                    pass

        sys.excepthook = _hook

    def _write_classification_crash_log(self, text: str) -> None:
        try:
            path = os.path.join(os.getcwd(), 'classification_crash.log')
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{text}\n")
        except Exception:
            try:
                print(text)
            except Exception:
                pass

    def _log_exception(self, context: str, exc: BaseException) -> None:
        text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._write_classification_crash_log(text)
        try:
            self.log(f"{context}: {exc}")
        except Exception:
            print(f"{context}: {exc}")

    def _setup_workflow_layout(self):
        """Build a stable tabbed Classification page once.

        TODO: Auto QC.
        TODO: Include/exclude samples.
        TODO: table filtering.
        TODO: confidence threshold.
        TODO: export labels CSV.
        TODO: export selected file list.
        TODO: copy selected files.
        TODO: embedding point selection linked to table.
        """
        if self._classification_workflow_ready:
            return
        self._setup_fresh_classification_page()
        return
        root = getattr(self.ui, 'classificationGraphicsViewWidget', None)
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        label_list = getattr(self.ui, 'ClassificationImportListWidget', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
        rule_label = getattr(self.ui, 'ClassificationImportRuleLabel', None)
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        import_btn = getattr(self.ui, 'ClassificationImportImportButton', None)
        predict_btn = getattr(self.ui, 'ClassificationImportClassifyButton', None)
        graphics = getattr(self.ui, 'ClassificationGraphicsView', None)
        dr_group = getattr(self.ui, 'DimensionalityReductionGroupBox', None)
        clf_group = getattr(self.ui, 'ClassificationGroupBox', None)
        log_browser = getattr(self.ui, 'classificationPageTextBrowser', None)
        if any(widget is None for widget in (root, table, label_list, path_btn, path_edit, rule_edit, import_btn, graphics, dr_group, clf_group)):
            return

        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
            QTabWidget, QFrame, QScrollArea, QSizePolicy, QAbstractItemView,
            QFormLayout, QTextEdit, QSplitter
        )

        def _clear_layout(layout):
            if layout is None:
                return
            while layout.count():
                layout.takeAt(0)

        def _detach(widget):
            parent = widget.parentWidget() if widget is not None and hasattr(widget, 'parentWidget') else None
            layout = parent.layout() if parent is not None else None
            if layout is not None:
                for idx in reversed(range(layout.count())):
                    item = layout.itemAt(idx)
                    if item is not None and item.widget() is widget:
                        layout.takeAt(idx)
            if widget is not None:
                widget.setParent(None)
            return widget

        old_layout = root.layout()
        _clear_layout(old_layout)
        if old_layout is None:
            root_layout = QVBoxLayout(root)
            root.setLayout(root_layout)
        else:
            root_layout = old_layout
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        page_body = QWidget(root)
        page_body.setObjectName("classificationPageBody")
        main_layout = QVBoxLayout(page_body)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        try:
            from PyQt5.QtWidgets import QGridLayout
            if isinstance(root_layout, QGridLayout):
                root_layout.addWidget(page_body, 0, 0, 1, 1)
            else:
                root_layout.addWidget(page_body)
        except Exception:
            root_layout.addWidget(page_body)

        splitter = QSplitter(Qt.Vertical, page_body)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, 1)
        main_layout.setStretch(0, 1)
        self._classification_main_splitter = splitter

        tabs = QTabWidget(splitter)
        tabs.setObjectName("ClassificationMainTabs")
        tabs.setMinimumHeight(360)
        splitter.addWidget(tabs)
        self._classification_main_tabs = tabs
        root.setStyleSheet("""
            QWidget#classificationGraphicsViewWidget {
                background: #f3f6fa;
            }
            QFrame[classificationCard="true"] {
                background: #ffffff;
                border: 1px solid #cfd8e3;
                border-radius: 6px;
            }
            QLabel[classificationTitle="true"] {
                font-weight: 700;
                color: #1f2937;
            }
            QTableWidget {
                background: #ffffff;
                gridline-color: #d8e1ec;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }
            QHeaderView::section {
                background: #edf2f7;
                border: 0;
                border-right: 1px solid #cfd8e3;
                border-bottom: 1px solid #cfd8e3;
                padding: 5px;
                font-weight: 600;
            }
            QPushButton {
                min-height: 28px;
            }
        """)

        def _card(title: str):
            card = QFrame()
            card.setProperty("classificationCard", True)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(10, 8, 10, 10)
            layout.setSpacing(6)
            title_label = QLabel(title)
            title_label.setProperty("classificationTitle", True)
            layout.addWidget(title_label)
            return card, layout

        dataset_tab = QWidget()
        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setFrameShape(QFrame.NoFrame)
        dataset_content = QWidget()
        dataset_layout = QVBoxLayout(dataset_content)
        dataset_layout.setContentsMargins(8, 8, 8, 8)
        dataset_layout.setSpacing(8)
        dataset_scroll.setWidget(dataset_content)
        dataset_outer = QVBoxLayout(dataset_tab)
        dataset_outer.setContentsMargins(0, 0, 0, 0)
        dataset_outer.addWidget(dataset_scroll)

        labels_card, labels_card_layout = _card("Labels")
        labels_card_layout.addWidget(QLabel("Label"))
        label_list.setMinimumWidth(300)
        label_list.setMinimumHeight(160)
        labels_card_layout.addWidget(_detach(label_list))
        label_buttons = QHBoxLayout()
        plus_btn = getattr(self.ui, 'ClassificationImportPlusButton', None)
        minus_btn = getattr(self.ui, 'ClassificationImportMinusButton', None)
        if plus_btn is not None:
            label_buttons.addWidget(_detach(plus_btn))
        if minus_btn is not None:
            label_buttons.addWidget(_detach(minus_btn))
        label_buttons.addStretch(1)
        labels_card_layout.addLayout(label_buttons)
        dataset_layout.addWidget(labels_card)

        source_card, source_card_layout = _card("Source")
        source_form = QFormLayout()
        source_form.setContentsMargins(0, 6, 0, 0)
        source_form.setSpacing(6)
        path_row = QWidget()
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(6)
        path_edit.setMinimumWidth(220)
        path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        path_row_layout.addWidget(_detach(path_edit), 1)
        path_row_layout.addWidget(_detach(path_btn))
        source_form.addRow("Path", path_row)
        rule_edit.setMinimumWidth(220)
        rule_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_form.addRow("Rule", _detach(rule_edit))
        source_card_layout.addLayout(source_form)
        dataset_layout.addWidget(source_card)
        if rule_label is not None:
            rule_label.hide()

        actions_card, actions_card_layout = _card("Actions")
        action_row = QHBoxLayout()
        self._classification_scan_button = QPushButton("Scan Files")
        self._classification_scan_button.clicked.connect(self._on_scan_files_clicked)
        action_row.addWidget(self._classification_scan_button)
        action_row.addWidget(_detach(import_btn))
        action_row.addStretch(1)
        actions_card_layout.addLayout(action_row)
        dataset_layout.addWidget(actions_card)
        dataset_layout.addStretch(1)

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(8)
        summary_card, summary_card_layout = _card("Dataset Summary")
        stats_widget = QWidget()
        stats = QGridLayout(stats_widget)
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setHorizontalSpacing(6)
        stats.setVerticalSpacing(6)
        for key, title in (
            ('total', 'Total files'),
            ('loaded', 'Loaded files'),
            ('labels', 'Labels'),
            ('selected', 'Current label'),
        ):
            frame = QFrame()
            frame.setStyleSheet(
                "QFrame { background: #f7f9fc; border: 1px solid #dce3ee; border-radius: 6px; }"
                "QLabel { background: transparent; border: none; }"
            )
            card_layout = QVBoxLayout(frame)
            card_layout.setContentsMargins(8, 5, 8, 5)
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #596579; font-size: 10px;")
            value_label = QLabel("-")
            value_label.setStyleSheet("font-weight: 700; font-size: 12px;")
            value_label.setWordWrap(True)
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            stats.addWidget(frame, 0 if key in ('total', 'loaded') else 1, 0 if key in ('total', 'labels') else 1)
            self._classification_status_card_frames[key] = frame
            self._classification_status_card_titles[key] = title_label
            self._classification_status_cards[key] = value_label
        summary_card_layout.addWidget(stats_widget)
        table_layout.addWidget(summary_card)
        table_card, table_card_layout = _card("Dataset Table")
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(260)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table_card_layout.addWidget(_detach(table), 1)
        table_layout.addWidget(table_card, 1)

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(6)
        preview_display_card, preview_display_layout = _card("Preview Display")
        graphics.setMaximumWidth(16777215)
        graphics.setMinimumHeight(300)
        preview_display_layout.addWidget(_detach(graphics), 1)
        preview_layout.addWidget(preview_display_card, 1)
        preview_controls_card, preview_controls_layout = _card("Display Controls")
        panel = getattr(self.ui, 'ClassificationPanelWidget', None)
        if panel is None:
            panel = QWidget(preview_tab)
            panel.setObjectName('ClassificationPanelWidget')
            self.ui.ClassificationPanelWidget = panel
        preview_controls_layout.addWidget(panel)
        preview_layout.addWidget(preview_controls_card)

        embedding_tab = QWidget()
        embedding_layout = QVBoxLayout(embedding_tab)
        embedding_layout.setContentsMargins(8, 8, 8, 8)
        embedding_controls_card, embedding_controls_layout = _card("Embedding Controls")
        embedding_controls_layout.addWidget(_detach(dr_group))
        self._compact_classification_controls(dr_group)
        embedding_layout.addWidget(embedding_controls_card)
        embedding_result_card, embedding_result_layout = _card("Embedding Result / Preview")
        embedding_result_layout.addWidget(QLabel("Use Show Embedding to open the existing embedding result window."))
        embedding_layout.addWidget(embedding_result_card)
        embedding_layout.addStretch(1)

        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(8, 8, 8, 8)
        classifier_card, classifier_layout = _card("Classifier Controls")
        classifier_layout.addWidget(_detach(clf_group))
        self._compact_classification_controls(clf_group)
        if predict_btn is not None:
            classifier_layout.addWidget(_detach(predict_btn))
            predict_btn.setMaximumWidth(180)
        model_layout.addWidget(classifier_card)
        result_card, result_layout = _card("Result Summary")
        self._classification_result_text = QTextEdit()
        self._classification_result_text.setReadOnly(True)
        self._classification_result_text.setPlaceholderText(
            "Train a classifier on loaded labeled samples. Results will appear here and in the log."
        )
        self._classification_result_text.setMaximumHeight(120)
        result_layout.addWidget(self._classification_result_text)
        model_layout.addWidget(result_card)
        model_layout.addStretch(1)

        for old_container_name in ('ClassificationImportGroupBox', 'ClassificationImportWidget'):
            old_container = getattr(self.ui, old_container_name, None)
            if old_container is not None:
                old_container.hide()

        tabs.addTab(dataset_tab, "Dataset")
        tabs.addTab(table_tab, "Table / Inspect")
        tabs.addTab(preview_tab, "Preview")
        tabs.addTab(embedding_tab, "Embedding")
        tabs.addTab(model_tab, "Model / Result")

        log_card = QFrame()
        log_card.setProperty("classificationCard", True)
        log_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_card.setMinimumHeight(44)
        log_card_layout = QVBoxLayout(log_card)
        log_card_layout.setContentsMargins(8, 6, 8, 8)
        log_card_layout.setSpacing(4)
        log_header = QHBoxLayout()
        log_header.setContentsMargins(0, 0, 0, 0)
        log_title = QLabel("Operation Log")
        log_title.setProperty("classificationTitle", True)
        log_title.setMaximumHeight(20)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_controls = QHBoxLayout()
        log_controls.setContentsMargins(0, 0, 0, 0)
        log_controls.setSpacing(4)
        log_controls.addStretch(1)
        clear_log_btn = QPushButton("Clear Log")
        toggle_log_btn = QPushButton("Collapse")
        clear_log_btn.setFixedHeight(24)
        toggle_log_btn.setFixedHeight(24)
        clear_log_btn.clicked.connect(self._clear_classification_log)
        toggle_log_btn.clicked.connect(self._toggle_classification_log_panel)
        log_controls.addWidget(clear_log_btn)
        log_controls.addWidget(toggle_log_btn)
        log_header.addLayout(log_controls)
        log_card_layout.addLayout(log_header)
        if log_browser is None:
            log_browser = QTextEdit(log_card)
            log_browser.setReadOnly(True)
            self.ui.classificationPageTextBrowser = log_browser
        log_browser = _detach(log_browser)
        log_browser.setMinimumHeight(110)
        log_browser.setMaximumHeight(16777215)
        try:
            from PyQt5.QtGui import QFont
            log_browser.setFont(QFont("Consolas", 9))
        except Exception as e:
            self._log_exception('[UI] Table header setup failed', e)
        log_card_layout.addWidget(log_browser)
        self._classification_log_browser = log_browser
        old_log_area = getattr(self.ui, 'classificationPageScrollArea', None)
        if old_log_area is not None:
            old_log_area.hide()
        alt_log = getattr(self.ui, 'classificationPagetextBrowser', None)
        if alt_log is not None:
            alt_log.hide()
        self._classification_log_panel = log_card
        self._classification_log_toggle_button = toggle_log_btn
        self._classification_log_expanded = True
        self._classification_log_last_sizes = [560, 160]
        splitter.addWidget(log_card)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        QTimer.singleShot(0, lambda: splitter.setSizes([560, 160]))

        self._classification_responsive_refs = {
            'root': root,
            'tabs': tabs,
            'splitter': splitter,
            'preview_tab': preview_tab,
            'stats_layout': stats,
            'preview_layout': preview_layout,
        }
        self._rename_workflow_buttons()
        self._classification_workflow_ready = True
        self._set_table_responsive_columns('stable')
        self._update_dataset_status_cards()

    def _safe_connect(self, signal, slot):
        try:
            signal.disconnect(slot)
        except Exception:
            pass
        try:
            signal.connect(slot)
        except Exception as e:
            self._log_exception('[UI] Signal connect failed', e)

    def _setup_fresh_classification_page(self):
        """Create a clean Classification page without reparenting old .ui widgets."""
        root = getattr(self.ui, 'classificationGraphicsViewWidget', None)
        if root is None:
            return

        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
            QTabWidget, QFrame, QScrollArea, QSizePolicy, QAbstractItemView,
            QFormLayout, QTextEdit, QSplitter, QListWidget, QLineEdit,
            QTableWidget, QGraphicsView, QComboBox, QSpinBox, QTextBrowser
        )
        from PyQt5.QtGui import QFont

        def _clear_root_layout():
            layout = root.layout()
            if layout is None:
                layout = QVBoxLayout(root)
                root.setLayout(layout)
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.hide()
                    widget.setParent(None)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            return layout

        def _card(title: str):
            card = QFrame()
            card.setProperty("classificationCard", True)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(10, 8, 10, 10)
            layout.setSpacing(8)
            title_label = QLabel(title)
            title_label.setProperty("classificationTitle", True)
            layout.addWidget(title_label)
            return card, layout

        root_layout = _clear_root_layout()
        page = QWidget(root)
        page.setObjectName("ClassificationPageRoot")
        root_layout.addWidget(page)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        root.setStyleSheet("""
            QWidget#ClassificationPageRoot, QWidget#classificationGraphicsViewWidget {
                background: #f3f6fa;
            }
            QFrame[classificationCard="true"] {
                background: #ffffff;
                border: 1px solid #cfd8e3;
                border-radius: 6px;
            }
            QLabel[classificationTitle="true"] {
                font-weight: 700;
                color: #1f2937;
            }
            QTableWidget {
                background: #ffffff;
                gridline-color: #d8e1ec;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }
            QHeaderView::section {
                background: #edf2f7;
                border: 0;
                border-right: 1px solid #cfd8e3;
                border-bottom: 1px solid #cfd8e3;
                padding: 5px;
                font-weight: 600;
            }
            QPushButton {
                min-height: 28px;
            }
        """)

        splitter = QSplitter(Qt.Vertical, page)
        splitter.setChildrenCollapsible(False)
        page_layout.addWidget(splitter, 1)
        self._classification_main_splitter = splitter

        tabs = QTabWidget(splitter)
        tabs.setObjectName("ClassificationMainTabs")
        splitter.addWidget(tabs)
        self._classification_main_tabs = tabs

        # Dataset tab
        dataset_tab = QWidget()
        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setFrameShape(QFrame.NoFrame)
        dataset_content = QWidget()
        dataset_layout = QVBoxLayout(dataset_content)
        dataset_layout.setContentsMargins(8, 8, 8, 8)
        dataset_layout.setSpacing(8)
        dataset_scroll.setWidget(dataset_content)
        dataset_outer = QVBoxLayout(dataset_tab)
        dataset_outer.setContentsMargins(0, 0, 0, 0)
        dataset_outer.addWidget(dataset_scroll)

        labels_card, labels_layout = _card("Labels")
        label_list = QListWidget()
        label_list.setObjectName("ClassificationImportListWidget")
        label_list.setMaximumHeight(220)
        label_list.setMinimumHeight(150)
        labels_layout.addWidget(label_list)
        label_button_row = QHBoxLayout()
        add_label_btn = QPushButton("Add Label")
        remove_label_btn = QPushButton("Remove Label")
        label_button_row.addWidget(add_label_btn)
        label_button_row.addWidget(remove_label_btn)
        label_button_row.addStretch(1)
        labels_layout.addLayout(label_button_row)
        dataset_layout.addWidget(labels_card)

        source_card, source_layout = _card("Source")
        source_form = QFormLayout()
        source_form.setContentsMargins(0, 0, 0, 0)
        source_form.setSpacing(8)
        path_row = QWidget()
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(6)
        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Folder or file path")
        choose_folder_btn = QPushButton("Choose Folder")
        path_row_layout.addWidget(path_edit, 1)
        path_row_layout.addWidget(choose_folder_btn)
        rule_edit = QLineEdit("*")
        source_form.addRow("Path", path_row)
        source_form.addRow("Rule", rule_edit)
        source_layout.addLayout(source_form)
        source_actions = QHBoxLayout()
        scan_btn = QPushButton("Scan Files")
        import_btn = QPushButton("Import Selected")
        source_actions.addWidget(scan_btn)
        source_actions.addWidget(import_btn)
        source_actions.addStretch(1)
        source_layout.addLayout(source_actions)
        dataset_layout.addWidget(source_card)
        dataset_layout.addStretch(1)

        # Table tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(8)
        summary_card, summary_layout = _card("Dataset Summary")
        stats_widget = QWidget()
        stats = QGridLayout(stats_widget)
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setHorizontalSpacing(6)
        stats.setVerticalSpacing(6)
        self._classification_status_cards = {}
        self._classification_status_card_frames = {}
        self._classification_status_card_titles = {}
        for pos, (key, title) in enumerate((
            ('total', 'Total files'),
            ('loaded', 'Loaded files'),
            ('labels', 'Labels'),
            ('selected', 'Current label'),
        )):
            frame = QFrame()
            frame.setStyleSheet(
                "QFrame { background: #f7f9fc; border: 1px solid #dce3ee; border-radius: 6px; }"
                "QLabel { background: transparent; border: none; }"
            )
            card_layout = QVBoxLayout(frame)
            card_layout.setContentsMargins(8, 5, 8, 5)
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #596579; font-size: 10px;")
            value_label = QLabel("-")
            value_label.setStyleSheet("font-weight: 700; font-size: 12px;")
            value_label.setWordWrap(True)
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            stats.addWidget(frame, pos // 2, pos % 2)
            self._classification_status_card_frames[key] = frame
            self._classification_status_card_titles[key] = title_label
            self._classification_status_cards[key] = value_label
        summary_layout.addWidget(stats_widget)
        table_layout.addWidget(summary_card)

        table_card, table_card_layout = _card("Dataset Table")
        table = QTableWidget()
        table.setObjectName("ClassificationImportTableWidget")
        table.setColumnCount(len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(300)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table_card_layout.addWidget(table, 1)
        table_layout.addWidget(table_card, 1)

        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)
        preview_card, preview_card_layout = _card("Preview Display")
        graphics = QGraphicsView()
        graphics.setObjectName("ClassificationGraphicsView")
        graphics.setMinimumHeight(300)
        graphics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_card_layout.addWidget(graphics, 1)
        preview_layout.addWidget(preview_card, 1)
        controls_card, controls_layout = _card("Display Controls")
        panel = QWidget()
        panel.setObjectName("ClassificationPanelWidget")
        controls_layout.addWidget(panel)
        preview_layout.addWidget(controls_card)

        # Embedding tab
        embedding_tab = QWidget()
        embedding_layout = QVBoxLayout(embedding_tab)
        embedding_layout.setContentsMargins(8, 8, 8, 8)
        embedding_layout.setSpacing(8)
        embedding_card, embedding_card_layout = _card("Embedding Controls")
        embedding_form = QFormLayout()
        embedding_form.setContentsMargins(0, 0, 0, 0)
        embedding_form.setSpacing(8)
        dim_method = QComboBox()
        dim_method.addItems(["PCA", "t-SNE", "UMAP"])
        dim_method.setMaximumWidth(220)
        target_dim = QSpinBox()
        target_dim.setRange(1, 50)
        target_dim.setValue(2)
        target_dim.setMaximumWidth(120)
        n_widget = QWidget()
        n_layout = QHBoxLayout(n_widget)
        n_layout.setContentsMargins(0, 0, 0, 0)
        n_label = QLabel("n_neighbors")
        n_spin = QSpinBox()
        n_spin.setRange(2, 200)
        n_spin.setValue(15)
        n_spin.setMaximumWidth(120)
        n_layout.addWidget(n_label)
        n_layout.addWidget(n_spin)
        n_layout.addStretch(1)
        embedding_form.addRow("Method", dim_method)
        target_dim_label = QLabel("Target Dim")
        embedding_form.addRow(target_dim_label, target_dim)
        embedding_form.addRow("", n_widget)
        embedding_card_layout.addLayout(embedding_form)
        embedding_buttons = QHBoxLayout()
        run_embedding_btn = QPushButton("Run Embedding")
        show_embedding_btn = QPushButton("Show Embedding")
        self._dr_status_label = QLabel("●")
        self._dr_status_label.setFixedWidth(16)
        embedding_buttons.addWidget(run_embedding_btn)
        embedding_buttons.addWidget(show_embedding_btn)
        embedding_buttons.addWidget(self._dr_status_label)
        embedding_buttons.addStretch(1)
        embedding_card_layout.addLayout(embedding_buttons)
        embedding_layout.addWidget(embedding_card)
        embedding_result_card, embedding_result_layout = _card("Embedding Result / Preview")
        embedding_result_layout.addWidget(QLabel("Run embedding, then use Show Embedding to inspect the result."))
        embedding_layout.addWidget(embedding_result_card)
        embedding_layout.addStretch(1)

        # Model tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(8)
        model_card, model_card_layout = _card("Classifier Controls")
        model_form = QFormLayout()
        model_form.setContentsMargins(0, 0, 0, 0)
        model_form.setSpacing(8)
        clf_method = QComboBox()
        clf_method.addItems(["KNN", "SVM"])
        clf_method.setMaximumWidth(220)
        clf_param_label = QLabel("n_neighbors:")
        clf_param_edit = QLineEdit("5")
        clf_param_edit.setMaximumWidth(120)
        model_form.addRow("Method", clf_method)
        model_form.addRow(clf_param_label, clf_param_edit)
        model_card_layout.addLayout(model_form)
        model_buttons = QHBoxLayout()
        train_btn = QPushButton("Train Classifier")
        predict_btn = QPushButton("Predict")
        save_btn = QPushButton("Save Model")
        load_btn = QPushButton("Load Model")
        for button in (train_btn, predict_btn, save_btn, load_btn):
            button.setMaximumWidth(160)
            model_buttons.addWidget(button)
        model_buttons.addStretch(1)
        model_card_layout.addLayout(model_buttons)
        model_layout.addWidget(model_card)
        result_card, result_layout = _card("Result Summary")
        self._classification_result_text = QTextEdit()
        self._classification_result_text.setReadOnly(True)
        self._classification_result_text.setPlaceholderText(
            "Train a classifier on loaded labeled samples. Results will appear here and in the log."
        )
        self._classification_result_text.setMinimumHeight(110)
        result_layout.addWidget(self._classification_result_text)
        model_layout.addWidget(result_card)
        model_layout.addStretch(1)

        tabs.addTab(dataset_tab, "Dataset")
        tabs.addTab(table_tab, "Table / Inspect")
        tabs.addTab(preview_tab, "Preview")
        tabs.addTab(embedding_tab, "Embedding")
        tabs.addTab(model_tab, "Model / Result")

        # Operation log
        log_panel = QFrame()
        log_panel.setProperty("classificationCard", True)
        log_panel.setMinimumHeight(140)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(8, 6, 8, 8)
        log_layout.setSpacing(4)
        log_header = QHBoxLayout()
        log_title = QLabel("Operation Log")
        log_title.setProperty("classificationTitle", True)
        clear_log_btn = QPushButton("Clear Log")
        toggle_log_btn = QPushButton("Collapse")
        clear_log_btn.setFixedHeight(24)
        toggle_log_btn.setFixedHeight(24)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_header.addWidget(clear_log_btn)
        log_header.addWidget(toggle_log_btn)
        log_layout.addLayout(log_header)
        log_browser = QTextBrowser()
        log_browser.setObjectName("classificationPageTextBrowser")
        log_browser.setFont(QFont("Consolas", 9))
        log_browser.setMinimumHeight(104)
        log_layout.addWidget(log_browser, 1)
        splitter.addWidget(log_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        QTimer.singleShot(0, lambda: splitter.setSizes([720, 180]))

        # Assign fresh widgets to the existing controller aliases.
        self.ui.ClassificationImportListWidget = label_list
        self.ui.ClassificationImportPlusButton = add_label_btn
        self.ui.ClassificationImportMinusButton = remove_label_btn
        self.ui.ClassificationImportFolderPathLabel = choose_folder_btn
        self.ui.ClassificationImportFolderPathValue = path_edit
        self.ui.ClassificationImportRuleLabel = QLabel("Rule")
        self.ui.ClassificationImportRuleValue = rule_edit
        self.ui.ClassificationImportImportButton = import_btn
        self.ui.ClassificationImportClassifyButton = predict_btn
        self.ui.ClassificationImportTableWidget = table
        self.ui.ClassificationGraphicsView = graphics
        self.ui.ClassificationPanelWidget = panel
        self.ui.DimensionalityReductionMethodCombox = dim_method
        self.ui.DimensionalityReductionTargetDimLabel = target_dim_label
        self.ui.DimensionalityReductionTargetDimValue = target_dim
        self.ui.nNeighborsWidget = n_widget
        self.ui.DimensionalityReductionNNeighborLabel = n_label
        self.ui.DimensionalityReductionNNeighborValue = n_spin
        self.ui.DimensionalityReductionStartButton = run_embedding_btn
        self.ui.DimensionalityReductionShowResultButton = show_embedding_btn
        self.ui.ClassificationMethodCombox = clf_method
        self.ui.ClassificationKNnnNneighborsLabel = clf_param_label
        self.ui.ClassificationKNnnNneighborsValue = clf_param_edit
        self.ui.ClassificationClassifyButton = train_btn
        self.ui.ClassificationSaveModelButton = save_btn
        self.ui.ClassificationLoadModelButton = load_btn
        self.ui.classificationPageTextBrowser = log_browser
        self.ui.classificationPagetextBrowser = log_browser

        self._classification_log_browser = log_browser
        self._classification_log_panel = log_panel
        self._classification_log_toggle_button = toggle_log_btn
        self._classification_log_expanded = True
        self._classification_log_last_sizes = [720, 180]
        self._classification_scan_button = scan_btn
        self._classification_responsive_refs = {
            'root': root,
            'tabs': tabs,
            'splitter': splitter,
            'preview_tab': preview_tab,
            'stats_layout': stats,
            'preview_layout': preview_layout,
        }

        self._safe_connect(label_list.itemSelectionChanged, self._on_list_selection_changed)
        self._safe_connect(label_list.itemDoubleClicked, self._on_item_double_clicked)
        self._safe_connect(label_list.itemChanged, self._on_item_renamed)
        self._safe_connect(add_label_btn.clicked, self._on_plus_clicked)
        self._safe_connect(remove_label_btn.clicked, self._on_minus_clicked)
        self._safe_connect(choose_folder_btn.clicked, self._on_choose_path_clicked)
        self._safe_connect(rule_edit.editingFinished, self._on_rule_edited)
        self._safe_connect(path_edit.editingFinished, self._on_path_edited)
        self._safe_connect(scan_btn.clicked, self._on_scan_files_clicked)
        self._safe_connect(import_btn.clicked, self._on_import_clicked)
        self._safe_connect(predict_btn.clicked, self._on_import_classify_clicked)
        self._safe_connect(table.cellClicked, self._on_table_cell_clicked)
        self._safe_connect(table.itemSelectionChanged, self._on_table_selection_changed)
        self._safe_connect(table.itemChanged, self._on_table_item_changed)
        self._safe_connect(dim_method.currentTextChanged, self._on_dim_method_changed)
        self._safe_connect(run_embedding_btn.clicked, self._on_dim_start_clicked_async)
        self._safe_connect(show_embedding_btn.clicked, self._on_dim_show_clicked)
        self._safe_connect(clf_method.currentTextChanged, self._on_clf_method_changed)
        self._safe_connect(train_btn.clicked, self._on_clf_start_clicked)
        self._safe_connect(save_btn.clicked, self._on_clf_save_clicked)
        self._safe_connect(load_btn.clicked, self._on_clf_load_clicked)
        self._safe_connect(clear_log_btn.clicked, self._clear_classification_log)
        self._safe_connect(toggle_log_btn.clicked, self._toggle_classification_log_panel)

        self._ensure_table_headers()
        self._on_dim_method_changed(dim_method.currentText())
        self._on_clf_method_changed(clf_method.currentText())
        self._set_dr_status_color('brown')
        self._classification_workflow_ready = True
        self._set_table_responsive_columns('stable')
        self._update_dataset_status_cards()
        self.log("[UI] Fresh Classification page initialized.")

    def _rename_workflow_buttons(self):
        names = {
            'ClassificationImportPlusButton': 'Add Label',
            'ClassificationImportMinusButton': 'Remove Label',
            'ClassificationImportFolderPathLabel': 'Choose Folder',
            'ClassificationImportImportButton': 'Import Selected',
            'ClassificationImportClassifyButton': 'Predict',
            'DimensionalityReductionStartButton': 'Run Embedding',
            'DimensionalityReductionShowResultButton': 'Show Embedding',
            'ClassificationClassifyButton': 'Train Classifier',
            'ClassificationSaveModelButton': 'Save Model',
            'ClassificationLoadModelButton': 'Load Model',
        }
        for attr, text in names.items():
            widget = getattr(self.ui, attr, None)
            if widget is not None and hasattr(widget, 'setText'):
                widget.setText(text)
                try:
                    widget.setMinimumHeight(30)
                    widget.setMinimumWidth(96)
                    widget.setMaximumWidth(16777215)
                except Exception:
                    pass

    def _compact_classification_controls(self, container):
        """Keep moved legacy controls usable inside compact cards."""
        if container is None:
            return
        try:
            from PyQt5.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit
            for combo in container.findChildren(QComboBox):
                combo.setMaximumWidth(220)
                combo.setMinimumWidth(120)
            for spin in container.findChildren((QSpinBox, QDoubleSpinBox)):
                spin.setMaximumWidth(120)
            for edit in container.findChildren(QLineEdit):
                edit.setMaximumWidth(220)
            for button in container.findChildren(QPushButton):
                button.setMinimumHeight(30)
                button.setMaximumWidth(180)
        except Exception as e:
            self._log_exception('[UI] Compact controls failed', e)

    def _clear_classification_log(self):
        try:
            tb = getattr(self, '_classification_log_browser', None)
            if tb is None:
                tb = getattr(self.ui, 'classificationPageTextBrowser', None)
            if tb is not None:
                tb.clear()
        except Exception as e:
            self._log_exception('[UI] Clear log failed', e)

    def _toggle_classification_log_panel(self):
        try:
            tb = getattr(self, '_classification_log_browser', None)
            btn = getattr(self, '_classification_log_toggle_button', None)
            splitter = getattr(self, '_classification_main_splitter', None)
            if tb is None:
                return
            expanded = bool(getattr(self, '_classification_log_expanded', True))
            panel = getattr(self, '_classification_log_panel', None)
            if expanded:
                if splitter is not None:
                    sizes = splitter.sizes()
                    if sizes and len(sizes) >= 2 and sizes[1] > 45:
                        self._classification_log_last_sizes = sizes
                    splitter.setSizes([max(1, sum(sizes) - 34) if sizes else 700, 34])
                tb.setVisible(False)
                if panel is not None:
                    panel.setMinimumHeight(34)
                    panel.setMaximumHeight(44)
                if btn is not None:
                    btn.setText("Expand")
                self._classification_log_expanded = False
            else:
                if panel is not None:
                    panel.setMaximumHeight(16777215)
                    panel.setMinimumHeight(120)
                tb.setVisible(True)
                if splitter is not None:
                    splitter.setSizes(getattr(self, '_classification_log_last_sizes', [560, 160]))
                if btn is not None:
                    btn.setText("Collapse")
                self._classification_log_expanded = True
        except Exception as e:
            self._log_exception('[UI] Toggle log panel failed', e)

    def _on_scan_files_clicked(self):
        path = self._get_cached('path', '') or ''
        rule = self._get_cached('rule', '*') or '*'
        if not path:
            path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
            path = normalize_path(path_edit.text()) if path_edit is not None else ''
        if not path:
            self.log('[Import] Please choose a folder or enter a path first.')
            return
        self._scan_and_list_files(path, rule)

    def _update_dataset_status_cards(self):
        cards = getattr(self, '_classification_status_cards', {})
        if not cards:
            return
        total = len(self.samples)
        loaded = sum(1 for s in self.samples if (s.preprocessed_data is not None) or (s.raw_data is not None))
        labels = len(self.import_cache)
        selected = self._get_current_name() or self.current_item_name or '-'
        values = {
            'total': str(total),
            'loaded': str(loaded),
            'labels': str(labels),
            'selected': selected,
        }
        for key, value in values.items():
            label = cards.get(key)
            if label is not None:
                label.setText(value)

    def eventFilter(self, watched, event):
        return super().eventFilter(watched, event)

    def _classification_available_width(self) -> int:
        refs = getattr(self, '_classification_responsive_refs', {})
        root = refs.get('root')
        fallback_widths = []
        try:
            if self.main_window is not None and self.main_window.isVisible() and self.main_window.width() > 400:
                return self.main_window.width()
        except Exception as e:
            self._log_exception('[UI] Import table header setup failed', e)
        try:
            if root is not None and root.isVisible() and root.width() > 400:
                return root.width()
        except Exception as e:
            self._log_exception('[UI] Table header setup failed', e)
        try:
            if self.main_window is not None and self.main_window.width() > 1000:
                fallback_widths.append(self.main_window.width())
        except Exception:
            pass
        try:
            if root is not None and root.width() > 1000:
                fallback_widths.append(root.width())
        except Exception:
            pass
        try:
            widget = root or self.main_window
            screen = widget.screen() if widget is not None and hasattr(widget, 'screen') else None
            if screen is not None:
                fallback_widths.append(screen.availableGeometry().width())
        except Exception:
            pass
        return max(fallback_widths) if fallback_widths else 1366

    def _remove_widget_from_layout(self, layout, widget) -> None:
        if layout is None or widget is None:
            return
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item is not None and item.widget() is widget:
                layout.takeAt(i)

    def _remove_tab_widget(self, tabs, widget) -> None:
        if tabs is None or widget is None:
            return
        idx = tabs.indexOf(widget)
        if idx >= 0:
            tabs.removeTab(idx)

    def _set_status_card_layout(self, mode: str) -> None:
        refs = getattr(self, '_classification_responsive_refs', {})
        layout = refs.get('stats_layout')
        if layout is None:
            return
        while layout.count():
            layout.takeAt(0)
        frames = getattr(self, '_classification_status_card_frames', {})
        labels = getattr(self, '_classification_status_cards', {})
        titles = getattr(self, '_classification_status_card_titles', {})
        order = ['total', 'loaded', 'labels', 'selected']
        compact_titles = {
            'total': 'Total',
            'loaded': 'Loaded',
            'labels': 'Labels',
            'selected': 'Current',
        }
        full_titles = {
            'total': 'Total files',
            'loaded': 'Loaded files',
            'labels': 'Number of labels',
            'selected': 'Current selected label',
        }
        for pos, key in enumerate(order):
            frame = frames.get(key)
            label = labels.get(key)
            if frame is None:
                continue
            title = titles.get(key)
            if title is not None:
                compact_card = mode in ('compact', 'stable')
                title.setText(compact_titles[key] if compact_card else full_titles[key])
                title.setStyleSheet("color: #596579; font-size: 10px;" if compact_card else "color: #596579; font-size: 11px;")
            if label is not None:
                label.setStyleSheet("font-weight: 700; font-size: 12px;" if mode in ('compact', 'stable') else "font-weight: 700; font-size: 14px;")
            if mode in ('compact', 'stable'):
                layout.addWidget(frame, pos // 2, pos % 2)
            else:
                layout.addWidget(frame, 0, pos)

    def _set_table_responsive_columns(self, mode: str) -> None:
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        if table is None:
            return
        from PyQt5.QtWidgets import QHeaderView, QAbstractScrollArea, QAbstractItemView
        try:
            table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
            for col in range(table.columnCount()):
                table.setColumnHidden(col, False)
            if mode == 'compact':
                for col in (TABLE_COL_SHAPE, TABLE_COL_PREDICTION, TABLE_COL_CONFIDENCE):
                    table.setColumnHidden(col, True)
            header = table.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(TABLE_COL_LABEL, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_TYPE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_FILES, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_LOADED, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_SHAPE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_STATUS, QHeaderView.Stretch)
            header.setSectionResizeMode(TABLE_COL_PREDICTION, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_CONFIDENCE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_PREVIEW, QHeaderView.ResizeToContents)
            if mode == 'compact':
                table.setMinimumHeight(260)
            else:
                table.setMinimumHeight(300)
        except Exception as e:
            self.log(f"[UI] Responsive table update failed: {e}")

    def _compact_preview_controls(self, mode: str) -> None:
        refs = getattr(self, '_classification_responsive_refs', {})
        preview_layout = refs.get('preview_layout')
        panel = getattr(self.ui, 'ClassificationPanelWidget', None)
        graphics = getattr(self.ui, 'ClassificationGraphicsView', None)
        try:
            if preview_layout is not None:
                margins = 4 if mode == 'compact' else 6
                preview_layout.setContentsMargins(margins, margins, margins, margins)
                preview_layout.setSpacing(4 if mode == 'compact' else 6)
            if panel is not None:
                panel.setMaximumHeight(150 if mode == 'compact' else 240)
                if panel.layout() is not None:
                    panel.layout().setContentsMargins(4, 4, 4, 4)
                    panel.layout().setSpacing(4 if mode == 'compact' else 6)
            if graphics is not None:
                graphics.setMinimumHeight(300 if mode == 'compact' else 220)
        except Exception as e:
            self.log(f"[UI] Preview responsive update failed: {e}")

    def _apply_classification_responsive_mode(self, force: bool = False):
        self._set_status_card_layout('stable')
        self._set_table_responsive_columns('stable')
        self._compact_preview_controls('stable')

    def _setup_connections(self):
        """连接 Classification 页面相关控件（Import/降维/分类）。"""
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        plus_btn = getattr(self.ui, 'ClassificationImportPlusButton', None)
        minus_btn = getattr(self.ui, 'ClassificationImportMinusButton', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
        import_btn = getattr(self.ui, 'ClassificationImportImportButton', None)
        import_clf_btn = getattr(self.ui, 'ClassificationImportClassifyButton', None)
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)

        # 降维
        dim_method = getattr(self.ui, 'DimensionalityReductionMethodCombox', None)
        dim_start = getattr(self.ui, 'DimensionalityReductionStartButton', None)
        dim_show = getattr(self.ui, 'DimensionalityReductionShowResultButton', None)

        # 分类
        clf_method = getattr(self.ui, 'ClassificationMethodCombox', None)
        clf_start = getattr(self.ui, 'ClassificationClassifyButton', None)
        clf_save = getattr(self.ui, 'ClassificationSaveModelButton', None)
        clf_load = getattr(self.ui, 'ClassificationLoadModelButton', None)

        if lw is not None:
            # 选择变化 -> 切换显示并加载缓存
            lw.itemSelectionChanged.connect(self._on_list_selection_changed)
            # 双击 -> 重命名
            lw.itemDoubleClicked.connect(self._on_item_double_clicked)
            # 监听重命名后
            lw.itemChanged.connect(self._on_item_renamed)

        if plus_btn is not None:
            plus_btn.clicked.connect(self._on_plus_clicked)

        if minus_btn is not None:
            minus_btn.clicked.connect(self._on_minus_clicked)

        if path_btn is not None:
            # 点击选择文件/目录
            path_btn.clicked.connect(self._on_choose_path_clicked)

        if rule_edit is not None:
            # 规则文本改变 -> 写回缓存
            rule_edit.editingFinished.connect(self._on_rule_edited)

        if path_edit is not None:
            # 路径文本改变 -> 写回缓存
            path_edit.editingFinished.connect(self._on_path_edited)

        if import_btn is not None:
            import_btn.clicked.connect(self._on_import_clicked)
        if import_clf_btn is not None:
            import_clf_btn.clicked.connect(self._on_import_classify_clicked)
        if table is not None:
            table.cellClicked.connect(self._on_table_cell_clicked)
            # 允许从表格选择同步到上方列表；允许在表格编辑类别名
            try:
                table.itemSelectionChanged.connect(self._on_table_selection_changed)
                table.itemChanged.connect(self._on_table_item_changed)
            except Exception:
                pass

        # 降维
        if dim_method is not None:
            dim_method.currentTextChanged.connect(self._on_dim_method_changed)
        if dim_start is not None:
            dim_start.clicked.connect(self._on_dim_start_clicked_async)
            # create status indicator label next to start button if possible
            try:
                parent = dim_start.parent()
                if parent is not None and parent.layout() is not None:
                    from PyQt5.QtWidgets import QLabel
                    self._dr_status_label = QLabel('●')
                    self._dr_status_label.setFixedWidth(14)
                    self._dr_status_label.setToolTip('DR status')
                    parent.layout().addWidget(self._dr_status_label)
                    self._set_dr_status_color('brown')
                else:
                    self._dr_status_label = None
            except Exception:
                self._dr_status_label = None
        if dim_show is not None:
            dim_show.clicked.connect(self._on_dim_show_clicked)
        else:
            # UI has no Show button; add one next to Start at runtime
            try:
                parent = dim_start.parent() if dim_start is not None else None
                if parent is not None and parent.layout() is not None:
                    from PyQt5.QtWidgets import QPushButton
                    self._dr_show_btn = QPushButton('Show Embedding')
                    self._dr_show_btn.setToolTip('Show DR result')
                    self._dr_show_btn.clicked.connect(self._on_dim_show_clicked)
                    parent.layout().addWidget(self._dr_show_btn)
            except Exception:
                pass

        # 分类
        if clf_method is not None:
            clf_method.currentTextChanged.connect(self._on_clf_method_changed)
        if clf_start is not None:
            clf_start.clicked.connect(self._on_clf_start_clicked)
        if clf_save is not None:
            clf_save.clicked.connect(self._on_clf_save_clicked)
        if clf_load is not None:
            clf_load.clicked.connect(self._on_clf_load_clicked)

        # Classification preview external window removed per request.

    def _initialize_ui(self):
        """初始化 Import UI 默认项与默认缓存"""
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
        rule_label = getattr(self.ui, 'ClassificationImportRuleLabel', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)

        if lw is None:
            return

        if self.import_cache:  # 有恢复的缓存 -> 直接按缓存构建
            self._rebuild_list_from_cache()
            # 同步输入显示
            self._on_list_selection_changed()
            self._sync_dynamic_attributes()
            return

        # 否则创建默认项
        default_name = 'Particle 1'
        if lw.count() == 0:
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(default_name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            lw.addItem(item)

        # 确保现有项可编辑
        for i in range(lw.count()):
            it = lw.item(i)
            it.setFlags(it.flags() | Qt.ItemIsEditable)

        # 初始化缓存默认项
        if default_name not in self.import_cache:
            self.import_cache[default_name] = {'path': '', 'rule': '*'}

        # 选中第一项
        if lw.currentItem() is None and lw.count() > 0:
            lw.setCurrentRow(0)
            self.current_item_name = lw.currentItem().text()

        # 填充显示
        if rule_edit is not None:
            rule_edit.setText(self._get_cached('rule', '*'))
        if path_edit is not None:
            path_edit.setText(self._get_cached('path', ''))
        self._refresh_labels()
        self._sync_dynamic_attributes()

    def _setup_classification_panel(self):
        """Create 1D/2D display controls inside ClassificationPanelWidget."""
        panel = getattr(self.ui, 'ClassificationPanelWidget', None)
        if panel is None:
            return
        try:
            from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QLabel, QCheckBox, QDoubleSpinBox, QComboBox
            if panel.layout() is None:
                layout = QVBoxLayout(panel)
                panel.setLayout(layout)
            else:
                layout = panel.layout()
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(6)

            info_label = QLabel('Select a row in Table / Inspect or click Preview.')
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
            self._panel_info_label = info_label

            controls = QGridLayout()
            controls.setContentsMargins(0, 0, 0, 0)
            controls.setHorizontalSpacing(10)
            controls.setVerticalSpacing(5)

            title_2d = QLabel('2D')
            title_2d.setStyleSheet('font-weight: 700;')
            title_1d = QLabel('1D')
            title_1d.setStyleSheet('font-weight: 700;')
            cb_auto = QCheckBox('Auto scale')
            cb_auto.setChecked(True)
            cb_auto.toggled.connect(self._on_image_auto_scale_toggled)
            cb_log = QCheckBox('Log scale')
            cb_log.setChecked(False)
            cb_log.toggled.connect(self._on_image_log_scale_toggled)
            sp_vmin = QDoubleSpinBox()
            sp_vmin.setDecimals(4)
            sp_vmin.setMinimum(-1e12)
            sp_vmin.setMaximum(1e12)
            sp_vmin.setSingleStep(0.1)
            sp_vmin.setValue(0.0)
            sp_vmin.setMaximumWidth(130)
            sp_vmax = QDoubleSpinBox()
            sp_vmax.setDecimals(4)
            sp_vmax.setMinimum(-1e12)
            sp_vmax.setMaximum(1e12)
            sp_vmax.setSingleStep(0.1)
            sp_vmax.setValue(1.0)
            sp_vmax.setMaximumWidth(130)
            sp_vmin.editingFinished.connect(self._on_image_vmin_editing_finished)
            sp_vmax.editingFinished.connect(self._on_image_vmax_editing_finished)
            sp_vmin.setEnabled(False)
            sp_vmax.setEnabled(False)

            cb_cmap = QComboBox()
            cb_cmap.addItems(['jet', 'gray', 'viridis', 'plasma', 'magma', 'inferno', 'turbo'])
            cb_cmap.setCurrentText(self._image_cmap_name)
            cb_cmap.setMaximumWidth(180)
            cb_cmap.currentTextChanged.connect(self._on_image_cmap_changed)

            cb_logy = QCheckBox('Log Y')
            cb_logy.setChecked(False)
            cb_logy.toggled.connect(self._on_curve_logy_toggled)

            controls.addWidget(title_2d, 0, 0)
            controls.addWidget(cb_auto, 0, 1)
            controls.addWidget(cb_log, 0, 2)
            controls.addWidget(QLabel('Colormap'), 1, 0)
            controls.addWidget(cb_cmap, 1, 1)
            controls.addWidget(QLabel('vmin'), 1, 2)
            controls.addWidget(sp_vmin, 1, 3)
            controls.addWidget(QLabel('vmax'), 1, 4)
            controls.addWidget(sp_vmax, 1, 5)
            controls.addWidget(title_1d, 2, 0)
            controls.addWidget(cb_logy, 2, 1)
            controls.setColumnStretch(6, 1)
            layout.addLayout(controls)

            self._panel_widgets = {
                '2d_auto': cb_auto,
                '2d_log': cb_log,
                'vmin': sp_vmin,
                'vmax': sp_vmax,
                '1d_logy': cb_logy,
                'cmap': cb_cmap,
            }
            return
        except Exception as e:
            self._log_exception('[UI] Preview controls rebuild failed', e)
            return
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QCheckBox, QDoubleSpinBox, QComboBox, QFormLayout
        # Initialize layout
        if panel.layout() is None:
            layout = QVBoxLayout(panel)
            panel.setLayout(layout)
        else:
            layout = panel.layout()
        # Clear existing
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # Info label
        info_label = QLabel('Current: —')
        layout.addWidget(info_label)
        self._panel_info_label = info_label
        # 2D controls
        gb2d = QGroupBox('2D Controls')
        gb2d_layout = QFormLayout()
        gb2d.setLayout(gb2d_layout)
        cb_auto = QCheckBox('Auto Scale (0.5–99.5%)')
        cb_auto.setChecked(True)
        cb_auto.toggled.connect(self._on_image_auto_scale_toggled)
        cb_log = QCheckBox('Log Scale')
        cb_log.setChecked(False)
        cb_log.toggled.connect(self._on_image_log_scale_toggled)
        sp_vmin = QDoubleSpinBox(); sp_vmin.setDecimals(4); sp_vmin.setMinimum(-1e12); sp_vmin.setMaximum(1e12); sp_vmin.setSingleStep(0.1); sp_vmin.setValue(0.0)
        sp_vmax = QDoubleSpinBox(); sp_vmax.setDecimals(4); sp_vmax.setMinimum(-1e12); sp_vmax.setMaximum(1e12); sp_vmax.setSingleStep(0.1); sp_vmax.setValue(1.0)
        # Trigger vmin/vmax updates on Enter (editingFinished), not on every change
        sp_vmin.editingFinished.connect(self._on_image_vmin_editing_finished)
        sp_vmax.editingFinished.connect(self._on_image_vmax_editing_finished)
        sp_vmin.setEnabled(False)
        sp_vmax.setEnabled(False)
        # Colormap selector
        cb_cmap = QComboBox()
        cb_cmap.addItems(['jet', 'gray', 'viridis', 'plasma', 'magma', 'inferno', 'turbo'])
        cb_cmap.setCurrentText(self._image_cmap_name)
        cb_cmap.currentTextChanged.connect(self._on_image_cmap_changed)
        # Arrange compactly for 300px width
        gb2d_layout.addRow(cb_auto, cb_log)
        gb2d_layout.addRow(QLabel('Colormap'), cb_cmap)
        gb2d_layout.addRow(QLabel('vmin'), sp_vmin)
        gb2d_layout.addRow(QLabel('vmax'), sp_vmax)
        layout.addWidget(gb2d)
        # 1D controls
        gb1d = QGroupBox('1D Controls')
        gb1d_layout = QHBoxLayout()
        gb1d.setLayout(gb1d_layout)
        cb_logy = QCheckBox('Log Y')
        cb_logy.setChecked(False)
        cb_logy.toggled.connect(self._on_curve_logy_toggled)
        gb1d_layout.addWidget(cb_logy)
        layout.addWidget(gb1d)
        # Store references
        self._panel_widgets = {
            '2d_auto': cb_auto,
            '2d_log': cb_log,
            'vmin': sp_vmin,
            'vmax': sp_vmax,
            '1d_logy': cb_logy,
            'cmap': cb_cmap,
        }

    def _update_panel_info(self, category: str):
        try:
            indices = [i for i, s in enumerate(self.samples) if s.category == category]
            total = len(indices)
            show_idx = self._category_show_index.get(category, 1)
            self._panel_info_label.setText(f"Current: {category} (showing {show_idx}/{total})")
        except Exception:
            pass

    def _rebuild_list_from_cache(self):
        """根据 import_cache 重建列表条目"""
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        if lw is None:
            return
        from PyQt5.QtWidgets import QListWidgetItem
        # 记录当前选择名
        current = self._get_current_name()
        lw.blockSignals(True)
        try:
            lw.clear()
            for name in self.import_cache.keys():
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                lw.addItem(item)
            # 恢复选择
            if current and current in self.import_cache:
                items = [lw.item(i).text() for i in range(lw.count())]
                if current in items:
                    lw.setCurrentRow(items.index(current))
            elif lw.count() > 0:
                lw.setCurrentRow(0)
            self.current_item_name = self._get_current_name()
            self._refresh_labels()
        finally:
            lw.blockSignals(False)
        # 列表重建后保持上下同步行数
        try:
            self._rebuild_table_grouped()
        except Exception:
            pass
        # 若缓存中已有路径，自动扫描刷新 Status
        try:
            self._refresh_status_for_all_categories()
        except Exception:
            pass

    def _refresh_status_for_all_categories(self):
        """遍历所有类别：若配置了路径，则扫描文件以更新 Status 的 M 计数。
        不加载数据，只建立文件列表（m 仍为已加载数量，通常为0）。
        """
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        if lw is None or lw.count() == 0:
            return
        current_row = lw.currentRow()
        try:
            for i in range(lw.count()):
                name = lw.item(i).text()
                cfg = self.import_cache.get(name, {})
                path = cfg.get('path', '') or ''
                rule = cfg.get('rule', '*') or '*'
                if not path:
                    continue
                # 切换当前项以让扫描归属到该类别
                lw.setCurrentRow(i)
                self.current_item_name = name
                self._scan_and_list_files(path, rule)
        finally:
            # 恢复选择并重建一次表格
            if current_row is not None and current_row >= 0 and current_row < lw.count():
                lw.setCurrentRow(current_row)
                self.current_item_name = lw.item(current_row).text()
            self._rebuild_table_grouped()

    # ---------------------------- 工具方法 ----------------------------
    def _get_current_name(self):
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        item = lw.currentItem() if lw is not None else None
        return item.text() if item is not None else None

    def _sanitize_attr_name(self, name: str) -> str:
        # 生成类似 particle1, gold, gold_2
        import re
        s = name.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip('_')
        if not s:
            s = 'item'
        return s

    def _get_cached(self, key: str, default=None):
        name = self._get_current_name() or self.current_item_name
        if name and name in self.import_cache:
            return self.import_cache[name].get(key, default)
        return default

    def _set_cached(self, key: str, value):
        name = self._get_current_name() or self.current_item_name
        if not name:
            return
        if name not in self.import_cache:
            self.import_cache[name] = {'path': '', 'rule': '*'}
        self.import_cache[name][key] = value
        self._sync_dynamic_attributes()

    def _sync_dynamic_attributes(self):
        """将缓存映射为控制器属性，例如 name='Particle 1' -> self.particle_1, self.particle_1_rule。"""
        # 清理：不删除旧属性，避免潜在引用断裂，只覆盖现值
        for name, data in self.import_cache.items():
            base = self._sanitize_attr_name(name)
            try:
                setattr(self, base, data.get('path', ''))
                setattr(self, f"{base}_rule", data.get('rule', '*'))
            except Exception:
                pass

        # 兼容历史命名：为 'Particle 1' 提供 self.particle1/self.particle1_rule
        if 'Particle 1' in self.import_cache:
            d = self.import_cache['Particle 1']
            self.particle1 = d.get('path', '')
            self.particle1_rule = d.get('rule', '*')

        # 广播参数变化并写入全局参数，立即保存到 user_parameters.json
        payload = {'import_cache': self.import_cache.copy()}
        self.parameters_changed.emit(payload)
        try:
            global_params.set_parameter('classification', 'import_cache', payload['import_cache'])
            global_params.save_user_parameters()
        except Exception as e:
            print(f"[classification] persist import_cache failed: {e}")

    def _refresh_labels(self):
        """根据当前项更新 Path/Rule 标签文本，让用户清楚对应关系。"""
        name = self._get_current_name() or self.current_item_name or '—'
        rule_label = getattr(self.ui, 'ClassificationImportRuleLabel', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)
        if path_btn is not None:
            path_btn.setText("Choose Folder")
            path_btn.setToolTip(f"Choose folder or file for {name}")
        if rule_label is not None:
            rule_label.setText("Rule:")

    # ---------------------------- 事件处理 ----------------------------
    def _on_list_selection_changed(self):
        self.current_item_name = self._get_current_name()
        # 切换显示
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
        if rule_edit is not None:
            rule_edit.setText(self._get_cached('rule', '*') or '*')
        if path_edit is not None:
            path_edit.setText(self._get_cached('path', '') or '')
        self._refresh_labels()
        self._update_dataset_status_cards()

    def _on_item_double_clicked(self, item):
        # 启动内联编辑
        lw = self.ui.ClassificationImportListWidget
        # 记录旧名，便于迁移缓存
        self._rename_old_name = item.text()
        lw.editItem(item)

    # _on_item_renamed enhanced version is defined later in the file.

    def _on_plus_clicked(self):
        from PyQt5.QtWidgets import QListWidgetItem
        lw = self.ui.ClassificationImportListWidget
        # 询问自定义名称
        text, ok = QInputDialog.getText(self.main_window, 'Add Item', 'Name:')
        if not ok:
            return
        name = text.strip() or self._make_default_name(lw)
        # 名称去重处理
        name = self._ensure_unique_name(lw, name)

        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsEditable)  # 可编辑
        lw.addItem(item)
        lw.setCurrentItem(item)

        # 初始化缓存
        self.import_cache.setdefault(name, {'path': '', 'rule': '*'})
        self.current_item_name = name
        self._refresh_labels()
        self._sync_dynamic_attributes()
        # 新增类别后立刻在表格中出现对应行
        self._rebuild_table_grouped()

    # _on_minus_clicked enhanced version is defined later in the file.

    def _make_default_name(self, lw):
        base = 'Particle'
        i = 1
        existing = {lw.item(j).text() for j in range(lw.count())}
        while f"{base} {i}" in existing:
            i += 1
        return f"{base} {i}"

    def _ensure_unique_name(self, lw, name):
        if name not in {lw.item(j).text() for j in range(lw.count())}:
            return name
        i = 2
        base = name
        while f"{base} ({i})" in {lw.item(j).text() for j in range(lw.count())}:
            i += 1
        return f"{base} ({i})"

    # ---- Path/Rule 编辑 ----
    def _on_choose_path_clicked(self):
        """点击 Path 按钮：弹出选择菜单 -> 选择文件或文件夹。"""
        # 没有选中条目时忽略
        if not self._get_current_name():
            self.status_updated.emit('Please select an item in the list first')
            return

        btn = self.ui.ClassificationImportFolderPathLabel
        menu = QMenu(btn)
        act_file = menu.addAction('Choose File')
        act_dir = menu.addAction('Choose Folder')
        action = menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft()))

        if action == act_file:
            self._choose_file()
        elif action == act_dir:
            self._choose_folder()

    def _choose_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            'Select Folder',
            '',
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder_path:
            folder_path = normalize_path(folder_path)
            # 更新显示与缓存
            path_edit = self.ui.ClassificationImportFolderPathValue
            path_edit.setText(folder_path)
            self._set_cached('path', folder_path)
            self.status_updated.emit(f"Selected folder: {folder_path}")
            # 扫描并填充表格
            self._scan_and_list_files(folder_path, self._get_cached('rule', '*'))

    def _choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            'Select File',
            '',
            'All Files (*);;CBF (*.cbf);;Images (*.png *.jpg *.jpeg *.tif *.tiff *.cbf);;Text (*.txt *.dat);;HDF5 (*.h5 *.hdf5)'
        )
        if file_path:
            file_path = normalize_path(file_path)
            path_edit = self.ui.ClassificationImportFolderPathValue
            rule_edit = self.ui.ClassificationImportRuleValue
            folder_path = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            # 按需求：路径指向文件夹；规则指向具体文件名
            path_edit.setText(folder_path)
            rule_edit.setText(filename)
            self._set_cached('path', folder_path)
            self._set_cached('rule', filename)
            self.status_updated.emit(f"Selected file: {file_path}")
            # 在该文件夹按文件名扫描
            self._scan_and_list_files(folder_path, filename)

    def _on_rule_edited(self):
        rule_edit = self.ui.ClassificationImportRuleValue
        rule = rule_edit.text().strip() or '*'
        self._set_cached('rule', rule)
        # 规则变化 -> 重新扫描当前路径
        path = self._get_cached('path', '') or ''
        if path:
            self._scan_and_list_files(path, rule)

    def _on_path_edited(self):
        path_edit = self.ui.ClassificationImportFolderPathValue
        path = normalize_path(path_edit.text())
        self._set_cached('path', path)
        if path:
            self._scan_and_list_files(path, self._get_cached('rule', '*'))

    # ---------------------------- 对外接口 ----------------------------
    def get_parameters(self):
        return {'import_cache': self.import_cache.copy()}

    def set_parameters(self, parameters):
        cache = parameters.get('import_cache') if isinstance(parameters, dict) else None
        if isinstance(cache, dict):
            self.import_cache = {k: {'path': v.get('path', ''), 'rule': v.get('rule', '*')} for k, v in cache.items()}
            # 尝试保持当前选择
            self._sync_dynamic_attributes()
            self._on_list_selection_changed()

    # ---------------------------- 日志 ----------------------------
    def log(self, message: str) -> None:
        ts = time.strftime('%H:%M:%S')
        line = f"[{ts}] {message}"
        tb = getattr(self.ui, 'classificationPageTextBrowser', None)
        if tb is None:
            tb = getattr(self.ui, 'classificationPagetextBrowser', None)
        if tb is not None:
            tb.append(line)
            try:
                bar = tb.verticalScrollBar()
                bar.setValue(bar.maximum())
            except Exception:
                pass
        self.status_updated.emit(message)

    # ---------------------------- 文件扫描 + 表格管理 ----------------------------
    def _allowed_extension(self, ext: str) -> Optional[str]:
        ext = ext.lower()
        if ext in ('.dat', '.txt'):
            return '1D'
        if ext in ('.edf', '.tif', '.tiff', '.cbf'):
            return '2D'
        return None

    def _scan_and_list_files(self, path: str, rule: str):
        path = normalize_path(path)
        if not path:
            return
        files: List[str] = []
        # Optional: support numeric index range rules for CBF files like "1-10" or "range: 5-12"
        def _parse_cbf_rule_indices(text: str) -> Optional[set]:
            if not text:
                return None
            s = (text or '').strip().lower()
            if s.startswith('range:'):
                s = s[len('range:'):].strip()
            m = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+))?", s)
            if not m:
                return None
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            if end < start:
                start, end = end, start
            return set(range(start, end + 1))
        cbf_indices = _parse_cbf_rule_indices(rule)
        def _extract_cbf_index(name: str) -> Optional[int]:
            m = re.search(r"(\d+)(?=\.cbf$)", name, re.IGNORECASE)
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for n in names:
                    f = os.path.join(root, n)
                    ext = os.path.splitext(n)[1]
                    dtype = self._allowed_extension(ext)
                    if dtype is None:
                        continue
                    # If rule specifies numeric range and file is CBF, filter by index
                    if cbf_indices is not None and ext.lower() == '.cbf':
                        idx = _extract_cbf_index(n)
                        if idx is None or idx not in cbf_indices:
                            continue
                    if rule and (('*' in rule) or ('?' in rule)):
                        if not fnmatch.fnmatch(n, rule):
                            continue
                    elif rule and rule not in n:
                        continue
                    files.append(f)
        else:
            n = os.path.basename(path)
            dtype = self._allowed_extension(os.path.splitext(n)[1])
            if dtype is not None:
                files.append(path)

        if not files:
            self.log('[Import] No files found.')
            return

        # 当前类别粒度处理：若文件群未变化则保持缓存，变化则清空并重建
        current_category = self._get_current_name() or 'Uncategorized'
        old_files = sorted([s.file_path for s in self.samples if s.category == current_category])
        new_files = sorted(files)
        if old_files == new_files:
            # 文件群未变，仅刷新状态显示
            self._rebuild_table_grouped()
            self.log(f"[Import] Category '{current_category}': files unchanged ({len(new_files)}). Kept cache.")
            return

        # 替换该类别样本为新的文件集合（清空旧缓存，Loaded 归零）
        self.samples = [s for s in self.samples if s.category != current_category]
        self._row_to_index.clear()
        self._path_to_index.clear()
        for f in new_files:
            name = os.path.basename(f)
            dtype = self._allowed_extension(os.path.splitext(name)[1])
            if dtype is None:
                continue
            self.samples.append(Sample(
                file_path=f,
                file_name=name,
                data_type=dtype,
                category=current_category,
            ))
        # Reset showing index for this category
        self._category_show_index[current_category] = 1
        self._rebuild_table_grouped()
        self.log(f"[Import] Category '{current_category}': listed {len(new_files)} files. Cache reset for this category.")

    # 原位置的实现已在文件后部增强版本中覆盖。

    def _apply_rule_to_all_rows(self):
        # 在聚合模式下，规则用于扫描阶段；此处保留占位以兼容直接规则修改时的重建需求
        self._rebuild_table_grouped()

    def _on_table_cell_clicked(self, row: int, col: int):
        # 同步选择到上方列表，但不强制选中表格行于列表切换时
        try:
            it = self.ui.ClassificationImportTableWidget.item(row, TABLE_COL_LABEL)
            if it is not None:
                name = it.text()
                lw = self.ui.ClassificationImportListWidget
                for i in range(lw.count()):
                    if lw.item(i).text() == name:
                        lw.setCurrentRow(i)
                        break
        except Exception as e:
            self._log_exception('[UI] Table click sync failed', e)
        self._preview_category_row(row)

    def _on_preview_index_entered(self, category: str, edit):
        text = (edit.text() or '').strip()
        try:
            pos = int(text)
        except ValueError:
            self.log('[Preview] Invalid index.')
            return
        indices = [i for i, s in enumerate(self.samples) if s.category == category]
        if not indices:
            return
        if pos < 1 or pos > len(indices):
            self.log(f"[Preview] Index out of range 1-{len(indices)}")
            return
        self._category_show_index[category] = pos
        self.show_sample(indices[pos - 1])

    # ---------------------------- 数据读取 ----------------------------
    def _on_preview_category_clicked(self, category: str, edit=None, *_args):
        if edit is not None:
            self._on_preview_index_entered(category, edit)
        else:
            indices = [i for i, s in enumerate(self.samples) if s.category == category]
            if indices:
                pos = max(1, min(self._category_show_index.get(category, 1), len(indices)))
                self._category_show_index[category] = pos
                self.show_sample(indices[pos - 1])
        try:
            tabs = getattr(self, '_classification_main_tabs', None)
            refs = getattr(self, '_classification_responsive_refs', {})
            preview_tab = refs.get('preview_tab')
            if tabs is not None and preview_tab is not None:
                idx = tabs.indexOf(preview_tab)
                if idx >= 0:
                    tabs.setCurrentIndex(idx)
        except Exception as e:
            self._log_exception('[Preview] Switch tab failed', e)

    def _preview_category_by_name(self, category: str, edit=None, switch_tab: bool = False, *_args):
        if edit is not None:
            self._on_preview_index_entered(category, edit)
        else:
            self._on_preview_category_clicked(category)
        if switch_tab:
            try:
                tabs = getattr(self, '_classification_main_tabs', None)
                refs = getattr(self, '_classification_responsive_refs', {})
                preview_tab = refs.get('preview_tab')
                if tabs is not None and preview_tab is not None:
                    idx = tabs.indexOf(preview_tab)
                    if idx >= 0:
                        tabs.setCurrentIndex(idx)
            except Exception as e:
                self._log_exception('[Preview] Switch tab failed', e)

    def _on_import_clicked(self):
        table = self.ui.ClassificationImportTableWidget
        try:
            table.setColumnCount(len(TABLE_HEADERS))
            self._ensure_table_headers()
        except Exception as e:
            self._log_exception('[UI] Table header setup failed', e)
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        if table.rowCount() == 0:
            self.log('[Import] No categories to import.')
            return
        # 以优先顺序确定当前类别：表格选择 -> 列表选择
        cat = None
        r = table.currentRow()
        if r is not None and r >= 0:
            it = table.item(r, TABLE_COL_LABEL)
            if it is not None:
                cat = (it.text() or '').strip()
        if not cat and lw is not None and lw.currentItem() is not None:
            cat = lw.currentItem().text()
        if not cat:
            self.log('[Import] Please select a category in list or table.')
            return

        # 仅导入该类别的样本
        indices = [i for i, s in enumerate(self.samples) if s.category == cat]
        total_files = len(indices)
        loaded_files = sum(1 for i in indices if (self.samples[i].preprocessed_data is not None) or (self.samples[i].raw_data is not None))
        failed = 0
        if total_files == 0:
            self.log(f"[Import] Category '{cat}' has no files.")
            self._rebuild_table_grouped()
            return
        # Multithread import using QThreadPool with queued signals
        pool = QThreadPool.globalInstance()
        pool.setMaxThreadCount(max(2, pool.maxThreadCount()))

        class _ImportSignals(QObject):
            result = pyqtSignal(int, bool, str)  # idx, ok, filename

        class _ImportTask(QRunnable):
            def __init__(self, controller, idx):
                super().__init__()
                self.controller = controller
                self.idx = idx
                self.signals = _ImportSignals()
            def run(self):
                ok = False
                name = ''
                try:
                    s = self.controller.samples[self.idx]
                    name = s.file_name
                    if (s.preprocessed_data is None) and (s.raw_data is None):
                        ok = bool(self.controller._load_sample_data(s))
                    else:
                        ok = True
                except Exception:
                    ok = False
                # Emit back to main thread
                try:
                    self.signals.result.emit(self.idx, ok, name)
                except Exception:
                    pass

        remaining = [i for i in indices if (self.samples[i].preprocessed_data is None and self.samples[i].raw_data is None)]
        if not remaining:
            self.log(f"[Import] [{cat}] No new files to import.")
            return
        self.log(f"[Import] [{cat}] Starting threaded import of {len(remaining)} files...")
        self.progress_updated.emit(0)
        loaded_ref = {'ok': loaded_files, 'fail': failed, 'done': 0}

        def _on_result(idx: int, ok: bool, fname: str):
            # main-thread UI updates
            if ok:
                loaded_ref['ok'] += 1
                self.log(f"[Import] [{cat}] Loaded {loaded_ref['ok']}/{total_files}: {fname}")
            else:
                loaded_ref['fail'] += 1
                self.log(f"[Import] [{cat}] Failed: {fname}")
            loaded_ref['done'] += 1
            pct = int(loaded_ref['done'] * 100 / len(remaining))
            self.progress_updated.emit(pct)
            if loaded_ref['done'] % max(1, len(remaining)//10) == 0 or loaded_ref['done'] >= len(remaining):
                self._rebuild_table_grouped()
            if loaded_ref['done'] >= len(remaining):
                self.log(f"[Import] [{cat}] Done: {loaded_ref['ok']} loaded, {loaded_ref['fail']} failed.")
                self._classification_active_tasks = [
                    task for task in self._classification_active_tasks
                    if not isinstance(task, _ImportTask)
                ]

        for i in remaining:
            task = _ImportTask(self, i)
            task.signals.result.connect(_on_result)
            self._classification_active_tasks.append(task)
            pool.start(task)

    def _load_sample_data(self, sample: Sample) -> bool:
        try:
            if sample.data_type == '1D':
                x, y = self._read_1d_data(sample.file_path)
                if x is None or y is None:
                    return False
                arr = np.column_stack([x, y])
                sample.raw_data = arr
                sample.preprocessed_data = arr.copy()
            else:
                img = self._read_2d_data(sample.file_path)
                if img is None:
                    return False
                sample.raw_data = img
                sample.preprocessed_data = img.copy()
            sample.preprocessing_desc = ''
            return True
        except Exception as e:
            self.log(f"[Import] Error reading {os.path.basename(sample.file_path)}: {e}")
            return False

    def _read_1d_data(self, path: str):
        xs: List[float] = []
        ys: List[float] = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('%'):
                        continue
                    parts = line.replace(',', ' ').split()
                    if len(parts) < 2:
                        continue
                    try:
                        x = float(parts[0]); y = float(parts[1])
                        xs.append(x); ys.append(y)
                    except ValueError:
                        continue
            if len(xs) == 0:
                return None, None
            return np.array(xs, dtype=float), np.array(ys, dtype=float)
        except Exception as e:
            self.log(f"[Import] 1D read error: {e}")
            return None, None

    def _read_2d_data(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in ('.tif', '.tiff'):
                try:
                    import imageio.v2 as imageio
                except Exception:
                    import imageio
                img = imageio.imread(path)
                return np.array(img, dtype=float)
            elif ext == '.cbf':
                try:
                    import fabio
                except Exception:
                    self.log('[Import] fabio is required to read CBF. Please install "fabio".')
                    return None
                cbf_image = fabio.open(path)
                data = getattr(cbf_image, 'data', None)
                if data is None:
                    return None
                return np.array(data, dtype=float)
            elif ext == '.edf':
                self.log('[Import] EDF reading not implemented, skipping.')
                return None
            else:
                from PIL import Image
                img = Image.open(path)
                return np.array(img, dtype=float)
        except Exception as e:
            self.log(f"[Import] 2D read error: {e}")
            return None

    # ---------------------------- 预览 ----------------------------
    def _preview_category_row(self, row: int):
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        if table is None:
            return
        it = table.item(row, TABLE_COL_LABEL)
        if it is None:
            return
        category = (it.text() or '').strip()
        indices = [i for i, s in enumerate(self.samples) if s.category == category]
        if not indices:
            return
        pos = self._category_show_index.get(category, 1)
        pos = max(1, min(pos, len(indices)))
        self.show_sample(indices[pos - 1])

    def _to_qpixmap_from_1d(self, arr2: np.ndarray) -> Optional[QPixmap]:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3, 2), dpi=120)
            if self._curve_log_y:
                ax.semilogy(arr2[:, 0], arr2[:, 1], lw=1.0)
            else:
                ax.plot(arr2[:, 0], arr2[:, 1], lw=1.0)
            ax.set_xlabel('q')
            ax.set_ylabel('I')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            image = QImage.fromData(buf.read(), 'PNG')
            return QPixmap.fromImage(image)
        except Exception as e:
            self.log(f"[Preview] 1D plot error: {e}")
            return None

    def _to_qpixmap_from_2d(self, img: np.ndarray) -> Optional[QPixmap]:
        try:
            m = img.astype(float)
            if self._image_log_scale:
                with np.errstate(divide='ignore', invalid='ignore'):
                    m = np.log1p(np.maximum(m, 0))
            # Determine vmin/vmax
            if self._image_auto_scale:
                try:
                    vmin = float(np.nanpercentile(m, 0.5))
                    vmax = float(np.nanpercentile(m, 99.5))
                except Exception:
                    vmin = float(np.nanmin(m))
                    vmax = float(np.nanmax(m))
                # Reflect auto-computed values in controls (without toggling auto)
                try:
                    self._update_vmin_vmax_controls(vmin, vmax, auto=True)
                except Exception:
                    pass
            else:
                # Manual mode: use stored values, or current spinboxes, or fallback to data min/max
                if self._image_vmin is not None and self._image_vmax is not None:
                    vmin = float(self._image_vmin)
                    vmax = float(self._image_vmax)
                else:
                    try:
                        sp_vmin = self._panel_widgets.get('vmin') if hasattr(self, '_panel_widgets') else None
                        sp_vmax = self._panel_widgets.get('vmax') if hasattr(self, '_panel_widgets') else None
                        vmin = float(sp_vmin.value()) if sp_vmin is not None else float(np.nanmin(m))
                        vmax = float(sp_vmax.value()) if sp_vmax is not None else float(np.nanmax(m))
                    except Exception:
                        vmin = float(np.nanmin(m))
                        vmax = float(np.nanmax(m))
                try:
                    self._update_vmin_vmax_controls(vmin, vmax, auto=False)
                except Exception:
                    pass
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(m))
                vmax = float(np.nanmax(m))
                if vmax <= vmin:
                    vmax = vmin + 1e-9
            with np.errstate(invalid='ignore'):
                norm = (m - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)
            # Apply colormap
            cmap_name = (self._image_cmap_name or 'jet').lower()
            if cmap_name in ('gray', 'grey'):
                mm = (norm * 255.0).astype(np.uint8)
                if mm.ndim == 3:
                    # collapse to grayscale by first channel if needed
                    mm = mm[..., 0]
                h, w = mm.shape[:2]
                qimg = QImage(mm.data, w, h, w, QImage.Format_Grayscale8)
                return QPixmap.fromImage(qimg.copy())
            else:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap(cmap_name)
                except Exception:
                    cmap = None
                if cmap is not None:
                    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
                    h, w = rgb.shape[:2]
                    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                    return QPixmap.fromImage(qimg.copy())
                else:
                    mm = (norm * 255.0).astype(np.uint8)
                    h, w = mm.shape[:2]
                    qimg = QImage(mm.data, w, h, w, QImage.Format_Grayscale8)
                    return QPixmap.fromImage(qimg.copy())
        except Exception as e:
            self.log(f"[Preview] 2D image error: {e}")
            return None

    def _update_vmin_vmax_controls(self, vmin: float, vmax: float, auto: bool):
        # Update vmin/vmax spin boxes and auto checkbox without recursive refreshes
        if not hasattr(self, '_panel_widgets'):
            return
        sp_vmin = self._panel_widgets.get('vmin')
        sp_vmax = self._panel_widgets.get('vmax')
        cb_auto = self._panel_widgets.get('2d_auto')
        if sp_vmin is None or sp_vmax is None or cb_auto is None:
            return
        try:
            sp_vmin.blockSignals(True)
            sp_vmax.blockSignals(True)
            cb_auto.blockSignals(True)
            # keep user-chosen auto state; only reflect enable/disable based on controller state
            sp_vmin.setEnabled(not self._image_auto_scale)
            sp_vmax.setEnabled(not self._image_auto_scale)
            sp_vmin.setValue(float(vmin))
            sp_vmax.setValue(float(vmax))
        finally:
            sp_vmin.blockSignals(False)
            sp_vmax.blockSignals(False)
            cb_auto.blockSignals(False)

    def show_sample(self, index: int) -> None:
        if index < 0 or index >= len(self.samples):
            return
        sample = self.samples[index]
        data = sample.preprocessed_data if sample.preprocessed_data is not None else sample.raw_data
        if data is None:
            return
        gv = getattr(self.ui, 'ClassificationGraphicsView', None)
        if gv is None:
            return
        scene = QGraphicsScene()
        pix: Optional[QPixmap] = None
        if sample.data_type == '1D':
            arr = data
            if arr.ndim == 1:
                arr = np.column_stack([np.arange(len(arr)), arr])
            pix = self._to_qpixmap_from_1d(arr)
        else:
            pix = self._to_qpixmap_from_2d(data)
        if pix is not None:
            scene.addPixmap(pix)
            gv.setScene(scene)
            gv.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._last_preview_index = index
        # Update panel info
        try:
            self._update_panel_info(sample.category)
        except Exception:
            pass

    # ---------------------------- 降维 ----------------------------
    def _on_dim_method_changed(self, text: str):
        # 输出维度控件（Target Dim）
        td_label = getattr(self.ui, 'DimensionalityReductionTargetDimLabel', None)
        td_spin = getattr(self.ui, 'DimensionalityReductionTargetDimValue', None)
        # 参数控件（t-SNE 的 perplexity / UMAP 的 n_neighbors）
        n_widget = getattr(self.ui, 'nNeighborsWidget', None)
        n_label = getattr(self.ui, 'DimensionalityReductionNNeighborLabel', None)
        n_spin = getattr(self.ui, 'DimensionalityReductionNNeighborValue', None)
        if td_label is None or td_spin is None:
            return

        # Target Dim 始终表示输出 embedding 维度
        td_label.setText('Target Dim')

        if text == 'PCA':
            # PCA 只需要目标维度，不需要 nNeighborsWidget
            td_spin.setRange(1, 50)
            if td_spin.value() < 1:
                td_spin.setValue(2)
            if n_widget is not None:
                n_widget.setVisible(False)
        elif text == 't-SNE':
            # t-SNE: 上面 Target Dim 设输出维度；右侧小控件用作 perplexity
            td_spin.setRange(2, 20)
            if td_spin.value() < 2:
                td_spin.setValue(2)
            if n_widget is not None:
                n_widget.setVisible(True)
            if n_label is not None:
                n_label.setText('Perplexity')
            if n_spin is not None:
                n_spin.setRange(5, 200)
                if n_spin.value() < 5:
                    n_spin.setValue(30)
        elif text == 'UMAP':
            # UMAP: Target Dim = 输出维度；右侧小控件用作 n_neighbors
            td_spin.setRange(2, 20)
            if td_spin.value() < 2:
                td_spin.setValue(2)
            if n_widget is not None:
                n_widget.setVisible(True)
            if n_label is not None:
                n_label.setText('n_neighbors')
            if n_spin is not None:
                n_spin.setRange(2, 200)
                if n_spin.value() < 2:
                    n_spin.setValue(15)
        else:
            # 其他未知方法：显示 Target Dim，隐藏 nNeighborsWidget
            td_spin.setRange(1, 50)
            if n_widget is not None:
                n_widget.setVisible(False)

    def _on_dim_start_clicked(self):
        pass

    def _set_dr_status_color(self, color: str):
        try:
            if getattr(self, '_dr_status_label', None) is not None:
                self._dr_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception:
            pass

    def _set_import_clf_button_state(self, ready: bool, running: bool = False):
        """Update ClassificationImportClassifyButton color to indicate model state.

        - running=True: red
        - ready=True and not running: green
        - otherwise: default/grey
        """
        btn = getattr(self.ui, 'ClassificationImportClassifyButton', None)
        if btn is None:
            return
        try:
            if running:
                btn.setStyleSheet("background-color: #d32f2f; color: white;")
            elif ready:
                btn.setStyleSheet("background-color: #4caf50; color: white;")
            else:
                btn.setStyleSheet("")
        except Exception:
            pass

    def _on_dim_start_clicked_async(self):
        method = self.ui.DimensionalityReductionMethodCombox.currentText()
        # 输出维度（Target Dim）
        td_spin = getattr(self.ui, 'DimensionalityReductionTargetDimValue', None)
        out_dim = int(td_spin.value()) if td_spin is not None else 2
        # 方法特定参数：t-SNE 的 perplexity / UMAP 的 n_neighbors
        n_spin = getattr(self.ui, 'DimensionalityReductionNNeighborValue', None)
        param_val = int(n_spin.value()) if n_spin is not None else out_dim
        X = self._build_feature_matrix()
        if X is None:
            self.log('[DR] No data to process. Ensure files are imported.')
            return
        self.log(f"[DR] Method={method}, X shape={X.shape}")

        pool = QThreadPool.globalInstance()

        class _DRSignals(QObject):
            finished = pyqtSignal(bool, object)  # ok, embedding or None

        class _DRTask(QRunnable):
            def __init__(self, method, param_val, out_dim, X):
                super().__init__()
                # m: 方法名；p: 方法特定参数（PCA: 忽略；t-SNE: perplexity；UMAP: n_neighbors）
                # out_dim: 输出 embedding 维度
                self.m = method; self.p = param_val; self.out_dim = out_dim; self.X = X
                self.signals = _DRSignals()
            def run(self):
                ok = False; emb = None
                try:
                    n_samples, n_features = self.X.shape
                    max_components = max(1, min(n_samples, n_features))
                    # 输出维度：统一由 out_dim 控制
                    target_dim = int(self.out_dim) if self.out_dim else 2
                    target_dim = int(max(1, min(target_dim, max_components)))

                    if self.m == 'PCA':
                        from sklearn.decomposition import PCA
                        # 如果样本数太少（<2），不跑真正的 PCA，直接返回前 target_dim 个特征
                        if n_samples < 2:
                            emb = self.X[:, :target_dim]
                        else:
                            model = PCA(n_components=target_dim, random_state=0)
                            emb = model.fit_transform(self.X)
                    elif self.m == 't-SNE':
                        from sklearn.manifold import TSNE
                        # t-SNE: p 为 perplexity
                        if n_samples <= 1:
                            # 样本太少，直接退化成取前 target_dim 个特征
                            emb = self.X[:, :target_dim]
                        else:
                            try:
                                perp = float(self.p)
                            except Exception:
                                perp = 30.0
                            max_perp = max(1.0, n_samples - 1e-3)
                            perp = max(1.0, min(perp, max_perp))
                            n_comp = max(2, target_dim)
                            model = TSNE(
                                n_components=n_comp,
                                perplexity=perp,
                                random_state=0,
                                init='pca',
                                learning_rate='auto'
                            )
                            emb = model.fit_transform(self.X)
                    elif self.m == 'UMAP':
                        try:
                            from umap import UMAP
                        except Exception:
                            self.signals.finished.emit(False, None)
                            return
                        if n_samples <= 2:
                            # 样本特别少，退化成取前 target_dim 个特征
                            emb = self.X[:, :target_dim]
                        else:
                            # UMAP: p 为 n_neighbors
                            try:
                                n_neighbors = int(self.p)
                            except Exception:
                                n_neighbors = 15
                            n_neighbors = max(2, min(n_neighbors, n_samples - 1))
                            n_comp = max(2, target_dim)
                            model = UMAP(
                                n_components=n_comp,
                                n_neighbors=n_neighbors,
                                random_state=0
                            )
                            emb = model.fit_transform(self.X)
                    else:
                        self.signals.finished.emit(False, None)
                        return
                    ok = True
                except Exception as e:
                    ok = False
                try:
                    self.signals.finished.emit(ok, emb)
                except Exception:
                    pass

        def _on_finished(ok: bool, emb_obj):
            self._set_dr_status_color('green' if ok else 'red')
            if not ok or emb_obj is None:
                self.log('[DR] Error or no embedding produced.')
                return
            try:
                emb = np.array(emb_obj)
                self._last_embedding = emb
                for i, s in enumerate(self.samples):
                    if i < len(emb):
                        s.embedding = np.array(emb[i])
                self.log(f"[DR] Done. Embedding shape={emb.shape}")
                # Auto-open DR window when finished (Show button not present in UI)
                try:
                    if emb.ndim == 1:
                        emb = emb.reshape(-1, 1)
                    self._open_dr_result_window(emb)
                except Exception:
                    pass
            except Exception as e:
                self.log(f"[DR] Finish update error: {e}")

        self._set_dr_status_color('brown')
        task = _DRTask(method, param_val, out_dim, X)
        task.signals.finished.connect(_on_finished)
        pool.start(task)

    def _on_dim_show_clicked(self):
        emb = getattr(self, '_last_embedding', None)
        if emb is None:
            self.log('[DR] No embedding to show.')
            return
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)
        try:
            self._open_dr_result_window(emb)
        except Exception as e:
            self.log(f"[DR] Show error: {e}")

    def _open_dr_result_window(self, emb: np.ndarray):
        from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QMenu, QTextEdit, QGroupBox, QLabel, QCheckBox, QComboBox
        from PyQt5.QtWidgets import QSpinBox
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import numpy as np
        import matplotlib.cm as cm

        class DRWindow(QMainWindow):
            def __init__(self, controller, emb):
                super().__init__(controller.main_window)
                self.c = controller
                self.setWindowTitle('DR Result')
                central = QWidget(); v = QVBoxLayout(central); self.setCentralWidget(central)
                # Controls: dims + color mode
                gb_ctrl = QGroupBox('View Controls')
                form = QFormLayout(gb_ctrl)
                self.sp_x = QSpinBox(); self.sp_x.setMinimum(1); self.sp_x.setMaximum(max(2, emb.shape[1])); self.sp_x.setValue(1)
                self.sp_y = QSpinBox(); self.sp_y.setMinimum(1); self.sp_y.setMaximum(max(2, emb.shape[1])); self.sp_y.setValue(min(2, emb.shape[1]))
                self.cb_color_seq = QCheckBox('Color by index (sequence)')
                self.cb_color_seq.setChecked(False)
                self.cb_show_decision = QCheckBox('Show decision boundary')
                self.cb_show_decision.setChecked(False)
                self.cb_cmap = QComboBox(); self.cb_cmap.addItems(['viridis','plasma','magma','inferno','turbo','jet','rainbow'])
                # Reset view button
                from PyQt5.QtWidgets import QPushButton
                self.btn_reset = QPushButton('Reset View')
                self.btn_reset.clicked.connect(self._reset_view)
                form.addRow(QLabel('x-dim'), self.sp_x)
                form.addRow(QLabel('y-dim'), self.sp_y)
                form.addRow(QLabel('Index colormap'), self.cb_cmap)
                form.addRow(self.cb_color_seq)
                form.addRow(self.cb_show_decision)
                form.addRow(self.btn_reset)
                v.addWidget(gb_ctrl)
                # Canvas
                self.fig = Figure(figsize=(5, 4)); self.canvas = FigureCanvas(self.fig)
                v.addWidget(self.canvas)
                self.ax = self.fig.add_subplot(111)
                # Ensure canvas captures key events (for ESC)
                try:
                    self.canvas.setFocusPolicy(Qt.StrongFocus)
                    self.canvas.setFocus()
                except Exception:
                    pass
                self.info = QTextEdit(); self.info.setReadOnly(True); v.addWidget(self.info)
                self.selected = set()
                self.emb = emb
                self.scatter = None
                self.highlight = None
                self._blink_on = True
                self.anim_timer = QTimer(self)
                self.anim_timer.setInterval(400)
                self.anim_timer.timeout.connect(self._toggle_blink)
                # Interaction state for pan
                self._panning = False
                self._pan_start = None  # (xdata, ydata, xlim, ylim)
                self._xlim = None
                self._ylim = None
                self._first_draw = True
                # events
                self.sp_x.valueChanged.connect(self._redraw)
                self.sp_y.valueChanged.connect(self._redraw)
                self.cb_color_seq.toggled.connect(self._redraw)
                self.cb_show_decision.toggled.connect(self._redraw)
                self.cb_cmap.currentTextChanged.connect(self._redraw)
                self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
                self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
                self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
                self.canvas.mpl_connect('scroll_event', self._on_scroll)
                self.canvas.mpl_connect('key_press_event', self._on_key)
                install_adaptive_window_profile(self, self._apply_screen_profile, apply_window_minimum=False)
                self._redraw()
                # Auto scale on first open so users don't need to click Reset
                try:
                    self._reset_view()
                except Exception:
                    pass
                self._first_draw = False

            def _apply_screen_profile(self, profile, screen):
                apply_density_profile(self, profile)

            def _current_xy(self):
                xdim = max(1, min(self.sp_x.value(), self.emb.shape[1])) - 1
                ydim = max(1, min(self.sp_y.value(), self.emb.shape[1])) - 1
                if xdim == ydim:
                    ydim = (ydim + 1) % self.emb.shape[1]
                xs = self.emb[:, xdim]
                ys = self.emb[:, ydim]
                return xs, ys

            def _colors(self, n):
                if self.cb_color_seq.isChecked():
                    cmap = cm.get_cmap(self.cb_cmap.currentText())
                    return cmap(np.linspace(0, 1, n))
                # color by category
                cats = [s.category for s in self.c.samples[:n]]
                uniq = {c: i for i, c in enumerate(sorted(set(cats)))}
                palette = cm.get_cmap('tab20')
                return [palette(uniq[c] % palette.N) for c in cats]

            def _redraw(self):
                self.ax.clear()
                xs, ys = self._current_xy()
                cols = self._colors(len(xs))
                # 先画决策边界（若有分类器且用户勾选）
                self._draw_decision_boundary(xs, ys)
                # 再画散点
                self.scatter = self.ax.scatter(xs, ys, s=20, c=cols)
                # draw highlight overlay if selection exists
                if self.selected:
                    sel_idx = np.array(sorted(self.selected), dtype=int)
                    hx = xs[sel_idx]
                    hy = ys[sel_idx]
                    self.highlight = self.ax.scatter(hx, hy, s=80, facecolors='none', edgecolors='yellow', linewidths=2)
                    if not self.anim_timer.isActive():
                        self.anim_timer.start()
                else:
                    self.highlight = None
                    if self.anim_timer.isActive():
                        self.anim_timer.stop()
                self.ax.set_xlabel(f'x-dim={self.sp_x.value()}')
                self.ax.set_ylabel(f'y-dim={self.sp_y.value()}')
                # preserve current limits across redraw
                if self._xlim is not None and self._ylim is not None:
                    try:
                        self.ax.set_xlim(self._xlim)
                        self.ax.set_ylim(self._ylim)
                    except Exception:
                        pass
                else:
                    # On initial draw, compute padded limits for visibility
                    try:
                        pad_x = (np.max(xs) - np.min(xs)) * 0.05 or 1.0
                        pad_y = (np.max(ys) - np.min(ys)) * 0.05 or 1.0
                        self._xlim = [float(np.min(xs) - pad_x), float(np.max(xs) + pad_x)]
                        self._ylim = [float(np.min(ys) - pad_y), float(np.max(ys) + pad_y)]
                        self.ax.set_xlim(self._xlim)
                        self.ax.set_ylim(self._ylim)
                    except Exception:
                        self.ax.relim(); self.ax.autoscale_view()
                self.fig.tight_layout()
                self.canvas.draw_idle()

            def _draw_decision_boundary(self, xs, ys):
                # 仅在有分类器且用户勾选时绘制
                if not self.cb_show_decision.isChecked():
                    return
                model = getattr(self.c, '_loaded_model', None)
                if not isinstance(model, dict):
                    return
                if model.get('feature_space') != 'embedding':
                    return
                clf = model.get('clf', None)
                if clf is None:
                    return
                try:
                    emb = self.emb
                    D = emb.shape[1]
                    emb_dim = int(model.get('embedding_dim', D))
                    if D != emb_dim:
                        return
                    # 当前 x/y 维度索引
                    xdim = max(1, min(self.sp_x.value(), D)) - 1
                    ydim = max(1, min(self.sp_y.value(), D)) - 1
                    if xdim == ydim:
                        ydim = (ydim + 1) % D

                    # 在当前坐标范围内生成网格
                    xlim = self.ax.get_xlim() if self._xlim is None else self._xlim
                    ylim = self.ax.get_ylim() if self._ylim is None else self._ylim
                    gx = np.linspace(xlim[0], xlim[1], 100)
                    gy = np.linspace(ylim[0], ylim[1], 100)
                    XX, YY = np.meshgrid(gx, gy)
                    base = emb.mean(axis=0)
                    grid_points = []
                    for xv, yv in zip(XX.ravel(), YY.ravel()):
                        v = base.copy()
                        v[xdim] = xv
                        v[ydim] = yv
                        grid_points.append(v)
                    grid_points = np.stack(grid_points, axis=0)

                    try:
                        Z = clf.predict(grid_points)
                    except Exception:
                        return

                    labels_unique = model.get('labels_unique', None)
                    if labels_unique is None or len(labels_unique) == 0:
                        return
                    # 将标签映射到整数
                    label_to_int = {lab: i for i, lab in enumerate(labels_unique)}
                    Z_int = np.vectorize(label_to_int.get)(Z)
                    Z_int = Z_int.reshape(XX.shape)

                    from matplotlib.colors import ListedColormap
                    # 使用 tab20 的前 N 个颜色
                    palette = cm.get_cmap('tab20')
                    colors = [palette(i % palette.N) for i in range(len(labels_unique))]
                    cmap = ListedColormap(colors)
                    self.ax.contourf(XX, YY, Z_int, alpha=0.15, cmap=cmap, levels=np.arange(len(labels_unique)+1)-0.5)
                except Exception:
                    return

            def _on_mouse_press(self, event):
                if event.button == 3:
                    self._show_menu(event)
                elif event.button == 1:
                    # select nearest point
                    if event.xdata is None or event.ydata is None:
                        return
                    xs, ys = self._current_xy()
                    d = np.hypot(xs - event.xdata, ys - event.ydata)
                    i = int(np.argmin(d))
                    mods = event.guiEvent.modifiers()
                    if mods & Qt.ControlModifier:
                        if i in self.selected:
                            self.selected.remove(i)
                        else:
                            self.selected.add(i)
                    else:
                        self.selected = {i}
                    self._update_info()
                    self._redraw()
                elif event.button == 2:
                    # middle button: start panning
                    if event.xdata is None or event.ydata is None:
                        return
                    self._panning = True
                    self._pan_start = (event.xdata, event.ydata, list(self.ax.get_xlim()), list(self.ax.get_ylim()))

            def _on_mouse_release(self, event):
                if event.button == 2:
                    self._panning = False
                    self._pan_start = None

            def _on_mouse_move(self, event):
                if not self._panning or self._pan_start is None:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                x0, y0, xlim, ylim = self._pan_start
                dx = event.xdata - x0
                dy = event.ydata - y0
                self._xlim = [xlim[0] - dx, xlim[1] - dx]
                self._ylim = [ylim[0] - dy, ylim[1] - dy]
                self.ax.set_xlim(self._xlim)
                self.ax.set_ylim(self._ylim)
                self.canvas.draw_idle()

            def _on_scroll(self, event):
                # zoom around cursor
                if event.xdata is None or event.ydata is None:
                    return
                # Scroll up (step>0) should zoom in (reduce span)
                scale = (1/1.2) if event.step > 0 else 1.2
                xlim = list(self.ax.get_xlim())
                ylim = list(self.ax.get_ylim())
                cx, cy = event.xdata, event.ydata
                new_w = (xlim[1] - xlim[0]) * scale
                new_h = (ylim[1] - ylim[0]) * scale
                self._xlim = [cx - new_w/2, cx + new_w/2]
                self._ylim = [cy - new_h/2, cy + new_h/2]
                self.ax.set_xlim(self._xlim)
                self.ax.set_ylim(self._ylim)
                self.canvas.draw_idle()

            def _reset_view(self):
                xs, ys = self._current_xy()
                try:
                    pad_x = (np.max(xs) - np.min(xs)) * 0.05 or 1.0
                    pad_y = (np.max(ys) - np.min(ys)) * 0.05 or 1.0
                    self._xlim = [float(np.min(xs) - pad_x), float(np.max(xs) + pad_x)]
                    self._ylim = [float(np.min(ys) - pad_y), float(np.max(ys) + pad_y)]
                except Exception:
                    self._xlim = None; self._ylim = None
                if self._xlim and self._ylim:
                    self.ax.set_xlim(self._xlim)
                    self.ax.set_ylim(self._ylim)
                self.canvas.draw_idle()

            def _on_key(self, event):
                if event.key == 'escape':
                    self.selected.clear()
                    self._update_info()
                    self._redraw()

            def keyPressEvent(self, e):
                # Qt-level ESC handling to ensure reliability
                try:
                    from PyQt5.QtCore import Qt as _Qt
                    if e.key() == _Qt.Key_Escape:
                        self.selected.clear()
                        self._update_info()
                        self._redraw()
                        e.accept()
                        return
                except Exception:
                    pass
                super().keyPressEvent(e)

            def _update_info(self):
                lines = []
                # 检查是否有可用于 embedding 空间的分类器
                model = getattr(self.c, '_loaded_model', None)
                clf = None
                labels_unique = None
                has_clf = False
                if isinstance(model, dict) and model.get('feature_space') == 'embedding':
                    clf = model.get('clf', None)
                    labels_unique = model.get('labels_unique', None)
                    try:
                        D = self.emb.shape[1]
                        emb_dim = int(model.get('embedding_dim', D))
                        has_clf = clf is not None and D == emb_dim
                    except Exception:
                        has_clf = False

                for i in sorted(self.selected):
                    if not (0 <= i < len(self.c.samples)):
                        continue
                    s = self.c.samples[i]
                    base = f"[{i}] {s.category} - {s.file_name}\n{s.file_path}"
                    extra = []
                    # 若存在分类器，给出该点在当前 embedding 空间中的预测结果（即所在分类区域）
                    if has_clf and 0 <= i < self.emb.shape[0]:
                        try:
                            vec = self.emb[i].reshape(1, -1)
                            pred = clf.predict(vec)[0]
                            extra.append(f"Classifier region: {pred}")
                            # 若样本已存有 predicted_label 或真实类别，可一并显示
                            if s.predicted_label is not None:
                                extra.append(f"Stored predicted: {s.predicted_label}")
                            if s.category:
                                extra.append(f"True category: {s.category}")
                            # 若支持概率输出，则给出每类概率
                            try:
                                if hasattr(clf, 'predict_proba'):
                                    proba = clf.predict_proba(vec)[0]
                                    # 使用分类器内部的 classes_ 顺序与概率列对齐
                                    try:
                                        classes = getattr(clf, 'classes_', None)
                                    except Exception:
                                        classes = None

                                    if classes is not None and len(classes) == len(proba):
                                        labels_for_proba = classes
                                    elif labels_unique is not None and len(labels_unique) == len(proba):
                                        # 回退：若 classes_ 不可用，则退回到保存的 labels_unique
                                        labels_for_proba = labels_unique
                                    else:
                                        # 最后兜底：用索引作为标签
                                        labels_for_proba = [str(i) for i in range(len(proba))]

                                    prob_str = ", ".join(
                                        f"{str(lab)}={float(p):.2f}" for lab, p in zip(labels_for_proba, proba)
                                    )
                                    extra.append(f"Probabilities: {prob_str}")
                            except Exception:
                                pass
                        except Exception:
                            pass

                    text = base
                    if extra:
                        text += "\n" + "\n".join(extra)
                    lines.append(text)

                self.info.setPlainText('\n\n'.join(lines))

            def _toggle_blink(self):
                if self.highlight is None:
                    return
                self._blink_on = not self._blink_on
                try:
                    self.highlight.set_edgecolor('yellow' if self._blink_on else 'orange')
                    self.canvas.draw_idle()
                except Exception:
                    pass

            def _show_menu(self, event):
                menu = QMenu(self)
                act_info = menu.addAction('Show Info')
                act_del = menu.addAction('Delete from group')
                act_move = menu.addAction('Move to moved/')
                act_copy = menu.addAction('Copy to moved/')
                a = menu.exec_(self.mapToGlobal(self.canvas.pos()))
                if a == act_info:
                    self._update_info()
                elif a in (act_del, act_move, act_copy):
                    self._apply_action(a == act_del, a == act_move, a == act_copy)

            def _apply_action(self, do_del, do_move, do_copy):
                import os, shutil
                sel = sorted(self.selected)
                if not sel:
                    return
                if do_del:
                    for i in reversed(sel):
                        # remove from controller and embedding
                        if 0 <= i < len(self.c.samples):
                            self.c.samples.pop(i)
                        try:
                            self.emb = np.delete(self.emb, i, axis=0)
                        except Exception:
                            pass
                    self.c._rebuild_table_grouped()
                    self.c.log(f"[DR] Deleted {len(sel)} samples from group.")
                    self.selected.clear(); self._redraw()
                else:
                    for i in sel:
                        s = self.c.samples[i]
                        folder = os.path.dirname(s.file_path)
                        moved = os.path.join(folder, 'moved')
                        os.makedirs(moved, exist_ok=True)
                        target = os.path.join(moved, s.file_name)
                        if do_move:
                            try:
                                shutil.move(s.file_path, target)
                                s.file_path = target
                                s.file_name = os.path.basename(target)
                            except Exception as e:
                                self.c.log(f"[DR] Move failed: {e}")
                        elif do_copy:
                            try:
                                shutil.copy2(s.file_path, target)
                                self.c.samples.append(Sample(
                                    file_path=target,
                                    file_name=os.path.basename(target),
                                    data_type=s.data_type,
                                    category=s.category,
                                ))
                            except Exception as e:
                                self.c.log(f"[DR] Copy failed: {e}")
                    self.c._rebuild_table_grouped()
                    self.c.log(f"[DR] {'Moved' if do_move else 'Copied'} {len(sel)} samples to moved/.")

        # Reuse single DR window instance
        try:
            if self._dr_window is None:
                self._dr_window = DRWindow(self, emb)
            else:
                # Update embedding and redraw in existing window
                self._dr_window.emb = emb
                # Update spinbox ranges according to new emb dims
                self._dr_window.sp_x.setMaximum(max(2, emb.shape[1]))
                self._dr_window.sp_y.setMaximum(max(2, emb.shape[1]))
                # keep current limits if set
                self._dr_window._redraw()
            if not self._dr_window.isVisible():
                move_window_to_cursor_screen(self._dr_window)
            self._dr_window.show()
            self._dr_window.raise_()
            self._dr_window.activateWindow()
        except Exception as e:
            self.log(f"[DR] Window error: {e}")

    def _build_feature_matrix(self) -> Optional[np.ndarray]:
        if not self.samples:
            return None
        h_min, w_min = None, None
        for s in self.samples:
            data = s.preprocessed_data if s.preprocessed_data is not None else s.raw_data
            if data is None:
                continue
            if s.data_type == '2D':
                h, w = data.shape[:2]
                h_min = h if h_min is None else min(h_min, h)
                w_min = w if w_min is None else min(w_min, w)
        self._feature_crop_shape_2d = (h_min, w_min) if (h_min and w_min) else None

        feats: List[np.ndarray] = []
        any_loaded = False
        for s in self.samples:
            data = s.preprocessed_data if s.preprocessed_data is not None else s.raw_data
            if data is None:
                continue
            any_loaded = True
            if s.data_type == '1D':
                arr = data
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    vec = arr[:, 1]
                else:
                    vec = arr.ravel()
            else:
                img = data
                if self._feature_crop_shape_2d is not None:
                    h, w = img.shape[:2]
                    ch, cw = self._feature_crop_shape_2d
                    y0 = max(0, (h - ch) // 2)
                    x0 = max(0, (w - cw) // 2)
                    img = img[y0:y0+ch, x0:x0+cw]
                vec = img.astype(float).ravel()
            vmin, vmax = float(np.min(vec)), float(np.max(vec))
            if vmax > vmin:
                vec = (vec - vmin) / (vmax - vmin)
            feats.append(vec.astype(np.float32))

        if not any_loaded or not feats:
            return None
        L = min(len(v) for v in feats)
        X = np.stack([v[:L] for v in feats], axis=0)
        return X

    # ---------------------------- 分类 ----------------------------
    def _on_clf_method_changed(self, text: str):
        label = getattr(self.ui, 'ClassificationKNnnNneighborsLabel', None)
        edit = getattr(self.ui, 'ClassificationKNnnNneighborsValue', None)
        if label is None or edit is None:
            return
        if text == 'KNN':
            label.setText('n_neighbors:')
            edit.setText('5')
        elif text == 'SVM':
            label.setText('C:')
            edit.setText('1.0')

    def _on_clf_start_clicked(self):
        """Start classification training on DR feature space in a background thread."""
        method = self.ui.ClassificationMethodCombox.currentText()
        param_text = self.ui.ClassificationKNnnNneighborsValue.text().strip()

        # 需要先有 DR 特征空间（仅作为开关，实际会在任务中重新按当前参数计算一次 DR）
        if getattr(self, '_last_embedding', None) is None:
            self.log('[CLS] No DR embedding available. Please run DR first.')
            return

        X_all = self._build_feature_matrix()
        if X_all is None or X_all.shape[0] == 0:
            self.log('[CLS] No data to process. Ensure files are imported.')
            return

        # 仅使用有数据且类别非空的样本
        cats = np.array([s.category if (s.category and s.category.strip()) else '' for s in self.samples])
        has_data = np.array([(s.preprocessed_data is not None) or (s.raw_data is not None) for s in self.samples])
        mask = (cats != '') & has_data
        if not np.any(mask):
            self.log('[CLS] No labeled samples with data found.')
            return

        labels_unique = np.unique(cats[mask])
        if len(labels_unique) == 0:
            self.log('[CLS] No category labels found.')
            return

        # 当前 DR 配置
        dr_method_widget = getattr(self.ui, 'DimensionalityReductionMethodCombox', None)
        dr_method = dr_method_widget.currentText() if dr_method_widget is not None else 'PCA'
        td_spin = getattr(self.ui, 'DimensionalityReductionTargetDimValue', None)
        n_spin = getattr(self.ui, 'DimensionalityReductionNNeighborValue', None)
        target_dim = int(td_spin.value()) if td_spin is not None else 2
        dr_param = int(n_spin.value()) if n_spin is not None else target_dim

        pool = QThreadPool.globalInstance()

        class _ClfSignals(QObject):
            finished = pyqtSignal(bool, object)  # ok, payload or error

        class _ClfTask(QRunnable):
            def __init__(self, X_all, cats, mask, method, param_text, dr_method, target_dim, dr_param):
                super().__init__()
                self.X_all = X_all
                self.cats = cats
                self.mask = mask
                self.method = method
                self.param_text = param_text
                self.dr_method = dr_method
                self.target_dim = max(1, int(target_dim) if target_dim else 2)
                self.dr_param = dr_param
                self.signals = _ClfSignals()

            def run(self):
                import traceback
                ok = False
                payload = {}
                try:
                    X = self.X_all
                    n_samples, n_features = X.shape
                    from sklearn.model_selection import train_test_split

                    # 根据 DR 方法在后台重新计算一次特征空间（用于分类 + 未来 transform）
                    dr_model = None
                    emb = None
                    dr_type = self.dr_method
                    tdim = max(1, min(self.target_dim, min(n_samples, n_features)))

                    if dr_type == 'PCA':
                        from sklearn.decomposition import PCA
                        dr_model = PCA(n_components=tdim, random_state=0)
                        emb = dr_model.fit_transform(X)
                    elif dr_type == 'UMAP':
                        try:
                            from umap import UMAP
                        except Exception:
                            payload = {'error': 'UMAP is not available. Please install umap-learn.'}
                            self.signals.finished.emit(False, payload)
                            return
                        try:
                            n_neighbors = int(self.dr_param)
                        except Exception:
                            n_neighbors = 15
                        n_neighbors = max(2, min(n_neighbors, max(2, n_samples - 1)))
                        dr_model = UMAP(n_components=max(2, tdim), n_neighbors=n_neighbors, random_state=0)
                        emb = dr_model.fit_transform(X)
                    elif dr_type == 't-SNE':
                        from sklearn.manifold import TSNE
                        if n_samples <= 1:
                            emb = X[:, :tdim]
                        else:
                            try:
                                perp = float(self.dr_param)
                            except Exception:
                                perp = 30.0
                            max_perp = max(1.0, n_samples - 1e-3)
                            perp = max(1.0, min(perp, max_perp))
                            dr_model = None  # t-SNE 无稳定 transform
                            emb = TSNE(
                                n_components=max(2, tdim),
                                perplexity=perp,
                                random_state=0,
                                init='pca',
                                learning_rate='auto'
                            ).fit_transform(X)
                    else:
                        # 无 DR：直接使用原始特征
                        dr_type = 'none'
                        emb = X

                    if emb is None:
                        payload = {'error': 'Failed to build feature space for classification.'}
                        self.signals.finished.emit(False, payload)
                        return

                    X_emb = emb[self.mask]
                    y = self.cats[self.mask]
                    labels_unique = np.unique(y)
                    counts = {lab: int(np.sum(y == lab)) for lab in labels_unique}
                    can_stratify = len(labels_unique) > 1 and all(c >= 2 for c in counts.values())
                    strat = y if can_stratify else None
                    Xtr, Xte, ytr, yte = train_test_split(
                        X_emb, y, test_size=0.2, random_state=0, stratify=strat
                    )

                    if self.method == 'KNN':
                        from sklearn.neighbors import KNeighborsClassifier
                        try:
                            n = int(float(self.param_text) if self.param_text else 5)
                        except Exception:
                            n = 5
                        clf = KNeighborsClassifier(n_neighbors=max(1, n))
                    elif self.method == 'SVM':
                        from sklearn.svm import SVC
                        from sklearn.calibration import CalibratedClassifierCV
                        try:
                            C = float(self.param_text) if self.param_text else 1.0
                        except Exception:
                            C = 1.0
                        # 若每个类别样本数足够，则使用外部概率校准；否则退回 SVC 内置概率
                        try:
                            # counts 和 labels_unique 在上方已计算
                            min_count = min(counts.values()) if counts else 0
                        except Exception:
                            min_count = 0
                        if min_count >= 3 and len(labels_unique) > 1:
                            n_folds = min(3, min_count)
                            base_svc = SVC(C=C, kernel='rbf', probability=False)
                            clf = CalibratedClassifierCV(base_svc, cv=n_folds, method='sigmoid')
                        else:
                            # 数据太少时，使用 SVC(probability=True) 避免交叉验证报错
                            clf = SVC(C=C, kernel='rbf', probability=True)
                    else:
                        payload = {'error': 'Unknown classification method.'}
                        self.signals.finished.emit(False, payload)
                        return

                    clf.fit(Xtr, ytr)
                    from sklearn.metrics import accuracy_score, confusion_matrix
                    ypred = clf.predict(Xte)
                    acc = accuracy_score(yte, ypred)
                    cm = confusion_matrix(yte, ypred, labels=labels_unique)

                    # 对所有已标注样本做预测
                    all_pred = clf.predict(X_emb)
                    masked_indices = np.where(self.mask)[0]

                    payload = {
                        'clf': clf,
                        'acc': float(acc),
                        'cm': cm,
                        'labels_unique': labels_unique,
                        'all_pred': all_pred,
                        'masked_indices': masked_indices,
                        'emb': emb,
                        'dr_type': dr_type,
                        'dr_model': dr_model,
                        'dr_params': {
                            'target_dim': int(tdim),
                            'param': self.dr_param,
                        },
                    }
                    ok = True
                except Exception:
                    payload = {'error': traceback.format_exc()}
                    ok = False
                try:
                    self.signals.finished.emit(ok, payload)
                except Exception:
                    pass

        def _on_finished(ok: bool, payload):
            self._set_import_clf_button_state(ready=False, running=False)
            if not ok or not isinstance(payload, dict):
                msg = payload.get('error') if isinstance(payload, dict) else 'Unknown error.'
                self.log(f"[CLS] Training failed: {msg}")
                return

            try:
                clf = payload['clf']
                acc = payload['acc']
                cm = payload['cm']
                labels_unique = payload['labels_unique']
                all_pred = payload['all_pred']
                masked_indices = payload['masked_indices']
                emb = np.array(payload['emb'])

                # 更新 embedding 到控制器（供 DRWindow 使用）
                self._last_embedding = emb
                for i, s in enumerate(self.samples):
                    if i < len(emb):
                        s.embedding = np.array(emb[i])

                # 写入预测标签
                for j, i_global in enumerate(masked_indices):
                    self.samples[i_global].predicted_label = str(all_pred[j])

                # 日志
                self.log(f"[CLS] {method} accuracy={acc:.3f}")
                self.log(f"[CLS] Confusion matrix labels={list(labels_unique)}")
                self.log(f"[CLS] Confusion matrix=\n{cm}")

                # 保存可复现的模型和 DR 信息
                self._loaded_model = {
                    'clf': clf,
                    'method': method,
                    'param': param_text,
                    'feature_crop_shape_2d': self._feature_crop_shape_2d,
                    'dr_type': payload.get('dr_type', dr_method),
                    'dr_params': payload.get('dr_params', {}),
                    'dr_model': payload.get('dr_model', None),
                    'feature_space': 'embedding',
                    'embedding_dim': int(emb.shape[1]),
                    'labels_unique': labels_unique,
                }

                self._set_import_clf_button_state(ready=True, running=False)
                self.classification_completed.emit({'accuracy': float(acc)})
            except Exception as e:
                self.log(f"[CLS] Error on finish: {e}")
            finally:
                try:
                    if task in self._classification_active_tasks:
                        self._classification_active_tasks.remove(task)
                except Exception:
                    pass

        # 启动后台训练
        self._set_import_clf_button_state(ready=False, running=True)
        task = _ClfTask(X_all, cats, mask, method, param_text, dr_method, target_dim, dr_param)
        task.signals.finished.connect(_on_finished)
        self._classification_active_tasks.append(task)
        pool.start(task)

    def _on_clf_save_clicked(self):
        if not self._loaded_model:
            self.log('[CLS] No trained model to save.')
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, 'Save Model', '', 'Joblib (*.joblib);;Pickle (*.pkl)')
        if not path:
            return
        try:
            import joblib
            joblib.dump(self._loaded_model, path)
            self.log(f"[CLS] Model saved to {path}")
        except Exception as e:
            self.log(f"[CLS] Save failed: {e}")

    def _on_clf_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self.main_window, 'Load Model', '', 'Joblib (*.joblib);;Pickle (*.pkl);;All (*.*)')
        if not path:
            return
        try:
            import joblib
            self._loaded_model = joblib.load(path)
            self.log(f"[CLS] Model loaded from {path}")

            # 恢复分类方法与参数
            try:
                method = self._loaded_model.get('method', None)
                param = self._loaded_model.get('param', '')
                clf_method = getattr(self.ui, 'ClassificationMethodCombox', None)
                clf_param_edit = getattr(self.ui, 'ClassificationKNnnNneighborsValue', None)
                if clf_method is not None and method is not None:
                    idx = clf_method.findText(method)
                    if idx >= 0:
                        clf_method.setCurrentIndex(idx)
                if clf_param_edit is not None:
                    clf_param_edit.setText(str(param))
            except Exception:
                pass

            # 恢复 DR 区域设置
            try:
                dr_type = self._loaded_model.get('dr_type', None)
                dr_params = self._loaded_model.get('dr_params', {})
                dr_method_widget = getattr(self.ui, 'DimensionalityReductionMethodCombox', None)
                td_spin = getattr(self.ui, 'DimensionalityReductionTargetDimValue', None)
                n_spin = getattr(self.ui, 'DimensionalityReductionNNeighborValue', None)
                if dr_method_widget is not None and dr_type is not None:
                    idx = dr_method_widget.findText(dr_type)
                    if idx >= 0:
                        dr_method_widget.setCurrentIndex(idx)
                        # 触发一次范围/可见性更新
                        self._on_dim_method_changed(dr_type)
                if td_spin is not None and 'target_dim' in dr_params:
                    td_spin.setValue(int(dr_params['target_dim']))
                if n_spin is not None and 'param' in dr_params:
                    try:
                        n_spin.setValue(int(dr_params['param']))
                    except Exception:
                        pass
            except Exception:
                pass

            # 有模型在内存中，标记 ImportClassify 按钮为可用（绿）
            self._set_import_clf_button_state(ready=True, running=False)
        except Exception as e:
            self.log(f"[CLS] Load failed: {e}")

    # ---------------------------- 其它 ----------------------------
    def validate_parameters(self):
        return True, 'OK'

    def reset_to_defaults(self):
        self.samples.clear()
        self._row_to_index.clear()
        self._path_to_index.clear()
        self._feature_crop_shape_2d = None
        self._loaded_model = None
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        if table is not None:
            table.setRowCount(0)
        self.log('[Reset] Classification page reset.')

    def _on_import_classify_clicked(self):
        """使用已训练/加载的模型对当前类别的新样本进行分类，并在需要时更新 DR embedding。

        要求：
        - 内存中存在 _loaded_model；
        - 若 dr_type 为 t-SNE，则仅提示不支持对新样本 transform；
        - 若 dr_type 为 PCA/UMAP/none，则利用 dr_model 或原始特征进行预测，
          并在 PCA/UMAP 情况下更新 _last_embedding 以便 DRWindow 显示。"""
        if not self._loaded_model:
            self.log('[CLS] No trained model in memory. Please train or load first.')
            return

        clf = self._loaded_model.get('clf', None)
        if clf is None:
            self.log('[CLS] Loaded model is invalid (no classifier).')
            return

        dr_type = self._loaded_model.get('dr_type', 'none')
        dr_model = self._loaded_model.get('dr_model', None)

        # 当前类别
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        current_category = lw.currentItem().text() if (lw is not None and lw.currentItem() is not None) else None
        if not current_category:
            self.log('[CLS] Please select a category in the list.')
            return

        # 构建特征矩阵，并建立“特征行 -> 样本索引”映射
        X_all = self._build_feature_matrix()
        if X_all is None or X_all.shape[0] == 0:
            self.log('[CLS] No data available to classify. Please import data first.')
            return

        # 有数据的样本索引（顺序与 _build_feature_matrix 中堆叠顺序一致）
        valid_indices = []
        for idx, s in enumerate(self.samples):
            if (s.preprocessed_data is not None) or (s.raw_data is not None):
                valid_indices.append(idx)
        if not valid_indices:
            self.log('[CLS] No loaded samples to classify.')
            return
        if len(valid_indices) != X_all.shape[0]:
            try:
                self.log('[CLS] Warning: feature matrix and sample mapping size differ; classification may be unreliable.')
            except Exception:
                pass

        # 属于当前类别的行位置（在特征矩阵中的行号）
        cat_row_indices = [row for row, idx in enumerate(valid_indices)
                           if self.samples[idx].category == current_category]
        if not cat_row_indices:
            self.log(f"[CLS] Category '{current_category}' has no loaded samples to classify.")
            return

        import numpy as _np

        if dr_type == 't-SNE':
            self.log('[CLS] Current model uses t-SNE DR. Applying to new samples is not supported (no transform). Please retrain with PCA/UMAP.')
            return

        # 根据 DR 类型构造用于分类和可视化的特征
        if dr_type in ('PCA', 'UMAP') and dr_model is not None:
            try:
                X_emb_all = dr_model.transform(X_all)
            except Exception as e:
                self.log(f"[CLS] DR transform failed on new data: {e}")
                return
            X_for_clf = X_emb_all
            # 更新 embedding 以便 DRWindow 展示
            self._last_embedding = _np.array(X_emb_all)
            # 按 valid_indices 映射回样本；若映射长度不一致则退化到旧策略
            if len(valid_indices) == len(X_emb_all):
                for row, idx in enumerate(valid_indices):
                    if idx < len(self.samples):
                        self.samples[idx].embedding = _np.array(X_emb_all[row])
            else:
                for i, s in enumerate(self.samples):
                    if i < len(X_emb_all):
                        s.embedding = _np.array(X_emb_all[i])
        else:
            # 无 DR：直接在原始特征空间分类
            X_for_clf = X_all

        # 取出当前类别对应的特征行
        X_cat = X_for_clf[cat_row_indices]
        try:
            y_pred = clf.predict(X_cat)
        except Exception as e:
            self.log(f"[CLS] Predict failed on new data: {e}")
            return

        # 将预测结果写回对应样本
        for idx_local, row in enumerate(cat_row_indices):
            if row < len(valid_indices):
                idx_global = valid_indices[row]
                if idx_global < len(self.samples):
                    self.samples[idx_global].predicted_label = str(y_pred[idx_local])

        self.log(f"[CLS] Applied model to category '{current_category}', classified {len(cat_row_indices)} samples.")
        self._rebuild_table_grouped()

        # 若我们刚刚更新了 embedding，可选择自动打开 DR 结果窗口
        try:
            if getattr(self, '_last_embedding', None) is not None and dr_type in ('PCA', 'UMAP'):
                emb = self._last_embedding
                if emb.ndim == 1:
                    emb = emb.reshape(-1, 1)
                self._open_dr_result_window(emb)
        except Exception:
            pass

    # ---------------------------- 覆盖增强：同步类别重建/重命名/删除 ----------------------------
    def _rebuild_table_grouped(self):
        """按列表中的类别一一对应显示：
        - 行数 = 上方列表中的类别数
        - 仅 Category 列可编辑；其他列只读
        - Status 显示 Loaded m/n
        """
        table = self.ui.ClassificationImportTableWidget
        try:
            self._ensure_table_headers()
        except Exception as e:
            self.log(f"[UI] Table header refresh failed: {e}")
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        names: List[str] = []
        seen = set()
        for category in self.import_cache.keys():
            if category not in seen:
                names.append(category)
                seen.add(category)
        if lw is not None:
            for i in range(lw.count()):
                category = lw.item(i).text()
                if category not in seen:
                    names.append(category)
                    seen.add(category)
        for sample in self.samples:
            category = sample.category
            if category and category not in seen:
                names.append(category)
                seen.add(category)
        current_category = self._get_current_name() or self.current_item_name
        self.log(f"[Table] headers={TABLE_HEADERS}")
        self.log(f"[Table] categories={names}")
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                for col in range(table.columnCount()):
                    widget = table.cellWidget(row, col)
                    if widget is not None:
                        table.removeCellWidget(row, col)
                        widget.deleteLater()
            self._row_to_index.clear()
            self._row_to_category.clear()
            table.clearContents()
            table.setColumnCount(len(TABLE_HEADERS))
            table.setHorizontalHeaderLabels(TABLE_HEADERS)
            table.setRowCount(len(names))
            self._ensure_table_headers()
            for row, category in enumerate(names):
                indices = [idx for idx, s in enumerate(self.samples) if s.category == category]
                self._row_to_category[row] = category
                if indices:
                    self._row_to_index[row] = indices[0]
                total = len(indices)
                loaded = sum(1 for i in indices if (self.samples[i].preprocessed_data is not None) or (self.samples[i].raw_data is not None))
                n1d = sum(1 for i in indices if self.samples[i].data_type == '1D')
                n2d = sum(1 for i in indices if self.samples[i].data_type == '2D')
                if n1d and n2d:
                    type_text = '1D/2D'
                elif n1d:
                    type_text = '1D'
                elif n2d:
                    type_text = '2D'
                else:
                    type_text = '-'
                if total == 0:
                    status_text = 'No files scanned'
                elif loaded > 0:
                    status_text = f"Loaded {loaded}/{total}"
                else:
                    status_text = f"Listed {total} files"

                it_label = QTableWidgetItem(category)
                it_label.setFlags(it_label.flags() | Qt.ItemIsEditable)
                table.setItem(row, TABLE_COL_LABEL, it_label)

                def _readonly(value: str) -> QTableWidgetItem:
                    item = QTableWidgetItem(value)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    return item

                shapes: List[str] = []
                predictions: Dict[str, int] = {}
                confidences: List[float] = []
                for i in indices:
                    sample = self.samples[i]
                    d = sample.preprocessed_data if sample.preprocessed_data is not None else sample.raw_data
                    if d is not None:
                        if sample.data_type == '1D':
                            shapes.append(f"[{d.shape[0]}]")
                        elif d.ndim == 2:
                            shapes.append(f"[{d.shape[0]},{d.shape[1]}]")
                        elif d.ndim == 3:
                            shapes.append(f"[{d.shape[0]},{d.shape[1]},{d.shape[2]}]")
                    if sample.predicted_label:
                        predictions[sample.predicted_label] = predictions.get(sample.predicted_label, 0) + 1
                    confidence = getattr(sample, 'confidence', None)
                    if confidence is None:
                        confidence = getattr(sample, 'prediction_confidence', None)
                    if confidence is not None:
                        try:
                            confidences.append(float(confidence))
                        except (TypeError, ValueError):
                            pass
                shape_text = ', '.join(sorted(set(shapes))) if shapes else '-'
                prediction_text = '-'
                if predictions:
                    prediction_text = ', '.join(f"{k}:{v}" for k, v in sorted(predictions.items()))
                confidence_text = '-'
                if confidences:
                    confidence_text = f"{float(np.mean(confidences)):.3f}"

                table.setItem(row, TABLE_COL_TYPE, _readonly(type_text))
                table.setItem(row, TABLE_COL_FILES, _readonly(str(total)))
                table.setItem(row, TABLE_COL_LOADED, _readonly(str(loaded)))
                table.setItem(row, TABLE_COL_SHAPE, _readonly(shape_text))
                table.setItem(row, TABLE_COL_STATUS, _readonly(status_text))
                table.setItem(row, TABLE_COL_PREDICTION, _readonly(prediction_text))
                table.setItem(row, TABLE_COL_CONFIDENCE, _readonly(confidence_text))
                self.log(f"[Table] row {row}: label={category}, files={total}, loaded={loaded}")

                from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
                cell = QWidget()
                hb = QHBoxLayout(cell)
                hb.setContentsMargins(0, 0, 0, 0)
                hb.setSpacing(4)
                le = QLineEdit()
                le.setPlaceholderText(f"1-{max(1, total)}")
                le.setMaximumWidth(54)
                try:
                    le.setText(str(self._category_show_index.get(category, 1)))
                except Exception:
                    le.setText('1')
                le.returnPressed.connect(partial(self._on_preview_index_entered, category, le))
                btn = QPushButton('Preview')
                btn.setMaximumWidth(82)
                btn.setEnabled(total > 0)
                btn.clicked.connect(partial(self._preview_category_by_name, category, le, True))
                le.setEnabled(total > 0)
                hb.addWidget(le)
                hb.addWidget(btn)
                table.setCellWidget(row, TABLE_COL_PREVIEW, cell)
            if current_category:
                for row, category in self._row_to_category.items():
                    if category == current_category:
                        table.selectRow(row)
                        break
        finally:
            table.blockSignals(False)
        self._update_dataset_status_cards()
        self._set_table_responsive_columns('stable')
        self.log(f"[Table] Rebuilt grouped table: {len(names)} labels, {len(self.samples)} samples, rows={table.rowCount()}, categories={names}")

    def _ensure_table_headers(self):
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        if table is None:
            return
        from PyQt5.QtWidgets import QHeaderView
        table.setColumnCount(len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        try:
            header = table.horizontalHeader()
            header.setSectionResizeMode(TABLE_COL_LABEL, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_TYPE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_FILES, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_LOADED, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_SHAPE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_STATUS, QHeaderView.Stretch)
            header.setSectionResizeMode(TABLE_COL_PREDICTION, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_CONFIDENCE, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(TABLE_COL_PREVIEW, QHeaderView.ResizeToContents)
        except Exception as e:
            self._log_exception('[UI] Table header setup failed', e)

    def _on_item_renamed(self, item):
        """重命名：迁移缓存 + 同步样本类别名 + 重建聚合表。"""
        if item is None:
            return
        if self._in_item_changed:
            return
        self._in_item_changed = True
        try:
            lw = self.ui.ClassificationImportListWidget
            new_name = (item.text() or '').strip()
            if not new_name:
                new_name = self._rename_old_name or self.current_item_name or 'Item'
                item.setText(new_name)
            if any(lw.item(i) is not item and lw.item(i).text() == new_name for i in range(lw.count())):
                unique = self._ensure_unique_name(lw, new_name)
                if unique != new_name:
                    new_name = unique
                    item.setText(new_name)
            old_name = self._rename_old_name or self.current_item_name
            if not old_name or old_name == new_name:
                self.current_item_name = new_name
                self._refresh_labels()
            else:
                if old_name in self.import_cache and new_name not in self.import_cache:
                    self.import_cache[new_name] = self.import_cache.pop(old_name)
                elif new_name in self.import_cache and old_name in self.import_cache:
                    for k in ('path', 'rule'):
                        if not self.import_cache[new_name].get(k):
                            self.import_cache[new_name][k] = self.import_cache[old_name].get(k, '')
                    self.import_cache.pop(old_name, None)
                else:
                    self.import_cache[new_name] = self.import_cache.get(old_name, {'path': '', 'rule': '*'})
                    self.import_cache.pop(old_name, None)
                if lw.currentItem() is item:
                    self.current_item_name = new_name
                self._refresh_labels()
            if lw.currentItem() is item:
                rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
                path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
                if rule_edit is not None:
                    rule_edit.setText(self._get_cached('rule', '*') or '*')
                if path_edit is not None:
                    path_edit.setText(self._get_cached('path', '') or '')
            if old_name and old_name != new_name:
                for s in self.samples:
                    if s.category == old_name:
                        s.category = new_name
                self._rebuild_table_grouped()
        finally:
            self._rename_old_name = None
            self._in_item_changed = False
            self._sync_dynamic_attributes()

    def _on_minus_clicked(self):
        lw = self.ui.ClassificationImportListWidget
        item = lw.currentItem()
        if item is None:
            return
        name = item.text()
        row = lw.row(item)
        lw.takeItem(row)
        self.import_cache.pop(name, None)
        if lw.count() > 0:
            lw.setCurrentRow(min(row, lw.count() - 1))
            self.current_item_name = lw.currentItem().text()
        else:
            self.current_item_name = None
        self._on_list_selection_changed()
        self._sync_dynamic_attributes()
        if self.samples and any(s.category == name for s in self.samples):
            self.samples = [s for s in self.samples if s.category != name]
        self._rebuild_table_grouped()

    # ---------------------------- 表格交互（选择/重命名） ----------------------------
    def _on_table_selection_changed(self):
        try:
            table = self.ui.ClassificationImportTableWidget
            r = table.currentRow()
            if r is None or r < 0:
                return
            it = table.item(r, TABLE_COL_LABEL)
            if it is None:
                return
            name = (it.text() or '').strip()
            if not name:
                return
            lw = self.ui.ClassificationImportListWidget
            for i in range(lw.count()):
                if lw.item(i).text() == name:
                    lw.setCurrentRow(i)
                    break
            self._preview_category_row(r)
        except Exception as e:
            self._log_exception('[Preview] Table selection update failed', e)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        # 仅处理第一列（Category）改名，其他列不允许编辑
        if item is None or item.column() != TABLE_COL_LABEL:
            return
        if self._in_table_item_changed:
            return
        self._in_table_item_changed = True
        try:
            table = self.ui.ClassificationImportTableWidget
            row = item.row()
            new_name = (item.text() or '').strip()
            lw = self.ui.ClassificationImportListWidget
            if lw is None or row < 0:
                return
            old_name = self._row_to_category.get(row)
            if not old_name:
                old_item = table.item(row, TABLE_COL_LABEL)
                old_name = (old_item.text() or '').strip() if old_item is not None else ''
            if not old_name:
                return
            if new_name == old_name or not new_name:
                # 回退显示
                item.setText(old_name)
                return
            # 通过上方列表的重命名逻辑来迁移缓存/样本
            for i in range(lw.count()):
                if lw.item(i).text() == old_name:
                    lw.item(i).setText(new_name)
                    break
            # _on_item_renamed 会被触发并负责重建表格
        finally:
            self._in_table_item_changed = False

    # ---------------------------- Panel control handlers ----------------------------
    def _on_image_auto_scale_toggled(self, value: bool):
        self._image_auto_scale = bool(value)
        try:
            # enable/disable vmin/vmax inputs when auto on/off
            if hasattr(self, '_panel_widgets'):
                self._panel_widgets['vmin'].setEnabled(not self._image_auto_scale)
                self._panel_widgets['vmax'].setEnabled(not self._image_auto_scale)
        except Exception:
            pass
        self._refresh_current_preview()

    def _on_image_log_scale_toggled(self, value: bool):
        self._image_log_scale = bool(value)
        self._refresh_current_preview()

    def _on_image_vmin_changed(self, value: float):
        self._image_vmin = float(value)
        # If user adjusts vmin/vmax, disable auto to apply manual range
        try:
            if self._image_auto_scale and hasattr(self, '_panel_widgets'):
                self._panel_widgets['2d_auto'].setChecked(False)
            self._image_auto_scale = False
        except Exception:
            self._image_auto_scale = False
        self._refresh_current_preview()

    def _on_image_vmax_changed(self, value: float):
        self._image_vmax = float(value)
        # If user adjusts vmin/vmax, disable auto to apply manual range
        try:
            if self._image_auto_scale and hasattr(self, '_panel_widgets'):
                self._panel_widgets['2d_auto'].setChecked(False)
            self._image_auto_scale = False
        except Exception:
            self._image_auto_scale = False
        self._refresh_current_preview()

    def _on_image_vmin_editing_finished(self):
        try:
            sp = self._panel_widgets.get('vmin') if hasattr(self, '_panel_widgets') else None
            if sp is not None:
                self._on_image_vmin_changed(float(sp.value()))
        except Exception:
            pass

    def _on_image_vmax_editing_finished(self):
        try:
            sp = self._panel_widgets.get('vmax') if hasattr(self, '_panel_widgets') else None
            if sp is not None:
                self._on_image_vmax_changed(float(sp.value()))
        except Exception:
            pass

    def _on_curve_logy_toggled(self, value: bool):
        self._curve_log_y = bool(value)
        self._refresh_current_preview()

    def _on_image_cmap_changed(self, name: str):
        self._image_cmap_name = str(name)
        self._refresh_current_preview()

    def _on_image_dim_x_changed(self, value: int):
        self._image_dim_x = int(value)
        self._refresh_current_preview()

    def _on_image_dim_y_changed(self, value: int):
        self._image_dim_y = int(value)
        self._refresh_current_preview()

    def _refresh_current_preview(self):
        try:
            table = getattr(self.ui, 'ClassificationImportTableWidget', None)
            if table is None:
                # Fallback: refresh last shown sample
                if self._last_preview_index is not None:
                    self.show_sample(self._last_preview_index)
                return
            r = table.currentRow()
            if r is None or r < 0:
                # Fallback to last shown sample
                if self._last_preview_index is not None:
                    self.show_sample(self._last_preview_index)
                return
            self._preview_category_row(r)
        except Exception:
            pass

    # Classification external preview removed.
