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
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtCore import QThreadPool, QRunnable, QEvent, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QMessageBox, QInputDialog, QMenu,
    QTableWidgetItem, QPushButton, QGraphicsScene
)
from core.global_params import global_params  # 全局参数管理器，用于持久化到 user_parameters.json


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

    # ---------------------------- 初始化与连接 ----------------------------
    def initialize(self):
        if self._initialized:
            return

        self._setup_connections()
        # Build classification panel with 1D/2D display controls
        try:
            self._setup_classification_panel()
        except Exception:
            pass
        # Prepare table headers/columns
        try:
            self._ensure_table_headers()
        except Exception:
            pass
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
        except Exception:
            pass
        self._initialized = True
        # 初始化时确保降维控件状态与当前方法一致（隐藏/显示 nNeighborsWidget 等）
        try:
            dim_method = getattr(self.ui, 'DimensionalityReductionMethodCombox', None)
            if dim_method is not None:
                self._on_dim_method_changed(dim_method.currentText())
        except Exception:
            pass

    def _setup_connections(self):
        """连接 Classification 页面相关控件（Import/降维/分类）。"""
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        plus_btn = getattr(self.ui, 'ClassificationImportPlusButton', None)
        minus_btn = getattr(self.ui, 'ClassificationImportMinusButton', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
        import_btn = getattr(self.ui, 'ClassificationImportImportButton', None)
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
            path_btn.setText(f"Path - {name}")
        if rule_label is not None:
            rule_label.setText(f"Rule - {name}:")

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
        path = path_edit.text().strip()
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
            it = self.ui.ClassificationImportTableWidget.item(row, 0)
            if it is not None:
                name = it.text()
                lw = self.ui.ClassificationImportListWidget
                for i in range(lw.count()):
                    if lw.item(i).text() == name:
                        lw.setCurrentRow(i)
                        break
        except Exception:
            pass
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
    def _on_import_clicked(self):
        table = self.ui.ClassificationImportTableWidget
        try:
            table.setColumnCount(6)
            self._ensure_table_headers()
        except Exception:
            pass
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        if table.rowCount() == 0:
            self.log('[Import] No categories to import.')
            return
        # 以优先顺序确定当前类别：表格选择 -> 列表选择
        cat = None
        r = table.currentRow()
        if r is not None and r >= 0:
            it = table.item(r, 0)
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

        for i in remaining:
            task = _ImportTask(self, i)
            task.signals.result.connect(_on_result)
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
        it = table.item(row, 0)
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
                self.cb_cmap = QComboBox(); self.cb_cmap.addItems(['viridis','plasma','magma','inferno','turbo','jet','rainbow'])
                # Reset view button
                from PyQt5.QtWidgets import QPushButton
                self.btn_reset = QPushButton('Reset View')
                self.btn_reset.clicked.connect(self._reset_view)
                form.addRow(QLabel('x-dim'), self.sp_x)
                form.addRow(QLabel('y-dim'), self.sp_y)
                form.addRow(QLabel('Index colormap'), self.cb_cmap)
                form.addRow(self.cb_color_seq)
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
                self.cb_cmap.currentTextChanged.connect(self._redraw)
                self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
                self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
                self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
                self.canvas.mpl_connect('scroll_event', self._on_scroll)
                self.canvas.mpl_connect('key_press_event', self._on_key)
                self._redraw()
                # Auto scale on first open so users don't need to click Reset
                try:
                    self._reset_view()
                except Exception:
                    pass
                self._first_draw = False

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
                for i in sorted(self.selected):
                    s = self.c.samples[i]
                    lines.append(f"[{i}] {s.category} - {s.file_name}\n{s.file_path}")
                self.info.setPlainText('\n'.join(lines))

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
        method = self.ui.ClassificationMethodCombox.currentText()
        param_text = self.ui.ClassificationKNnnNneighborsValue.text().strip()
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
        X = X_all[mask]
        y = cats[mask]
        labels_unique = np.unique(y)
        if len(labels_unique) == 0:
            self.log('[CLS] No category labels found.')
            return
        try:
            from sklearn.model_selection import train_test_split
            # 仅在各类样本数>=2时启用分层抽样
            counts = {lab: int(np.sum(y == lab)) for lab in labels_unique}
            can_stratify = len(labels_unique) > 1 and all(c >= 2 for c in counts.values())
            strat = y if can_stratify else None
            Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
                X, y, np.arange(len(y)), test_size=0.2, random_state=0, stratify=strat
            )
            if method == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                n = int(float(param_text) if param_text else 5)
                clf = KNeighborsClassifier(n_neighbors=max(1, n))
            elif method == 'SVM':
                from sklearn.svm import SVC
                C = float(param_text) if param_text else 1.0
                clf = SVC(C=C, kernel='rbf', probability=True)
            else:
                self.log('[CLS] Unknown method.')
                return
            clf.fit(Xtr, ytr)
            from sklearn.metrics import accuracy_score, confusion_matrix
            ypred = clf.predict(Xte)
            acc = accuracy_score(yte, ypred)
            cm = confusion_matrix(yte, ypred, labels=labels_unique)
            self.log(f"[CLS] {method} accuracy={acc:.3f}")
            self.log(f"[CLS] Confusion matrix labels={list(labels_unique)}")
            self.log(f"[CLS] Confusion matrix=\n{cm}")
            # 对所有样本做预测（仅对mask内样本）
            all_pred = clf.predict(X)
            masked_indices = np.where(mask)[0]
            for j, i_global in enumerate(masked_indices):
                self.samples[i_global].predicted_label = str(all_pred[j])
            self._loaded_model = {
                'clf': clf,
                'method': method,
                'param': param_text,
                'feature_crop_shape_2d': self._feature_crop_shape_2d,
            }
            self.classification_completed.emit({'accuracy': float(acc)})
        except Exception as e:
            self.log(f"[CLS] Error: {e}")

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

    # ---------------------------- 覆盖增强：同步类别重建/重命名/删除 ----------------------------
    def _rebuild_table_grouped(self):
        """按列表中的类别一一对应显示：
        - 行数 = 上方列表中的类别数
        - 仅 Category 列可编辑；其他列只读
        - Status 显示 Loaded m/n
        """
        table = self.ui.ClassificationImportTableWidget
        # Ensure headers/columns
        try:
            self._ensure_table_headers()
        except Exception:
            pass
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        table.setRowCount(0)
        self._row_to_index.clear()
        names: List[str] = []
        if lw is not None:
            names = [lw.item(i).text() for i in range(lw.count())]
        # 构建每个类别的统计
        for category in names:
            indices = [idx for idx, s in enumerate(self.samples) if s.category == category]
            row = table.rowCount()
            table.insertRow(row)
            if indices:
                self._row_to_index[row] = indices[0]
            total = len(indices)
            loaded = sum(1 for i in indices if (self.samples[i].preprocessed_data is not None) or (self.samples[i].raw_data is not None))
            n1d = sum(1 for i in indices if self.samples[i].data_type == '1D')
            n2d = sum(1 for i in indices if self.samples[i].data_type == '2D')
            type_text = f"1D:{n1d}, 2D:{n2d}"
            status_text = f"Loaded {loaded}/{total}"

            # Category（可编辑）
            it_cat = QTableWidgetItem(category)
            it_cat.setFlags(it_cat.flags() | Qt.ItemIsEditable)
            table.setItem(row, 0, it_cat)
            # Type（只读）
            it_type = QTableWidgetItem(type_text)
            it_type.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 1, it_type)
            # Category（重复列，只读，保持 UI 原布局）
            it_cat2 = QTableWidgetItem(category)
            it_cat2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 2, it_cat2)
            # Status（只读）
            it_status = QTableWidgetItem(status_text)
            it_status.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 3, it_status)
            # Shape（只读）
            shapes: List[str] = []
            for i in indices:
                d = self.samples[i].preprocessed_data if self.samples[i].preprocessed_data is not None else self.samples[i].raw_data
                if d is None:
                    continue
                if self.samples[i].data_type == '1D':
                    shapes.append(f"[{d.shape[0]}]")
                else:
                    if d.ndim == 2:
                        shapes.append(f"[{d.shape[0]},{d.shape[1]}]")
                    elif d.ndim == 3:
                        shapes.append(f"[{d.shape[0]},{d.shape[1]},{d.shape[2]}]")
            shape_text = ' '.join(sorted(set(shapes))) if shapes else ''
            it_shape = QTableWidgetItem(shape_text)
            it_shape.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 4, it_shape)
            # 预处理摘要（只读）
            pp_descs = {self.samples[i].preprocessing_desc for i in indices if self.samples[i].preprocessing_desc}
            pp_text = ','.join(sorted(pp_descs)) if pp_descs else ''
            it_pp = QTableWidgetItem(pp_text)
            it_pp.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 5, it_pp)
            # 预览输入框+按钮
            from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
            cell = QWidget()
            hb = QHBoxLayout(cell)
            hb.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setPlaceholderText(f"1-{max(1, total)}")
            try:
                le.setText(str(self._category_show_index.get(category, 1)))
            except Exception:
                pass
            le.returnPressed.connect(lambda c=category, e=le: self._on_preview_index_entered(c, e))
            btn = QPushButton('Show')
            btn.clicked.connect(lambda _, r=row: self._preview_category_row(r))
            hb.addWidget(le)
            hb.addWidget(btn)
            table.setCellWidget(row, 5, cell)

    def _ensure_table_headers(self):
        table = getattr(self.ui, 'ClassificationImportTableWidget', None)
        if table is None:
            return
        from PyQt5.QtWidgets import QHeaderView
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(['Category', 'Type', 'Category', 'Status', 'Shape', 'Preview'])
        try:
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        except Exception:
            pass

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
                renamed_any = False
                for s in self.samples:
                    if s.category == old_name:
                        s.category = new_name
                        renamed_any = True
                if renamed_any:
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
            it = table.item(r, 0)
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
        except Exception:
            pass

    def _on_table_item_changed(self, item: QTableWidgetItem):
        # 仅处理第一列（Category）改名，其他列不允许编辑
        if item is None or item.column() != 0:
            return
        if self._in_table_item_changed:
            return
        self._in_table_item_changed = True
        try:
            table = self.ui.ClassificationImportTableWidget
            row = item.row()
            new_name = (item.text() or '').strip()
            lw = self.ui.ClassificationImportListWidget
            if lw is None or row < 0 or row >= lw.count():
                return
            old_name = lw.item(row).text()
            if new_name == old_name or not new_name:
                # 回退显示
                item.setText(old_name)
                return
            # 通过上方列表的重命名逻辑来迁移缓存/样本
            lw.item(row).setText(new_name)
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

