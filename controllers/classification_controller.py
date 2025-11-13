"""
Classification 控制器 - Import 区域逻辑（从零实现）
"""

import os
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QMenu
from core.global_params import global_params  # 全局参数管理器，用于持久化到 user_parameters.json


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

    # ---------------------------- 初始化与连接 ----------------------------
    def initialize(self):
        if self._initialized:
            return

        self._setup_connections()
        # 优先恢复缓存，再初始化UI，避免默认项覆盖
        try:
            params = global_params.get_module_parameters('classification')
            cache = params.get('import_cache') if isinstance(params, dict) else None
            if isinstance(cache, dict) and cache:
                self.import_cache = {k: {'path': v.get('path', ''), 'rule': v.get('rule', '*')} for k, v in cache.items()}
        except Exception as e:
            print(f"[classification] restore import_cache failed: {e}")
        self._initialize_ui()
        self._initialized = True

    def _setup_connections(self):
        """只连接 Import 区域相关控件，移除旧的无效信号。"""
        lw = getattr(self.ui, 'ClassificationImportListWidget', None)
        plus_btn = getattr(self.ui, 'ClassificationImportPlusButton', None)
        minus_btn = getattr(self.ui, 'ClassificationImportMinusButton', None)
        path_btn = getattr(self.ui, 'ClassificationImportFolderPathLabel', None)
        rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
        path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)

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

    def _on_item_renamed(self, item):
        """名称变更后迁移缓存，并刷新标签。"""
        if item is None:
            return
        if self._in_item_changed:
            return

        self._in_item_changed = True
        try:
            lw = self.ui.ClassificationImportListWidget
            new_name = (item.text() or '').strip()
            # 空名则回退为旧名
            if not new_name:
                new_name = self._rename_old_name or self.current_item_name or 'Item'
                item.setText(new_name)

            # 若重名，确保唯一
            if any(lw.item(i) is not item and lw.item(i).text() == new_name for i in range(lw.count())):
                unique = self._ensure_unique_name(lw, new_name)
                if unique != new_name:
                    new_name = unique
                    item.setText(new_name)

            # 选择用于迁移的旧名
            old_name = self._rename_old_name or self.current_item_name

            if not old_name or old_name == new_name:
                self.current_item_name = new_name
                self._refresh_labels()
            else:
                if old_name in self.import_cache and new_name not in self.import_cache:
                    self.import_cache[new_name] = self.import_cache.pop(old_name)
                elif new_name in self.import_cache and old_name in self.import_cache:
                    # 合并（保留新名，补全空字段）
                    for k in ('path', 'rule'):
                        if not self.import_cache[new_name].get(k):
                            self.import_cache[new_name][k] = self.import_cache[old_name].get(k, '')
                    self.import_cache.pop(old_name, None)
                else:
                    # 新名首次出现
                    self.import_cache[new_name] = self.import_cache.get(old_name, {'path': '', 'rule': '*'})
                    self.import_cache.pop(old_name, None)

                # 若该项为当前选中，更新 current_item_name
                if lw.currentItem() is item:
                    self.current_item_name = new_name
                self._refresh_labels()

            # 更新显示（如果是当前项）
            if lw.currentItem() is item:
                rule_edit = getattr(self.ui, 'ClassificationImportRuleValue', None)
                path_edit = getattr(self.ui, 'ClassificationImportFolderPathValue', None)
                if rule_edit is not None:
                    rule_edit.setText(self._get_cached('rule', '*') or '*')
                if path_edit is not None:
                    path_edit.setText(self._get_cached('path', '') or '')
        finally:
            self._rename_old_name = None
            self._in_item_changed = False
            self._sync_dynamic_attributes()

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

    def _on_minus_clicked(self):
        lw = self.ui.ClassificationImportListWidget
        item = lw.currentItem()
        if item is None:
            return
        name = item.text()
        row = lw.row(item)
        lw.takeItem(row)
        self.import_cache.pop(name, None)
        # 选择新的当前项
        if lw.count() > 0:
            lw.setCurrentRow(min(row, lw.count() - 1))
            self.current_item_name = lw.currentItem().text()
        else:
            self.current_item_name = None
        self._on_list_selection_changed()
        self._sync_dynamic_attributes()

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

    def _choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            'Select File',
            '',
            'All Files (*);;Images (*.png *.jpg *.jpeg *.tif *.tiff);;Text (*.txt *.dat);;HDF5 (*.h5 *.hdf5)'
        )
        if file_path:
            path_edit = self.ui.ClassificationImportFolderPathValue
            rule_edit = self.ui.ClassificationImportRuleValue
            path_edit.setText(file_path)
            # 规则自动更新为 *.ext
            ext = os.path.splitext(file_path)[1].lstrip('.')
            auto_rule = f"*.{ext}" if ext else '*'
            rule_edit.setText(auto_rule)
            self._set_cached('path', file_path)
            self._set_cached('rule', auto_rule)
            self.status_updated.emit(f"Selected file: {file_path}")

    def _on_rule_edited(self):
        rule_edit = self.ui.ClassificationImportRuleValue
        rule = rule_edit.text().strip() or '*'
        self._set_cached('rule', rule)

    def _on_path_edited(self):
        path_edit = self.ui.ClassificationImportFolderPathValue
        path = path_edit.text().strip()
        self._set_cached('path', path)

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

