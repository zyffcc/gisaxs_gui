# 初始化期间避免触发方法调用的修复

## 问题描述

在应用启动时，用户什么都没做的情况下会弹出"Please import an image first."的警告对话框。这是由于在初始化期间设置UI控件状态时触发了信号连接，导致相关方法被意外调用。

## 根本原因

1. **初始化顺序问题**：原来的初始化顺序是先设置信号连接，再初始化UI状态
2. **信号触发**：在设置复选框、输入框等控件的初始值时，触发了已连接的信号
3. **方法调用**：信号触发导致`_perform_cut()`、`_on_fit_display_option_changed()`等方法被调用
4. **缺少数据检查**：这些方法在没有图像数据时会弹出警告对话框

## 解决方案

### 1. 调整初始化顺序

**修改位置**: `initialize()`方法

```python
def initialize(self):
    """初始化控制器"""
    if self._initialized:
        return
        
    # 先初始化UI状态（不触发信号）
    self._initialize_ui()
    # 然后设置信号连接
    self._setup_connections()
    # 会话管理已移到主控制器统一处理
    self._initialized = True
    self._initializing = False  # 初始化完成
```

### 2. 使用信号阻塞机制

**新增方法**: `_initialize_fit_checkboxes()`

```python
def _initialize_fit_checkboxes(self):
    """初始化拟合相关复选框状态（阻塞信号避免触发方法调用）"""
    try:
        # 初始化fitCurrentDataCheckBox（默认不勾选）
        if hasattr(self.ui, 'fitCurrentDataCheckBox'):
            self.ui.fitCurrentDataCheckBox.blockSignals(True)
            self.ui.fitCurrentDataCheckBox.setChecked(False)
            self.ui.fitCurrentDataCheckBox.blockSignals(False)
        # ... 其他复选框同样处理
```

### 3. 添加初始化状态标志

**修改位置**: 构造函数

```python
# 初始化标志
self._initialized = False
self._initializing = True  # 标记正在初始化中
```

### 4. 在关键方法中添加初始化检查

**修改的方法**:
- `_on_fit_display_option_changed()`
- `_on_cutline_parameters_changed()`  
- `_on_parameter_display_changed()`

```python
def _on_fit_display_option_changed(self):
    """拟合显示选项改变时的处理"""
    try:
        # 如果正在初始化中，跳过处理
        if getattr(self, '_initializing', True):
            return
        # ... 原有逻辑
```

### 5. 会话恢复时的信号阻塞

**新增方法**: `_restore_fit_checkboxes()`

```python
def _restore_fit_checkboxes(self, session_data):
    """恢复拟合复选框状态（阻塞信号避免触发方法调用）"""
    try:
        # 恢复fitCurrentDataCheckBox
        if hasattr(self.ui, 'fitCurrentDataCheckBox'):
            self.ui.fitCurrentDataCheckBox.blockSignals(True)
            self.ui.fitCurrentDataCheckBox.setChecked(session_data.get('fit_current_data', False))
            self.ui.fitCurrentDataCheckBox.blockSignals(False)
        # ... 其他复选框同样处理
```

## 修改的文件

- `controllers/fitting_controller.py`
  - 调整`initialize()`方法的初始化顺序
  - 添加`_initialize_fit_checkboxes()`方法
  - 添加`_restore_fit_checkboxes()`方法
  - 在关键方法中添加初始化状态检查
  - 添加`_initializing`状态标志

## 技术要点

### 1. `blockSignals()`机制

```python
widget.blockSignals(True)   # 阻塞信号
widget.setChecked(value)    # 设置状态（不触发信号）
widget.blockSignals(False)  # 恢复信号
```

### 2. 初始化状态管理

```python
self._initializing = True   # 构造函数中设置
# ... 初始化过程
self._initializing = False  # 初始化完成后清除
```

### 3. 防御性编程

```python
if getattr(self, '_initializing', True):
    return  # 安全的默认值处理
```

## 效果

1. **消除启动警告**：不再在启动时弹出"Please import an image first."
2. **保持功能完整**：所有正常的用户交互功能都保持不变
3. **提高稳定性**：避免了初始化期间的意外方法调用
4. **保持性能**：初始化顺序优化，减少不必要的处理

## 测试验证

- ✅ 启动应用不会弹出警告对话框
- ✅ 复选框状态保存和恢复正常工作
- ✅ 用户交互触发的方法调用正常工作
- ✅ Cut操作自动勾选功能正常工作