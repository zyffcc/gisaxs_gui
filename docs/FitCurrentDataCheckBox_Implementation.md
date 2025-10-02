# FitCurrentDataCheckBox功能实现说明

## 功能需求实现

### 1. 自动勾选fitCurrentDataCheckBox
**实现位置**: `_perform_cut()`方法开头
```python
# 0. 自动勾选fitCurrentDataCheckBox
if hasattr(self.ui, 'fitCurrentDataCheckBox'):
    self.ui.fitCurrentDataCheckBox.setChecked(True)
```

当用户点击`gisaxsInputCutButton`按钮执行Cut操作时，会自动勾选`fitCurrentDataCheckBox`复选框。

### 2. 状态保存到用户参数
**实现位置**: `get_session_data()`方法
```python
# 添加拟合选项状态保存
'fit_current_data': self.ui.fitCurrentDataCheckBox.isChecked() if hasattr(self.ui, 'fitCurrentDataCheckBox') else False,
'fit_log_x': self.ui.fitLogXCheckBox.isChecked() if hasattr(self.ui, 'fitLogXCheckBox') else False,
'fit_log_y': self.ui.fitLogYCheckBox.isChecked() if hasattr(self.ui, 'fitLogYCheckBox') else False,
'fit_norm': self.ui.fitNormCheckBox.isChecked() if hasattr(self.ui, 'fitNormCheckBox') else False
```

以下复选框的状态会保存到`config/user_parameters.json`的`fitting.last_session`中：
- `fitCurrentDataCheckBox`
- `fitLogXCheckBox` 
- `fitLogYCheckBox`
- `fitNormCheckBox`

### 3. 状态恢复功能
**实现位置**: `restore_session()`方法
```python
# 恢复拟合选项状态
if hasattr(self.ui, 'fitCurrentDataCheckBox'):
    self.ui.fitCurrentDataCheckBox.setChecked(session_data.get('fit_current_data', False))
if hasattr(self.ui, 'fitLogXCheckBox'):
    self.ui.fitLogXCheckBox.setChecked(session_data.get('fit_log_x', False))
if hasattr(self.ui, 'fitLogYCheckBox'):
    self.ui.fitLogYCheckBox.setChecked(session_data.get('fit_log_y', False))
if hasattr(self.ui, 'fitNormCheckBox'):
    self.ui.fitNormCheckBox.setChecked(session_data.get('fit_norm', False))
```

应用程序重启时会自动恢复这些复选框的上次状态。

### 4. 条件化的显示选项处理
**实现位置**: `_on_fit_display_option_changed()`方法
```python
def _on_fit_display_option_changed(self):
    """拟合显示选项改变时的处理"""
    try:
        # 检查fitCurrentDataCheckBox的状态
        if hasattr(self.ui, 'fitCurrentDataCheckBox') and self.ui.fitCurrentDataCheckBox.isChecked():
            # fitCurrentDataCheckBox被勾选时，重新执行cut操作
            self._perform_cut()
            self.status_updated.emit("Fit display options changed - Cut results updated")
        else:
            # fitCurrentDataCheckBox未勾选或不存在时，暂不处理（之后可扩展）
            self.status_updated.emit("Fit display options changed - fitCurrentDataCheckBox not checked, no action taken")
            
    except Exception as e:
        self.status_updated.emit(f"Fit display option change error: {str(e)}")
```

**行为逻辑**:
- 当`fitCurrentDataCheckBox`**被勾选**时：
  - `fitLogXCheckBox`、`fitLogYCheckBox`、`fitNormCheckBox`状态改变会调用`self._perform_cut()`重新执行Cut操作
- 当`fitCurrentDataCheckBox`**未勾选**时：
  - 这些复选框状态改变时暂不执行任何操作（为后续扩展预留）

### 5. UI连接设置
**实现位置**: `_setup_connections()`方法
```python
# 连接拟合显示选项复选框
if hasattr(self.ui, 'fitCurrentDataCheckBox'):
    # fitCurrentDataCheckBox不需要特殊处理，只需要保存状态
    pass
    
if hasattr(self.ui, 'fitLogXCheckBox'):
    self.ui.fitLogXCheckBox.toggled.connect(self._on_fit_display_option_changed)
    
if hasattr(self.ui, 'fitLogYCheckBox'):
    self.ui.fitLogYCheckBox.toggled.connect(self._on_fit_display_option_changed)
    
if hasattr(self.ui, 'fitNormCheckBox'):
    self.ui.fitNormCheckBox.toggled.connect(self._on_fit_display_option_changed)
```

## 使用流程

1. **执行Cut操作**：用户点击`gisaxsInputCutButton`
   - 自动勾选`fitCurrentDataCheckBox`
   - 执行正常的Cut处理流程

2. **调整显示选项**：用户改变`fitLogXCheckBox`、`fitLogYCheckBox`、`fitNormCheckBox`状态
   - 如果`fitCurrentDataCheckBox`已勾选：立即重新执行Cut显示更新
   - 如果`fitCurrentDataCheckBox`未勾选：暂不处理

3. **状态持久化**：应用退出时自动保存所有相关复选框状态

4. **状态恢复**：应用启动时自动恢复上次的复选框状态

## 扩展性设计

`_on_fit_display_option_changed()`方法中预留了扩展空间，当`fitCurrentDataCheckBox`未勾选时可以添加其他处理逻辑，如：
- 仅更新显示格式而不重新计算
- 缓存显示选项用于下次Cut操作
- 其他自定义行为

## 错误处理

所有方法都包含了适当的异常处理：
- 使用`hasattr()`检查UI控件是否存在
- 捕获并记录异常信息
- 提供默认值防止程序崩溃