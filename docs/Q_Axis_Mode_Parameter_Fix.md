# Q轴模式切换时参数异常修复

## 问题描述

用户报告：刚启动应用后，第一次点击Cut按钮，然后修改Detector Parameters里的参数，Cut Vertical/Parallel和Center的数值会被修改成很大很大的数值。需要Auto Finding一下或手动修改一下第二次才正常。

## 根本原因分析

### 1. 初始化状态问题
- `_last_q_mode`在构造函数中初始化为`None`
- 当用户第一次修改探测器参数时，`_update_parameter_values_for_q_axis()`方法被调用
- 由于`_last_q_mode`为`None`，代码误认为Q轴显示模式发生了改变

### 2. 错误的坐标转换
- 代码错误地执行了坐标转换（像素↔Q空间）
- 小的像素值（如10像素）转换成Q空间坐标后变成很大的数值
- 这导致UI控件显示异常大的数值

### 3. 触发条件
```python
# 问题代码逻辑
if hasattr(self, '_last_q_mode') and self._last_q_mode == show_q_axis:
    return  # 跳过转换
# 当_last_q_mode=None时，不会跳过，导致执行转换
```

## 解决方案

### 1. 修改模式检查逻辑

**修改位置**: `_update_parameter_values_for_q_axis()`方法

```python
def _update_parameter_values_for_q_axis(self):
    """根据Q轴显示状态切换时转换参数数值并更新显示"""
    try:
        show_q_axis = self._should_show_q_axis()
        
        # 如果是第一次调用，直接设置当前模式但不进行转换
        if not hasattr(self, '_last_q_mode') or self._last_q_mode is None:
            self._last_q_mode = show_q_axis
            print(f"DEBUG: 首次初始化Q模式状态 = {show_q_axis}")
            self.status_updated.emit(f"Q轴显示模式已设置: {'Q坐标' if show_q_axis else '像素坐标'}")
            return
        
        # 检查当前参数是否已经是目标模式（避免重复转换）
        if self._last_q_mode == show_q_axis:
            print("DEBUG: 模式未变化，跳过转换")
            return
        # ... 其余转换逻辑保持不变
```

### 2. 添加Q模式状态初始化方法

**新增方法**: `_initialize_q_mode_state()`

```python
def _initialize_q_mode_state(self):
    """初始化Q模式状态，避免第一次调用时误触发转换"""
    try:
        # 获取当前Q轴显示状态并设置为初始状态
        current_q_mode = self._should_show_q_axis()
        self._last_q_mode = current_q_mode
        print(f"DEBUG: 初始化Q模式状态 = {current_q_mode}")
    except Exception as e:
        # 如果获取状态失败，默认设置为像素模式
        self._last_q_mode = False
        print(f"初始化Q模式状态失败，默认为像素模式: {e}")
```

### 3. 在初始化流程中调用

**修改位置**: `_initialize_ui()`方法

```python
# 初始化Cut Line标签的单位
self._update_cutline_labels_units()

# 初始化Q模式状态（避免第一次调用时误触发转换）
self._initialize_q_mode_state()

# 检查依赖库
self._check_dependencies()
```

### 4. 在会话恢复时重新初始化

**修改位置**: `restore_session()`方法

```python
# 更新显示
self._update_stack_display()

# 重新初始化Q模式状态（避免恢复后误触发转换）
self._initialize_q_mode_state()

# 如果AutoShow启用且有文件，自动显示
if (session_data.get('auto_show', False) and 
    last_file and os.path.exists(last_file)):
    # ... 自动显示逻辑
```

## 修复效果

### 1. 启动时行为
- ✅ 应用启动后正确识别当前Q轴显示模式
- ✅ `_last_q_mode`被正确初始化为当前模式状态
- ✅ 不会误触发坐标转换

### 2. 首次使用行为
- ✅ 用户第一次点击Cut按钮正常工作
- ✅ 修改探测器参数时不会异常转换坐标
- ✅ 参数数值保持合理范围

### 3. 模式切换行为
- ✅ 真正的模式切换时正确执行坐标转换
- ✅ 重复的模式切换操作被正确跳过
- ✅ 转换后的数值准确

### 4. 会话恢复行为
- ✅ 恢复会话后Q模式状态正确初始化
- ✅ 避免恢复过程中的误转换

## 技术要点

### 1. 状态初始化策略
```python
# 初始化时立即设置当前状态，避免None值
self._last_q_mode = self._should_show_q_axis()
```

### 2. 首次调用检测
```python
# 检测是否为首次调用
if not hasattr(self, '_last_q_mode') or self._last_q_mode is None:
    # 首次调用：设置状态但不转换
    self._last_q_mode = show_q_axis
    return
```

### 3. 防御性编程
```python
# 异常处理确保有合理的默认值
except Exception as e:
    self._last_q_mode = False  # 默认像素模式
```

## 相关文件

- `controllers/fitting_controller.py` - 主要修复
  - `_update_parameter_values_for_q_axis()` - 修改核心逻辑
  - `_initialize_q_mode_state()` - 新增初始化方法
  - `_initialize_ui()` - 添加初始化调用
  - `restore_session()` - 添加恢复时初始化

## 测试场景

1. **启动测试**：启动应用→点击Cut→修改Detector Parameters→检查参数是否正常
2. **模式切换测试**：真正切换Q轴显示模式→检查参数是否正确转换
3. **会话恢复测试**：保存会话→重启应用→恢复会话→检查状态是否正常