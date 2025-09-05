# Cut参数实时更新功能实现

## 功能概述
实现了Cut Line Center参数和Detector Parameters参数修改后自动重新计算和更新Cut图片的功能。

## 主要改进

### 1. **Cut Line参数实时更新**
- **触发条件**: 修改Cut Line Center的四个数值（Vertical/Parallel Center 和 Vertical/Parallel CutLine）
- **前提条件**: 必须已经执行过Cut操作，显示区域有图像
- **功能**: 参数修改后自动重新执行Cut操作，同时更新主UI和独立窗口的显示

### 2. **Detector Parameters实时更新**
- **触发条件**: 在Detector Parameters弹窗中修改任何参数
- **功能**: 参数修改后如果已有Cut结果，自动重新计算Cut图片
- **更新内容**: 同时更新主界面fitGraphicsView和独立matplotlib窗口

### 3. **智能步长调整**

#### Cut Line参数步长（根据模式自动调整）
- **Q-space模式**: 步长 = 0.01
- **Pixel模式**: 步长 = 1.0
- **动态更新**: 模式切换时自动调整步长

#### Detector Parameters步长（固定设置）
- **Distance**: 步长 = 0.1
- **Grazing angle**: 步长 = 0.01  
- **Wavelength**: 步长 = 0.001
- **Beam Center X/Y**: 步长 = 0.01，显示两位小数
- **Pixel size X/Y**: 步长 = 0.1

## 实现细节

### 代码修改位置

#### `controllers/fitting_controller.py`
1. **`_on_cutline_parameters_changed()`方法增强**
   - 添加Cut结果检查逻辑
   - 自动重新执行`_perform_cut()`
   - 智能状态提示

2. **`_on_detector_parameters_changed()`方法增强**
   - 添加步长更新调用
   - 添加自动Cut重新计算功能
   - 改进状态反馈

3. **新增方法**
   - `_is_q_space_mode()`: 检测当前显示模式
   - `_update_cutline_step_sizes()`: 动态更新Cut Line参数步长

4. **参数配置优化**
   - 移除硬编码的步长设置
   - 添加动态步长更新调用

#### `ui/detector_parameters_dialog.py`
1. **所有Detector参数控件步长设置**
   - Distance: `setSingleStep(0.1)`
   - Grazing Angle: `setSingleStep(0.01)`
   - Wavelength: `setSingleStep(0.001)`
   - Beam Center: `setSingleStep(0.01)` + `setDecimals(2)`
   - Pixel Size: `setSingleStep(0.1)`

## 用户体验改进

### 1. **实时反馈**
- 参数修改立即看到Cut结果变化
- 主界面和独立窗口同步更新
- 智能状态提示区分有无Cut数据

### 2. **操作便利性**
- 滚轮调整步长符合使用习惯
- 不同模式下自动适配步长大小
- 避免手动重新点击Cut按钮

### 3. **精度控制**
- Q-space模式精细调整（0.01步长）
- Pixel模式整数调整（1.0步长）
- Detector参数针对性步长设置

## 技术特点

### 1. **智能检测**
- 检查是否已有Cut数据
- 检查当前显示模式
- 避免无效操作

### 2. **自动同步**
- 主界面和独立窗口实时同步
- 参数变化立即反映到显示
- 保持数据一致性

### 3. **错误处理**
- 完整的异常捕获和提示
- 状态信息清晰反馈
- 稳定性保证

## 使用流程

1. **加载数据**: 导入GISAXS数据
2. **首次Cut**: 设置参数并点击Cut按钮
3. **参数调整**: 修改Cut Line Center或Detector Parameters
4. **自动更新**: 系统自动重新计算和显示Cut结果
5. **实时预览**: 在主界面和独立窗口查看更新结果

## 性能优化

- 只在有Cut数据时才触发重新计算
- 避免重复无效计算
- 高效的参数检测机制
- 优化的状态更新逻辑
