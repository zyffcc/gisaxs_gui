# 参数独立性实现文档

## 问题描述

原系统中 Trainset 模块和 Fitting 模块的探测器参数（特别是 beam center）使用相同的全局参数空间，导致两个模块的参数相互影响，用户修改其中一个模块的参数时会影响另一个模块。

## 解决方案

实现独立的参数命名空间，让两个模块使用完全独立的参数存储空间。

### 参数命名空间设计

1. **Trainset 模块参数**: `trainset.detector.*`
   - `trainset.detector.beam_center_x`
   - `trainset.detector.beam_center_y`
   - `trainset.detector.distance`
   - `trainset.detector.preset`
   - 等其他探测器参数

2. **Fitting 模块参数**: `fitting.detector.*`
   - `fitting.detector.beam_center_x`
   - `fitting.detector.beam_center_y`
   - `fitting.detector.distance`
   - `fitting.detector.preset`
   - 等其他探测器参数

## 修改的文件

### 1. core/global_params.py
- 增加了 `trainset.detector` 和 `fitting.detector` 参数组
- 确保两个模块的参数完全独立存储

### 2. controllers/trainset_controller.py
- 修改所有参数读取从 `trainset.detector.*` 空间
- 修改所有参数设置到 `trainset.detector.*` 空间
- 确保 UI 控件同步使用正确的参数空间

### 3. controllers/fitting_controller.py
- Q轴计算使用 `fitting.detector.beam_center_x/y` 参数
- 确保 Fitting 模块独立于 Trainset 参数

### 4. ui/detector_parameters_dialog.py
- 探测器参数对话框专门管理 `fitting.detector.*` 参数
- 与 Trainset 模块参数完全隔离

### 5. main.py
- Trainset UI 控件同步使用 `trainset.detector.*` 参数
- 确保启动时参数正确加载

## 参数独立性验证

### 测试步骤
1. 在 Trainset 模块中修改 beam center 参数
2. 切换到 Fitting 模块，打开 "Detector Parameters" 对话框
3. 确认 Fitting 模块的 beam center 参数没有改变
4. 在 Fitting 模块中修改 beam center 参数
5. 切换回 Trainset 模块，确认参数没有受影响

### 预期结果
- 两个模块的参数完全独立
- 修改其中一个模块不会影响另一个模块
- 参数在程序重启后能正确保存和恢复

## 技术细节

### 参数同步机制
- Trainset 模块的 UI 控件直接与 `trainset.detector.*` 参数同步
- Fitting 模块的对话框直接与 `fitting.detector.*` 参数同步
- 移除了原有的跨模块参数同步代码

### 向后兼容性
- 首次运行时，如果新参数空间为空，会从默认值初始化
- 保持了原有的参数结构和默认值

## 注意事项

1. **参数初始化**: 确保每个模块在首次使用时正确初始化其参数空间
2. **UI 同步**: 确保 UI 控件的信号连接使用正确的参数空间
3. **持久化存储**: 参数会自动保存到各自的命名空间中

## 未来扩展

如果需要添加更多模块，可以按照相同的模式创建独立的参数空间：
- `module_name.detector.*`
- `module_name.sample.*`
- 等等

这种设计确保了各模块间的完全独立性。
