# 多文件预测功能实现说明

## 功能概述

为GISAXS GUI应用程序实现了完整的多文件预测功能，包括结果列表、进度跟踪、过滤排序、导出等功能。

## 主要组件

### 1. MultiFilePredictResultsWidget
- **位置**: `controllers/multifile_predict_results.py`
- **功能**: 多文件预测结果展示和管理的主要UI组件
- **特性**:
  - 实时结果列表更新
  - 状态过滤（Pending/Running/Completed/Failed/Cancelled）
  - 文件名搜索过滤
  - 多列排序（文件名、状态、耗时、置信度）
  - 进度条显示
  - 右键菜单支持

### 2. PredictResult数据类
- **功能**: 单个预测结果的数据模型
- **字段**:
  - 文件路径和名称
  - 预测状态（枚举）
  - 开始/结束时间
  - 处理耗时
  - 错误信息
  - 预测数据和置信度

### 3. 表格模型系统
- **PredictResultsTableModel**: 基础表格数据模型
- **PredictResultsFilterModel**: 支持过滤和排序的代理模型

### 4. MultiFilePredictManager
- **功能**: 多文件预测的后台管理器
- **特性**:
  - 使用ThreadPoolExecutor进行批量处理
  - 实时状态更新信号
  - 支持取消操作

### 5. ExportDialog
- **功能**: 导出配置对话框
- **选项**:
  - 导出范围：全部/选中/当前显示
  - 导出格式：JSONL/JPG/ASCII 1D曲线

## UI布局设计

### Predict-2D Tab布局
```
+----------------------------------+----------------------------------+
| predict2dGraphicsView            | predict2dParameterWidget         |
| (图像显示区域)                     | ┌──────────────────────────────┐ |  
| (0,0)                           | │ Color Scale & Export Controls │ |
|                                 | │ (现有参数控件)                  │ |
|                                 | ├──────────────────────────────┤ |
|                                 | │ MultiFilePredictResultsWidget│ |
|                                 | │ (多文件结果列表)               │ |
|                                 | │ - 只在multifile模式显示        │ |
|                                 | └──────────────────────────────┘ |
| (0,1 - 充满整个右侧)               |                                 |
+----------------------------------+----------------------------------+
```

- **内嵌设计**: 结果列表直接添加到predict2dParameterWidget内部
- **位置**: Export按钮下方，与现有控件垂直布局
- **自适应**: 只在多文件模式下显示，不影响单文件模式

## 集成到主控制器

### 修改的文件
- `controllers/gisaxs_predict_controller.py`

### 主要修改点

1. **UI集成**:
   - 在`_setup_multifile_ui()`中直接获取predict2dParameterWidget
   - 将结果Widget添加到参数控件的布局中，位于export按钮下方
   - 内嵌式设计，不改变整体布局结构

2. **预测逻辑重构**:
   - `_predict_multi_files()`使用新的队列处理系统
   - 支持文件范围过滤
   - 实时状态更新

3. **导出功能增强**:
   - `_on_predict_export_clicked()`自动检测单文件/多文件模式
   - 多文件模式调用专用的导出界面
   - 支持多种导出格式

4. **事件处理**:
   - `_on_multifile_result_selected()`: 快速预览功能
   - `_on_multifile_export_requested()`: 导出处理
   - 预测开始/完成/进度更新的信号处理

## 用户交互流程

### 多文件预测流程
1. 选择"Multi Files"模式
2. 选择输入文件夹
3. 可选：设置文件范围（如"1-10,15,20-25"）
4. 点击"Predict"开始批量预测
5. 实时查看结果列表中的状态更新
6. 可以随时点击完成的结果进行快速预览

### 结果管理
- **过滤**: 按状态或文件名过滤
- **排序**: 点击列标题排序，支持升序/降序
- **预览**: 点击已完成的结果行查看预测结果
- **右键菜单**: 
  - "Export This Result": 导出单个结果
  - "Retry Prediction": 重试失败的预测（预留功能）

### 导出功能
1. 点击"Export..."按钮
2. 选择导出范围：
   - All Results: 所有结果
   - Selected Results: 选中的结果
   - Current Display: 当前过滤显示的结果
3. 选择导出类型：
   - **JSONL**: 结构化数据，每行一个文件的完整信息
   - **JPG**: 预测结果图像，保存在专用文件夹
   - **ASCII**: 1D曲线数据，所有文件的曲线合并到一个表格

## 导出格式说明

### JSONL格式
```json
{"index": 0, "filename": "file001.cbf", "filepath": "/path/to/file001.cbf", "timestamp": "2026-01-15T10:30:00", "processing_time": 1.23, "confidence": 0.95, "prediction_data": {...}}
```

### JPG格式
- 文件夹: `prediction_images_20260115_103000/`
- 文件: `file001.cbf_0001_hr.jpg`

### ASCII格式
```
# Prediction 1D Curves Export
# Generated: 20260115_103000
# Columns: file001.cbf_h | file001.cbf_r | file002.cbf_h | file002.cbf_r
# Index	file001.cbf_h	file001.cbf_r	file002.cbf_h	file002.cbf_r
0	1.234567	2.345678	1.123456	2.234567
1	1.234568	2.345679	1.123457	2.234568
...
```

## 测试

运行测试脚本验证功能：
```bash
python test_multifile.py
```

## 兼容性

- **单文件模式**: 完全保留原有功能，不受影响
- **多文件模式**: 新功能，向后兼容
- **UI适配**: 自动根据模式调整布局

## 性能特性

- **异步处理**: 使用线程池避免UI阻塞
- **内存优化**: 大型数组只保存关键信息到JSONL
- **渐进更新**: 每完成一个文件立即更新UI
- **可取消**: 支持中途取消预测任务

## 扩展性

代码设计支持未来扩展：
- 添加更多导出格式
- 支持并行预测（当前单线程顺序处理）
- 添加更多过滤和排序选项
- 集成更复杂的预测状态管理