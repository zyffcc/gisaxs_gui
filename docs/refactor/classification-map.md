# Classification 渐进迁移地图

## 当前基线

- `controllers/classification_controller.py`（1807 行）同时承担页面装配、数据源编辑、导入、训练、预测、降维、模型保存、会话持久化、绘图和文件对话框。
- `controllers/classification_data_service.py`（603 行）混合目录扫描、文件读取、QC、预处理和特征矩阵构造。
- `controllers/classification_training_service.py`（590 行）直接创建 sklearn/UMAP 对象并执行交叉验证、排名和最终训练。
- `controllers/classification_workers.py`（258 行）是 Qt `QRunnable`，直接持有数据与训练实现；训练仍发生在 GUI 进程内的线程中。
- `controllers/classification_models.py`（262 行）主要是纯 dataclass/Enum，允许 NumPy，适合优先迁入 domain。

## 职责与迁移目标

| 职责 | 当前入口 | 当前依赖 | 目标边界 | 迁移顺序 |
| --- | --- | --- | --- | --- |
| 数据源与样本结构 | `DatasetSource`、`ClassificationSample` | stdlib、NumPy | `domain.models` | 1 |
| 数据扫描与读取 | `scan_source`、`load_sample`、`read_data` | 文件系统、h5py、fabio/Pillow | `DatasetRepository` + local adapter | 2 |
| QC 与特征矩阵 | `validate_dataset`、`build_feature_matrix` | NumPy；当前与读取混合 | application use case 调用 data port；稳定数值规则逐步下沉 domain | 2 |
| 降维/embedding | `EmbeddingWorker.run`、`_projection_step` | sklearn、UMAP | `EmbeddingPort` 的 JobRunner adapter | 3 |
| 多算法训练与排名 | `compare_algorithms`、`_evaluate_algorithm` | sklearn、UMAP | `TrainClassifiers` + `ClassifierTrainerPort`；真实实现走 JobRunner | 3 |
| 模型预测 | `PredictionWorker.run` | 文件读取、sklearn pipeline | `PredictClassification` + predictor/data ports | 3 |
| 模型保存与加载 | `_save_active_model`、`_load_model` | QFileDialog、joblib、文件系统 | presentation 选择路径；`ModelRepository` adapter 读写 | 4 |
| 页面状态与 commands | controller 全部 `_start_*`/`_on_*` | PyQt、服务、global_params | `ClassificationViewModel` 调用 use cases | 4 |
| 绘图与用户交互 | `_render_*`、`QMessageBox`、`QFileDialog` | PyQt、Matplotlib | 保留 presentation | 4 |

## 关键输入输出

- 导入：`DatasetSource[]` → `ClassificationSample[]` + `DatasetSummary`。
- 特征化：samples + `PreprocessingConfig` → `FeatureMatrix(X, y, samples, input_shape, warnings)`。
- 训练：feature matrix + algorithms + validation + projection + ranking metric → `ExperimentResult`。
- 预测：paths + `SavedModelPackage` → `PredictionResult[]`。
- embedding：feature matrix + method → 二维 NumPy array。
- 持久化：`SavedModelPackage` ↔ 用户选择的模型文件；文件选择不进入 application。

## 必须保持的行为

- 算法 ID、默认参数、随机种子 42、验证策略、排名指标和 class balance 行为不变。
- 当前支持的 1D、图像、CBF/EDF、NumPy、HDF5 文件及其预处理语义不变。
- 模型包字段、joblib 文件兼容性、预测标签/置信度/decision score 不变。
- 页面布局、按钮、进度信息和原 controller 公共入口继续有效。
- 训练、embedding 等 ML runtime 按需加载；测试可用 fake port 且不导入 sklearn/UMAP。

## Characterization tests

1. 纯 domain dataclass 默认值、`DatasetSummary.status`、ranking/best result。
2. fake dataset port 的导入、特征化和文件错误结构化结果。
3. fake classifier trainer 的训练请求、成功/失败/取消状态及 ViewModel 转换。
4. JobRunner adapter 的序列化边界与 fake runner 测试；独立小数据集进程 smoke test。
5. fake embedding/predictor/model repository，确保 application 测试不加载 sklearn。
6. legacy controller 静态门禁：不新增 sklearn/UMAP/joblib 直接导入，Qt dialog 仍只在 presentation。

## 兼容层策略

旧 `controllers.classification_*` 模块和 `ClassificationController` 暂时保留。每引入一个新 use case，先让旧 Qt worker/controller 委托新 API；确认回归后再删除对应旧实现。本阶段不重设计页面，也不一次性删除动态连接的 legacy 方法。
