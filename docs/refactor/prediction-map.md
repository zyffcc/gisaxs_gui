# Prediction 渐进迁移地图

## 范围与基线

- Legacy 入口：`controllers/gisaxs_predict_controller.py`，当前 4,549 行。
- 多文件 UI/队列：`controllers/multifile_predict_results.py`。
- 模块目录：`modules/<module>/module.yaml`，当前包含五个 GISAXS prediction module。
- 已存在的新边界：`src/gimap/features/prediction` 中有最小 `RunPrediction` 和
  `Predictor` port；`src/gimap/integrations/tensorflow` 已通过 JobRunner 隔离模型加载和预测。
- 本轮必须保持 `module.yaml`、单文件、范围选择、多文件 `Every=N` stack、模型发现、
  输出名称/方向和现有 Qt 入口兼容。

## 当前启动与入口

`MainController` 创建 `GisaxsPredictController`，延迟调用 `initialize()`。初始化依次创建
图像场景、连接控件、恢复 `global_params`、扫描 modules、建立模型状态和多文件结果窗口。

主要用户入口：

- 选择文件/目录：`_choose_gisaxs_file`、`_choose_gisaxs_folder`；
- 数据加载：`_load_single_stack`、`_load_multi_sequence`、`_start_image_loading`；
- module 选择：`_initialize_modules_ui`、`_refresh_modules`、`_on_module_selected`；
- 模型加载：`_on_model_import_clicked`、`_on_model_load_finished`；
- 预测：`_run_gisaxs_predict` → `_execute_prediction`；
- 多文件预测：`_predict_multi_files` → `MultiFilePredictManager`；
- 导出：`_export_results_jsonl`、`_export_results_jpg`、`_export_results_ascii`。

## 职责分组

### 1. Module discovery 与 `module.yaml`

当前方法：

- `_modules_root`、`_scan_modules`；
- `_parse_module_yaml`、`_extract_spec_from_dict`、`_extract_spec_fallback`；
- `_populate_module_combo`、`_on_module_selected`；
- `_write_model_path_to_yaml`、module edit watch/reload。

输入是 module 根目录或 YAML 路径；输出是职责混杂的 `dict`，包含 id/name、framework、
model path、preprocess entry/steps/params、I/O shape、outputs、folder/yaml path。

依赖：PyYAML（可选）、正则 fallback parser、文件系统、Qt combo/timer/dialog。

优先 seam：

- domain：typed `PredictionModule`、`ModelSpec`、`PreprocessSpec`、`OutputSpec`；
- application：`DiscoverPredictionModules`、`LoadPredictionModule`；
- port：`ModuleRepository`；
- infrastructure：`YamlModuleRepository`，保留 fallback 和 Windows 路径字符串兼容；
- presentation：ViewModel 只保存 module list/current module/status。

必须保持：现有五个 YAML 均可读取；空 model path 合法；`outputs` 同时支持 list 和
`sf_4_parameters` mapping；相对 mask/preprocess 路径以 module folder 为基准。

### 2. 图像/sequence 加载

当前方法：

- `_scan_directory_for_cbf`、`_extract_index`、`_parse_range_text`；
- `_load_single_stack`、`_load_multi_sequence`；
- `_start_image_loading` 使用 fitting feature 的 `AsyncImageLoader`；
- `_load_cbf_file_sync`、`_load_cbf_stack_sync` 在多文件路径重复实现 Fabio I/O。

输入：CBF/TIF/TIFF 文件、stack count、范围文本、Every 值。输出：float32 二维数组及
源文件列表。

依赖：Qt QThread、Fabio、文件系统、`FittingController.AsyncImageLoader`。

优先 seam：纯 range/index/batch 规则进入 domain；`PredictionImageRepository` port 和本地
adapter 统一单文件/stack；prediction 不再跨 feature 导入 fitting presentation/controller。

必须保持：stack 使用求和而不是平均；自然/现有排序规则；Every=N 只处理完整 batch，
尾部不足 N 个文件继续跳过；文件索引来自文件名末尾数字。

### 3. Preprocessing

当前方法：

- `_preprocess_for_module`；
- `_collect_preprocess_steps`；
- `_prepare_model_input`、`_normalize_input_rank`、`_coerce_array_to_shape`、`_resize_nhwc`。

当前行为优先动态加载 `module_folder/preprocess.py` 的 entry function，并传入原始 YAML
preprocess block；若不能加载则走 controller 内置 fallback。部分 resize fallback 在 GUI
进程直接 import TensorFlow。

输入：二维 detector image + module spec；输出：模型输入 ndarray 和可选 step snapshots。

优先 seam：

- domain 只保存稳定 array shape/coercion 规则；
- `Preprocessor` application port；
- `ModuleEntryPreprocessor` adapter 负责动态 Python module import；
- resize fallback 使用稳定数值/image adapter，不在 controller import TensorFlow。

必须保持：step 顺序、mask 值、crop orientation、log normalization、NHWC rank、双通道
SF 模型输入和所有 module 自定义 entry 的调用签名兼容。

### 4. Model discovery/load/prediction

当前方法：

- `detect_available_frameworks`、`is_framework_compatible`；
- `_select_model_folder`、`_on_model_import_clicked`、`_on_model_load_finished`；
- `_predict_with_current_model`、`_normalize_parameter_prediction`。

已有 `TensorFlowPredictor` 和 `TensorFlowModelProxy`，但 controller 仍持有 proxy 并直接做
输出归一化；某些兼容分支直接 import TensorFlow。

优先 seam：

- application：`InspectPredictionModel`、`PredictImage`；
- ports：`Predictor`、`Preprocessor`；
- infrastructure：现有 TensorFlow integration；
- domain：模型无关 output normalization 与 typed prediction payload。

必须保持：模型按需加载；GUI 启动不加载 TensorFlow；`.keras`/SavedModel 均可发现；
1-output、list-output、dict-output、HR/H/R 和 SF parameters 的名称、shape 和数值不变。

### 5. 单文件 prediction workflow

当前路径：

`_execute_prediction` → `_preprocess_for_module` → `_predict_with_current_model` →
`_display_prediction`。

输入：当前 image/module/model；输出：prediction payload 和展示 tabs。

优先 seam：`PredictSingleImage` use case 组合 image、preprocessor、Predictor；ViewModel
管理 loading/running/ready/error，不操作 widgets。

必须保持：当前展示 tab 次序、progress 语义、parameter inverse scaling、输出 array
orientation 和 `prediction_completed` payload。

### 6. 多文件 prediction workflow

当前方法：

- `_predict_multi_files` 建立 files/batches；
- `_predict_single_file_for_batch` 和 `_execute_single_file_prediction` 临时修改 controller
  的 current state；
- `MultiFilePredictManager` 使用 `ThreadPoolExecutor` 调用 controller bound method；
- result widget 同时包含 domain result、table model、窗口、filter/export actions。

输入：folder/range/Every；输出：逐文件或逐 stack 的结果、状态、耗时和错误。

优先 seam：

- domain：`PredictionBatch`、`PredictionItemResult`、状态；
- application：`BuildPredictionBatches`、`PredictMultipleFiles`；
- 长任务统一通过 `JobRunner`，worker 消息只包含可序列化数据；
- presentation 只把 progress/result 映射到现有 table widget。

必须保持：取消在当前文件后停止；单个文件失败不退出整个 GUI；完整 batch 规则；结果选择
仍能恢复原图/stack 并显示 prediction；单文件和多文件调用同一个 prediction use case。

### 7. Display 与 export

当前方法：

- `_display_prediction`、`_render_predict_panel`、HR/curve/image render；
- `_serialize_prediction_data`；
- JSONL/JPG/ASCII export；
- `MultiFilePredictResultsWidget` 的 heatmap/trend/filter/table。

展示属于 presentation。JSONL/ASCII/JPG 写入应通过 export port/adapter；纯 output-to-row
转换可放 domain/application。Qt pixmap/QImage/matplotlib canvas 不进入 application/domain。

必须保持：JSONL 字段、ASCII header/列顺序、JPG 命名、parameter name fallback、1D 曲线
和 2D distribution 展示。

### 8. Settings/global state

当前 `_load_saved_parameters`、`_persist_parameters` 直接访问 `global_params`。新 feature
应通过 `AppContext.settings` 注入；兼容 controller 可在迁移期同步旧 key，不能创建新全局。

## 四轮迁移顺序

1. **纯 domain**：module/value objects、range parsing、index extraction、batch grouping、
   model input shape coercion、output normalization；建立 controller 前后数值/shape 回归。
2. **Use cases/ports**：module discovery、image load、single prediction、batch construction、
   multi prediction、export；全部用 fake predictor/preprocessor/repositories 测试。
3. **Adapters**：YAML repository、Fabio image repository、dynamic module preprocessor、
   TensorFlow predictor composition、local export；保留 module.yaml 和文件格式。
4. **ViewModel/兼容层**：typed state/commands；controller 只保留 Qt dialogs、signals、render，
   旧方法委托新 use cases；单/多文件入口继续有效。

每轮只在上一轮 focused tests 通过后开始。

## Characterization tests

- 五个现有 `module.yaml` 解析快照：id/name/framework/model/preprocess/io/outputs；
- Windows model path 原样保留，写回时不破坏 YAML 其他字段；
- range：单值、正序、反序、非法文本；文件名末尾 index；
- Every=1/2/3，尾部不足 batch 的行为；
- CBF stack 是 float32 求和，单文件与 stack source files 正确；
- 每个 module preprocess entry 对固定小图的 shape、dtype、finite/mask 特征；
- input rank/shape：2D、HWC、NHWC、单通道和双通道；
- predictor fake 覆盖 dict/list/single output 与 SF parameter inverse scaling；
- 单文件 use case 成功、预处理错误、模型错误；
- 三文件 workflow：中间错误继续、取消、progress 可序列化；
- export JSONL/ASCII 的 header 和数值；
- 无 QApplication、无 TensorFlow 时 application tests 可运行；
- legacy controller 禁止再 import fitting controller、TensorFlow/Keras runtime。
