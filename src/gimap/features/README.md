# `features`

每个业务能力是一个 feature，拥有自己的 `domain`（纯科学逻辑）、`application`（用例与 ports）、
`infrastructure`（adapters）、`presentation`（Qt 页面、ViewModel、bindings）和 `bootstrap.py`（组装）。
（2026-10-06 核对）

| Feature | 在界面里 | 做什么 |
| --- | --- | --- |
| `analyze` | 主工作区 Analyze | 打开帧 → 几何 → 掩膜与校正 → 切线 → 结果 → 导出；Series 热图、批量导出 |
| `fitting` | 主工作区 Fitting；Tools ▸ 1D Predict — Fit Many Curves… | 一维模型拟合：Single（一条曲线）与 In-situ series；快速物理拟合 `create_quick_fit()` 也供其他功能注入 |
| `compare` | 主工作区 Compare | 多个序列并排：奇异帧、阶段、变化快慢、哪些相似 |
| `assistant` | Analyze 的 Run Automatic Analysis（无需 AI）；Tools ▸ Process with AI…、Analyze 的 Ask AI… | 标准流程（GIWAXS 与 GISAXS）和 AI 对话；`tools/gimap_agent.py` 的无界面版本 |
| `calibration` | Tools ▸ Geometry Calibration… | 用标样（AgBh、LaB6…）标定束流中心与探测器距离；无界面标定也注入自动分析 |
| `format_converter` | Tools ▸ Format Converter…、Convert Current File… | NXS、CBF、TIFF、HDF5 探测器图像的格式转换 |
| `xrr` | Tools ▸ XRR Series Extractor… | 从 NXS / CBF 角度序列提取 XRR 强度 |
| `prediction` | Labs ▸ 2D Prediction | 二维图样的机器学习预测 |
| `trainset` | Labs ▸ Trainset Build | 模拟训练集 |

规则（`tests/test_architecture_dependencies.py` 检查，详见 `docs/architecture/dependency-rules.md`）：

- feature 之间不互相导入；需要组合时在 `src/gimap/app/`（组合根）注入。
- presentation 不导入 domain 或 infrastructure；application 不导入 Qt、presentation 或 infrastructure；
  domain 不导入 Qt、TensorFlow、Keras、BornAgain。
- 多个 feature 稳定需要的科学能力放到 `src/gimap/shared/`，不是另一个 feature 里。
