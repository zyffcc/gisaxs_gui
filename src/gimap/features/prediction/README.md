# Prediction feature

Prediction 负责模型无关的预测请求和结果。application 只依赖 `Predictor` port；
TensorFlow 的具体实现位于 `gimap.integrations.tensorflow`。

Prediction workspace layout、专用 cards、preview layout、typed UI state 和 ViewModel 位于
`presentation/`。旧主窗口仍生成现有 controls，并由 feature-owned workspace 原样重组；
生产运行时直接构造 `PredictionViewBinding`，旧 controller 名称仅为 import 兼容别名。
