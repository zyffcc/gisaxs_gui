# Prediction feature

Prediction 负责模型无关的预测请求和结果。application 只依赖 `Predictor` port；
TensorFlow 的具体实现位于 `gimap.integrations.tensorflow`。
