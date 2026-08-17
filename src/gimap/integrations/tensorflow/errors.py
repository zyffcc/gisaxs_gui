"""TensorFlow integration 的明确错误类型。"""


class TensorFlowIntegrationError(RuntimeError):
    """TensorFlow adapter base error。"""


class TensorFlowNotInstalledError(TensorFlowIntegrationError):
    """当前 worker 环境未安装 TensorFlow。"""


class TensorFlowModelError(TensorFlowIntegrationError):
    """模型 artifact 缺失、损坏或不兼容。"""


class TensorFlowWorkerError(TensorFlowIntegrationError):
    """隔离 worker 异常、超时或崩溃。"""
