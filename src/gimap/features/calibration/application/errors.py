"""可由 presentation 处理、但不依赖具体图像库的 application errors。"""


class AmbiguousImageDatasetError(ValueError):
    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__("Multiple detector image datasets are plausible: " + ", ".join(paths))


class CalibrationCancelledError(RuntimeError):
    """用户请求取消 Calibration。"""
