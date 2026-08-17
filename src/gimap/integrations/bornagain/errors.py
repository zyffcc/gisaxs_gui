"""BornAgain integration 的可诊断错误。"""


class BornAgainError(RuntimeError):
    pass


class BornAgainNotInstalledError(BornAgainError):
    pass


class BornAgainBrokenError(BornAgainError):
    pass


class BornAgainUnsupportedVersionError(BornAgainError):
    pass
