# BornAgain integration

BornAgain 只在 Job worker 中 import。主进程通过 `BornAgainSimulator` 调用统一的
`SimulationPort`，并得到明确的 not installed、broken、unsupported version 错误。
当前 adapter 明确支持 BornAgain 24.1；API 构建和数值输出由 trainset 回归测试保护。
