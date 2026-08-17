# Job contracts

Job contracts 是 GUI 与 worker process 之间的稳定边界。Request、progress、result 和
error 均只能包含 JSON 可序列化值。`JobRunner` 的实现负责进程生命周期、取消、超时和
崩溃隔离；presentation 只消费进度和最终状态。
