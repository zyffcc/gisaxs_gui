# Local process jobs

`LocalProcessJobRunner` 使用独立 Python process 执行由 `module:function` 标识的 handler。
worker 只发送普通 dict/list/scalar 消息。父进程负责观察进度、终止取消/超时任务，并把
异常退出转换为 `JobError`，不会让 GUI process 退出。
