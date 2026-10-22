# Local process jobs

`LocalProcessJobRunner` 使用独立 Python process 执行由 `module:function` 标识的 handler。
worker 只发送普通 dict/list/scalar 消息。父进程负责观察进度、终止取消/超时任务，并把
异常退出转换为 `JobError`，不会让 GUI process 退出。

spawn 出的子进程会先把父进程的主模块重新跑一遍（`__mp_main__`）。GUI 的 `main.py` 在顶层导入 PyQt5：
PyQt5 自带的旧 msvcp140.dll 一旦先进入 worker，BornAgain 24.1（更新的 MSVC 编译）在 `simulate()` 里崩溃
（0xC0000005）。所以 `light_main.light_main()` 在启动 worker 时把 `__main__` 暂时指向空模块
`worker_main.py`；Analyze 的批处理 worker（`frame_workers.py`）用同一个 helper 和同一把锁。
`src/gimap/app/__init__.py` 按需（PEP 562）导入，导入 `src.gimap.app.jobs` 不会带进 Qt。
