"""Save the automatic analysis with its inputs and replay recipe."""

AUTOMATIC_REPLAY_ERRORS = {
    "Automatic analysis exited with code {code}.": "复现自动分析的进程已退出，退出码为 {code}。",
    "Automatic analysis stopped during replay: {error}": "复现时自动分析中止：{error}",
}

AUTOMATIC_REPLAY_ZH = {
    **AUTOMATIC_REPLAY_ERRORS,
    "Replayable report": "可复现报告",
    "Repeat a recorded reduction, fit or automatic analysis and compare the results":
        "按记录重新归约、拟合或运行自动分析，并比较结果",
}
