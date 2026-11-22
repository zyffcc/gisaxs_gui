"""Chinese of the chained series fit's rules and Start column, the converging Refine, Fit New Frames in Fitting with
the watch, the measured curves of an NXcanSAS file of fits and the Series map kept in a project (merged by
``zh_features.py``)."""

SERIES_CHAIN_ZH = {
    # -- where a frame's fit started (fitting/presentation/single/start_texts.py, application/series_chain.py) --
    "Started from": "起点",
    "Where the frame's kept fit started: the result of frame n (the frame before; the frame after in the backward pass), "
    "Predict parameters where the chain broke (a failed or unconverged fit, a new stage, a jump of χ²ᵣ), or the model "
    "of Single analysis":
        "这一帧所保留拟合的起点：第 n 帧的结果（上一帧；反向拟合时为下一帧）、链断开处（拟合失败或未收敛、进入新阶段、χ²ᵣ 跳变）"
        "的“预测参数”，或单条分析中的模型",
    "continued from frame {n}": "接续第 {n} 帧",
    "continued from frame {n} (backward pass)": "接续第 {n} 帧（反向拟合）",
    "continued from the frame before": "接续上一帧",
    "continued from the frame after (backward pass)": "接续下一帧（反向拟合）",
    "from the model in Single analysis": "从单条分析中的模型开始",
    "restarted from Predict ({reason})": "从“预测参数”重新开始（{reason}）",
    "{start} (the chain broke: {reason})": "{start}（链断开：{reason}）",
    "Predict's solution: {rule}": "“预测参数”的解：{rule}",
    "the fit failed ({why})": "拟合失败（{why}）",
    "the fit did not converge": "拟合未收敛",
    "a new stage begins": "进入新阶段",
    "χ²ᵣ jumped from {before} to {after}": "χ²ᵣ 从 {before} 跳到 {after}",
    "the closest to frame {frame} of the solutions within {factor}× of the most probable (distance {distance})":
        "在概率不低于最可能解 1/{factor} 的解中最接近第 {frame} 帧的解（距离 {distance}）",
    "the most probable solution (no previous result to follow)": "最可能的解（没有可接续的上一帧结果）",
    "What to know before trusting the row: failed, not converged, a poor fit, values at a bound of their range, the beam "
    "tail at an edge, a value the curve does not determine (none of them breaks the chain)":
        "采信这一行之前需要知道的：失败、未收敛、拟合差、参数停在范围边界、光束拖尾在范围边缘、曲线确定不了的参数（这些都不会使链断开）",
    "Each frame from the result of the frame before; where that chain breaks — a failed or unconverged fit, a new stage, "
    "χ²ᵣ more than 3× the frame before's (a value at a bound does not break it) — Predict parameters fits the frame "
    "again from its solution closest to the frame before, and the better fit is kept":
        "每一帧从上一帧的结果开始；在链断开处——拟合失败或未收敛、进入新阶段、χ²ᵣ 超过上一帧的 3 倍（参数停在边界不会使链断开）——"
        "用“预测参数”从最接近上一帧的解重新拟合这一帧，并保留较好的结果",
    # -- Refine goes on while it improves (fitting/presentation/single/run.py) --
    "The least squares went on from its own result {count} times (each time χ²ᵣ fell by more than {share} %).":
        "最小二乘从自己的结果继续了 {count} 次（每次 χ²ᵣ 都下降了 {share}% 以上）。",
    # -- Fit New Frames in Fitting… (analyze/presentation/bindings/watch_online.py, single/series_watch.py) --
    "Fitting ▸ In-situ series fits the curves written so far and watches for new ones: every new frame is fitted from "
    "the result of the one before.":
        "“拟合 ▸ 原位序列”正在拟合目前写好的曲线并监视新曲线：每个新帧都从上一帧的结果开始拟合。",
    "Watch for new curves is on: set the model up on a frame in Single analysis, then press Start — every curve listed "
    "and every new one is fitted.":
        "“监视新曲线”已开启：请先在单条分析中用一帧设置好模型，然后按“开始”——已列出的和新出现的曲线都会被拟合。",
    # -- an NXcanSAS file of fits opened again (fitting/presentation/single/page.py, curve_import.py) --
    "{name} holds {count} measured curves (and the models fitted to them). Which one is to be fitted?":
        "{name} 含有 {count} 条测量曲线（以及拟合它们的模型）。要拟合哪一条？",
    "NXcanSAS file of fits: its {count} measured curves are imported, not the models beside them (Q, I and Idev as σ); "
    "the rows shown are those of the first curve in the file.":
        "拟合结果的 NXcanSAS 文件：导入其中的 {count} 条测量曲线，不导入旁边的模型（Q、I，Idev 作为 σ）；显示的行是文件中第一条曲线的。",
    # -- the Series map in a project and the autosave (analyze/presentation/bindings/project.py) --
    "The Series map ({rows} frames) was too large to keep in the project: Build Map again.":
        "序列热图（{rows} 帧）太大，没有保存在项目中：请重新构建热图。",
    "The Series map of the project could not be read: Build Map again.": "无法读取项目中的序列热图：请重新构建热图。",
}

__all__ = ["SERIES_CHAIN_ZH"]
