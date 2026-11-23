"""Chinese text for drift in reduced χ² during a chained series fit."""

CHAIN_DRIFT_ZH = {
    "χ²ᵣ rose from {before} at frame {frame} to {after}": "χ²ᵣ 从第 {frame} 帧的 {before} 升至 {after}",
    "Retry with Predict after a failed or unconverged fit or at the start of a new stage. Also retry if χ²ᵣ is "
    "above 2 and exceeds 3× its value at either the previous frame in this pass or the last chain start. "
    "Keep the better fit.":
        "若拟合失败、未收敛或进入新阶段，则用“预测参数”重试。若 χ²ᵣ 超过 2，且超过本轮前一帧或本段链起始帧的 χ²ᵣ "
        "的 3 倍，也会重试。保留较好的拟合结果。",
    "The kept fit started from the neighboring frame, Predict parameters, or the model in Single analysis. "
    "A failed or unconverged fit, a new stage, or a jump or gradual rise in χ²ᵣ can break the chain.":
        "本帧保留的拟合从相邻帧的结果、“预测参数”或单条分析中的模型开始。拟合失败、未收敛、进入新阶段、χ²ᵣ 跳升或逐步升高时，"
        "链可能断开。",
    "fit failure, non-convergence, a new stage, or χ²ᵣ above {floor} and {ratio}× its value at "
    "either the previous frame in this pass or the last chain start":
        "拟合失败、未收敛、进入新阶段，或 χ²ᵣ 超过 {floor} 且超过本轮前一帧或本段链起始帧的 χ²ᵣ 的 {ratio} 倍",
}
