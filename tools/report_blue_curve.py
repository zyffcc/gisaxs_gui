"""Publish this pilot's scope, immutable evaluation and direct-prediction figure."""

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.blue_curve_distillation import OUT, AUDIT


def main():
    evaluation = json.loads((OUT / "evaluation.json").read_text())
    refs = json.loads((OUT / "real_references.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    history = json.loads((OUT / "training_history.json").read_text())
    physics = json.loads((OUT / "physics_checks.json").read_text())
    previous = np.load(AUDIT / "comparison_native.npz")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, side in enumerate(("positive", "negative")):
        r = next(v for v in refs if v["frame"] == "00033" and v["side"] == side)
        new = next(
            v
            for v in evaluation["real"]
            if v["frame"] == "00033" and v["side"] == side and v["model"] == "curve_supervised"
        )
        q = np.array(r["q"])
        reference = np.array(r["reference"])
        curve = np.array(new["forward_curve"])
        ax = axes[0, col]
        ax.scatter(q, r["y"], s=5, c="#90959b", alpha=0.6, label="Measured")
        ax.plot(q, previous[f"{side}_nn_rank1"], c="#d79537", lw=1.5, label="Previous general NN")
        ax.plot(q, reference, c="#2468b5", lw=2, label="Blue numerical reference")
        ax.plot(q, curve, c="#b33c69", lw=1.6, label="New local NN, no refinement")
        ax.set_yscale("log")
        ax.set_title(f"{side.capitalize()} / full curve")
        ax.set_xlabel("q (nm^-1)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        ax = axes[1, col]
        sel = (q > 0.7) & (q < 2.4)
        ax.scatter(q[sel], np.array(r["y"])[sel], s=7, c="#90959b", alpha=0.6)
        ax.plot(q[sel], previous[f"{side}_nn_rank1"][sel], c="#d79537", lw=1.5)
        ax.plot(q[sel], reference[sel], c="#2468b5", lw=2)
        ax.plot(q[sel], curve[sel], c="#b33c69", lw=1.6)
        ax.set_title(f"Peak zoom | full-curve NN/blue error: {new['vs_reference_logrmse']:.3f}")
        ax.set_xlabel("q (nm^-1)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.15)
    fig.suptitle(
        "CBF 00033 development case: learn the physical curve shape, not random noise\nSingle random-cylinder pilot; no component discovery or calibrated multi-solution distribution"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "blue_target_comparison.png", dpi=160)
    plt.close(fig)
    realrows = "\n".join(
        f"| {r['frame']} | {r['side']} | {r['observed_logrmse']:.4f} | {r['reference_logrmse']:.4f} | {r['vs_reference_logrmse']:.4f} | {r['sidepeak_q_error_nm_inverse']:.4f} | {r['sidepeak_height_relative_error']:+.1%} | {1000 * r['warm_median_seconds']:.1f} |"
        for r in evaluation["real"]
        if r["model"] == "curve_supervised"
    )
    synthetics = "\n".join(
        f"| {name} | {r['clean_curve_median']:.4f} | {r['clean_curve_mean']:.4f} | {r['clean_curve_p90']:.4f} |"
        for name, r in evaluation["models"].items()
    )
    chosen = min(history, key=lambda r: r["validation_mean"])
    text = f"""# 蓝线目标：直接参数预测的小规模训练验证

日期 2026-09-21。用户最新目标：曲线接近之前蓝色数值参考的峰位、峰形和整体趋势；不要求原始含噪数据 lnRMSE 必须小于 0.05。

本轮实际生成数据并训练了一个**单随机圆柱专用分支**。输入含计数噪声，输出 11 个物理坐标，再用版本化正演产生曲线；预测阶段没有 least_squares 或其他数值精修。这不是只预测一条没有参数的平滑曲线。
该分支尚未替换 GUI 的通用 V5 网络；CLI 可直接回放。GUI 已移除固定 0.05 判红及强制通过/失败状态，完整观测残差仍显示并导出。

![00033 直接预测与蓝线](blue_target_comparison.png)

## 数据、物理与训练

- 4096 TRAIN / 512 VALIDATION / 512 TEST，三个独立 Latin hypercube 种子。没有把任何真实 CBF 强度或数值拟合参数作为训练样本。
- 物理参数域由 00033 开发案例确定，因此 00033 不是独立泛化证据。支持 R 1.1–2.2 nm、h 6–24 nm、D 为 max(3,2.002R)–5.5 nm，sigma_Res 0.009–0.026 nm^-1，nu_Res 2–4；幅度/背景独立。完整坐标范围见 `protocol.json`。
- 单 RC，structure factor 开；只预测一个候选，不进行组分识别、固定参数条件预测或概率校准。不能继承旧模型的“最多四组分”适用范围。
- 干净标签由 Gauss 48 分布节点 / 96 取向节点的正演生成。随机抽查局部参数域 32 个样本，节点加倍的最大曲线差 {physics["quadrature_max_logrmse"]:.6g}；TF/NumPy 最大差 {physics["tf_numpy_max_logrmse"]:.6g}；自动微分与中心差分核对通过。这是局部采样验证，不能外推到旧 R/h 全范围。
- 用两侧原生 q 模板、有效像素计数模拟 Poisson 输入，计数倍率 0.5–2，另随机缺失 2.5% 的观测列。网络特征为有效像素加权分箱、实际 q 偏移、曝光和缺失标志；缺失箱没有插值填充。分箱仅为编码特征，最终评分回到全部原生观测。
- MLP：256 → 256 → 128 → 11，swish 隐层、sigmoid 有界参数输出。先参数监督（最多 100 epochs，验证早停），再 1600 步物理曲线监督：`mean((ln(forward(pred))-ln(clean))²) + 0.002 * parameter_MSE`。
- 曲线阶段每批 8 条，每条随机 32 个原生 q；使用 TF 正演的真实梯度。每 100 步检查事前固定的 64 条验证子集，在第 {chosen["step"]} 步选中权重，验证平均误差 {chosen["validation_mean"]:.4f}。未用 TEST 或真实帧选择权重。
- 数据生成用时约 {protocol["seconds"]:.1f} s；参数预训练及曲线阶段约 {history[-1]["seconds"]:.1f} s，本地 CPU 完成。未使用 Maxwell，未更改旧权重。

## 留出合成 TEST：监督曲线是否有益

共 512 条独立合成 TEST；目标是生成时已知的干净物理曲线，不是噪声点。参数监督基线和曲线监督使用同一网络结构、同一数据及参数预训练起点。

| 训练方式 | clean lnRMSE 中位数 | 均值 | P90 |
|---|---:|---:|---:|
{synthetics}

参数可辨识性没有因此得到保证；不同 R/h/宽度/幅度组合仍可生成相近曲线。本轮没有测试完整多峰后验覆盖。
该 TEST 的归一化参数 RMSE 中位数由约 0.1494 变为 0.1538，尽管曲线误差下降：本轮证据支持曲线预测改善，不支持所有参数更接近生成真值。

## 真实帧：全部直接预测，无精修

00033 为开发帧；00005 / 00045 未用于训练或 checkpoint 选择，但来自同一序列，也已经参与上一轮噪声诊断，不能当成全新独立实验。三帧使用相同保存几何、mask 和提取区域，没有为网络重新找中心或改 ROI。

蓝线参考：单 RC、独立幅度、较宽参数域的收敛积分数值拟合；其他帧从 00033 参数启动后分别拟合。参考线不是已知真实干净曲线，参考参数不是 ground truth。

| 帧 | 侧 | 新 NN 对观测 lnRMSE | 蓝线对观测 | 新 NN 对蓝线 | 峰位差 nm^-1 | 峰高差 | 暖调用 ms |
|---|---|---:|---:|---:|---:|---:|---:|
{realrows}

峰位/峰高在预先指定的 q=1–2 nm^-1 侧峰窗口计算；全部 lnRMSE 使用全部原生有效点。暖调用包含特征编码、网络前向、原生点物理正演，五次中位数；不含首次 TensorFlow 导入/加载，不含图形渲染。没有数值精修。

![三帧对照](direct_comparison.png)

## 使用、记录及下一步

`curve_best.weights.h5` 是验证集选定的曲线监督权重；`parameter_only.weights.h5` 为消融基线，`feature_scaling.npz` 与 `protocol.json` 必須一起使用。
`evaluation.json` 含全部评分和参数；`real_references.json` 含完整参考曲线；`training_history.json`、`physics_checks.json` 和 `dataset.npz` 保存研究证据。旧 `conditional_v2/` 保持原始 SHA。

从 GUI 根目录运行：

```text
python tools/predict_blue_curve.py validation/blue_curve_distill_20260921/cbf_00033_input.npz output/blue_prediction.json
```

输入 NPZ 的 `q` 单位 nm^-1，可含正负侧；`intensity` 为有效像素平均计数；`count` 为每点有效像素曝光数。不能把任意归一化文本强度、未知曝光或稀疏 Cut 当成本分支的已验证输入。

复现训练：`python tools/blue_curve_distillation.py generate`，然后 `python tools/blue_curve_distillation.py train --steps 1600`。前者检测到 dataset 时会复用；后者会重新训练并覆盖同名实验权重，历史研究归档应先保留。评估：`python tools/evaluate_blue_curve.py evaluate`。

后续应保留本轮“含噪输入 → 参数 → 干净物理曲线”的训练目标，再扩展到多组分和多候选，并验证更广参数/q 域及不同实验。需让多个不同参数模式分别拟合，不能仅平均它们。应以独立曲线形状和峰位验证泛化，保留观测残差而不强行追逐噪声。

相关检查：22 个 focused tests 通过；GUI 真主窗口 offscreen 启动通过（Windows 重定向日志需 UTF-8）；新脚本及改动模块 Ruff 通过。首次 smoke 因 GBK 无法输出既存 Unicode 日志导致初始化失败，设置 PYTHONIOENCODING=utf-8 后通过，没有修改无关初始化代码。
"""
    (OUT / "REPORT_zh.md").write_text(text, encoding="utf-8")
    source = json.loads((AUDIT / "inputs.json").read_text())
    np.savez_compressed(
        OUT / "cbf_00033_input.npz",
        q=np.r_[-np.array(source["negative"]["q"])[::-1], source["positive"]["q"]],
        intensity=np.r_[source["negative"]["y"][::-1], source["positive"]["y"]],
        count=np.r_[source["negative"]["count"][::-1], source["positive"]["count"]],
    )
    print(json.dumps(dict(report=str(OUT / "REPORT_zh.md"), selected_step=chosen["step"])))


if __name__ == "__main__":
    main()
