"""Build the measured-cause report from completed audit records (no new fits)."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_cbf_causes import Model, OUT, ARMS


def read(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def best(records, side, arm, resolution=None):
    return min(
        (
            r
            for r in records
            if r["task"]["side"] == side
            and r["task"]["arm"] == arm
            and (resolution is None or r["task"].get("resolution", True) == resolution)
        ),
        key=lambda r: r["logrmse"],
    )


def prediction(record, data):
    t = record["task"]
    m = Model(
        data,
        t["arm"],
        t["types"],
        t["gates"],
        t.get("resolution", True),
        t.get("quadrature", "legacy"),
    )
    return m.predict(record["z"])


def main():
    inputs = read("inputs")
    range_rows = read("range")
    completed = {k: len(read(k)) for k in ("range", "mixture", "targeted", "gauss", "gauss48")}
    bank = range_rows + read("mixture") + read("targeted")
    network, replay = read("network_checks"), read("replay_checks")
    gauss = read("gauss48")
    convergence = read("convergence_checks")
    noise = json.loads((ROOT / "validation/preprocess_mask_20260921/noise_budget.json").read_text())
    result = dict(
        target=0.05,
        metric="sqrt(mean(ln(I_forward / I_observed)**2)) on ALL native positive observations",
        completed_optimization_counts=completed,
        network=network,
        noiseless_replay=replay,
        provenance=read("provenance"),
        noise_budget=noise,
        sides={},
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [2.1, 1, 1]}, sharex="col"
    )
    native = {}
    for col, side in enumerate(("positive", "negative")):
        d = inputs[side]
        q, y = np.array(d["q"]), np.array(d["y"])
        original = best(bank, side, "train_exact", True)
        broad = best(bank, side, "resolution_wide")
        continuous = best([r for r in gauss if "split" not in r["task"]], side, "resolution_wide")
        auto = next(
            r
            for r in network
            if r["side"] == side and r["conditions"] == "automatic" and not r["numerical"]
        )
        folder = OUT / "network" / f"{side}_automatic_False"
        solutions = json.loads((folder / "solutions.json").read_text())
        raw = np.load(folder / "candidates.npz")
        curve = (
            raw["curves"][0, solutions["solutions"][0]["candidate_index"], raw["mask"][0] > 0]
            * y.max()
        )
        models = [
            (curve, "#dd7733", f"NN rank 1: {auto['rank1_logrmse']:.3f}"),
            (
                prediction(original, d),
                "#36a078",
                f"Original bounds, multistart: {original['logrmse']:.3f}",
            ),
            (
                prediction(continuous, d),
                "#2469bd",
                f"Converged integral, wide resolution: {continuous['logrmse']:.3f}",
            ),
        ]
        ax = axes[0, col]
        ax.scatter(
            q, y, s=5, color="#767d84", alpha=0.55, label=f"Measured: {len(y)} native points"
        )
        for pred, color, label in models:
            ax.plot(q, pred, color=color, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_title(f"{side.capitalize()} side — full-curve lnRMSE")
        ax.set_ylabel("Intensity (counts / valid pixel)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.15)
        for row, (pred, color, label) in enumerate((models[0], models[2]), 1):
            ax = axes[row, col]
            ax.axhspan(-0.05, 0.05, color="#36a078", alpha=0.13)
            ax.axhline(0, color="#555", lw=0.8)
            ax.scatter(q, np.log(pred / y), s=6, color=color, alpha=0.6)
            ax.set_ylabel("ln(fit / observed)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.15)
        axes[2, col].set_xlabel(r"$|q_y|$ (nm$^{-1}$)")
        cv = [r for r in gauss if r["task"]["side"] == side and "split" in r["task"]]
        result["sides"][side] = dict(
            points=len(q),
            q_range=[float(q.min()), float(q.max())],
            best_single_component_by_arm={arm: best(range_rows, side, arm) for arm in ARMS},
            best_original_bounds_resolution_on=original,
            best_original_bounds_allow_resolution_off=best(bank, side, "train_exact"),
            best_expanded_mixture=broad,
            best_converged_integral_single_component=continuous,
            interleaved_holdout=cv,
            whole_curve_pass=False,
        )
        for label, arr in [
            ("q", q),
            ("observed", y),
            ("nn_rank1", curve),
            ("original_multistart", models[1][0]),
            ("converged_integral", models[2][0]),
        ]:
            native[f"{side}_{label}"] = arr
        # A noiseless replay remains unchanged by the network's positivity floor.
        truth = prediction(original, d)
        result["sides"][side]["noiseless_replay_feature_floor_count"] = int(
            np.sum(truth < 0.1 * np.array(d["sigma"]))
        )
    fig.suptitle(
        "CBF 00033: identical mask, geometry and native observations\nTarget < 0.05 is NOT met. Shading denotes ±0.05 log residual, not a per-point acceptance test.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "cause_comparison.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(OUT / "comparison_native.npz", **native)
    result["quadrature_convergence"] = convergence
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    p, n = result["sides"]["positive"], result["sides"]["negative"]
    ablation = "\n".join(
        f"| {arm} | {p['best_single_component_by_arm'][arm]['logrmse']:.6f} | {n['best_single_component_by_arm'][arm]['logrmse']:.6f} |"
        for arm in ARMS
    )
    nn_rows = "\n".join(
        f"| {r['side']} | {r['conditions']} | {r['numerical']} | {r['rank1_logrmse']:.6f} | {r['best_bank_logrmse']:.6f} | {r['candidate_count']} |"
        for r in network
    )
    replay_rows = "\n".join(
        f"| {r['side']} | {r['numerical']} | {r['rank1_logrmse']:.6f} | {r['best_bank_logrmse']:.6f} |"
        for r in replay
    )
    cv_rows = "\n".join(
        f"| {r['task']['side']} | {r['task']['split']} | {r['train_logrmse']:.6f} | {r['heldout_logrmse']:.6f} | {r['converged']} |"
        for r in gauss
        if "split" in r["task"]
    )
    report = f"""# CBF 00033：优化、参数范围、正演积分和噪声的对照诊断

日期：2026-09-21。结论：**现有模型仍未达到用户要求的原生全曲线 lnRMSE < 0.05，不能宣布可实用。**
本轮完成原因验证，没有训练、替换权重或修改 GUI 的生产正演。新积分函数仅用于研究诊断。

## 结论及证据强度

1. **逆向预测／有限优化不足已经证实。** 同一输入，自动纯网络排第一的误差为 0.372308 / 0.319465；原有参数范围内多初值数值优化可达 0.168553 / 0.149688（均开启 resolution）。即使检查全部网络候选，最好也只有 0.319952 / 0.283253。排序不是唯一原因。
2. **不是“真实参数肯定在范围外”。** 原范围的多组分组合已经能达到约 0.15–0.17，故单组分范围对照不能用于否定整个原模型。放开整体幅度、背景和 resolution 范围后，一个随机圆柱也接近这一水平；原有 nuisance 参数化限制了简单解释。进一步扩大颗粒 R/h/D 等范围帮助很小。本次没有读取原始大型训练集逐条统计；确认的是冻结输出域及回放覆盖失败，不能声称已证明训练样本从未覆盖某一区间。
3. **发现正演数值积分精度问题，而非证明解析公式错误。** 原随机圆柱只有 13×13 个分布节点、24 个均匀角节点。保持参数不动加密节点，曲线会显著改变；参数曾部分补偿离散误差。独立自适应积分验证了新 Gauss 求积在测试点的正确性；实际宽 resolution 最佳解再加密到 96/192、192/384 节点后变化低于 1e-12 lnRMSE，但重新拟合仍为 0.161560 / 0.144046。
4. **噪声贡献很大，但尚不能宣布严格下限。** 原始六行像素的弱信号段离散程度与 Poisson 计数近似相符；单帧计数近似和模拟给出约 0.14–0.16 的全曲线随机波动量级。低 q 行间梯度、正侧残余结构及非平稳帧提示仍有系统项，不能把所有残差归于噪声。

## 数据与评分契约

- CBF：`TestSAXSdata/jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf`。
- 使用用户确认的上次保存几何：距离 1456.7 mm，波长 0.1033 nm，入射角 0.4°，172 µm 像素，beam Y 370.75；优化后的 Center X 791.32，Yoneda Y 505。
- 提取原始数组区域（含端点）：行 1171–1176、列 201–1381。预处理坏点及 3 px guard mask 不变，33 个全无效列剔除，正侧 571 / 负侧 577 个原生观测。坏点不插值，未按残差删点，未为降低分数另裁 q。
- 评分为 `sqrt(mean(ln(I_forward / I_observed)^2))`；本样本所有保留 I>0。自然对数，不是 log10；全部原生有效 q 点参与。q=0 不参与。
- q 单位 nm^-1；R、h、D 单位 nm；sigma_R/sigma_h/sigma_D 是相对标准差。权重不是组分后验概率或质量分数。
- 10% 相对 sigma 是工作容差，不是测得的计数噪声。本轮从 sigma 中减去其方差项，并用原 CBF 有效像素数独立核对。
- 精确输入快照、来源 SHA256、几何和 mask 见 `inputs.json`、`provenance.json`。图和评分都使用这些观测，避免与旧预处理结果混比。

![同输入曲线及残差](cause_comparison.png)

## 1. 优化与范围消融

单组分共完成 360 次：两侧 × 五个范围臂 × 三种形状 × structure factor 开/关 × 六个初值；每次最多 450 次 residual evaluation，SciPy bounded `least_squares`，目标为等权自然对数残差平方和。此处次数是 max_nfev，数值导数另有额外正演开销。记录在 `range.json`。

| 单组分范围臂 | 正侧 | 负侧 |
|---|---:|---:|
{ablation}

范围定义：

- train_exact：冻结 V5 正演、归一化混合权重、原 BG/resolution 比率和范围、固定观察归一化幅度；对参考正演的逐值一致性测试通过。
- scale_free：仅额外放开整体强度尺度。
- amplitude_free：颗粒、背景、resolution 使用独立正幅度，sigma_Res 仍为 0.007–0.013 nm^-1、nu_Res 仍为 5–10。
- resolution_wide：独立幅度，sigma_Res 扩至 0.001–0.1，nu_Res 扩至 1–20，颗粒范围不变。
- particle_wide：进一步扩 R 1–100 到 0.2–200，h 2–500 到 0.2–1000，D 上限 500 到 1000 nm，分布宽度也扩大，细节见脚本。

多组分补充：完成初始 mixture 扫描 25/176 次后，因发现积分误差中止该完整扫描，保留所有已完成记录；另完成平衡两侧的 targeted 20 次，每次 350 次 evaluation。

| 方案 | 正侧 | 负侧 |
|---|---:|---:|
| 原 GUI 快速物理拟合 | 0.166058 | 0.146463 |
| 原范围多初值，resolution 开 | {p["best_original_bounds_resolution_on"]["logrmse"]:.6f} | {n["best_original_bounds_resolution_on"]["logrmse"]:.6f} |
| 原范围，允许 resolution 关 | {p["best_original_bounds_allow_resolution_off"]["logrmse"]:.6f} | {n["best_original_bounds_allow_resolution_off"]["logrmse"]:.6f} |
| 扩幅度/resolution，多组分，旧积分 | {p["best_expanded_mixture"]["logrmse"]:.6f} | {n["best_expanded_mixture"]["logrmse"]:.6f} |
| 扩幅度/resolution，单圆柱，收敛积分 | {p["best_converged_integral_single_component"]["logrmse"]:.6f} | {n["best_converged_integral_single_component"]["logrmse"]:.6f} |

原范围较好解分别为两随机圆柱、球+三竖直圆柱；这说明存在可达曲线，不证明这些组分是真实结构。不同形状/幅度可能互相补偿。记录为有限多初值局部优化结果，**不是全局最优的证明**。大量增加当前数值预算的收益已远小于从 0.15 降至 0.05 所需的改进。

## 网络回放：候选缺失与排序均有贡献

same-input 直接运行当前 `WorkflowEngine`，自动模式默认预算 12 种 discovery、3 种条件组合；numerical=True 为现有四步数值校正，不是本报告中的充分优化。全候选库计算独立 lnRMSE，以避免只看排序前列。当前生产排序使用带 sigma 的 signed Huber，因此其 rank 1 不一定是 lnRMSE 最小者。

| 侧 | 先验 | 四步校正 | 排名 1 lnRMSE | 全候选最小 lnRMSE | 候选数 |
|---|---|---|---:|---:|---:|
{nn_rows}

`fit_derived_conditions` 来自本次较好数值解，是诊断先验，绝不是实验已知 ground truth。

进一步用上述**原范围、原冻结正演**的数值参数合成无噪声曲线，提供精确组分、sigma_Res、nu_Res 和归一化值，让网络回放。理论上生成参数本身能产生零残差，但网络结果如下。保留原观测 sigma 作为接口权重输入，因此这是当前接口／网络管线的恢复能力测试，不是只测试裸网络层。两侧合成均值均未触发 0.1 sigma 的特征正值下限。

| 侧 | 四步校正 | 排名 1 lnRMSE | 全候选最小 lnRMSE |
|---|---|---:|---:|
{replay_rows}

该失败说明：存在冻结域内可精确表达、当前网络仍找不到的曲线。实验噪声和范围外不是充分解释。尚不能由此区分训练分布稀疏、预处理分布偏移、网络容量、训练目标或优化策略各自占比；需要匹配分布的受控训练实验。此处只有两条针对性案例，不能当成总体失败率。

## 正演数值核验

`numerics_checks.json`：固定旧单圆柱最佳参数，将原采样数分别加密 2/4/8 倍，8 倍曲线相对原曲线的 lnRMSE 为 0.14545 / 0.10451，说明旧离散积分尚未收敛。独立实现的原采样 factor=1 与冻结正演一致。

`audit_gauss_forward.py` 在 cos(alpha) 上做均匀取向积分，用 Gauss-Legendre 处理截断高斯分布。24 个分布节点仍不足，所以探索阶段 `gauss.json` 只完成 2 条便停止；随后 `gauss48.json` 八条全部完成，含四条全数据拟合和四条留出检验。分布 48/取向 96 在 R=1.6、h=12 的 q=0.1、1、4 测试点与独立嵌套 adaptive quad 在 rtol=1e-7 内一致。

`convergence_checks.json` 进一步核验了每个实际拟合参数，而非仅理论测试点。宽 resolution 全数据最佳解及其留出解积分稳定。窄 resolution 全数据解在加密后还有 0.00155 / 0.00470 的曲线变化，更高阶负侧仍有约 6.8e-5 变化；其 48/96 阶分数不能宣称完全收敛。**48/96 不是整个 R/h 参数域的通用保证**；新生产正演应按 qR/qh 与宽度自适应选阶或用预验证规则。

原单圆柱旧积分的 h 约 11.97 / 10.22 nm；宽 resolution 收敛积分的 h 约 16.36 / 11.68 nm，显示曲线相近时参数也会漂移。解析假设是否真实仍需外部物理标准和独立数据验证。

同一帧交错列留出：只用一半原生 q 拟合，另一半评分，固定形状初值且幅度初始化只看训练半。完整 q 参考网格保持不变。全观察最大值仅用于宽幅度边界；这是内部留出，不是独立实验。

| 侧 | 训练列奇偶 | 训练 lnRMSE | 留出 lnRMSE | 优化器收敛 |
|---|---|---:|---:|---|
{cv_rows}

## 2. 噪声与系统残差检查

`counting_checks.json` 直接读取原始 CBF 并使用同样的 mask，逐值断言六行列均值等于本轮拟合输入。

- q=1–2、2–3、3–5 nm^-1 段，六行像素的 Pearson dispersion / df 正侧约 1.020 / 0.962 / 0.974，负侧 1.151 / 1.058 / 1.045。在“同列各行均值相同”的近似下，弱尾部与 Poisson 计数波动相容。
- q<1 部分该值为 22.19 / 5.36；行向梯度和真实二维形态会增加该统计量，不应直接当作独立曝光噪声。无法据此排除几何、背景、形态或物理模型的系统误差。
- 同目录 00005、00045 两帧，即使缩放强度，与 00033 仍有明显差异；标准化差异 RMS 约 21.3、4.17。它们不是已验证平稳重复曝光，未用于平均降噪或严格噪声下限标定。
- 方差传播近似给出全曲线 log 波动 0.1512 / 0.1406；相邻二阶差分给出 0.1616 / 0.1308，但真实曲率也会污染差分估计。
- 在独立 Poisson、当前拟合均值正确的条件下，1000 次模拟的原计数水平 lnRMSE 中位数为 0.1602 / 0.1469；9 倍计数为 0.0510 / 0.0474；16 倍为 0.0382 / 0.0354。这是条件模拟，不是增加曝光后必达标的承诺，系统误差不会按该规则下降。
- 同时对数据和正演做 8 列计数加权合并，诊断误差降到 0.0628 / 0.0570；16 列为 0.0638 / 0.0396。合并改变 q 分辨率，**不能用来宣称原始全曲线通过 0.05**。正侧未持续下降也提示系统残差还存在。

因此，保留用户的 0.05 原生全曲线门槛，本样本当前仍标记 FAIL。未来应另报告网络对潜在干净曲线的逼近误差与原始观测残差，不能通过平滑、删有效点或换指标假装达标。

## 后续实施顺序

1. 先建立版本化、全目标 q/参数域验证的收敛正演，独立检查实验幅度、背景和 resolution 语义；**不能把新积分直接替换到旧权重下**。固定几何之外的物理假设仍需验证。
2. 用该正演生成与实验 q、mask、计数噪声及幅度/resolution 分布匹配的训练/验证数据。先做小规模可重复的同分布学习对照与本次噪声自由回放，再决定是否扩量/换网络。保留独立未调参 CBF 作为外部验证；本 00033 已反复用于开发。
3. 将有限多初值优化提供的不同可行参数模式作为候选监督，评估候选覆盖、已知先验下回归、以及指定误差内的多样性；排序目标和用户的全曲线门槛要明确分开。不能把数值可行的复杂组分直接标成真实组分。
4. 用真正同状态重复曝光标定噪声、低 q 系统残差和长期漂移；再判断原生 0.05 是否需要更多计数。如果要重分箱，必须预先声明 q 分辨率并同时积分 forward。

## 文件、复现与验证

完整机器记录见 `summary.json`；曲线数值见 `comparison_native.npz`。冻结资产不变。
研究工具位于 GUI 项目 `tools/audit_cbf_*.py`、`tools/audit_gauss_forward.py`、`tools/report_cbf_causes.py`。
复制整个 GUI 项目可携带输入快照和记录；仅复制模型目录时，开发 evidence 子目录含必要快照与脚本副本，重跑脚本需放回 GUI 根目录 tools 中以满足 imports。

```text
python tools/audit_cbf_causes.py --phase range --workers 4
python tools/audit_cbf_causes.py --phase targeted --workers 3
python tools/audit_cbf_causes.py --phase gauss48 --workers 3
python tools/audit_cbf_counting.py
python tools/audit_cbf_numerics.py
python tools/audit_cbf_convergence.py
python tools/audit_cbf_network.py
python tools/audit_cbf_replay.py
python tools/report_cbf_causes.py
python -m pytest tests/test_cbf_cause_audit.py tests/test_workflow_v5.py -q
```

优化脚本会跳过已保存的任务；本次 mixture/gauss 的部分完成状态如实保留，不能把其列为完整扫描。
本轮 focused tests：19 passed（含正演逐值契约、独立积分对照和冻结资产）；相关研究脚本 Ruff 通过。
未重新声明整个仓库测试通过：此前全套检查受 Windows 不支持旧 PosteriorV8 的 `resource` 导入阻挡，仓库另有既存 WAXS Ruff 错误，本轮没有改动这些无关模块。
"""
    assert all(r["noiseless_replay_feature_floor_count"] == 0 for r in result["sides"].values())
    (OUT / "REPORT_zh.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(completed=completed, report=str(OUT / "REPORT_zh.md"), full_curve_pass=False)
        )
    )


if __name__ == "__main__":
    main()
