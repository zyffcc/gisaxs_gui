# 当前 1D 预测 workflow · GUI 集成版

2026-09-22 范围纠正。GUI 的 **Fitting → Fit → 1D Predict** 默认方法恢复为 **General V5 (experimental)**，用于提出多种组分候选；这不表示旧模型已达到实用准确率。此前将局部 RC 分支称为默认 Stable 的发布表述已撤回。
原因与当前边界见 [范围纠正记录](development/SCOPE_CORRECTION_20260922_zh.md)。
已有设置和 Recipe 的显式方法选择不强制改写。打开 **Parameters / batch… → Fit method** 检查当前方法；只有已知单随机圆柱时才选择 **Single RC specialist (experimental)**，并将 Complete composition 明确设为一个 Random cylinder。兼容配置键仍为 `method="stable"`，不是质量认证。

- **Fit curve**：按保存的 Fit method 执行。General V5 使用旧多组分候选网络及可选四步校正。RC specialist 仅在显式 `components=[2]` 且计数／q 范围适用时使用局部网络，默认只校准线性幅度、固定形状参数；输入或曲线检查不适合时转数值拟合。
- **Calibrate intensity amplitudes**：RC specialist 专用选项，默认开启，与 General V5 四步校正独立；关闭后仍可能因曲线检查转数值回退。
- **General predict only**：显式使用原 V5 神经网络候选，不做四步数值校正；它仍为实验性模型。
- **Quick physical fit**：新增实验性数值拟合，放宽分辨函数范围并独立求解幅度。Auto 仅比较单组分形状；指定完整组分后可拟合混合物。它不是免精修网络。详见 `development/CBF_USABILITY_20260921_zh.md`。
- **Parameters / batch…**：设置组分、固定 resolution、噪声、单位；`Add curves…` 一次选择多份文本，点击 `Fit N files` 批处理。
- 一般无需选择模型文件。RC specialist 只产生一个单 RC 候选，不自动判断组分。若该方法下组分为 Auto／`[]`，将数值回退并只筛单组分族，不能据此覆盖未知混合物；指定其他完整组合或固定 resolution 时执行遵守这些条件的数值拟合。
- 输出自动保存在 GUI 启动目录的 `AI_Fitting_Output/v5_日期_时间/`；每次运行新建目录。

## 输入与输出

文本支持空格、Tab、CSV，2 列 `q intensity` 或 3 列 `q intensity sigma`，可有表头。
文件 q 单位在参数中设置（默认 nm⁻¹；Å⁻¹ 会乘 10）。当前 cut / 已导入曲线沿用主界面明确保存的单位，转换到 nm⁻¹。
每侧要求 8–1000 个不同的有效 q 点；q=0 不送入模型。两侧独立拟合，不折叠平均。
负强度保留用于评分，网络内部才使用正值代理。重复 q、无效 sigma、点数越界会报错，不自动伪造或插值训练输入。

没有 sigma 时，默认 `sqrt((0.1*abs(I))² + floor²)`；floor 默认为输入峰值绝对值的 0.1%，可指定。
CBF 在统一 detector preprocessing 中屏蔽无效像素及默认 3 px 邻域；设置在 Preprocessing → CBF bad-pixel guard，随 in-situ recipe 保存。cut 与拟合均使用同一 AnalysisImage 的原生有效像素均值及计数噪声近似，再叠加相对/绝对容差。每点实际有效像素数独立保存，并与 ROI／删点／分侧排序保持一致，不从叠加了容差的 sigma 反推。500 点仅用于正演显示。
计数误差采用 Poisson 近似，额外容差是工作假设。RC specialist 使用 counts/pixel 原始幅度；强度 normalizer 仅用于 General V5。选用 specialist 时，文本缺少原生 CBF 计数契约，以及 threshold、实际镜像替换或 stack 改变计数条件的输入，将转数值拟合。

长度 R/h/D 为 nm，sigma_R/h/D 是相对标准差，另给绝对宽度 nm。sigma_Res 是 nm⁻¹，nu_Res 无量纲。
组分混合权重不是后验概率、质量分数或体积分数。

每条曲线保存：

- `solutions.json`：物理参数、固定条件、幅度、误差、单位、实际运行阶段与回退原因。界面 Stage 显示 `Neural model`、`Model + amplitude` 或 `Numerical fallback`，悬停可查看原因和限制；误差没有通过/失败标签。
- `candidates.npz`：全部原始候选参数与原始点正演；同组分的不同参数模式保留。
  此文件用于 General V5；RC specialist 与物理拟合的参数、独立幅度及原生点正演完整保存在 `solutions.json`。
- `display.npz`、`fitting_curves.csv`：物理参数与幅度保持不变，计算 500 个显示点。General V5 另冻结测量网格的参考系数。
- 运行根目录 `request.json`、`top20_candidates.json`（沿用兼容文件名，不限于 20 行）、`batch_summary.json` 和耗时汇总。

500 点只用于显示。排名基于原始测量点，先评分再去重；不拿“去重后的第一头”替代原先应保留的好解。
logRMSE 使用自然对数，只统计原始正强度点；RMS/σ 统计包括负强度在内的全部点。
GUI 直接绘制本 workflow 的正演结果，不调用旧手工模型公式重新解释这些参数。

## In-situ

先在 Single 中确认代表帧的检测器与 cut，进入 In-situ：选择文件夹 → `Use current setup` → `Start Process` / `Start Watch`。
页面只有 Source / Analysis settings / Results 三步。逐帧使用捕获到 recipe 的方法；未保存其他方法时默认 General V5 (experimental)。
在 `1D parameters…` 中检查 Fit method 和完整组分条件；若选 RC specialist，须明确一个 Random cylinder 才允许神经快速路径。修改保存到未来帧的 recipe，不静默改写旧 Recipe。
Analysis settings 可选择拟合曲线或只提取曲线；specialist 的幅度校准独立于旧 V5 四步校正。修改后点击 `Save analysis settings`。
检测器/预处理/cut 放在可展开设置中。`1D parameters…` 保存为未来帧使用的新配置快照，不反向改写 Single。
正负侧最佳解、参数、耗时和结果路径随帧保存；失败可以继续或停止，暂停/取消复用原序列调度器。
Recipe 保存实际科学几何的完整精度，不能把 Center X 等值经显示控件舍入后替换；真正的用户编辑仍会捕获。相同帧的 Single／in-situ 必须保留相同原生 q、强度、有效像素数和 ROI。
in-situ 当前每帧使用独立 worker；多文本文件批处理在一个 worker 中复用模型。

## 科学适用范围

RC specialist 是局部单随机圆柱模型，带 structure factor；训练域来自 00033 开发案例，仅允许显式 `components=[2]`。它没有证明未知组分识别、新样品泛化、多参数模式覆盖或概率校准。数值回退不会扩大网络的泛化证据。[2026-09-22 早期发布记录](development/evidence/stable_blue_20260922/RELEASE_zh.md) 保留为历史测量；其中默认 Stable 的定位由本次范围纠正替代。
数值回退 Auto 比较单组分球、随机圆柱和竖直圆柱；指定完整组合时支持最多四组分及重复类型，遵守固定 resolution。该宽范围回退仍使用历史 V5 求积，输出明确标注，局部神经分支的积分／拟合验证不能推广到它。

General V5 是原有 sphere / random cylinder / vertical cylinder 多候选模型，最多四组分，可重复。恢复它为默认只恢复未知组分候选入口，不解决其已发现的拟合不足。历史独立验证针对**完整真实组分和两个 resolution 值已知**的条件预测：128 条中纯 NN 84 条、加四步校正 114 条达到 clean logRMSE < 0.05；不代表未知组分自动发现准确率或实验验收概率。

General V5 自动模式默认在分类器提出的 12 种组合中产生候选，再对按有符号残差选择的 3 种组合做条件预测。
参数入口可扩大到最多 34 种组合；增加预算会增加耗时，不能保证发现所有可行解。
自动 resolution 是估计值；一部分先验已固定时，返回的候选必须遵守固定值，不保留违反它的发现阶段候选。
排序不表示概率；良好的曲线拟合不证明参数真实或唯一。本次范围纠正未改训练权重，局部 RC 与旧 `conditional_v2/` 权重均保持原样，也未产生新的泛化验证结论。
用户已将目标调整为接近经过积分核验的数值参考曲线：峰位和整体形状准确，不追逐随机噪声。
原始全曲线 logRMSE 保留显示，0.05 不再作为强制通过/失败线；作业完成不证明参数真实。
噪声诊断与后续工作顺序见
`development/PREPROCESS_MASK_AND_TARGET_20260921_zh.md`。

最新原因对照见 `development/evidence/cbf_cause_audit_20260921/REPORT_zh.md`：
已区分网络候选不足、幅度/resolution 范围限制、随机圆柱求积离散误差和计数噪声；
该报告记录当时的 0.05 目标；当前目标已按用户反馈调整。记录、输入快照和诊断工具随开发资料保存。

向蓝线形状靠拢的新训练验证见 `development/evidence/blue_curve_distill_20260921/REPORT_zh.md`。
该分支以 GUI **Single RC specialist (experimental)** 保留；NumPy 网络层与原 TensorFlow 使用同一权重。早期 CBF 开发帧在幅度校准下观测 lnRMSE 为 **0.17120 / 0.15523**，worker 双侧约 **0.68 s**，主窗口约 **3.70 s**，单帧 in-situ 约 **3.07 s**。这些历史开发回放既不是 General V5 默认方法的指标，也不证明未知组分或新样品泛化；本次更名／路由限制没有重新训练或改善这些指标。

## 换电脑

复制**整个 GUI 项目**，包括本目录。路径按项目目录解析，不依赖 E:、D:、原研究目录或 SSH。
使用 Python 3.10，安装项目依赖及本目录的 `requirements-runtime.txt`；推荐独立环境。
RC specialist 推理与数值回退使用 NumPy／SciPy，不加载 TensorFlow；默认 General V5 与继续训练仍需要 TensorFlow 2.15.1。局部分支资产目录 `stable_blue_rc_v1/` 与内部 `stable` 方法键为兼容旧记录而保留，应与源码、manifest 和开发记录一起迁移。
无需 Maxwell 即可推理。启动 GUI 后直接进入 1D Predict。

验证命令（从 GUI 项目根目录）：

```text
python tools/check_stable_predict_workflow.py --mode all
python -m pytest tests/test_stable_metadata_ui.py tests/test_stable_blue_core.py tests/test_stable_predict_workflow.py
python tools/check_workflow_v5.py
python tools/smoke_workflow_v5_ui.py
python -m pytest tests/test_workflow_v5.py
```

名称含 stable 的检查工具保留兼容性，局部分支验证需显式指定单 RC。其历史 GUI 回放覆盖按钮、单帧 in-situ、输入一致性、批处理失败继续和取消；旧 V5 工具重放随包 Cut 开发案例。这些既有样例和测试范围未因默认方法纠正而变成独立泛化证据，真实长序列／NXS 全流程仍未覆盖。
`conditional_v2/` 保留原始模型包和条件预测 CLI；开发日志见 `development/HANDOFF_zh.md`。
