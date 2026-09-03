# Deep learning 多解散射反演论文骨架

- **Status**：working paper outline；实验结果尚未填入，不构成已完成论文或性能声明
- **Scope**：用户范围条件化的一维 GISAXS/SAS 混合模型多解候选生成与精确正演验收
- **Related code**：`utils/ML_Fitting_1D_GISAXS/PosteriorV8/`
- **Related tests**：`utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_*.py`
- **Related protocol**：`docs/architecture/multisolution-inversion.md`
- **Real-case inventory**：`docs/research/real-data-case-inventory.md`
- **Last verified**：2026-09-03

## 暂定题目

**Forward-Model-Verified Amortized Multisolution Inversion for User-Constrained 1D GISAXS Curves**

中文工作题目：**面向用户约束 GISAXS 曲线的深度学习多解反演与精确正演验收**。

“forward-model-verified”只表示候选通过本项目版本化的非神经经验正演模型；在没有独立物理模型和实验
ground truth 前，不使用“physics-verified”“找到全部解”或“真实后验”等更强表述。

## 中心问题

给定观测曲线、q 网格、测量不确定度和用户对 Sphere/Cylinder/Vertical Cylinder、D、Resolution、BG、
k、`Int_i`、`int_Res` 的范围，能否在固定 exact-forward 调用或 wall-clock 预算内，比传统多起点方法找到
更多互不重复、满足范围与硬物理约束、且与观测兼容的参数等价簇？

这里的网络不是最终求解器。它摊销学习“先搜索哪些离散分支、从哪些连续位置开始”，随后每个候选都经过
同一个有界 variable-projection refinement 和 authoritative exact-forward verification。

## 物理问题的形式化

对观测切线 `y(q)`、测量信息 `sigma(q)` 和用户范围 `B`，外部枚举有限 hard branch `b`（shape
multiset、各分量 D presence、Resolution presence）。每个 branch 的权威经验正演为 `F_b(theta; q)`，合法
候选集合定义为：

```text
S(y, B, tau) = { (b, theta) : theta in B,
                 hard_physics_constraints(b, theta) pass,
                 d_log(F_b(theta; q), y, sigma) <= tau }.
```

这里 `theta` 中的 `k` 与 `Int_i/int_Res` 只通过有效系数 `a_i=k*Int_i`、`a_res=k*int_Res`
进入曲线，因此输出空间先对 exact shared-k gauge 取商；只有 geometry、D policy/presence 和对应 `Int_i`
范围完全相同的同 shape slots 才允许置换。论文评价对象不是 `S` 的数学完备枚举，而是由冻结、network-free、
逐步饱和搜索得到的 compatible quotient-space clusters 的有限代表集。

神经网络学习的是条件 proposal `q_phi(b, u | y, B)` 及有限搜索协议下的 search-yield ranking，其中 `u`
是 branch/query-local unit coordinate。精确求解器 `R_b` 把 proposal seed 映射到局部优化结果，最终系统返回：

```text
M_hat_N,B = Deduplicate({ R_b(u_j) : cumulative exact-forward calls <= B,
                                      exact/range/physics gates pass })[:N].
```

因此主要科学问题是固定输出上限 `N` 和 exact-forward 预算 `B` 下的可发现解簇覆盖，而不是单点参数回归
误差。有限 reference search 的遗漏、模型 proposal 的不确定性和测量 compatibility 的不确定性分别报告。

## 预期贡献

1. 将混合 topology、D/Resolution hard branches 与逐参数用户范围统一成有限 branch catalog 加连续
   set-valued inverse problem；`BG/k/Int_i/int_Res` 每个 active 轴独立采样范围 regime，并由 168D
   authoritative named-coordinate contract 追溯，避免共享 range regime 泄漏伪相关。
2. 用 branch-conditioned neural density 与 search-yield ranking 一次给出多组不同的优化初值，并以
   Sobol/retrieval rescue 保持有限预算下的稳健性。
3. 解析 profile `BG` 与有效线性幅度，同时正确保留 GUI 的共享 `k` gauge、`Int_i` 范围和可行 `k`
   区间。
4. 用 network-free、逐步饱和的 reference bank，在相同 exact-forward 预算下评价解簇
   `recall@N,B`，而不是只评价生成参数的单点误差；正式统计推断以多个独立 randomized-Sobol
   scramble 的 replicate mean 为单位，不把同一低差异序列内的点误当作 IID。
5. 给出 synthetic ID、acquisition OOD、alternate-forward/BornAgain challenge 与冻结后的真实 Cut_Data
   盲测，明确区分经验正演一致性和真实物理泛化。

## 可证伪假设

- H1：range-conditioned neural proposal + exact refinement 的 reference-representative recall AUC 高于
  Sobol-only 和 retrieval-only，且使用完全相同的 exact-call ledger。
- H2：解析幅度 profiling 比直接预测或联合优化所有幅度具有更短 TTFC 和更高 compatible success。
- H3（仅在真正实现并版本化 matched-set/mass-aware objective 后启用）：该目标比单点 MSE 和当前
  `MDN NLL + hard best-of-M` 基线提高 best-of-N 覆盖，同时把 duplicate rate 控制在 10% 以下。
- H4：完整 geometry+D+`Int_i` slot equivalence 与 varying-only refinement 必须通过置换不变性、fixed
  参数零漂移和 exact-call ledger 正确性测试；它们是方法成立的前置条件，不作为与已知错误实现比较的
  性能贡献。K2–K4 的可发表架构对照只比较两种都满足这些约束的合法表示。
- H5：模型对未见 q grid、crop、mask、noise 与 forward discrepancy 的性能下降可被显式测量，而不会被
  nominal synthetic inverse crime 掩盖。

## 实验递进与停止规则

1. Phase A：单一 K1 branch 记忆能力与计算图连通性，只是工程诊断。
2. Phase B：同一 K1 branch 的 stochastic proposal + exact refinement 吞吐诊断，仍不代表 K1 泛化。
3. K1-C/E1：先用与 gate 完全不重叠的 K1 train/tuning blocks 训练覆盖全部 12 branches 的模型；不能把
   Phase-A 的单一 Sphere 记忆模型冒充 K1 模型。随后在 3 种 shape × D absent/present × Resolution
   absent/present 共 12 branches 上，使用独立
   clean-parent Sobol blocks 和逐轴 mixed range stress；门槛为 branch recall@4 ≥95%、any-compatible ≥95%、
   reference recall@N16/B4096 ≥85%、三类合规率 100%、duplicate <10%，并胜过等预算 Sobol/retrieval。
4. K2 后才允许检验 slot permutation、多分量弱可见性和真正的 topology ambiguity；K1-C 未过则不扩数据。
5. all34 先做 plumbing，再做最多三 seed 的工程 learning curve；只有冻结 validation 持续增益才扩到论文
   规模，正式结果至少运行 5 个预声明 training seeds。
6. checkpoint、calibration threshold 和模型架构全部冻结后，才打开 test、OOD 和真实曲线盲测。
7. 正式 compatibility calibration 使用按冻结 strata 独立抽取、可交换的 IID clean-parent recipes，不能用
   RQMC 点拟合 split-conformal 阈值；test/reference/OOD 则使用彼此隔离的冻结 independent-RQMC 设计。
   正式设计每个 split 至少 8 个独立 scramble replicate、目标 16 个。主区间由 paired replicate means
   计算；单个 scramble 内的 recipe 不作 IID bootstrap。
8. reference cohort 在搜索前冻结。headline 明确写成“conditional on frozen reference-search
   qualification”；同时报告完整预选 cohort 的零填充保守下界、unresolved 取 1 的通用上界和按 K/弱分量/歧义度/
   range width 分层的 qualification。若预定主 cohort 未达到冻结 qualification 门槛，则不作主性能声明。

## 图表清单

- Figure 1：curve + user ranges → hard-branch enumeration → neural proposals → exact refinement → verified set。
- Figure 2：不同 topology/参数却产生近等价曲线的物理例子，以及 shared-k gauge fiber。
- Figure 3：主 `recall@N,B` 与 TTFC 曲线，逐个报告预声明 training seed；在每个 scramble 内先平均
  冻结 seed 集合的配对差，再对 independent-RQMC replicate means 给区间；不对单一 Sobol 前缀内的
  recipe 作 IID bootstrap。
- Figure 4：K、shape、Resolution、range width、noise、q window、weak component 分层性能。
- Figure 5：reference-search saturation、候选簇和真实 Cut_Data 拟合叠图。
- Table 1：方法能力与相同预算对照；Table 2：主结果；Table 3：消融；Table 4：OOD/真实案例与资源。

## 结果写作红线

- 未通过冻结 gate 的结果按负结果如实报告，不移动门槛。
- 未饱和 reference bank 时不报告“全部解召回率”。
- refined candidates 不称 posterior samples；search-yield score 不称 branch-existence probability。
- 真实曲线没有 traceable ground truth 时只报告拟合一致性、多样性、稳定性和专家盲评，不报告参数准确率。
- 每个主表结果必须追溯到 source/data/model/protocol SHA、seed、Slurm job、forward calls 与 wall time。
- “95% interval”必须同时标注其 estimand 和随机化单位；固定 cohort 的 recipe-resampling 结果只能叫
  empirical-cohort descriptive interval，不能解释为 population/generalization interval。
- refined candidate 的主召回只匹配实际返回给用户的单个、类型化、可复放 parameter representative；
  query/branch/source/参数/精确曲线摘要和边界/物理状态进入审计哈希。簇内隐藏 member 的最近匹配只能作为
  secondary diagnostic。
- alternate-forward/BornAgain challenge 在生成器、参数映射、样本量、失败/abstention 规则全部冻结并
  实际运行前，只是待完成的 inverse-crime control，不写成已经验证的贡献。
