# 多解散射反演论文：相关工作与主张边界

- **Status**：working literature map；不是系统综述，也不是数学意义穷尽检索
- **Scope**：用户范围条件化、散射逆问题、多模态/集合预测、神经 proposal 与物理 refinement
- **Related code**：`utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py`
- **Related tests**：`utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_study_protocol.py`
- **Paper outline**：`docs/research/multisolution-inversion-paper-outline.md`
- **Last verified**：2026-09-03

本文档保存论文设计所依据的最接近工作和 claim guardrails。机器可执行的 endpoint、预算、split、
checkpoint 与 acceptance 定义仍只由 `study_protocol.py` 管理；科学公式只由
`docs/architecture/fitting-scientific-model.md` 管理。

## 最接近的散射工作

| 工作 | 已覆盖的能力 | 本项目必须补足或区别的部分 |
|---|---|---|
| [Starostin et al., PANPE, 2025](https://arxiv.org/abs/2407.18648)，[Science Advances DOI](https://doi.org/10.1126/sciadv.adr9668) | XRR/NR；曲线与用户 prior bounds 条件化；normalizing flow 表达多模态；再以真实 likelihood 做 IS/MCMC 修正 | 有限混合 topology、固定大小且去重的候选集合、同等 exact-forward 调用预算下的解簇覆盖 |
| [Zhdanov et al., 2022](https://arxiv.org/abs/2210.01543) | BornAgain GISAXS；CVAE 表征与 conditional normalizing flow；摊销后验 | 用户逐参数范围、跨 topology、逐候选权威正演 refinement、reference-cluster recall |
| [Kim & Lee, 2021](https://doi.org/10.1107/S1600576721009043) | XRR Gaussian MDN；输出并聚类多个高概率参数组 | 更灵活的 proposal、相同 exact budget 的 refinement 和完整性评价；是必做 MDN baseline |
| [Munteanu et al., 2024](https://doi.org/10.1107/S1600576724002115) | 将曲线和每个参数上下界一起输入网络；单点回归后可常规拟合 | 多分支而非窄范围内单点；不能声称首次 range-conditioned inversion |
| [Roberts et al., 2025](https://doi.org/10.1039/D5DD00059A) | 常见 nanoparticle SAS 的层级分类与参数回归，包含 sphere/cylinder，并展示跨 topology 退化 | 同一曲线的多 topology 参数解与逐候选验真；是必做 classifier/regressor baseline |
| [Yildirim et al., SASformer, 2024](https://doi.org/10.1039/D3DD00225J) | 55 个 SasView 模型、top-3 model suggestion 与多任务参数回归 | 参数解簇而不只是 top-k 类型；相同 exact budget 下的成功率和覆盖率 |
| [Cao et al., 2026](https://doi.org/10.1021/photonsci.5c00053) | sphere、polydispersity、power-law/background 合成训练；真实 SAXS；树 ensemble UQ | 树间方差不等于多组物理解；需与 range-conditioned 单点回归比较 |
| [Zhai et al., 2026](https://doi.org/10.1021/photonsci.6c00015) | DWBA 生成的大规模 2D GISAXS；CNN 反演粒子联合分布 | 样品内粒径分布不等于逆问题的多个参数解；本项目是一维经验正演族，不得暗示完整 DWBA |
| [Leng et al., FFSAS, 2022](https://doi.org/10.1107/S1600576722006379) | constrained nonlinear programming；向用户提供不同拟合水平的近优解；真实 SAXS/SANS | 摊销 proposal、异质混合体系、用户范围条件和固定预算 coverage；是重要 exact-only baseline |
| [Heil et al., CREASE, 2022](https://doi.org/10.1021/acscentsci.2c00382) | GA 搜索结构，机器学习 surrogate 加速 scattering forward | surrogate 调用不能冒充 exact-forward 调用；GA 应使用同一权威 solver 与预算单独比较 |

## 相邻方法学先例

- [Sharony et al., MISO](https://arxiv.org/abs/2411.02158)：网络一次产生多个优化初值，再选择或并行运行
  下游优化器。它与“一键多候选 + 精确求解器”概念最接近，但不是散射问题，也没有用户范围与解簇覆盖。
- [Dax et al., Neural Importance Sampling, 2023](https://doi.org/10.1103/PhysRevLett.130.171403)：
  neural posterior proposal 后用真实 likelihood importance weights 修正，并用 ESS 诊断。因此不能声称首次
  “神经 proposal + authoritative correction”。
- [Ren et al., NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html)：
  将可生成候选数/时间纳入 inverse-model 评价，并比较 cINN、cVAE、tandem 与 neural-adjoint；它只取最佳
  误差且使用 learned forward surrogate，但构成预算曲线评价先例。
- [Hagemann et al., 2022](https://doi.org/10.1137/21M1450604)：conditional stochastic normalizing
  flow 与 MCMC，包含 optical scatterometry；其 scatterometry forward 为 surrogate PDE。
- [Rupprecht et al., Multiple Hypothesis Prediction, 2017](https://doi.org/10.1109/ICCV.2017.388) 与
  [Zhang et al., Deep Set Prediction Networks, 2019](https://arxiv.org/abs/1906.06565)：分别是多 hypothesis
  与 permutation-aware set output 的基础架构先例；不提供物理可行性或解集完整性保证。
- SBI 基础引用包括 [Papamakarios & Murray, 2016](https://arxiv.org/abs/1605.06376) 和
  [Greenberg et al., APT/SNPE-C, 2019](https://proceedings.mlr.press/v97/greenberg19a.html)。

## 论文必须采用的对照

所有方法使用相同 query cohort、用户范围、观测噪声、候选去重和局部 solver。候选初筛、优化及最终验真
所调用的每一次权威 forward 都计入预算 `B`；神经或 surrogate 调用另行报告。

1. 用户范围内 Sobol/均匀随机多起点 + 相同 exact local solver。
2. Differential Evolution 或 GA + 相同 exact solver。
3. range-conditioned 单点 MSE 回归 + 相同 exact polishing。
4. MDN top-N/聚类 + 相同 exact refinement。
5. conditional flow/PANPE-like proposal + 相同 exact refinement；likelihood 定义充分时再报告 IS/MCMC。
6. topology classifier + per-topology regressor；允许 top-k topology，但 exact budget 相同。
7. retrieval seed + 相同 exact refinement。
8. prior importance sampling 或 parallel-tempered MCMC/ABC 作为传统 Bayesian 参考。

主指标不是 generating parameter 的单点误差，而是冻结参考等价簇的 `recall@N,B`、log2-budget AUC、
time-to-first-compatible、独立 compatible 簇数、exact-forward logRMSE，以及 bounds/physics 合规率。

## 可辩护的新意

截至本轮定向检索，尚未发现一项工作在同一协议中同时实现以下组合：

> 对预声明的有限 1D GISAXS/SAS 混合 topology 目录，以测量曲线和用户逐参数范围为条件，摊销地产生
> 去重候选集；随后使用同一权威非神经正演逐候选执行有界优化与验真；并在预注册、逐调用记账的
> exact-forward 预算下，以参数等价类或解簇覆盖率评价多解发现能力。

这只是本轮检索支持的保守组合性主张，不能写成绝对 “first-ever”。

## 禁止或尚无证据的表述

- 不称“找到所有数学解”；只称覆盖冻结搜索协议发现的 reference-compatible equivalence classes。
- 不称首次使用深度学习做 GISAXS/SAS、首次支持用户范围、首次输出多解或首次做物理修正。
- 未做归一化 posterior 与 SBC/coverage 前，不把 candidate score 称为 posterior probability 或 Bayesian
  calibration。
- 模拟参数与 exact-search 结果属于 simulation-/search-supervised，不称 self-supervised。
- 无全局最优证书时称 authoritative exact-forward evaluation 与 bounded refinement，不称 exact/global
  solver。
- “通用”只限于冻结的 shape/topology、参数范围、背景、Resolution 经验项和 q/噪声/acquisition policy。
- 在真实 Cut_Data、OOD acquisition、alternate-forward 与配对预算检验完成前，不声称真实实验泛化或
  优于传统方法。
