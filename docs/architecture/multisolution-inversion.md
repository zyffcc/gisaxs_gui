# GISAXS 多解反演科学与模型契约

- **Status**：Methods template — V5.2 geometry/amplitude-range-conditioned contract implemented；具体训练、模型、reference 与 holdout artifact 尚未冻结，本文件不构成外部 preregistration
- **Scope**：Sphere/Cylinder/Vertical Cylinder 混合曲线的多解定义、模型输出、物理验收与训练数据
- **Related code**：`src/gimap/features/fitting/domain/scattering_model.py`、
  `utils/ML_Fitting_1D_GISAXS/PosteriorV8/`；论文数值门禁的机器可读唯一版本位于
  `PosteriorV8/study_protocol.py`
- **Related tests**：`utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_*.py`
- **Related research**：`docs/research/multisolution-inversion-literature.md`、
  `docs/research/multisolution-inversion-paper-outline.md`
- **Last verified**：2026-09-03

本文定义“一键多解”的科学目标和必须在 holdout 前冻结的设计。它不是已经完成的实验注册，也不表示
具体 run/model/reference artifact 已经冻结。它不复制具体散射公式；正向模型、单位和参数顺序仍以
[`fitting-scientific-model.md`](fitting-scientific-model.md) 为唯一权威。
论文中的对象必须称为“指定经验正演族下的一维 GISAXS cut”，不能暗示完整 DWBA GISAXS；当前
`Resolution` 是 legacy additive low-q component，不是仪器分辨率卷积。

相关工作已经分别覆盖 scattering 深度反演、用户 prior/range 条件化、多模态后验、多个优化初值和
神经 proposal 后的物理 likelihood 修正，因此文章不得声称这些单项的首次提出，也不得使用绝对
“first-ever”。允许检验的新意严格限定为它们在同一协议中的组合：有限混合 topology 目录、逐参数用户
范围、set-valued 候选、权威正演有界 refinement，以及在相同 exact-forward 调用预算下对操作性等价解簇
覆盖率的配对评价。range-conditioned 单点回归、top-k topology、MDN/conditional flow、Sobol、
Differential Evolution/GA、retrieval 和分层 Bayesian 方法都必须作为相同查询与预算下的对照或诊断。

## 目标与多解定义

字面意义上的“全部参数解”通常是连续无穷集合。当前模型至少存在以下等价性：

- `k` 与所有 `Int_i`、`int_Res` 可以反向缩放而不改变曲线；
- 一个分量可以拆成参数相同、强度相加的多个分量；
- 不可见、窗口外或强度极弱的分量可以加入或删除；
- 未观察到结构峰时，多组 `D`、`sigma_D` 可以在实验噪声内等价。

因此，一键预测的有限目标定义为：

> 在给定 q 窗口、实验不确定度、用户范围、物理约束、误差阈值和搜索预算内，返回所有已发现的、
> 经冻结距离和确定性 complete-linkage 聚类得到的 exact-compatible 参数簇代表。

输出必须区分：

- **declared topology**：候选搜索时使用的 K 与 shape multiset；
- **effective topology**：去除不可观测幅度分量后的 K 与 shapes；
- **complete-linkage diameter-δ compatible parameter representative**：按冻结参数距离和确定性
  complete linkage 得到的 exact-compatible 参数簇代表；complete linkage 只保证每个簇的直径
  `<=δ`，不同簇所选代表之间不保证 pairwise distance `>δ`。这是有限预算下的操作性单位，不宣称是
  数学或 posterior mode；
- **curve-equivalence group**：在实验不确定度内产生等价曲线的一组参数代表。

生成时使用的单一 K/type 只能作为模拟 provenance，不能在歧义曲线上当作唯一真值。
连续 ridge 可能被操作性阈值切成多个代表，因此论文只报告阈值敏感性和 reference-search saturation，
不得把有限列表描述成“所有数学解”。

## 独立 GUI 尺度与解析幅度

固定所有非线性几何参数后，权威模型写成：

```text
I(q) = BG + k * (sum_i Int_i * P_i(q) * S_i(q) + int_Res * R(q))
     = BG + sum_i a_i F_i(q) + a_res R(q)
```

其中 `F_i=P_i*S_i`，有效系数为 `a_i=k*Int_i`、`a_res=k*int_Res`。GUI 的 `k` 与每个 `Int_i`、
`int_Res` 是彼此独立的参数和用户范围；`Int_i` 不要求和为 1，也绝不能假设 `k=sum_i a_i`。内部线性
求解使用 `[BG,a_1,...,a_K,a_res]`，但必须保留一个共享辅助变量 `κ` 的存在性：

```text
k_low <= κ <= k_high
Int_i_low * κ <= a_i <= Int_i_high * κ
int_Res_low * κ <= a_res <= int_Res_high * κ     # Resolution present only
```

`BG` 仍使用自己的独立范围。实现通过精确消去 `κ` 得到 coefficient polytope，并在导出候选时从完整
feasible-`κ` interval 选择一个确定性 witness，保存显式 `k=κ`、`Int_i=a_i/κ` 和
`int_Res=a_res/κ`，再逐项执行真实 GUI 范围 gate。两个不同的 `(k,Int_i,int_Res)` witness 若产生完全相同的
有效系数，只是共享尺度 gauge 的不同参数化，不应制造新的解簇；但不能因此归一化掉可观测的绝对曲线尺度。
Resolution 不存在时不创建 `a_res` 轴并输出 `int_Res=0`。任何可接受候选仍必须满足粒子有效幅度总和
`sum_i a_i>0`。

该范围转换、polytope 和逐项验收审计必须携带同一版本标识；固定几何的加权线性 amplitude profile 与
exact-log amplitude polish 在收到 GUI 幅度范围时都在完整 polytope 内求解并返回审计。未提供该范围时
保持原有非负无约束行为。`BG`、各粒子有效系数和 Resolution 有效系数应在每个候选上用有界非负线性
求解初始化，再用与 GUI 数值实现一致的 forward 在 log residual 上验收。这里的“exact forward”仅指
没有使用神经网络近似且与该版本 GUI 公式数值一致，不代表完整散射物理是精确的。

约束线性 profile 使用缩放后的凸二次目标。若 SLSQP 对一个已有可行 witness 的问题报告 line-search 或
constraint-compatibility failure，不能据此把物理 branch 判为不可行。版本化恢复规则只在 contract witness
与有限、仍通过同一 coefficient polytope 的 SLSQP terminal 之间选择加权线性目标较低者，随后仍执行
authoritative GUI forward 与全部门禁；`solver_status=-101` 和 message 必须明确记录来源及“不是最优性证明”。
不存在可检查的有限可行点时仍然失败关闭，不能放松用户范围或把失败改写成 negative label。

当前线性幅度求解优化的是加权线性 residual，而外层优化和验收使用 log residual，因此它是高效的
profile approximation，不称为严格的解析 profile likelihood。论文必须在分层小样本上与 joint
geometry + log-space amplitude optimization 从相同 seeds 比较，量化此近似对最优误差、branch/mode
assignment 和运行时间的影响。只在几何结束后 polish 幅度不能代替这项 gold comparison。
当前 gold 路径在同一个有界变量中联合优化 K=1…4 的 branch-codec 几何坐标和非负幅度，并对每次
目标计算调用权威 forward；它拥有独立的 exact-forward 硬预算并返回预算内 best valid seen。该路径
只用于小样本 reference/benchmark，不直接替换速度更快的生产 proposal refiner。

## 统计误差契约

有逐点实验不确定度时，主 compatibility score 为
`RMS((log I_pred - log I_obs) / sigma_log)`，其中小相对误差下
`sigma_log ~= sigma_I / I_obs`。模拟噪声包含 counting noise、乘性扰动和数值 floor；因此该分数是
版本化的 weighted log-residual pseudo-likelihood，不宣称服从未经检验的严格概率分布。

验收阈值必须只由独立 calibration split 按 acquisition design cycle slot、noise policy 和 q-window policy 的
**实际可达 60 个** acquisition-only strata 校准，随后冻结。不能把三个维度的名义笛卡尔积误写成 80：当前
deterministic observation generator 的 design-N 与 q-window phase 耦合，冻结 universe 字段为
`(design_point_count_cycle_slot,q_window_id,noise_id)`，其 version 与 canonical SHA 必须从实现常量写入
calibration、checked identity 和 study artifact。artifact 的历史 wire 字段 `point_count` 明确表示 mask/crop
前的 design N；每个 calibration observation 还要单独记录实际进入 standardized score 的
`effective_valid_point_count`。后者与 grid kind、crop、mask 一样在层内边缘化，不得偷偷扩张主 stratum。
K 也不进入主 calibration stratum；K-conditional calibration 只作 label-conditional sensitivity，不能解释为
topology precision。

主 paper estimand 对每个 clean-parent physical recipe × acquisition stratum 只允许一个预先确定的 deterministic
observation view，直接以该 score 计算 split-conformal order statistic，不再对相关 views 取最大值。相同
recipe-stratum 出现第二个 view 时必须 fail closed；若未来要校准 whole-acquisition 的多 view 统计量，必须
另建并在使用前冻结 estimand，不能复用本 artifact。每条 observation 和 artifact 都必须携带版本化的完整
`acquisition_policy_id`，覆盖 grid、mask、crop、view 与 measurement-sigma provenance。不仅 stratum 必须
出现，该完整 policy ID 也必须确实属于该 stratum 的 calibration 成员；未知的同-stratum nuisance policy 仍
fail closed。未见过的 stratum 必须拒绝覆盖声明，pooled fallback 只可作描述性参考。目标覆盖率、经验分位数和
校准样本不确定性的描述性区间必须随 artifact 保存；它们不替代方法比较所需的独立 RQMC replicate
推断。raw logRMSE 始终同时报告。有 sigma 的曲线用已校准 standardized score 作 compatibility
gate；没有可靠 measurement sigma 的真实 Cut_Data 不得进入 calibration，只报告描述性 raw 指标，不得
声称统计覆盖率。

正式 `paper_full_calibrated` 搜索协议必须加载 checked calibration artifact，并同时绑定其 canonical artifact
SHA、文件 SHA、输入 SHA、dataset-manifest SHA、固定值为 `calibration` 的 reserved-split ID、split SHA、
schema/version、目标覆盖率及 stratum contract。每条 observation 在任何拟合前只从完整
`acquisition_policy_id` 提取 design N、noise ID 和 q-window ID，执行 exact-stratum lookup，并把 stratum、
阈值及 lookup digest 纳入 task identity；未见 stratum、缺失 acceptance sigma 或 artifact 漂移均 fail closed，
不得使用 pooled fallback。允许标量 standardized/raw 阈值的 `engineering_pilot` 协议只用于吞吐与接线，不能
产生正式 compatibility claim，也不能被 full trainer 接受。

神经预处理需要一个正的 uncertainty feature，但这不允许把人为填充值伪装成实验误差。真实曲线没有
逐点 sigma 时，可以显式记录一个仅供 encoder 使用的 relative-sigma proxy；传给 compatibility/gold
验收的 `sigma_log` 必须保持缺失。只有随数据提供并通过单位检查的 sigma 才能进入 standardized score。
V5 模型另接收独立的三态 `uncertainty_provenance` one-hot：`measured_sigma`、`simulated_sigma` 或
`encoder_proxy_missing_sigma`；它进入 conditional curve state，不能从曲线数值或填充值隐式猜测。

compatibility calibration 只回答“精确曲线在该噪声模型下是否可兼容”，不校准神经 density，也不控制
经过搜索选择后的错误 branch 假阳性率。未经 refinement 的神经 proposal samples 必须另外对冻结 target
construction distribution 做 held-out NLL、coverage、rank 或局部 classifier two-sample diagnostics；只有满足
后文生成条件时才称为 SBC。refinement 后的候选不得继续称为 posterior sample。
正式 95% 层内阈值使用至少 200 个、目标 500 个独立 physical recipes，而不仅是数学上勉强可计算的
20 个。

## 离散结构

shape 集合固定为：

```text
sphere, cylinder, vertical_cylinder
```

K=1…4 的无序、可重复 shape multiset 一共 34 类。V8 使用稳定 topology ID 和 canonical component
顺序，不再同时预测 `TYPE_EMPTY` 与独立 existence，也不允许可微训练用 soft shape 混合替代一个
真实的 hard topology。

重复 shape 不自动意味着槽位可交换。只有两个槽的完整 GUI geometry bounds（包含 D policy）、当前
hard branch 的 D-presence 以及各自 `Int_i` bounds 都完全相同时，交换才是同一物理查询中的 label
symmetry；仅 shape 相同或范围重叠都不够。连续 target、proposal、exact refinement、参数距离和
reference matching 必须复用这一等价定义，并把几何与对应有效幅度系数一起置换。调用方缺少 amplitude
context 时采用保守策略：每个槽都是 singleton，不做任何交换。

`D` presence 和 Resolution presence 是离散分支。Resolution absent 必须精确令 `a_res=0`；若
Resolution present 候选没有可观测证据，输出应标记为 observability unknown，不能擅自
降成 effective absent。粒子的 effective topology 不能按
系数 fraction 判断，因为不同 shape 的 unit basis 绝对归一化不同；应以删除该分量、重新 profile 其余幅度
后的 exact log-score 增量为主，并同时报告尺度不变的 curve/uncertainty-normalized contribution。
降阶模型仍然 compatible 是“该项不必要”的正证据；只有 K1 对穷尽的 K0
BG-only null（或 Resolution shape 已被用户范围固定后的 BG+Resolution 非负线性 null），或具有
适用性证明的独立全局 certificate，才可把降阶失败解释为
“该项必要”。K≥2 固定几何删除、D-toggle 或 Resolution-toggle 在有限搜索中未找到解，
只能标记 `provisional_or_unknown`。这些候选仍保留在 exact-compatible 列表中展示，不得被
可观测性门禁静默删除。

## 唯一版本化 codec

模型包必须声明并严格匹配：

- forward contract ID；
- input preprocessing ID；
- physical-to-latent codec ID；
- topology catalog ID；
- q 单位和支持范围；
- artifact schema version。

训练、验证、一键预测、用户范围和 refinement 必须调用同一 codec。加载器遇到缺失版本或版本
不一致时应拒绝运行，不能回退到 legacy normalization。

机器可读 methods template 必须直接绑定当前实现的 forward、preprocessing、branch codec、V5.2 model、
TensorFlow-free objective identity、compatibility calibration/universe、formal observation policy、formal
production source/stage/shard/plan/membership/runtime、exact-search executor、query-local parameter distance、
contextual reference bank/deduplication、paper budget evaluator/matching、每个 amplitude 轴独立的 range
assignment、168D authoritative Sobol coordinate contract、paper RQMC design/evaluator/frozen training-artifact
set，以及 one-click branch ranking/round-robin seed scheduler 的 schema/version/SHA 或稳定语义标识。能安全导入的身份从 owner 模块读取；会造成
import cycle 或加载 TensorFlow 的身份用明确字符串绑定，并由从安全导入方向执行的一致性测试核对。任何一项
漂移都必须生成新 protocol version；methods template 的 digest 不能冒充尚未生成的具体 run/model artifact。

V8 latent 中所有 distribution width 使用对应 mean 的 fraction；在 GUI 边界显式转换：Sphere 和
Cylinder 的 `sigma_R/sigma_h/sigma_D` 为绝对宽度，Vertical Cylinder 的 `sigma_R` 保持现有相对
宽度语义。用户输入的物理范围先经相同转换进入 latent；候选导出前再反向转换并验证范围。

## 一键推断

`V5BoundsQuery` 只描述一个 canonical topology；产品层的“一键”查询不能把这一点误当成用户已经选择了
K/type。跨 topology 入口必须对同一份预处理 observation 和 uncertainty，接收用户明确允许的多组
topology-specific geometry/amplitude queries，按稳定 `topology_id` 顺序为每组完整枚举所有 codec-feasible
wire branches。全局 branch key 固定为 `(topology_id, pattern_id)`；另用绑定 geometry/amplitude query 与
该 branch 幅度 polytope 的 context digest 区分不同用户范围。一个 topology 内不得按模型分数预先丢 branch，
不同 topology 的 component slot 和 `Int_i` 范围不得映射到共同的无类型参数槽。显式 topology 子集限制必须
进入审计。最终参数代表的去重只能在 exact-forward 验收后、同 declared topology 与合法 slot 等价类内使用
参数距离；不同 topology 的槽位绝不对齐，只允许用 curve-equivalence group 表达跨 topology 曲线等价。
branch key 也不能代替物理解去重。该 framework-neutral 组合契约位于
`PosteriorV8/universal_query_v5.py`。

```text
curve + uncertainty provenance + geometry/amplitude user bounds
  -> versioned preprocessing
  -> contextual policy-allowed hard-branch enumeration
  -> per-branch frozen-search-yield ranking
  -> branch-conditioned local-coordinate MDN samples
  -> Sobol or retrieval fallback seeds
  -> profiled non-negative amplitudes
  -> bounded exact-forward multi-start refinement
  -> deterministic complete-linkage diameter-δ cluster representatives
  -> independent curve-equivalence grouping
  -> diverse, exact-verified candidates
```

神经网络是 amortized proposal generator，不是最终科学判据。最终排序首先使用 GUI-consistent forward
误差和范围/物理约束；模型 proposal score 单独报告，未校准前不得命名为 posterior probability。
若第一轮没有达到目标，workflow 可以自动扩大 contextual branch beam、local proposal samples 或 restart budget。

当前 global-target 工程模型的连续输出属于 full-domain codec。安全转换链固定为：

```text
model global u
  -> full-domain codec decode
  -> physical user-bound validation
  -> user-bounds codec encode
  -> bounded exact refinement
```

同一个数值 `u` 不得直接交给 user-bounds codec，也不得用 clip 修补越界。由于 GUI width coupling 和
hard-core D 约束，物理用户范围通常也不是简单的 global 26-D box。正式产品在 local-target 模型冻结前
默认让网络使用 full-range condition；越界 proposal 被拒绝并由 user-codec Sobol/retrieval seeds 补足。
若确有经过验证的 global-coordinate box，工程模型可对 logistic-normal 做精确截断，再用
`low + (high-low)u` 映射；该兼容路径不得宣称是 local-target density。

搜索预算至少分开记录 topology beam、branch beam、mixture/sample 数、单候选 forward 上限与整轮
forward 上限。单个困难候选不能耗尽整轮预算；任何一个 refinement 异常只产生一条失败审计并继续后续
候选。网络、retrieval、Sobol proposal 来源分别统计，最终科学排序仍只使用 exact score、范围和物理门禁。
论文输出上限 `N` 只对在当前预算 prefix 内已完成最终 exact verification、通过 compatibility/range/physics
门禁并作为 cluster representative 返回的候选计数。incompatible 或 unverified 尝试仍消耗其已经发生的
exact-forward calls，但不占用 `N`；顺序必须是先按 `B` 截断 call ledger、再筛 exact-compatible verified
returns、最后按冻结且与 reference 无关的 ranking 取前 `N`。

V5.2 的 proposal-to-verification 交接由 `candidate_refinement_v5.py` 负责：proposal 的
`branch_batch_index` 必须同时绑定 `V5CandidateContextBatch` 中的 user-local branch codec 和同一对象身份的
`amplitude_constraints[index]`。该 `GuiAmplitudeConstraint` 不仅用于最终 gate，而是原样传入初始 profile、
每次 nonlinear optimizer residual 以及终点复核，使所有 `[BG,a_i,a_res]` 求解始终位于同一耦合 coefficient
polytope。每个 authoritative GUI-forward 调用按 seed profile、optimizer residual 和 terminal verification
分项记账。这里冻结的一个 exact-forward 预算单位是一次完整 geometry-objective evaluation，包括该几何点的
design/basis 构造、同一 polytope 内的 amplitude profile 和 authoritative GUI-forward verification；预算在这些
计算开始前占用，失败的已开始 evaluation 仍计入，内部 basis/profile 计算不是免费预算。候选内 hard limit
耗尽、数值 refinement 失败、以及总预算耗尽导致下一个 seed 未开始，分别使用独立 attempt status 和 ledger
count；候选内耗尽仍表示该 seed 已处理，而未开始 seed 会令 `all_input_seeds_processed=false`。单个候选的校验或
refinement 失败只写入该候选审计并继续。成功输出仍只是可交给 query-bound evaluator 的 exact candidate，
尚未形成 measurement-compatibility、`no_solution` 或 posterior 结论。

精确 refinement 的优化向量只能包含 branch codec 的 `varying_indices`，不能使用包含 user-fixed 轴的
`active_indices`。固定但结构上存在的 R、width、D 或 Resolution 参数保留在 seed template 中，并在整个
优化期间逐位不变；它们不占有限差分维数。单 seed 的 solver 预算也必须用同一个 varying dimension count
计算。若 `varying_count=0`，直接做一次 authoritative profile/verification，不进入 SciPy。该规则避免零
Jacobian 列浪费 exact-forward budget，也防止 K 较大或 mixed/fixed 范围查询被系统性判成难解。

候选生成层只能报告 `raw_target_reached`、`raw_partial` 或
`no_candidate_found_within_budget`；它不判断解是否与观测兼容。verified 层每新增一个 raw candidate 后，
必须调用与产品 query 绑定的唯一权威 evaluator，以 exact compatibility gate 和 complete-linkage 参数代表
去重后的数量决定
是否达到目标。incompatible candidate 或重复代表不能提前结束 retrieval/Sobol fallback。有限搜索返回零个
候选只表示给定预算内没有找到，绝不能写成 `no_solution`；后者只允许由独立、版本化且适用当前范围的
不可行性 certificate 给出。

V5.2 universal 路径的权威实现固定为
`PosteriorV8/query_bound_evaluation_v5.py::evaluate_v5_query_bound_candidates()`：每个候选先由
`V5UniversalCandidateContext` 与其 `V5UniversalCandidateProvenance.global_branch_key` 绑定到唯一 contextual
task；evaluator 自身先核对 context encoder 的 q/intensity/uncertainty provenance 与传入 exact
`ObservedCurve`，再重验 topology、D/Resolution pattern、user-local codec 与同一幅度 polytope；任何
curve/context 或 provenance/context 不一致均 fail closed。`ReferenceMode` 从自身 topology、逐槽 D presence
和 Resolution presence 映射到当前 query 中唯一 branch，缺失、歧义、非 canonical pattern 或越出
geometry/amplitude query 均 fail closed。跨 contextual branch 的 parameter distance 为无穷。聚类、early
stop、参数代表和 reference matching 全部调用同一个 query-local distance contract；主 reference matching
只比较实际返回给用户的 parameter-mode representative，不能使用簇内隐藏 member。该 evaluator 的
schema/version/SHA 写入 universal result audit，并绑定当前 bounds/amplitude query schema、V5.2-R2 model
identity 与 numeric-policy contract。原 `evaluation.py::evaluate_candidates()` 的 global、scale-quotiented、
all-same-shape-permutation 行为仅保留为旧 workflow 与 correctness regression baseline，不得用于 V5.2
universal 或论文 endpoint。

搜索停止与主 success status 只由 exact-compatible、按冻结 parameter 距离去重后的候选数决定。
component observability/minimality 是独立的次级证据：它可以把分量标为 unneeded、needed 或 unknown，
但不得过滤主 compatible 候选、不得把 K>=2/D/Resolution 的 unknown 改写为失败，也不得阻止达到候选
目标。observability 必须在主搜索停止后使用独立预算运行，其 exact-forward calls 不进入论文主预算 `B`。

主 parameter distance 是绑定完整用户 query 的、gauge-invariant 但 scale-sensitive 的版本化距离：几何轴按
branch codec 的用户范围归一化；线性部分比较 `[BG,a_i,a_res]` 的 composition，并逐轴比较由同一 GUI
amplitude polytope marginal interval 定义的 zero-safe 对数绝对尺度。总距离取结构/composition RMS 与绝对
尺度 RMS 的较大值，fixed 轴不贡献维数。它只商掉产生相同有效系数的 shared-`k` GUI gauge，不商掉真实
曲线幅度。fixed interval 中相同值距离为零，不同值为无穷；跨 contextual branch 距离为无穷。exact-search
去重、reference bank 和 endpoint matching 必须绑定同一 distance schema/version/SHA。

## 训练数据

物理采样器首先产生 **identifiability candidates**，不能因生成标签已知就直接称为 identifiable。
混合 shape 共享相同的尺寸 prior；只有重复的同 shape 分量可以按明确 canonical order 做间隔采样，
不得让 canonical shape slot 与尺寸区间形成可被网络利用的伪相关。

候选经过固定预算的 delete、merge、D-toggle、Resolution-toggle 和 shape-swap competing-model refit
后，训练集才分为两个明确子集：

- **identifiable core**：每个分量在观测窗口内有足够贡献；删除、合并或换 shape 后不能在噪声内
  重新拟合；
- **ambiguity bank**：主动包含弱分量、重叠、窗口外、D 不可见和 Resolution 补偿，并通过 hard
  topology 多启动求解保存有效参数模态与临界 hard negatives。

数据生成保存 compact parameter recipe、seed、q/noise provenance；不同噪声、mask、crop、N、q window
和 grid 视图可在读取时生成，避免为百万级训练重复保存 padded 曲线。仪器/q-window policy 只能依赖独立
observation seed，不能读取未知的 R/h/D 来保证峰落在窗口内；看不到特征的窗口应保留为 ambiguity/OOD。
V5 的 observation/data-view 层固定先由 `recipe_seed + view_index` 选择 grid、noise、mask、crop 以及
sigma-present/missing policy，再在该 grid 上调用唯一权威 exact forward；不得用曲线或隐藏物理参数改变
采集选择。训练视图只包含 `simulated_sigma` 与 `encoder_proxy_missing_sigma` 两种：前者的模拟测量 sigma
可进入 acceptance，后者只向 encoder 提供带显式 provenance 的 relative-sigma proxy，传给 acceptance 的
`sigma_log` 必须为缺失。每条 view 同时记录 mask/crop 前的 design N、实际有效的 effective N、完整
`acquisition_policy_id` 和 3 维 uncertainty-provenance one-hot；同一 clean recipe 的所有 views 继承同一
recipe-level split ID。该职责位于 `PosteriorV8/observation_v5.py`，不包含 shard 或训练循环。
正式 cross-topology 标签不把这两类 view 混成同一个分类目标。它在看见曲线、生成参数或搜索结果之前，
仅由 `recipe_seed` 和冻结的候选 view pool（当前为 `(0, 1)`）执行 curve-blind 选择，并要求恰好选出一个
带 measurement sigma 的 view；零个或多个均 fail closed。所选 view、候选 pool、每个候选的 uncertainty
状态、policy ID/SHA 与选择审计必须进入全局生产计划和每个 shard membership。缺 sigma 的 view 仍可用于
encoder robustness/OOD 或真实曲线描述性分析，但不产生正式 compatibility 正负标签。
同一 clean recipe 的全部 observation views 必须在同一 split；train/tuning-validation/calibration/test 按
参数空间 cell 或 Sobol block 隔离，不能只依赖互不相同的随机 seed。calibration split 不得用于 early
stopping、architecture 选择或阈值之外的调参。

粗 parameter-cell hash 只能防止同 cell 重复，不能证明相邻 cell 与 train 分离。论文 test 还必须使用
contiguous Sobol blocks 或 guard bands，并报告 train-test nearest-neighbour distance；现有四路 grouped
split 在此之前只称 leakage-controlled interpolation split，不称强外推测试。

若模型把用户范围作为条件输入，主表必须单列 full-range。范围数据应先采样合法范围，再从该范围内的 prior
采样参数；当前“从 truth 向两侧生成范围”的 Phase-2 candidate 只可作工程 smoke，因为 narrow range 会泄漏
真值位置。连续 density 应预测范围内局部坐标并映射为 `low + (high-low)u`，同时覆盖非对称、贴边、固定参数、
排除 generating truth 但含另一有效解和真正 no-solution 的范围 OOD；否则
产品契约必须明确为“网络给全域 proposal，用户范围只由 Sobol fallback 和 bounded refinement 强制”。

V5.2 的幅度范围不能再用一个全局 regime 同时控制所有轴。`BG`、`k`、每个 active `Int_i` 以及
Resolution 存在时的 `int_Res` 各自拥有独立的 named `regime/width/position` 坐标，分别覆盖
`full/wide/narrow/fixed/edge_low/edge_high`；inactive 轴仍保留在坐标字典并进入审计，但不参与物理映射。
`Int_i` 是相互独立的 GUI 输入，不作 simplex 归一化。当前 authoritative direct-Sobol coordinate contract
固定为 168D，并以 schema/version/SHA 绑定 recipe、RQMC design、训练数据和模型 artifact；任何坐标变换语义
变化都必须改变 contract SHA，不能只比较坐标名字。

direct-Sobol 的范围生成、branch codec、hard-core、Resolution、target 和模型 embedding 必须显式绑定
固定 Decimal 数值策略：binary64 输入通过 `Decimal.from_float` 精确进入局部 context，使用
`precision=80`、`ROUND_HALF_EVEN`、固定 `Emin/Emax` 与 `clamp=0` 计算 `log/exp/sqrt/hypot/atan`，再转换
为 binary64。不得依赖 ambient Decimal context、环境变量或调用点 override。普通 GUI 查询仍显式绑定快速
platform-libm 策略，避免让高精度审计路径拖慢交互拟合；geometry/amplitude query 必须使用同一策略并把版本
和完整 policy payload 的 SHA 写入各自 canonical hash；policy payload 是 precision、rounding、指数范围、
traps、输入/输出规则与 `atan` range-reduction 算法的唯一事实来源，recipe、研究协议、模型契约和 exact-search
证据只能引用该 payload/SHA，不能各自复制一份可漂移的 context。十进制有效位后处理不能证明相邻
binary64 一定落入同一舍入单元，因此不再作为
跨平台保证。每个拟晋级的冻结 Sobol 设计仍必须从同一不可变源码快照分别在 macOS 和 Maxwell worker 上
生成不含运行时元数据的科学 manifest，并逐字节比较。门禁覆盖 0..1023 的完整生成 recipe/physics、固定
10 个 probe 的全部 34 topology query、78 维 geometry bounds embedding，以及 intensity reference 为 1 和
10000 时的 21 维 amplitude embedding；只允许替换被运行时版本污染的 Sobol design/point identity，禁止
量化或归一化任何科学浮点和派生 query hash。这只证明该有限设计在声明平台上的复现性，不外推为
任意实数超越函数的普适正确舍入。用户在 GUI 中直接输入的物理范围不会被十进制量化。

V4 bounds-first pilot 将上述要求固定为可审计生成顺序：先用独立 `bounds_seed` 在真实
`GuiComponentBounds/ResolutionBounds` 物理域采 full/wide/narrow、非对称、贴边和部分 fixed 范围，经
`ProfiledBranchCodec` 验证 hard-core `D` 可行性；然后才用独立 `local_target_seed` 在该 codec 的 local
unit coordinate 中采 truth。artifact 同时保存 local target、全域 codec 参考坐标、GUI 物理 truth、物理范围的
规范 JSON/SHA256 和 78 维 `(low, high, present)` bounds embedding。fixed 与 inactive coordinate 都编码为
0.5，但由独立 `local_varying_mask` 区分。该 embedding 明确不是简单 global 26-D latent box；`no_solution`
和 OOD 标签必须带独立理由与 certificate，且禁止携带伪造 inverse truth。V4 位于
`PosteriorV8/bounds_first_contract.py` 与 `PosteriorV8/bounds_first_dataset.py`；原 V3 数据语义保留作基线。
用于 continuous density 的 varying local target 从与 logistic-normal 支持一致的开区间均匀采样；即使物理
range 本身贴住全域边界，也不在主训练中制造网络无法表示的 0/1 概率原子。精确落在 closed boundary 的
情况进入独立 optimizer stress set，并由有界 refinement 验证；若未来需要网络直接给出边界质量，必须改用
显式 zero/one-inflated density 并重新版本化，不能靠 clipping 冒充。
其 solution-only sharded 扩展按每个 topology 自己的 occurrence，对 18 种 range/placement 与该 topology
全部有效 D/Resolution branch 做版本化全因子循环；同一 clean recipe 的 observation views 共享 bounds、
truth 和 split。当前四路 hash split 必须在 metadata 标为 `recipe_grouped_interpolation_pilot` 和
`no_parameter_guard_band`，不能称为论文 strong holdout/OOD；正式研究仍需独立 Sobol block/guard-band
split 与 train-test nearest-neighbour audit。sharded artifact 不生成 no-solution/OOD 标签。
从 recipe index 0 开始覆盖各阶段所有 topology 的最慢完整全因子周期分别需要 216 个 K1 recipes、
1,296 个 K1–K2 recipes 和 14,688 个 all34 recipes；这些是对相同 shape 槽位的离散 D 状态取排列商后
的 schedule 覆盖下限，不是统计充分性结论，
observation views 也不能当作额外独立物理 recipes。

正式 bounds-conditioned 模型必须有不同于 global-target 工程模型的 model name、manifest schema 和输入输出
坐标语义。其条件输入是上述真实 GUI physical bounds embedding，而不是伪造的 global 26-D box；连续输出是
由当前 user-bounds codec 定义的 local unit coordinate。加载器、训练器和推断器都必须 fail closed，禁止
V1 global artifact、V2 engineering artifact 与正式 bounds-first artifact 交叉加载。固定或 codec-effective
不可变的坐标不进入 continuous density loss，采样时保持规范值 0.5。

V5.2 local proposal density 使用给定 query/branch 下经 exact verification 的 compatible local targets。
target 的选择与采样分布属于必须版本化的操作性训练分布；除非另行证明它来自已声明 prior/likelihood 的
无偏 conditional samples，否则该 MDN 不称为 posterior。显式多解搜索用于监督 provenance、独立验收、
hard-case mining 和覆盖率测量；“未搜索到”只能在冻结搜索确实完成后作为 operational search-yield 负标签，
不能当作数学不可解标签。

当前 grouped solution-stage 的 known-truth positive 只足以做 generating-topology/branch 的 local-density
warmup、codec wiring 与 memorization；即使同 topology 的其他 wire branch 被保存为 `unverified`，它也没有
提供跨 topology model-selection 监督，不能报告 search-yield ranking 或 K/type 选择能力。正式
cross-topology verified-search 阶段必须把同一 observation 与每个预先冻结的 topology-specific
geometry/amplitude query 的全部 feasible wire branches 配对，并对每个 branch 独立执行不读取待评估模型分数
的冻结 bounded exact search。只有该搜索完成得到的正/负 outcome 才进入 search-yield BCE；不得从 generating
topology 或 generating parameters 推导 competing-topology 标签，generating mismatch 也绝不是自动负例。
完整阶段与 claim 边界的机器可读契约位于 `PosteriorV8/universal_training_contract_v5.py`。

V5.2 的 cross-topology 搜索标签不写回上述 solution-stage grouped artifact，而由
`search_supervision_contract_v5.py`、`search_supervision_sidecar_v5.py` 与
`search_supervision_overlay_v5.py` 管理独立、不可覆盖且与父 artifact 精确哈希绑定的 sidecar。
一个 sidecar 只属于一个 `train` 或 `tuning_validation` 父 shard；正式生产 sidecar 对每个 clean recipe 只覆盖
由冻结 curve-blind policy 从 `(0,1)` 选出的唯一 sigma-present observation view，而不是全部 training
augmentation views；并为该 selected observation 下每个显式选择 topology 的全部 codec-feasible wire branch 保存
唯一 `(clean_group, universal_query, global_branch, context)` 行。branch 行独占所有 candidate-owned
`MODEL_V5_INPUT_KEYS`；`x`、`point_mask`、`global_features` 和 `uncertainty_provenance` 只从父 observation
严格 join，sidecar 不复制第二份模型输入事实来源。缺失 observation、重复 branch、部分 catalog、跨 split、
父 artifact/hash、query/context/protocol 不一致均 fail closed。正式 full search-yield training 默认拒绝任何
`unverified`；completed negative 只表示同一冻结、模型无关、等 branch 预算搜索已按声明 termination 完成，
不表示数学上无解。

正式 frozen-search executor 使用一份保存确切 bytes、顺序、SciPy 版本和 digest 的 direct 26-D local-unit
Sobol 日程。每个 branch 先完成固定 scout prefix，再仅按所选 exact metric 与 Sobol index 稳定排序 scout
起点，并在同一 codec 和同一 `GuiAmplitudeConstraint` 中做 multistart refinement。论文模式禁止 positive
early stop：发现首个 compatible candidate 后仍继续，直到该 branch 的 exact-forward budget 完整耗尽，再以
冻结的 query-local、gauge-invariant、scale-sensitive parameter distance 和 complete linkage 形成全部已发现
cluster representatives。正/负结果均保存完整
budget-prefix ledger；异常、日程未完成、预算不足或 artifact 不可重放只能产生 `unverified`。executor artifact
独占创建，保存候选 local 参数、exact 曲线分数、范围/物理门禁、代表曲线、逐阶段 ledger、所有相关版本/hash，
且明确记录没有读取 neural proposal score。写入完成后 executor 必须立即作 task-bound 语义回读，逐个从
local 参数与线性系数快照经权威 GUI forward 重建候选曲线，并核验 exact 曲线 hash、raw/standardized/selected
metric、compatibility、完整任务 provenance 和代表集。因此即使 completed negative 没有代表曲线，其“全部
候选均未过阈值”也不是仅信任 artifact 中预存的分数字段。

替代 topology 的用户范围不能由 generating truth、源 topology 范围或 agent 猜测。collector 要求调用方提供
版本化 query-catalog artifact ID/SHA，并在每个 observation 行冻结每个 topology 的完整 geometry 与 amplitude
canonical JSON；同一 clean group 的所有 view 必须引用完全相同的 catalog。当前 direct-Sobol universal query
design 可作为该显式来源，未知范围没有 fallback。exact runner 也不能从归一化 encoder tensor 反解物理曲线：
每个 task 显式携带由 `PreprocessedCurve.valid_arrays()` 对齐的 `ObservedCurve(q, intensity, sigma_log)` 及 digest。
其中 `sigma_log` 只能来自 observation 的 acceptance evidence；encoder-only proxy 永不进入 exact compatibility。
正式模式有 acceptance sigma 时使用该 observation 所属 acquisition stratum 的冻结 standardized natural-log
RMSE threshold；缺失 sigma 时只能形成 raw 描述性记录或 `unverified`，不能形成正/负训练标签。显式 raw
natural-log RMSE fallback 仅保留在标记为 `engineering_pilot` 的吞吐协议中，并在 query/search record 审计其
非正式 claim boundary。

一个 positive branch 可以引用同一版本化 representative-set artifact 中多个 cluster 代表；该 artifact 必须按
冻结 distance、δ 和确定性 complete-linkage 规则证明每个 cluster 的直径 `<=δ`，但不得声称所选代表彼此
pairwise `>δ`。sidecar 保存全部 representative references，
training overlay 按带 local target 的代表展开 MDN 样本，同时把原 branch 的 `candidate_weight` 均分，避免
search-yield BCE 因多代表重复而加权。当前实现已包含模型无关的生产 executor、完整
schema/collector/validator、训练 overlay，以及 `frozen_search_pipeline_v5.py` 所有 branch 的本地流水线。
流水线从冻结 split/design 重放 clean recipe 与 physical observation，要求 `generating_only=False` 父 shard，
并把每个 direct-Sobol universal query design 作为真实 catalog artifact 以 ID/SHA 绑定。已有父 shard 只有在
manifest 和全部数组严格重放相等时才复用；executor evidence 先独占创建，完整 sidecar 原子发布后必须再
发布逐 task 回放的 evidence receipt，completion 最后发布；异常只发布独立 failure audit，不会把缺 receipt
的结果声明为可训练标签。`freeze_exact_search_schedule_v5.py` 必须在 Maxwell 指定的
TensorFlow/SciPy 环境中一次冻结 local-Sobol checked artifact；`launch_frozen_search_v5.py` 默认 dry-run，
只在登录节点做 hash/path 检查和提交 train/tuning-validation 两个 CPU arrays。两个 array 先以 hold 状态提交，
两者及最终 fingerprint 都成功后才一起 release；任一步失败会取消全部已提交作业并保存审计。worker 在任何
物理生成前重新核对 source bundle、split plan、Sobol design、calibration 以及 local schedule 的
logical/file/artifact/manifest SHA，并在 sidecar/receipt 发布前重新核对 source bundle，防止排队或长搜索期间
路径内容漂移。search sidecar `/v5` 与 K1 diagnostic result 的 `source_sha256` 均覆盖全部
`PosteriorV8/*.py`，并额外绑定权威 `scattering_model.py` 与 `physical_constraints.py`；不得用只列入口脚本的
局部哈希代替这一闭包。receipt 同时绑定全局 launch-plan SHA 与当前 shard-plan SHA。当前 launcher 只允许完整 K1、
每 shard 一个 recipe，并根据显式 pilot 吞吐率、calls/branch、queries/shard 和安全因子拒绝预计超过 20 小时的
24 小时 worker；失败 shard 只作审计且必须换新 run root 重跑，真实 pilot 审核前禁止扩容。manifest 仅给出
未自动执行的 `afterok` dependency template；当前 pilot 或 formal-calibration contract smoke 都不能传给 full
trainer。Maxwell 真实吞吐 pilot 和 E1 标签挖掘尚未执行，
因此本地 K1 contract smoke 不能称为工程或论文效果结果。calibration/test split 也刻意不进入该训练 sidecar，
未来若需要保存评估搜索证据，必须使用独立版本化 artifact。

两阶段 trainer 的数据源不能混用：warmup 从全部 grouped parent shard 只读取 known positive；full 阶段只从
显式提供 sidecar 的 parent 子集读取 verified positive/completed negative。这样允许较大的连续参数 warmup 与较小
但完整的 cross-topology 搜索标注集共存，不要求为每个 warmup shard 挖掘昂贵标签。sidecar 依据其内部 parent
artifact/manifest SHA 在同一 development split 中唯一匹配，命令行顺序不构成关联证据。full 开始前，train 与
tuning-validation 两个实际标注子集都必须整体满足 positive `>0`、completed negative `>0`、unverified `=0`，
且所有 sidecar 使用同一冻结 protocol；单个 recipe batch 不要求恰好同时含正负。warmup 与 full 分别拥有
`steps_per_epoch` 和 `full_steps_per_epoch`，full sampler 永不访问没有 sidecar 的 recipe，并必须拒绝
`engineering_pilot` protocol tier；本轮 `paper_full_calibrated` 只验证逐 observation 校准阈值与搜索证据契约，
其 purpose、prefix、scope 与 receipt 都明确标记为 contract smoke 且不可训练。未来 full 阶段还必须使用新的
可重放 paper-scale production launch contract，并取得显式 `full_training_eligible` receipt；协议 tier 本身
不构成训练资格。

该正式生产合同由 `formal_production_search_contract_v5.py`、
`formal_production_search_plan_v5.py` 与 `formal_production_search_membership_v5.py` 定义：全局计划显式列出
K1、K1–K2、all34 三阶段的 train/tuning-validation shard、clean-parent/Sobol 成员、每个 parent 的唯一
sigma-present view、完整 query digest、输出相对路径和逐 branch exact-forward 预算。计划要求所有阶段/分片
互不重叠且能从冻结 split/design/query/schedule/calibration/source 重放；receipt 还必须先通过文件级证据回读，
再证明属于同一 global plan 的唯一 shard。当前该 membership proof 明确不单独授权训练，生产 launcher、
worker 重放和 trainer collection gate 完成前，`TRAINING` receipt 仍硬关闭。

当前 checked ZIP reader 不把不同 shard 的大数组合并，但会让已验证 shard arrays 常驻单进程；trainer 在读取前按
manifest 计算 declared array bytes，并以 64 GiB 为 fail-closed 安全门禁。超过该门禁的 paper-scale run 必须先
采用仍能逐 array 校验 SHA 的 streaming/mmap shard reader，不能靠扩大内存或跳过 artifact 验证绕过。

## 论文问题与实验协议

以下内容是 **methods template / frozen-design-before-holdout**，不是外部时间戳 preregistration，也不是实验
结果。数值阈值、模型选择规则、具体 split/design/calibration/reference/model artifact 身份和排除规则必须在
第一次查看 synthetic test 或真实曲线结果之前写入不可覆盖的版本化 run protocol manifest；未冻结或未执行的
比较必须标为 planned，不能仅凭本模板声称已经预注册或得到论文性能结果。

### 论文定位与预期贡献

工作定位为 **forward-model-verified amortized multi-solution inversion**：深度模型只降低发现候选解盆地
的成本，科学结论限定于版本化经验正演、物理/范围约束、受限多启动优化和独立 reference search。这里的
verified 不代表完整 GISAXS/DWBA 物理正确，也不消除 synthetic train/test 使用同一 forward 的 inverse
crime 限制。论文不以“恢复唯一生成参数”为中心，而检验在相同 exact-forward 或 wall-clock 预算内能否
发现更多与数据兼容的 complete-linkage diameter-δ cluster 参数代表。不同 cluster 的代表不保证 pairwise
distance `>δ`。预期方法贡献限定为以下四项，任何一项都必须由
独立消融支持后才能在摘要中声明：

1. topology/branch-conditioned 的多模态 proposal distribution；
2. 直接在用户物理范围 codec 的 local coordinate 中学习和采样，覆盖窄、非对称、贴边与 fixed 范围；
3. neural、retrieval、低差异 seeds 与 exact constrained refinement 组成的预算化发现流程；
4. 将 proposal uncertainty、measurement compatibility 与 search incompleteness 分开校准和报告。

同一版本 forward 生成 train/test 的 ID synthetic 结果只能支持“在 nominal empirical forward 下的算法发现
效率”，不能单独支持散射物理泛化。物理泛化 claim 之前必须增加至少一套与训练实现独立或明确扰动物理项的
alternate-forward OOD，并在参数映射有效的范围内加入 BornAgain 生成的 challenge set。alternate-forward、
BornAgain 和真实案例都不得用于训练、compatibility calibration、checkpoint/family selection；nominal ID、
alternate-forward OOD 与真实案例必须分表报告，并把模型失配和失败案例保留在预先声明的分母中。
在 alternate-forward challenge 的生成器版本、参数映射、样本量、分母、失败/abstention 规则成为可执行冻结
artifact 前，这一项仍标记为 pending，不能写成已完成贡献。冻结因子至少包括 q calibration offset、非恒定
背景、异方差及相关噪声、Resolution/convolution mismatch，以及 form/structure-factor mismatch；BornAgain
只覆盖参数映射可追溯的子集。若查看某个 OOD 结果后改变模型，必须生成新的 untouched challenge family。

若简单 diagonal MDN 已达到冻结门禁，则它是主模型；只有 paired validation 表明稳定收益时才升级更复杂
density family。文章的创新性不依赖堆叠网络复杂度，而依赖问题定义、物理验收、搜索完整性审计和可复现的
多解指标。

当前可发表定位是“面向受约束经验一维散射正演的、由 contextual branch-conditioned amortized proposal
加速的多解发现”，而不是万能 GISAXS 反演器、完整 posterior sampler 或唯一结构恢复。可主张的结果必须是
相同 exact-forward/wall-clock 预算下的 representative-discovery 效率、范围合规与可复现失败边界；模型
复杂度本身不构成贡献。已实现的 V5.2 合同同时接收 geometry 与 amplitude ranges；只有冻结具体模型 artifact、
冻结搜索监督、独立 reference bank 和冻结统计计划后，才能开始 paper holdout 并形成论文级性能结论。

### Related-work 文献占位类别

投稿前应为以下类别补充经核验的一手文献；当前只冻结检索/论证位置，不在本契约内伪造引用：

- GISAXS/SAS 一维曲线的非唯一性、identifiability 与实验 q-window 限制；
- scattering inverse problem 中的传统全局优化、多起点、贝叶斯/ensemble 与不确定性方法；
- 深度学习用于 SAXS/GISAXS/reflectometry 参数反演及其 synthetic-to-real/inverse-crime 限制；
- amortized inference、conditional density estimation、MDN、normalizing flow 与 simulation-based inference；
- label switching、exchangeable components、set-valued prediction 与 permutation-invariant matching；
- separable nonlinear least squares、variable projection、线性幅度 profiling 与约束 polytope 优化；
- Sobol/低差异搜索、global optimizer benchmark、搜索饱和与 anytime/budgeted evaluation；
- conformal calibration、measurement-error compatibility，以及 selection 后 calibration 的边界；
- scientific-ML benchmark 的 grouped split、数据泄漏、seed-level uncertainty、independent-RQMC replicate
  inference、固定 cohort 的描述性 paired bootstrap 与多重比较；
- 训练/推断 latency、计算资源、能耗和碳排的可复现报告规范。

### 研究问题与可证伪假设

论文的核心问题不是“网络能否回归生成参数”，而是：在混合几何、离散结构未知和连续多解并存时，
amortized proposal 能否在相同计算预算内提高 exact-compatible 参数代表覆盖率，同时由版本化 forward
验收保证候选与该经验模型一致。
必须在 holdout 前冻结的主要假设为：

1. 解析消去线性幅度比直接预测 `BG/Int/int_Res/k` 提高收敛率和跨强度尺度泛化；
2. contextual hard-branch-conditioned density proposal 比固定多头回归提供更高的 best-of-N reference-representative
   recall；
3. proposal 加 exact refinement 在相同 wall-clock 预算下优于 solver-only Sobol/retrieval；
4. ambiguity-bank active learning 提高弱分量、重叠峰和错误 topology hard negative 上的覆盖率，且不
   牺牲 identifiable core 精度。

### 基线、概率模型阶梯与 go/no-go

所有方法必须接收相同曲线、sigma 可用性、用户范围与 branch catalog，并在相同 exact-forward 调用数和
wall-clock 两种预算下比较。最低基线包括 legacy 单点/固定多头回归、retrieval-seeded exact solver、
Sobol/global solver-only、proposal-only 以及 proposal + exact refinement。V5.2 的正式接口不再从曲线单独
输出 topology/branch logits：
每个用户查询先由外部 contextual catalog 枚举 policy-allowed hard branches，再由权威 codec 排除连续不可行
分支；网络对每个 `(curve, physical bounds, candidate branch)` 分别输出一个
`proposal_search_yield_logit` 和该 branch 的 local-coordinate diagonal logistic-normal MDN。该 logit 的
冻结 estimand 是：在指定生成/精修协议和 exact-forward 调用预算下，该 query/branch 是否至少找到一个
exact-compatible 参数代表。它只用于 proposal 排序/分配搜索预算；不得称为 posterior probability、数学上
的 branch 可解性或 `no_solution` certificate，也不能替代 exact-forward compatibility gate。

V5 search-yield 监督固定为 `compatible_found`、
`no_compatible_found_within_frozen_search_budget` 和 `unverified` 三态。前两态必须绑定同一冻结 search
protocol 的版本/哈希、完整 run artifact、evaluator、metric、threshold、exact-call budget/used、终止原因
和 compatible-representative count；负态只表示该协议在该预算内完成且未找到 compatible 代表，不表示分支
不存在解。正态还必须携带 exact-compatible representative artifact 与 gate 结果。只有前两态进入
search-yield BCE，`unverified` 的 BCE 权重严格为零；BCE 保留为 operational-yield calibration 项，正式阶段
同时在每个 clean recipe 内对全部 completed positive-negative 组合使用确定性 logistic pairwise ranking，warmup
阶段该 ranking 权重固定为零。local MDN NLL 只使用正态且存在 local target 的样本；full sidecar 中一个 branch
的全部 representative local targets 必须逐条展开，并把原 candidate weight 均分。每个展开 target 还使用所有
mixture medians 的权重参与的 negative-temperature log-sum-exp masked local RMS coverage loss，使位置正确但
proposal mass 极低、实际 Top-4 不会被尝试的 ghost mode 继续受到梯度惩罚。hard best-of-M RMS 只保留为诊断，
不再进入总 loss；另用 pairwise-sigmoid soft rank 构造可微 Top-4 对齐项，按 target-distance affinity 聚合后
惩罚覆盖 mixture 的 soft rank 超出 `L+0.5`，该项明确进入总 loss，hard argsort 只用于诊断。训练同时报告
与冻结推理策略一致的 Top-4 recall、hit rank、mass、entropy、effective-mode
utilization 和 duplicate fraction。MDN 与 coverage 都由 `varying_dimension_mask` 排除 fixed/inactive 轴。
该版本仍是逐 target 的 exchangeable density/coverage objective，不声称实施了 Hungarian 一对一匹配；相邻
target 仍可能由同一 mixture 覆盖。同一 clean recipe 内先按 candidate weight
归一化，再按 clean-recipe weight 作 macro mean，避免分支/候选多的 recipe 支配训练。batch 缺少某一类时
对应 head 稳定贡献零并报告审计 count。generating recipe 不匹配不得自动产生负标签。

branch-ranking recall@L 的统计单位是一个 query 的完整 codec-feasible contextual catalog；正分支是在同一
冻结 search protocol 下得到 `compatible_found` 的分支。只有 catalog 中每个可行分支都有 completed outcome
的 query 才进入条件 ranking endpoint；部分 `unverified` 的 query 不许删掉难分支后计算虚高 recall，而应从
条件 endpoint 排除并在完整预选 query sensitivity 中记零，同时报告 qualification fraction。该 recall 仍是
“冻结协议能找到”的 operational ranking，不是数学 branch-existence recall。
若完整 catalog 没有正分支，条件 ranking recall 无定义并排除，但在上述完整预选 sensitivity 中记零；另报
zero-positive fraction。catalog 大小小于 `L` 时使用全部 `min(L, catalog size)` 个分支。用于定义正负 outcome
的冻结 search 不得读取被评估模型的 score 或由该模型分配 branch 预算，避免模型参与制造自己的 ranking
标签；per-branch budget schedule 与 score 相同的 tie-break 也必须在看 reference 或 outcome 前冻结。

V5.2 使用 78 维 GUI geometry bounds embedding 与独立 21 维 amplitude bounds embedding，并显式区分
query 中物理轴可用的 `available_dimension_mask`、
当前 hard branch 选中的 `active_dimension_mask` 和 branch codec 中实际可变的 `varying_dimension_mask`。
因此 optional D/Resolution 可以在用户范围中 available、同时在某个候选 branch 中 absent。同 shape 分量
只有在完整 GUI bounds/policy 相同时才作槽位置换商；异质或仅重叠 bounds 的 single-D assignments 必须作为
不同 hard branches 分别评分。wire ID 仍为 `0..31`，contextual catalog 大小随查询变化，不使用固定 418
branch 假设。论文 proposal-density 基线是这种 per-hard-branch conditioned diagonal logistic-normal MDN；
其 mixture weight 在校准前只叫 proposal weight。

V5.2 必须显式接收用户的 `BG`、`k`、每个 `Int_i` 以及 Resolution 存在时的 `int_Res` 范围；这些输入与
exact profile/refinement 使用的 GUI amplitude coefficient polytope 来自同一份查询事实。否则相同几何
branch 在不同幅度可行域下会错误复用 search-yield score 和 local proposal。模型 contract 因此固定为新的
schema/version，并要求 artifact digest；开始 paper training 前必须验证 78D `geometry_bounds_embedding` 与
21D `amplitude_bounds_embedding` 均在实际模型图中生效，开始 holdout 前还要冻结具体 artifact identity。
主消融需在仍强制 exact polytope 的前提下单独移除 amplitude-range conditioning，从而区分“网络是否看见
范围”和“最终优化是否遵守范围”。

更强模型按下列顺序逐级开放，避免同时改变数据、模型和搜索预算：

1. diagonal MDN 先通过 codec、holdout 前冻结的 target-distribution diagnostics、held-out compatible-representative
   recall 与运行时门禁；
2. 只有残余失败呈稳定的参数相关性时，才比较 low-rank/full-covariance mixture；
3. 只有 mixture 数增长仍不能覆盖弯曲或相连模态时，才比较相同 per-hard-branch 条件下的 normalizing flow；
4. diffusion/score model 只在训练数据规模、采样延迟和重复采样成本均可接受时进入正式比较。

每次晋级需要在冻结 validation 上提高 paired budget-recall AUC，且在 holdout 前冻结的 target-coverage diagnostics 与
范围满足率上不
退化；所需最小提升和最大延迟增幅必须预先冻结。若 paired independent-RQMC replicate-mean 的 95%
Student-t interval 跨过零，保留更简单的
模型。最终 test 不参与架构晋级。

论文模型必须包含至少一个 full-search-supervision epoch；`full_epochs=0` 的 warmup-only run 永远不具备论文
模型资格。每个完成的 full epoch checkpoint 都要以 model/source SHA 不可覆盖地保留为候选。最终 checkpoint
只在 `tuning_validation` 上选择：首先最大化 recipe-macro N=16 exact-forward log2-budget recall AUC，再最小化
首次 exact-compatible verified return 的 median exact-call TTFC，随后比较 wall-clock TTFC，仍相同则选更早
epoch 和稳定 SHA 顺序。training validation loss 仅作优化诊断，不能单独选择最终 checkpoint；test、reference、
calibration 或真实数据均不得参与选择。在训练器实现 epoch retention 与该 selector 前，正式训练 promotion
保持关闭。

### Maxwell 分阶段执行矩阵

规模按独立 clean-parent physical recipe 计数；多个 noise/mask/crop view 只是相关增强。训练 augmentation
view 数与正式 search-label view 数是两件事：正式标签始终从冻结候选 pool `(0,1)` curve-blind 地选出恰好
一个 sigma-present view。每一级失败先定位 contract、目标函数或 encoder，不用扩大下一级数据掩盖问题。

| 阶段 | topology 范围 | clean recipes | augmentation / candidate views | 正式 label/eval views | 允许的结论与进入下一阶段门禁 |
|---|---|---:|---:|---:|---|
| contract smoke | K1 | 216 | pool 2 | 1 selected | 仅证明完整 schedule、codec/forward/manifest 与 100% bounds/physics 合规 |
| memorization | 单一 K1 branch：Sphere、D absent、Resolution absent | 512 | 1 clean | warmup oracle | 只隔离检查表示能力与目标函数，不声明完整 K1 覆盖；MDN single-draw local RMS median <0.05，best-of-32 median <0.01、p90 <0.03；exact post-refine raw logRMSE p90 <1e-3 且 compatible rate ≥99% |
| engineering E1 | K1 | 13,824 | 3 | 1 | frozen-search-yield ranking recall@4 ≥95%，至少一个 exact-compatible candidate rate ≥95%，reference recall@N16/B4096 ≥85%，主 AUC 优于 Sobol/global 与 retrieval |
| engineering E2 | K1–K2 | 82,944 | 3 | 1 | ranking recall@8 ≥90%，至少一个 exact-compatible candidate rate ≥90%，reference recall@N16/B4096 ≥75%，并建立 200–500 曲线 reference bank |
| all34 plumbing | K1–K4 | 14,688 | 3 | 1 | 覆盖 34 topology 和随查询变化的 contextual catalogs；只检查吞吐、loss、NaN 及四类 V5 指标可计算，不能当效果实验 |
| engineering E3 | K1–K4 | 235,008 | 3 | 1 | ranking recall@16 ≥80%，exact-compatible rate ≥80%，reference recall@N16/B4096 ≥65%；报告 AUC learning curve 与 failure strata |
| paper ID train | K1–K4 | 最低 940,032，正式结果至少 5 个预声明 seeds | 2–3 | 1 | 报告 14,688/58,752/235,008/940,032 clean-parent recipe learning curve；3 seeds 只作工程探索，最后一次扩容 AUC 增益仍 >2 percentage points 才开放更大规模 |
| paper ID test | K1–K4 | 最低 70k，目标 140k | 1 | 1 | 冻结独立 test；普通 generating diagnostic 与 reference-representative 结果分开 |
| calibration | 实际可达 60 acquisition-only strata | 最低 64k，目标 160k | 1 | 1 | 每层至少 200，目标 500 个独立 recipes；只拟合 compatibility threshold |
| gold reference | 分层 ambiguity set | 工程 200–500，论文 500–1000 curves | — | 1 | starts 1/2/4/8/16/32 饱和；最后翻倍新增代表 <1%，三组 seeds 的代表集 Jaccard ≥0.98 |
| OOD | acquisition、support、alternate-forward/model-discrepancy | 各自独立集合 | frozen per set | 1 | 分开报告鲁棒性、abstention 与 compatible success，不并入 ID headline |

K1-C/E1 的机器可读 gate 由 `k1_phase_c_contract_v5.py`、`k1_phase_c_plan_v5.py` 与
`k1_phase_c_evaluation_v5.py` 管理。它覆盖 12 个 K1 hard branches 和 25 个 range×observation stress
cells；夹具通过只证明 evaluator wiring，`formal` 与 paper acceptance 始终分离。Phase-A 单一 Sphere
memorization 模型不能直接充当 K1-C 的待测模型，必须先用与 gate 不重叠的全 K1 train/tuning blocks 训练。

输出上限固定报告 `N={1,4,8,16}`，主 N 为 16；主预算曲线固定报告 exact-forward calls
`B={256,512,1024,2048,4096}`。候选只有在 refinement 与最终 exact verification 均于累计调用数
`<=B` 完成时才进入该 budget prefix；跨过 B 的候选不计。每个 prefix 先筛 exact-compatible verified
returns，再按冻结 ranking 取前 N；incompatible/unverified 尝试消耗实际 exact calls，但不占 N。GUI 默认预算
只能在冻结模型后按实测 latency 选择。若 E1/E2 在数据翻倍后 paired recall-AUC 提升不足 2 percentage points 且仍未过门禁，停止扩数据，
转入 branch/continuous/encoder 的 oracle 诊断。

进入 K1–K2 扩容前还必须消除真正 exchangeable slot 的任意标号：只有同 shape 且完整 GUI bounds/policy
完全相同的 slot 才由 contextual catalog 作置换商；同 shape 但异质或仅部分重叠范围的 D assignment 是不同
物理分支，必须保留。连续目标采用与该 contextual equivalence class 一致的可复现 assignment；指标和
reference matching 只在同一个合法等价类内 permutation invariant。

### 消融设计

一次只去掉一个机制，并保持其余 clean-parent recipes、seeds、候选数和 exact-forward 预算一致。分别比较：
去掉 query-bounds 条件、把 local coordinates 换成 global coordinates、去掉 contextual candidate-branch
enumeration、去掉 frozen-search-yield ranking、把 profiled-amplitude 初始化换成 direct/joint-only 初始化、
在搜索内把 coupled GUI polytope 换成 naive coefficient axis box（最终 exact GUI range gate 保持不变）、
折叠三态 uncertainty provenance、去掉 exact refinement、只去掉 rescue、去掉 ambiguity-bank mining，以及
训练规模。另做一项 V5.2 专门消融：网络不再接收 `BG/k/Int_i/int_Res` ranges，但 exact solver 仍使用原
polytope，以测量 amplitude-range conditioning 自身的贡献。bounds 消融必须分别报告
full/wide/narrow、非对称、贴边与 fixed；不能把只测 full-range 的结果解释为用户范围泛化。
另外预注册：每个 amplitude axis 独立 range regime 对 shared-regime；known-generating-only 对 completed-search
supervision；search-label budget 加倍后的 label flip/stability；BCE+pairwise 对 class-balanced/listwise ranking；
mixture 数、diagonal 对 low-rank/flow；top-weight medians 对实际 stochastic sampling；index Conv1D 对
q-aware/log-q encoder；K2/all34 的合法 position-labelled 对 contextual permutation-equivariant component
representation。已版本化的 mass-aware 方案应与旧 `MDN NLL + hard best-of-M` 训练基线作受控消融；
生产 objective 使用 `MDN NLL + mass-aware soft coverage`，hard best-of-M 在其中仅作不进入梯度的诊断。
Hungarian 一对一 matched-set loss 只有真正实现、版本化并通过 focused regression 后才进入后续方法阶梯。
correctness bug 的旧实现不作为科学消融基线。

### 独立划分与模型冻结

固定随机种子只用于复现实验，不允许根据 test 结果选择 architecture、阈值或数据比例。architecture 与
early stopping 只看 train/tuning-validation，compatibility threshold 只看独立 calibration split；
synthetic test 使用未见过的 parameter cells，并额外设置超出训练 noise/q-window 组合的 OOD test。真实
Cut_Data 在模型和阈值冻结后才做盲测。相同底层 clean recipe 的不同 noise、mask、crop 或 grid view
必须留在同一 split，避免曲线近邻泄漏。

版本化 V5 split plan 的单一 scramble + guard-band 连续区块只保留为工程插值设计。正式论文训练可继续使用
低差异序列；`test`、`reference` 和 `ood` 必须各自使用冻结且互不重用的 scramble families，并各含至少
8 个独立 randomized-Sobol replicate（目标 16 个）。compatibility `calibration` 必须另用按冻结 strata
独立抽取且可交换的 IID clean-parent recipes，不能用 RQMC 点拟合有限样本 split-conformal 阈值。
guard band 只防近邻泄漏，不产生统计独立性。同一 clean parent 的所有
noise/mask/crop/grid views、contextual branches 和候选标签必须随 parent 分组，不能跨 split。`reference`
只用于 network-free gold search，不能参与模型或阈值选择。真实案例是冻结后的外部 case-study cohort，
不是第七个 synthetic Sobol block。OOD 子块分别预先声明 topology、range width、weak component 与
acquisition-policy holdout，不能把普通随机 holdout 重新命名为 OOD；记录 test/reference/OOD 到 train 的
最近邻距离。

### 独立参考解库与搜索完整性

参考多解库由跨 topology/D/Resolution 分支的大规模低差异多起点和 GUI-consistent refinement 构建，
并在小规模子集使用不同的 global optimizer 交叉核验，避免算法同源偏差。只有当新增
起点或搜索预算不再显著增加 complete-linkage diameter-δ cluster representative 数时，才把结果标记为“reference-discovered
representatives”；不能称为数学完备解。论文同时给出 representative-discovery saturation curve，使
recall 的参考集依赖透明可见。

主 reference bank 完全禁止使用待评估网络 proposal，由低差异多起点、GUI-consistent refinement 和至少
一种独立 global optimizer 的 gold subset 构建；所有被评估方法的 union 只可作 leave-one-method-out
sensitivity。至少一个分层 gold subset 使用 joint geometry + linear-coefficient exact optimization，并把
`[BG,a_1,...,a_K,a_res]` 严格限制在由同一用户 `BG/k/Int_i/int_Res` ranges 转换出的完整耦合 GUI amplitude
polytope 内；转换保留共享 `κ` witness 的存在性，而不是令 `k=sum_i a_i`，也不能退化为独立 coefficient
axis boxes。它与 sequential profile/refine 从相同 branch 和 seed
作 paired 比较。两者都记录 best-seen、exact-forward calls、wall time 和停止原因。reference-bank
saturation 是搜索完整性诊断，不是 posterior calibration。

主 reference population 包含所有 exact-compatible 且 bounds/physics-pass 的参数代表，包括
observability `unknown` 和 `confirmed_redundant`；`confirmed_effective` strict-minimal reference 仅是次级
sensitivity。reference schema 必须分别保存 primary eligibility、strict-minimal eligibility、逐候选与逐簇
observability 标签、阈值来源、搜索来源/预算和 saturation 状态，不能再用一个布尔值混合这些语义。
reference-search 运行前必须冻结完整预选 cohort、其 recipe ID/source SHA，以及每个 recipe 恰好一个的
typed qualification status；资格结果、协议和来源都必须形成可复放的内容哈希。主结果全名必须包含
“conditional on frozen reference-search qualification”，并同时给出：完整预选 cohort 中把未合格项记零的
保守下界、将 unresolved 项置于 1 得到的通用上界，以及按 K、歧义度、弱分量和 range width 的
qualification/saturation。若预定主 cohort 未达到冻结的分层 qualification 门槛，则该 headline 不成立，
不能通过删除困难 recipe 修复。

### 指标、匹配与统计推断

论文主指标固定命名为
`conditional-on-frozen-reference-search-qualification reference-discovered complete-linkage-diameter-cluster compatible parameter-representative recall@N`
的 log2-budget AUC。该名称不暗示不同 cluster 的代表彼此距离必然大于 δ。
对每个 clean recipe 和 `N={1,4,8,16}`、`B={256,512,1024,2048,4096}` 形成 N×B 矩阵；主 scalar 使用
N=16。令 `R_B` 为当前 B 下最大一对一匹配数除以 reference 数 `M_ref`，则每个 recipe 的主值为：

```text
AUC_log2B = (0.5 R_256 + R_512 + R_1024 + R_2048 + 0.5 R_4096) / 4
```

先在每个 training seed 和独立 evaluation scramble 内对 qualified clean recipes 做不加权 macro mean，再逐
seed 与逐 scramble 报告；
不得把具有更多 reference representatives 的 recipe 当作更多独立样本。方法 crash、超时或算法失败在全部 B
记零，不能作为 missing 删除。每个 B 内先保留 exact-compatible verified emissions，再按与 reference 无关的
冻结产品 ranking 选前 N 个 cluster representatives；incompatible/unverified emission 即使消耗了 exact calls
也不占 N。不得为提高 recall 用 reference 重排。若 `M_ref>N`，分母仍是 `M_ref`，并同时报告理论 ceiling
`min(N,M_ref)/M_ref`。

候选集与 reference set 使用 one-to-one bipartite matching：先最大化阈值内匹配基数，再最小化总 normalized
parameter distance；一个预测代表最多召回一个 reference 代表。主指标只用实际展示或返回给用户的
emission representative 与 reference 匹配，不能借用预测簇中隐藏的 accepted member 提高召回；“簇内任一
member 的 oracle 最近距离”只能作为明确标注的 secondary diagnostic。实际 emission 必须是恰好一组类型化、
可复放的物理参数；其 query、global branch、source artifact、规范化
参数、exact-intensity 摘要及 bounds/physics 状态进入内容哈希和 exact-call ledger，不能用隐藏 member 集合
代替单个输出。reference bank 在统计前使用同一 query-local scale-sensitive metric 和 complete-linkage 定义
去重。complete linkage 的保证是簇内任意两
member 的最大距离 `<=0.08`，不是输出代表之间 pairwise `>0.08`。操作阈值冻结为 parameter clustering
`0.08`、reference matching `0.10`、raw curve equivalence logRMSE `0.02`，并报告 `0.5×/1×/2×`
sensitivity；这些数是 holdout 前冻结的操作阈值，
不是物理常数。

主 query-local parameter metric 要求每个 `ReferenceMode` 都有 profiled `linear_solution`。它比较用户范围归一化
几何、有效系数 composition，以及绝对 `BG` 和粒子总有效幅度；只商掉 exact shared-`k` gauge。缺失
`linear_solution` 时该 reference artifact fail closed，不能在 pairwise distance 中静默删掉 composition 或
absolute-scale 维度；geometry-only 必须使用另一个版本化
secondary metric。reference search 未饱和或不合格的预选 recipe 不得替换：它不进入 qualified-only 条件主
估计，同时在“未合格记零”的完整预选集合 sensitivity 中计零，并始终报告 qualification fraction。reference
为空但没有独立不可行性 certificate 时只能标为 unresolved；有 certificate 的 no-solution 进入独立 abstention
endpoint。缺少 sigma 或 calibration stratum 未见过时，只报告 raw 描述性结果。

`accepted/proposed` 只可命名为 compatible candidate yield，去重后另报 unique-compatible-representative
yield。由于 reference bank 有限，未匹配但 exact-compatible 的预测称为 unmatched compatible
representative，不直接计为 false positive；`matched/predicted` 只能作为保守 reference-match-fraction lower
bound。论文不报告无法识别的“unique-candidate precision”，也不设置 precision 门禁。

每个主要指标报告 paired independent-randomized-Sobol replicate interval、按物理 strata 的分布和
wall-clock/forward-evaluation budget。正式 test/reference/OOD 各自由至少 8 个相互独立 scramble（目标
16 个）组成；同一 scramble 内的低差异点不是 IID，不能逐 recipe 作 population bootstrap。compatibility
calibration 使用单独的 IID/exchangeable clean-parent 抽样。若只对固定 benchmark
cohort 重采样，结果必须命名为 empirical-cohort descriptive interval。所有表格由版本化 audit JSON 自动生成，并保存代码、模型、数据
recipe、环境、Slurm job ID 和 SHA256；失败 run 也保留，避免只选择最好一次训练。

统计推断先在每个独立 scramble 内按 clean-parent recipe 作固定设计平均；每个 training seed 单独形成配对
结果，同时在同一 scramble 内先平均完整冻结 seed 集合的配对差，再以这些 paired scramble-replicate means
估计 seed-family 方法差异和 Student-t 区间。所有方法必须运行相同 points、预算、scramble 和冻结的 training
artifacts。正式论文至少使用 5 个预声明 training seeds；主 family estimand 明确为这组冻结 seeds 的平均条件
性能。seed 间分布另报，不能把单一 Sobol 点 bootstrap 当作训练随机性，也不能只选
最佳 seed。若要推断“重新训练一次”的期望性能，必须另行冻结包含 training seed 的交叉/分层推断方案。
主 confirmatory contrast 为 V5+exact refine/rescue 对 solver-only Sobol/global；其余同一 endpoint family
的比较使用 Holm family-wise correction，探索性 strata 明确标注为 exploratory。多个 views、候选或
reference representatives 都不是额外独立重复。

generating parameter RMSE 只用于可恢复 synthetic truth 的 oracle diagnostic，不能作为多解任务主指标。
每个候选还必须报告 exact raw logRMSE；仅在 sigma 真实存在且对应 calibration stratum 有冻结阈值时报告
standardized log score 与 compatibility。V5.2 不存在 curve-only topology classifier，因此离散候选层报告
frozen-protocol search-yield branch-ranking recall、Brier/NLL 与 reliability；连续候选报告 representative
recall、compatible/unique yield、reference-match lower bound、diversity/duplicate rate 和范围/物理约束
满足率。统计单位是 clean-parent physical recipe，而不是它的多个 noisy views；
正式结果至少报告 5 个预先指定的独立训练 seeds；每个 seed 单独给 paired independent-RQMC
replicate-mean 95% Student-t interval、分层结果和失败率，并报告“每个 scramble 内先平均冻结 seed 集合，
再跨 scrambles 推断”的 family summary，不能只选择最佳
seed。对固定已实现 benchmark cohort 的 clean-parent resampling 只能作为 empirical-cohort descriptive
interval，不能代替 population/generalization interval。

### 三类不确定性与校准边界

论文必须分别命名并验证三类量：

- **neural proposal uncertainty**：frozen-protocol search-yield logit 用 reliability/Brier/NLL；
  branch-conditioned continuous density 对冻结 target-construction distribution 报告 held-out NLL、coverage、
  rank 与必要时 classifier two-sample diagnostics。只有 target 确实来自预先声明 prior/likelihood 的无偏
  joint simulator draws 时，才把 pre-refinement 诊断命名为 SBC；
- **measurement compatibility**：只由真实 `sigma_log` 和独立、按冻结 strata IID/exchangeable 抽取的
  calibration split 上的冻结 split-conformal/pseudo-likelihood threshold 定义；RQMC 样本不能用于拟合该
  有限样本阈值；
- **search incompleteness**：用 reference-bank saturation、restart discovery curve 和 optimizer
  disagreement 描述。

三者不能互相替代：compatibility threshold 不会把 proposal weight 校准为 posterior probability；target
coverage 或适用时的 SBC 通过也不能证明已发现全部解；没有实验 sigma 的真实曲线不能获得频率学 coverage
声明。refinement 后候选是
优化解，不再是原始 posterior sample，必须在报告中分开。

### 真实 Cut_Data 盲测案例

模型冻结前发布不可变 case inventory、基于采集 metadata 而非拟合结果的纳入/排除规则，并记录每例 user
bounds 的制定者及依据。案例至少覆盖：结构特征清晰、弱分量或
重叠导致的多解、受限 q window，以及缺少可靠 sigma 的曲线。每例固定用户范围和运行预算，对比 GUI 人工
初始化、solver-only 与最终一键流程；保存候选叠图、raw/standardized residual（后者仅在合法时）、参数簇、
effective topology、observability 与失败审计。至少两位专家在隐藏方法标签下独立评价物理可解释性和候选
多样性；随机隐藏方法标签和候选顺序，冻结评分 rubric，并报告 weighted kappa/ICC、分歧和 adjudication。
GUI 人工 baseline 还要冻结操作者人数、经验、时间与尝试预算。若存在重复测量、另一 q window 或独立表征，仅作预先声明的外部一致性验证。真实
案例不能回流训练或阈值校准，也不得用主观“看起来很好”替代 exact residual。
这些结果只定位为 blinded case study：没有可追溯 ground truth 时不得报告 parameter accuracy、recovery
error 或“找回真实结构”的比例，也不得从少量案例外推总体性能。独立表征只报告预先声明的 concordance；
只有其测量链和不确定度足以定义 traceable ground truth 时，才可在另行声明的 endpoint 中讨论准确率。

### 计算与可复现性报告

每个表格行对应版本化 audit manifest，至少保存代码/data/model SHA256、simulation/codec/preprocessing
contract ID、随机 seed、Slurm job ID、CPU/GPU 型号与数量、软件/CUDA 环境、训练样本和 simulator 调用数、
总 GPU-hour/core-hour、峰值显存/内存、checkpoint 选择规则以及失败/重启 run。推断同时报告端到端 latency、
神经采样时间、exact-forward calls、refinement 时间和每个有效独特候选成本；给 median、p90 和冷/热启动
条件。主比较既按相同 forward-call budget，也按相同 wall-clock budget；不能用更多硬件隐藏算法成本。
若集群能提供计量，同时报告 kWh、计量方法与覆盖时段；碳排只作描述性 CO2e，并给出地区、时间、电力来源
和 emission-factor provenance。没有可靠能耗计量时明确记为 missing，不用 GPU TDP 乘时长冒充实测值。

Maxwell 正式任务不得直接从一个可变 working tree 运行。每轮先生成内容寻址、逐文件 SHA256 验证的源码
快照，上传到 `/data/dust/user/zhaiyufe/` 下新的版本目录，远端复核 archive 与 manifest 后只读解包；run
manifest 同时记录源码快照 SHA、环境和 Slurm job ID。快照必须包含当前 untracked 的 PosteriorV8 新源码，
因此不能使用只收集 Git index 的 archive。旧数据、模型、日志和源码目录一律不覆盖。

## 验收与论文实验

所有主指标使用 hard topology 和权威 exact forward，并按 K、shape、Resolution、q window、noise
和 weak-component stratum 分层。V8 扩容前的门禁为：

- physical/latent/GUI codec round-trip 误差 `<1e-6`；
- 训练 forward 与权威 exact forward 的 logRMSE `<1e-4`；
- train/inference preprocessing 数组一致；
- 用户范围与硬物理约束满足率 `100%`；
- paper model 必须在新 schema/version 下显式条件化全部 geometry 与 `BG/k/Int_i/int_Res` 用户范围，且
  proposal 与 exact solver 对同一 GUI amplitude polytope 的审计覆盖率为 `100%`；V5.2 模型图依赖审计与
  artifact identity 未冻结前禁止 paper training/holdout；
- K1 clean memorization：已给定 contextual branch 的 single-draw local RMS median `<0.05`，best-of-32
  median `<0.01`、p90 `<0.03`；exact post-refine raw logRMSE p90 `<1e-3` 且 compatible rate `>=99%`；
- K1 engineering：frozen-search-yield ranking recall@4 `>=95%`，exact-compatible candidate rate `>=95%`，
  reference recall@N16/B4096 `>=85%`；
- K1–K2 pilot：ranking recall@8 `>=90%`、exact-compatible candidate rate `>=90%`、reference
  recall@N16/B4096 `>=75%`，并报告主 log2-budget AUC；
- K1–K4 engineering：ranking recall@16 `>=80%`、exact-compatible candidate rate `>=80%`、reference
  recall@N16/B4096 `>=65%`，all34 plumbing 本身不允许效果 claim；
- noisy calibration/test：生成真值 compatibility 的目标覆盖率先在独立 calibration split 分层校准；
  阈值冻结后再在未使用的 test split 报告
  compatible-curve success，不能用 noise-free raw gate 代替；
- 建立饱和的 network-free reference bank 后，N=16、B=4096 的 reference-discovered compatible
  representative recall `>=80%`；log2-budget AUC 按冻结公式作为主 scalar；
- compatible candidate yield、unique-compatible-representative yield、保守 reference-match-fraction lower
  bound 和 unmatched-compatible 数量必须报告，不设置无法识别的 precision 门禁；
- 重复 parameter representative 比例 `<10%`；
- 有适用范围的独立不可行性 certificate 时才允许返回 `no_solution`；否则搜索耗尽必须返回
  `no_candidate_found_within_budget` 或 `no_compatible_representative_found_within_budget`，且绝不能输出越界候选。

论文至少比较：旧单解/固定多头回归、solver-only Sobol/global、retrieval-seeded solver、branch-conditioned
MDN-only、V5.2 search-yield-ranked MDN+exact refinement，以及再加 rescue 的完整方法；完整消融按上文冻结。
真实 Cut_Data 盲测保存曲线、残差、候选参数、运行时间和专家审查；无 traceable ground truth 时不报告
参数准确率，也不以单一 synthetic generating-parameter error 代替多解覆盖率。
