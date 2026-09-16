# GISAXS 多解反演科学与模型契约

- **Status**：Methods template — V5.2 geometry/amplitude-range-conditioned contract implemented；具体训练、模型、reference 与 holdout artifact 尚未冻结，本文件不构成外部 preregistration
- **Scope**：Sphere/Cylinder/Vertical Cylinder 混合曲线的多解定义、模型输出、物理验收与训练数据
- **Related code**：`src/gimap/features/fitting/domain/scattering_model.py`、
  `utils/ML_Fitting_1D_GISAXS/PosteriorV8/`；论文数值门禁的机器可读唯一版本位于
  `PosteriorV8/study_protocol.py`
- **Related tests**：`utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_*.py`
- **Related research**：`docs/research/multisolution-inversion-literature.md`、
  `docs/research/multisolution-inversion-paper-outline.md`
- **Last verified**：2026-09-06

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

为调优和论文预算时间轨迹提供的可选 `exact_forward_call_observer(index, phase)` 在预算准入后、
科学计算开始前调用；index 是整个 refinement batch 内从 1 起连续的已占用预算序号。观察器只能记录
审计数据，不参与候选选择或科学计算；未准入的调用不会发出事件。观察器异常以独立的
`V5ExactForwardObservationError` 中止整批执行，不允许被 optimizer 的数值恢复或单候选失败路径吞掉。
未设置观察器时保持原行为；固定和变化参数回归要求设置观察器前后的参数、幅度、正向数组及预算完全
一致。该接口本身不生成完整 tuning trace、不允许补造调用或开启正式训练授权。
`run_v5_universal_one_click_inference` 将相同观察器接入每次 refinement，以已消耗预算为偏移，
输出跨候选、跨分支连续的全局调用序号；零预算不产生事件，观察器失败不得被 rescue 路径恢复为
普通候选失败。该扩展不改变原有目标数量提前停止或搜索顺序，因此不能单独冒充必须完成固定预算的
论文调优 runner。
可选 `verified_report_observer(global_calls_used, report)` 只在 query-bound evaluator 真正产生
不可变报告之后、下一候选搜索之前调用，记录当时已验证的参数模式、代表解和累计预算。后续候选
不得被倒填进早期快照；零预算不产生报告事件。审计写入失败通过独立异常中止搜索，不生成缺失事件
的成功轨迹。该观察接口不改变排序、聚类、阈值、提前停止或返回结果，也不等于已经实现固定预算
调优 runner；模型绑定、实际输出事件序列与完整预算执行仍须由后续 adapter 严格实现。
可选 `evaluated_candidate_observer(global_calls_used, candidate, provenance, report)` 在新候选
完成权威评价后、report observer 之前接收真实 CandidateInput（含只读 float64 正演数组）、分支来源
和当时报告。它不把候选自动标为兼容，兼容状态仍由报告决定；持久化失败中止搜索。该接口用于
即时捕获无损 payload，禁止从最终报告补建早期候选。默认关闭时数值输出与调用账本不变。
`tuning_search_recorder_v5.py` 消费实际调用和候选事件，按当时 parameter modes 顺序捕获代表解；
它核对连续预算、候选/report/provenance 身份与单调时间，未完成真实预算时拒绝结束，不补造调用。
完整预算内无候选时显式记录空输出；记录器不加载模型、不授权训练，也不替代模型与协议绑定。
`tuning_checkpoint_model_v5.py` 对 retained checkpoint 先核验 0400/nlink1 和外部文件 SHA，
再经既有 V5 graph validator 加载模型、复用既有权重摘要算法核对实际权重，并比较加载前后完整文件身份。
此入口不授权梯度或科学晋级。
`universal_tuning_adapter_v5.py` 将实际 sealed checkpoint 与 materialized query 接入通用 trace runner。
独立协议 `checkpoint_universal_full_budget_snapshot_v1` 绑定全部预算、阈值与 inference version；
固定足够长的 Sobol horizon，并将 compatible target 设为最大预算 B+1，使候选达到目标的提前停止
在预算内不可达。它不修改 GUI 默认停止语义，不根据 reference 参数安排搜索，也不补造缺失调用；
种子耗尽或零调用失败导致短轨迹时仍拒绝完成。adapter 重算 checkpoint/source/seed/protocol binding，
核对 reference envelope 与实际 query context，加载实际 V5 模型，记录调用和候选出生快照，再核对
运行后文件、权重、query alignment 与精确调用账本。外层 producer 仍负责 worker placement、sealed
query publication replay、calibration、cohort isolation 与完整模型选择；该接口不授权梯度或论文晋级。
可选 `checkpoint_universal_calibrated_query_snapshot_v2` 接线要求 checked calibration 与逐 query
observation views 同时提供且精确覆盖 query 集合。它从真实 view 重建 exact observation，逐位绑定
曲线/acceptance sigma/context，再使用现有完整 acquisition-policy 校准查表；每条 observation、
校准来源和阈值进入协议摘要，实际搜索使用该 query 的 standardized 阈值，运行前后重新核对绑定。
可选 `calibration_path` 将实际封存文件接入该 adapter：严格要求 0400、单硬链接、无符号链接，
同时核验文件 SHA 与逻辑 artifact SHA，并在每次搜索前后复核完整文件身份；相同字节的文件替换也拒绝。
文件守卫版本进入协议摘要，未提供路径的内存接线不声称文件验证。外层仍须重放独立 cohort 与来源证明；
这些接线本身不授予正式训练或验收权限。
新 K1 source inventory 显式包含模型加载、事件记录、tuning runtime/trace/summary、代表解 history/store
和预算评价/选择模块；缺少其中任何文件都拒绝新的 source fingerprint。历史发布物仍按其封存清单与
完整 archive/tree 校验，不把后来扩充的 inventory 反向套用到旧数据。
`paper_representative_history_v5.py` 的独立 history/v1 契约记录可替换的有序代表解快照；按预算取
最后一个已发生的快照，而非历次代表解的并集。它绑定底层实际参数 emission/call ledger、query、
method、reference 与快照内容哈希，拒绝未来、未知、不兼容的代表解以及越界时间/预算；空快照表示
撤回全部代表解。history 不是旧 append-only trace 的子类，不能被旧写入器隐式降级。
history JSON 回放要求精确字段、自哈希和独立验证的底层 trace 全部身份一致；即使重新计算自哈希，
也不能替换 query、method、reference 或物理轨迹绑定。JSON 仅保存快照与绑定，不复制参数轨迹。
`paper_representative_history_store_v5.py` 复用独占只读 JSON 发布器，发布后读取同一文件回放；
读取要求外部 file/history SHA、0400/nlink1、无 symlink、稳定文件身份与唯一规范字节编码，
拒绝重复 JSON 键和 provisional trace 的零 artifact SHA。此 sidecar 不替代底层无损轨迹或
completion-last 标记，调用方仍须独立验证物理轨迹，并将 sidecar 纳入最终发布清单。
独立 snapshot-budget evaluation/v1 复用原一对一匹配、距离门禁、recall 和 AUC 实现，仅以预算时刻
快照替换追加式候选选择；首次可用时间来自真实非空快照，结果另绑定 history/reference/config SHA。
距离匹配只访问评估预算点实际可见且输出上限内的代表解，不读取隐藏 inventory member 来提高指标，
也不把参数较早生成的时间当作较晚实际展示的时间。
追加式数值入口保持原行为。预算 evaluator v7、tuning trace/runtime v2、checkpoint summary/selector v3
现在显式区分 append_only 与 budget_snapshot；选择策略进入 checkpoint-bound method v3 哈希，
所有 epoch/query 必须遵循同一冻结策略。快照 runner 不得缺省回退为追加式输出。
调优 runtime 将原始快照嵌入其封存 trace，作为该流程唯一快照来源（不另写重复 sidecar）；
summary 绑定完整 history SHA，读取器从无损 emission 重建 trace/history 并核对预算点代表解、首次可见时间
及完成时间。trace、summary、completion 均复用原子独占发布器封存为 0400/nlink1；读取器使用稳定文件
身份检查，拒绝可写、权限偏离或多硬链接文件，completion-last 不接受尚未封存的前置文件。
快照可以撤回或替换代表解，故其 recall 可下降，AUC 使用实际曲线；仅旧追加策略保留
recall 单调性检查。隐藏的 compatible inventory 不算首次可见输出。匹配公式、门禁及 B4096 不变。
本地替身 runner 的完整预算→持久化→读取→选择器链路已覆盖延迟显示、撤回和隐藏候选；
真实模型 live producer 和 Phase-C 回放仍未接通，不授权正式指标或训练。旧发布物不按新 schema 晋级。

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
闭区间物理端点经过 `log -> binary64 -> exp` 反变换时，精确编码的上下端点必须直接解码为原始物理
端点，不能保留 platform libm 向区间内侧或外侧产生的相邻 binary64 值；非端点值若仅因舍入落在区间外
不超过 8 ULP，也规范为相应端点，超过该 roundoff envelope 则失败关闭。该规则只处理可证明的端点
反变换舍入，不是把一般越界 proposal 裁剪回用户范围，并由 parameter-codec 版本和冻结
source/reference SHA 共同绑定。
component-slot canonicalization 保存的 truth 是所存 local coordinate 的精确 binary64 decode 代表；输入物理值
只在冻结 round-trip tolerance 内用于选择该代表，不能把相差一个 ULP 的 pre-encode 值另存为“真值”。

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
V5.2 的默认连续目标权重固定为 `local MDN NLL : mass-aware coverage : operational Top-4 alignment =
1 : 100 : 1`，并逐项进入训练配置与结果哈希。这个比例来自冻结 K1 checkpoint 上的非验收消融：旧
`1 : 1 : 1` 会通过放大 logistic-normal scale 降低 NLL，却不改善候选中心；纯 coverage 虽能改善中心，
却使 NLL 明显退化。`1 : 100 : 1` 同时改善 mixture-median RMS 与 NLL，因此作为下一版联合目标；它仍须
通过全新源码、数据与模型链的 Phase-A/Phase-B 门禁，诊断 checkpoint 本身不得晋级或充当正式模型。
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
全 K1 数据入口由 `k1_branch_forcing_v5.py` 将每个 branch-owned Sobol block 的 shape、D policy、
Resolution policy 与 singleton branch selector 固定到公开 bucket 中点；其余连续坐标逐位保持原始
Sobol 值，并同时绑定变换前后坐标 SHA。`k1_balanced_dataset_plan_v5.py` 为 train 与
`tuning_validation` 的 12 个分支分别建立独立 scramble，且与正式 Phase-C scramble 集合分离；该
write-free plan 不自行决定尚未冻结的数据规模或 seed，也不能替代生成后基于实际 recipe SHA 的
train/tuning/holdout disjointness receipt，更不能单独授权训练或验收。

实际 all-K1 grouped shard 由 `build_k1_balanced_grouped_shard_v5.py` 在 worker 上按一个 branch block
独占发布。`k1_forced_sobol_recipe_v5.py` 同时保存原始 Sobol 坐标、只改四个 categorical selectors
后的坐标、block/plan 身份和完整 direct-physics payload；构造时必须按 block scramble 与 local index
随机访问重放原始点，persisted audit 还会重新执行 forcing 与 physics 映射。每个 shard 只能含一个
split、一个 branch block 和一个 Sobol design，不能把不同分支拼成无法追责的父代集合。
`k1_forced_universal_query_v5.py` 是 balanced parent 进入 full-search 的 query bridge：先严格重放完整
persisted recipe，再从其 forced named-coordinate vector 生成 all-and-only K1 topology queries，并要求
generating topology 的 geometry、amplitude 与 singleton wire branch 逐字节回到 persisted physics。
该 query-set hash 同时绑定 recipe、plan、block、forcing、forced-coordinate bytes 和每个 topology query；
底层 named-coordinate adapter 不宣称 forced vector 是未经变换的原始 Sobol point，projection provenance
仍唯一归 forced-recipe contract 所有。query bridge 不读取 curve/model score，也不授权训练。
`k1_dataset_disjointness_v5.py` 从 0400/nlink1 checked artifacts 的小型 identity arrays 读取实际
recipe SHA 与 clean-group ID，不依赖仅有 seed 的推断；它要求 train、tuning-validation 与 Phase-C
holdout 在两类身份上全部两两不交叉，并分别给出 train/tuning claim 与 Phase-C exclusion claim。
该 receipt 只证明数据成员隔离，不能替代 full-search supervision、训练完成、Phase-C 评估或模型验收。

正式数据生产入口由 `k1_balanced_dataset_launch_plan_v5.py`、
`launch_k1_balanced_dataset_dag_v5.py`、`k1_balanced_dataset_worker_v5.py` 与
`k1_balanced_dataset_collector_v5.py` 共同约束。调用方必须显式给出 train/tuning 每分支父代数、两个
master scramble seed、每 shard 父代数和 observation view 集合；计划层不从 E1/Phase-C 的 13,824
holdout 规模反推训练规模。launcher 只在 `max-wgs` 创建新根并把 dataset array 与其 afterok collector
全部 held 提交，逐项 readback 后按 collector→dataset array 的逆序 release；worker 只在 Slurm CPU
节点生成 branch-pure shard，先封存 artifact 再写 0400/nlink1 completion。collector 必须收到全部
completion，重新从真实 checked artifact 读取 recipe SHA/clean-group ID 后才可发布 train–tuning
不相交 receipt，最后再写 collection completion。这个两总体 receipt 明确令
`phase_c_exclusion_proven=false`；只有 Phase-C 实际 clean-parent identities 也物化并由三总体 receipt
复核后，才可以填入训练 inventory 的 Phase-C exclusion 字段。

2026-09-06 获得自主冻结授权后，K1 E1 模型开发数据的唯一正式配置固定为：train 每分支
`1152` 个 clean parents（总计 `13,824`），`tuning_validation` 每分支 `288`（总计
`3,456`），augmentation views 为 `(0,1,2)`，每 shard `288` parents。该 train 数是独立的
capacity-bounded all12 训练总体选择，不复用也不标识 Phase-C 的同规模 holdout；调优总体为训练总体的
25%，两者仍必须用真实 recipe/group identities 与 Phase-C 做三方不相交复核。train 与 tuning master
scramble seed 分别为 SHA-256 ASCII 标签
`gisaxs.posterior_v8.k1.all12.train.v1`、
`gisaxs.posterior_v8.k1.all12.tuning_validation.v1` 的前四字节大端 uint32，即
`2844333258` 与 `3110101137`。`k1_balanced_dataset_plan_v5.py` 暴露单一 formal builder；生产
submit 会同时要求该 plan、三视图和 shard 大小完全匹配，否则在创建 Slurm 作业前 fail closed。首轮
`k1_balanced_all12_dataset_dag_v1` 因 authoring 端使用短包名、worker 使用仓库规范包名而在生成前
自哈希拒绝；第二轮 v2 纠正规范入口后，48 个 shard 完成，但 12 个含极端 coupled D–R–h query 的
shard 揭示 `BranchCodec.varying_mask` 以近 0/1 探针触及数值边界。v2 数据与全部失败证据保持历史
不可变。当前 codec 仍优先使用原近端点判定，只有该判定因耦合边界不可行时才改用 `0.25/0.75`
内部探针；内部探针若也不可行仍 fail closed。这个回退只判定 active axis 是否具有非零自由度，不改变
query、target、物理约束、exact forward 或验收阈值。数据 plan/recipe/shard/task/collector/launch 与
train–tuning receipt 同步提升为 v2；生产 submit 还要求规范 runtime package。下一次正式 run root
必须使用全新 `k1_balanced_all12_dataset_dag_v3`，日志前缀固定为
`k1-balanced-all12-v2-*`；该阶段仍只生产训练/调优数据，不能授权模型或 Phase-C 结论。

正式 Phase-C holdout 的身份生产与 train/tuning 数据生产保持独立。
`k1_phase_c_stress_v5.py` 先在 branch forcing 后按冻结的 5×5 range×observation cell 只改对应的
range-decision 坐标，并把 clean/noisy/masked/cropped/q-grid acquisition policy 在曲线生成前绑定；
shape、D、Resolution、continuous truth 与 amplitude composition 不得由观测结果反向选择。
`k1_phase_c_holdout_recipe_v5.py` 保存原始、branch-forced 与 stress-forced Sobol 坐标、完整 exact
physics、observation design、Phase-C plan/block 及全部 nested hashes；严格 decoder 必须重放每一层。
`k1_phase_c_holdout_shard_v5.py` 只发布 branch-pure recipe identities，不执行模型评估，也不产生科学
验收结论。

生产链由 `k1_phase_c_holdout_launch_plan_v5.py`、`launch_k1_phase_c_holdout_dag_v5.py`、worker 与
collector 共同约束：正式配置固定使用 Phase-C 的 12×1152 parents，并以每 shard 288 个形成 48 个
CPU worker tasks；array 与 afterok collector 必须全部 held 提交、readback 后按 collector→array 逆序
release。worker 逐任务复核只读 source/archive、Phase-C plan 和 train/tuning receipt，先封存 0400/nlink1
recipe shard，再写 completion。collector 必须交叉核对每个 shard 的 block、branch、Sobol window、
selection、nested hashes 和 completion-last，复核每分支全部 25 stress cells 后，才可从实际 recipe SHA
与 clean-group IDs 发布 train/tuning/Phase-C 三方不相交 receipt。该 receipt 只令
`phase_c_exclusion_proven=true`；`training_authorization_granted` 仍为 false，直至 tuning exact-budget
summary、正式训练授权和 full-search-supervision 契约另行通过。holdout 身份生产本身不是 Phase-C 模型
评估，也不授权 K1 generalization claim。

`k1_training_identity_runtime_v5.py` 是上述四份不可变发布物之间的唯一 identity handoff：它同时重放
balanced dataset completion、train/tuning receipt、Phase-C holdout completion 和三总体 receipt，要求
两个 completion 内声明的 receipt 路径就是实际读取的 0400/nlink1 文件，并逐项核对 file/self/claim
SHA、总体 artifact/manifest inventories、recipe/group set hashes、parent counts、输入 pre/post identity
与 completion-last。生成的 `k1_training_identity_authorization/v1` 随训练 inventory v2 原样嵌入；缺少该
对象时 inventory 必须令 Phase-C exclusion 为 false，不能再凭一个裸 receipt hash 声称隔离成立。该
authorization 只证明三个实际总体的成员身份互斥，固定声明
`full_search_supervision_complete=false`、`tuning_exact_budget_summary_complete=false` 和
`training_authorization_granted=false`，因此本身绝不开放梯度训练。

`k1_balanced_full_search_plan_v5.py` 将 identity handoff 具体展开为 60 个不可变 parent-shard 搜索
任务：train 固定为 48 shards、`tuning_validation` 固定为 12 shards，每个 shard 288 parents，并逐分支
核对 4:1 的 shard balance、13,824/3,456 的总体计数、artifact/manifest inventory、task completion-last
身份和 0400/nlink1 声明。plan 同时绑定 all-and-only K1 的 formal calibrated stage、4096 calls/branch
预算、固定 observation policy、候选 views `(0,1)` 以及全新且互异的 task output roots；任何 parent、stage
或授权总体替换都会破坏自哈希或 replay。该 plan 只允许 worker 执行 model-free full search，初始固定
`full_search_supervision_complete=false`、`gradient_training_authorized=false` 和
`tuning_exact_budget_summary_complete=false`；只有 60 个搜索结果与 completion 全部由 collector 重放、
调优 exact-budget summary 通过并产生新的正式授权后，训练入口才可开放。
`k1_balanced_full_search_plan_runtime_v5.py` 是实际发布物到该科学 plan 的唯一 authoring bridge。它从磁盘
重读并严格固定 balanced dataset launch plan、collection completion、train/tuning receipt、Phase-C
completion 和三总体 receipt 的显式 file/self/claim SHA，重放原 dataset source/authoring identity，再逐一
核验 60 个 task completion 的自哈希、输入 pre/post、selection、artifact path/hash/size、`0400/nlink1`
和 collector 中的有序 completion inventory；只有这些身份全部一致才构造 60 个 parent bindings。调用方另行
提供的新 search source SHA 只决定本次执行代码身份，不能替代或弱化旧 dataset publication 的谱系。
authoring worker 还读取每个 sealed grouped artifact，核对实际 file/manifest/byte receipt，解析完整
acquisition policy 并要求噪声算法版本与当前重建一致；不兼容时在生成批量计划前拒绝。该预检不替代
下游完整 policy、逐数组投影或校准身份检查，也不要求历史 source 与当前源码逐文件相同。
balanced dataset、Phase-C holdout、IID calibration 与 full-search 的 collector held-readback 必须存在唯一
`Dependency` 字段，且完整值只能是指定 array 的 `afterok`（允许 Slurm 的 `_*` 整数组表示和
`(unfulfilled)` 状态）。不得用 raw 文本中的作业号子串代替检查；单 task、其他 dependency 类型、
额外 AND/OR 条件或重复字段均拒绝释放，保留已提交作业及失败审计。
带噪声版本预检的 plan authoring execution version 为 v6；full-search submission 的
held receipt/launch completion schema、审计文件和日志前缀为 v3。科学搜索 plan 与预算定义不因
提交器修复而改变；新执行必须使用全新 run root，旧版本证据保留而不改写。
balanced dataset、Phase-C holdout 与 IID calibration 的提交入口允许精确匹配的
`max-wgs[0-9]*` 或 `max-fs-display[0-9]*` 登录主机，不接受相似前缀的其他主机，也不允许在
Slurm allocation 内再次提交。对应 generation/collection Python 入口和 sbatch wrapper 同时禁止
两类登录节点执行计算；伪造 Slurm 环境变量不能解除主机禁令。该变化只扩展已授权的提交位置，
不改变 worker 资源、观测生成或验收科学定义。
该提交修复的 balanced/IID 作业与日志前缀为 v3，holdout 为 v2；未改变的数据与计划
持久化格式保留原 schema，执行源码由新 content-addressed snapshot 绑定，输出使用全新 run root。
每个任务的搜索父代由 `k1_balanced_full_search_parent_v5.py` 从原始 0400 shard 内的 canonical forced
recipe 重建，而不是再次按 seed 采样；完整 decoder 同时恢复 physics 与 grid，并再次执行 forcing、nested
hash、范围、物理和幅度约束。随后只依据 `recipe_seed` 与候选 views `(0,1)` 运行冻结的曲线盲 policy，
为每个 clean parent 选择唯一含 measurement sigma 的 observation。重建前先用 recipe seed/view index
重放 acquisition policy 身份；噪声算法版本也属于该身份，历史 v1 与 overflow-safe v2 不可混用。
身份不匹配时保留原始发布物并明确拒绝，不通过改写 ID、放宽数组容差或套用不同 policy 的校准继续。
投影后的 clean/candidate 行必须与原
shard 逐值一致，选中 observation 的 audit、acquisition policy 和全部 model inputs 也必须是原三视图
shard 的精确子集；它不能改变 recipe 或制造新的观测。原始 shard 继续作为总体身份来源，投影 parent
另行绑定 source artifact、recipe/query-set/selection hashes，供 sidecar 与后续训练证据使用。
`k1_balanced_full_search_authorization_v5.py` 再把原始 shard selection、该投影、逐 recipe/query/观测
选择成员和全局 60-task plan 组合为唯一 task-plan SHA，并据此派生现有 formal search evidence 所需的
逐 shard role authorization。train 与 tuning 的 consumer role 仍严格隔离；授权对象只允许对应 worker
执行搜索，固定声明 search evidence 尚未生成且 gradient training 未授权。只有实际 parent/sidecar、
全部 branch executor evidence 和精确预算由既有 receipt 审计重放后，该 shard 才能成为训练或调优标签。
`k1_balanced_full_search_runtime_v5.py` 将已授权投影适配成 frozen-search pipeline 的只读 executable
task；query catalog 使用 forced recipe 重放得到的 all-K1 query set，父代发布仍逐数组任务使用全新输出
路径。底层 pipeline 对旧 Sobol formal shard 与新 forced balanced shard 共用同一 executor、sidecar 和
task-bound receipt 实现，从而不产生第二套 exact-search、预算或证据语义。
`k1_balanced_full_search_worker_v5.py` 是这一适配器唯一允许的批量执行入口：非 dry-run 必须处于 Slurm
array worker，登录节点直接拒绝；worker 在任何输出前复核 plan、source archive/snapshot、search schedule、
calibration、原始 grouped shard 与其 completion-last，并在搜索后逐项重算同一输入身份。每个新输出根先
写入 task authorization，再调用共用 frozen-search pipeline；forced query catalog 使用独立 artifact namespace，
正式 sidecar 使用 training-supervision scope，不能冒充旧 contract smoke。pipeline 全部文件先封为
`0400`、子目录封为 `0500`，最后才写本任务 `0400/nlink1` completion 并封闭输出根。该 completion 仍明确
`full_search_supervision_complete=false` 和 `gradient_training_authorized=false`；60 个任务的 collector、
tuning exact-budget summary 与新的正式训练 authorization 缺一不可。
`k1_balanced_full_search_collector_v5.py` 在独立 afterok worker 上逐任务重放 plan-derived authorization、
projected parent、training sidecar、task-bound receipt 和每个 executor evidence，并在发布前完整执行第二遍
重放以排除汇总期间的替换。它严格核对 48 train + 12 tuning shards、13,824/3,456 clean parents 以及每条
formal authorization 的精确 calls 总账；先封存带逐 shard 路径与 hash 的 inventory，最后才发布 collection
completion。该 completion 只把 `full_search_supervision_complete` 提升为 true，仍令 tuning summary、梯度
训练与模型验收为 false；任何中途失败只写全新 0400 failure audit，绝不复用或覆盖已有输出。
`launch_k1_balanced_full_search_dag_v5.py` 是这 60 个任务的唯一生产提交事务：dry-run 会重放只读 source
snapshot/archive、local-Sobol schedule 和 calibration 的语义与文件身份而不写文件；submit 只允许在
Maxwell 的 `max-wgs*` 或 Photon Science `max-fs-display*` 提交节点执行，并把全局 plan 独占发布为
`0400/nlink1`。这一入口扩展只改变登录/提交主机，不改变 worker、数据、搜索或科学门禁；其 launch
receipt/completion、job/log prefix 与 evidence filename 均提升为 v2。array 与 afterok collector 必须全部用
user hold 提交，逐项回读 `JobHeldUser` 和 collector 的精确 array job dependency 后，先封存 held receipt，
再按 collector→array 逆序 release，最后写 launch completion。任一提交、回读或 release 步骤失败只新增
failure audit，已提交作业保持原状且绝不自动取消；plan、receipt 与 completion 分别绑定科学自哈希、实际
文件 SHA/权限/链接数和提交前后外部输入身份。

计划编制本身会重放 60 份 dataset publication，因此也必须运行在 Slurm worker，不能在 `max-wgs*` 或
`max-fs-display*` 登录/提交节点执行。`author_k1_balanced_full_search_plan_v5.py` 与独立 CPU wrapper
固定 K1 的 `N=16` direct-scout、每 seed 最多 `256` 次 exact-forward、每 branch 总预算 `B=4096`，
在 worker 上连续两次重放全部 publication；只有输入身份前后不变且两次 plan 完全一致时，才在独立
authoring root 中先封存 `0400/nlink1` candidate plan、最后封存 completion。未来 production search root
此时必须尚不存在，真正 launcher 只读取 candidate，轻量重放 source/schedule/calibration 后再以 `O_EXCL`
发布同一 plan。search array 与 collector wrapper 也显式拒绝两类登录节点，防止伪造 Slurm 环境后误跑。
search worker 与 collector 的 Python 入口执行相同主机禁令，不依赖调用者经过 sbatch wrapper。
`k1_balanced_full_search_replay_v5.py` 在 worker 上只读回放已完成集合：校验调用者绑定的 plan、
inventory/completion 文件 SHA、0400/nlink1、源输入身份、全部 60 项 task-bound 证据及总数，
再次回放任务和输入以拒绝运行中漂移。它复用 collector 的逐任务规则，不重写输出，不授予梯度权限；
训练端还必须根据回放后的实际 parent/sidecar 与三方隔离身份建立训练清单。
跨版本消费搜索集合时，可显式传入原 balanced dataset launch plan 路径；消费端先核对搜索计划绑定的
该计划 file/self SHA、实际发布路径及 source root/archive/bundle 身份，再复用历史数据发布回放，
校验生产时逐文件清单与完整只读 archive/tree。旧 bundle 是已绑定的生产声明，不以消费端新增模块
重新计算或改写；前后回放与 task/source 输入身份仍须完全一致。新搜索 worker 的默认源码检查
仍要求当前完整清单，不提供跳过源码校验的参数。此只读兼容入口不授予梯度或科学验收。
K1 training job staging v3 将作业独立暂存和运行缓存限定在用户指定的 dust 根内；
不再接受 Slurm/系统临时目录回退。seed/collector wrapper 显式绑定 scratch base，并为每个作业
创建新的 owner-private 随机目录；原有 one-shot mint、输入只读身份和前后重放规则不变。
该存储迁移不启用尚未完成的正式训练授权，旧发布物保持原样。
这两个 wrapper 的 bootstrap、归档验证和训练/收集入口统一使用指定的 GISAXS R2 Python，
不再通过旧 `tf` 环境或 PATH 选择解释器；解释器不存在时在复制输入前失败。
`k1_search_training_inputs_v5.py` 复用此集合回放和既有 search evidence reader，逐分片核验
parent/sidecar/receipt 的文件身份、角色、manifest、实际 recipe 分支计数与 clean-group 集合，
拒绝重复及 train/tuning 交集。返回的 typed artifact 绑定 receipt **文件** SHA，不混用逻辑自哈希；
前后重新回放整个集合。此只读准备入口仍不授予 Phase-C 隔离、梯度或科学验收权限。
提供四份原始隔离 publication 的精确路径时，该入口还会在处理所有分片前后重放
balanced completion、train/tuning receipt、Phase-C completion 和三方 receipt；逐文件核验
冻结 file SHA、0400/nlink1 与文件身份，重新签发的身份必须与搜索计划中的原始授权完全一致。
即使字节相同，期间替换文件导致 inode 等身份变化也拒绝。未提供这些路径不冒充已完成此回放。
仅四份原始发布物的前后回放成功后，输入准备入口才返回由实际 artifact descriptors 构建的
v3 training inventory；其 source 仍绑定该搜索集合的冻结 source，原始 receipt 身份和 projection
证据全部保留。未提供原始 publication 路径时 inventory 为 null；此返回值不写文件或授予梯度权限。
同模块的 worker-only publication CLI 将成功准备的清单发布到全新目录：逐项显式绑定搜索输入及
四份原始 publication 路径，严格回放后排他写入 training inventory v3，fsync 并封存为 0400/nlink1，
再次检查原始发布物和清单身份后最后封存 v1 completion。失败残留目录不得复用；completion 缺失
即不构成完成发布。完成凭据保留真实文件身份、投影证据 SHA 与 worker 来源，不授予梯度或验收权限。
K1 training input inventory v3 可同时绑定原始 identity authorization 和 typed projection evidence：
逐角色核对原始 artifact/manifest、clean-parent 总数及 group-set SHA，并逐项匹配搜索后文件的
路径、artifact/manifest 和 parent 数量。消费代码的 source 身份与原始生产 source 分开保存，
不得通过改写原授权的哈希来适配搜索后文件。带 projection 的清单显式要求 live collection replay，
在正式运行适配器完成该回放前，Phase-C 隔离声明为 false，正式训练继续 fail closed。
准备结果另携带原始分片到投影分片的逐 task 来源映射：两端实际 recipe、pattern、split、clean-group
数组必须按顺序与 dtype 完全一致，两端文件与 manifest 身份分别绑定且 pre/post 不变。
原始隔离授权中的 artifact 哈希不会被投影哈希覆盖；来源映射本身也不能冒充新的隔离授权。
调用者提供原始 typed identity authorization 时，还必须与封存搜索计划中的 authorization SHA 一致；
逐 role 核对原始 artifact/manifest 清单、数量与投影后的实际 group-set SHA，拒绝换源或总体漂移。
这只是后续版本化训练授权的输入校验，不会直接放行梯度。
来源映射通过 `k1_search_projection_evidence_v5.py` 的 v1 strict typed evidence 返回，绑定搜索 plan、
inventory/completion 文件 SHA 和可选原始授权 SHA；独立自哈希、精确字段、任务顺序、双总体和路径唯一性
都必须重放，禁止在反序列化时抬升梯度、隔离或科学验收声明。尚不替代训练 inventory 的原始隔离 gate。
后续 K1 training seed 与 tuning-handoff collector 的 Python 执行入口和 sbatch wrapper 同样拒绝
`max-wgs*`、`max-fs-display*`；非 dry-run 在读取计划或创建 staging 前检查，即使存在 Slurm job ID
也不能绕过登录节点限制。该防护不改变模型、训练目标或验收定义。
底层 grouped trainer、其独立 CLI（包括会读取训练数据的 dry-run）和 GPU wrapper 也拒绝两类
登录节点；直接调用 trainer 不能绕过 K1 wrapper 的主机保护，且在解析训练文件前失败。
首次冻结的 plan-author v1 作业因 CLI 路径参数名未映射到 dataclass 的 `*_path` 字段而在任何 publication
重放或输出写入前 fail closed；该作业、authoring root 与日志只作历史证据。v2 显式冻结每个 CLI option
到 config 字段的映射，并以 parser-level 回归覆盖全部十二个路径参数；后续 authoring 与 search 均使用新根。
v2 worker 随后在首次路径检查发现契约的 `PurePosixPath` 根不能直接执行文件系统 `resolve`；
v3 在 authoring 与 publication replay 的公共入口及 I/O 边界显式转换为 `Path`，保证下游收到具体
文件系统路径；保留原有严格存在性和根目录包含检查，回归覆盖两种根类型、只读排他发布、默认根向
下游传递以及 CLI 到 config 的实际入口。v2 作业与根保持不变；此修复不改变科学定义或计划载荷 schema。
历史 balanced dataset 的 source replay 使用其封存 launch plan 中的逐文件清单，而不是消费端后来
扩展的 training source inventory；新计划仍必须满足当前完整清单。逐文件内容和权限、固定 archive SHA、
完整 archive/tree manifest 与目录集合继续严格核验。历史 bundle SHA 是已绑定计划的发布声明，不冒充
按当前清单重算的 bundle；跨阶段调用方必须先匹配外部预期 plan file/self SHA。此兼容修复不改变数据、
科学门禁或持久化 schema。回归覆盖消费端新增必需模块后旧发布仍可回放、新计划仍拒绝缺失模块、
历史文件篡改仍拒绝。失败的 authoring v4 根与日志保持不变，新部署仍需重新冻结源码并使用新根。

K1 seed binding 与 tuning handoff 的载荷升至 v3：训练 worker 在本次新输出完整性校验后，
先检查整棵输出树的 owner、文件类型与链接数，拒绝 symlink、hardlink 和特殊文件，再将产物封存为
`0400/nlink1`，重放绑定的 result、model 与全部 checkpoint，最后排他写入只读 seed completion。
collector 要求同样的精确权限与实际文件 SHA，并在 handoff 发布前复查 seed completion 字节。
调优 checkpoint reader 从已绑定的训练 result 清单读取完整 full-epoch 顺序，拒绝漏 epoch、越界路径、
warmup-only 或已冒充论文选中的状态；随后实际加载 V5 图、核对权重 SHA 与文件 pre/post 身份。
这些交接验证不授权梯度或科学验收，不改变训练目标、参数可行域或 checkpoint 选择规则。
`tuning_reference_executor_v5.py` 从外部 file SHA 绑定的 `0400/nlink1` executor artifact 提取全部
typed reference 参数。提取前必须调用现有 task-bound executor reader 重放全部候选的权威曲线、
指标、约束和分簇，提取后复查文件身份；完成的负分支返回空集合，不伪造真值参考。
该转换保留原始代表及其 branch/query/source 身份，不重新选取或合并；调用方仍须验证完整调优总体、
全部合法分支和正式校准协议，单分支转换成功不能代替完整 reference-set 或论文验收。
同模块的 query reader 要求原始 universal context 的全部 branch index 恰好各出现一次，
逐项核对 observation、query catalog、protocol 与 calibrated threshold 身份，并跨整个回放周期
检查所有封存输入的 pre/post 文件身份。负分支仍完整回放；若整个 query 没有兼容代表，必须显式失败，
不能从调优总体静默删除。该入口不发布 reference bank，也不代替总体覆盖或正式训练授权。
其独立 publication 入口使用 `tuning_query_executor_reference/v1` 封存单查询参考，
只允许 worker 在新的 dust 文件路径排他发布。文件包含全部 branch task/file 身份、原始物理代表、
comparison protocol 和显式 false 的训练/科学授权标志；发布后的实际字节 SHA 作为
`V5FrozenReferenceSet.reference_set_sha256`，query/pairing 身份取自 observation/clean group。
读取时从外部绑定的全部 executor 文件重新执行 task-bound 回放，并逐字节重建参考文件，
最后返回现有调优运行器所需的 `V5FrozenReferenceSet`，不从文件中的参数声明直接授信。
这不替代完整 tuning cohort、正式校准协议或原始 train/tuning/holdout 三方隔离验证。

K1 compatibility threshold 使用完全独立的 IID/exchangeable 生产链，不能从上述 Sobol train、tuning 或
Phase-C 总体拟合。正式规模冻结为 60 个 acquisition-only strata 各 `2,667` 个独立 clean parents，总计
`160,020`；这既超过预注册的 160k 目标，又让各层严格等量。每个样本先从 12 个合法 K1 branch 作独立
均匀离散抽取，再在 domain-separated pseudorandom seed 下只按该 branch 的物理可行性和预先指定 stratum
是否存在真实 measurement sigma 作拒绝抽样；拒绝逻辑不得读取曲线、模型或 score。master seed 固定为
`14483825437891294498`，其 ASCII 标签派生 SHA-256 为
`c900e27f4646f52228feac313658ab0480340c2fbfb94593be1e00576d556d68`。每个 clean parent 只贡献一个
确定 observation；split-conformal target coverage 保持 `0.95`，每层最低样本要求保持 `200`。

`k1_iid_calibration_plan_v5.py`、`k1_iid_calibration_shard_v5.py` 与
`k1_iid_calibration_launch_plan_v5.py` 固定上述总体、逐样本身份和 60-task array。worker 只在 Slurm CPU
节点生成 standardized truth-vs-noisy score shard，先将 shard 封为 `0400/nlink1`，最后写 task completion；
每条记录可从 master seed、stratum、sample ordinal、branch、recipe seed、view 与完整 acquisition policy
严格重放。`k1_iid_calibration_collector_v5.py` 必须重读全部 60 份 completion-last 证据，核对总数、每层
计数、全局 duplicate-free 与 12-branch support，才能拟合并封存 calibration artifact。collector 同时把
实际 calibration recipe/group identities 与既有 train/tuning/Phase-C 三方 receipt 比较，六项交集必须
全部为零，并发布紧凑的四方 disjointness receipt；该 receipt 绑定 60 份 shard 的 file/self/manifest SHA，
不复制一份可变的 160k identity 清单。只有 calibration artifact、四方 receipt 与输入 pre/post identity
全部复放后，collection completion 才可令 `compatibility_threshold_authorization_granted=true`；训练、模型
选择与 Phase-C 验收仍保持 false。`launch_k1_iid_calibration_dag_v5.py` 沿用 held-readback/reverse-release
事务，任何失败均保留新根证据且不自动取消已提交作业。

首个 v1 生产数组揭示了合法极小强度下的不确定度数值边界：直接计算
`sqrt(sigma_poisson**2 + sigma_relative**2 + sigma_floor**2)` 会让三个都有限且非负的 binary64 项在
平方时下溢为零，进而把 measurement sigma 误判成非正。v2 保持同一个 root-sum-square 科学定义，但用
逐项 `hypot` 的尺度归一实现，避免平方的下溢/上溢；noise application、calibration plan/shard/task/
collection/receipt/launch schema、文件名和日志前缀均随之版本化。v1 根、58 个失败 task 的日志、两个成功
shard 和被依赖系统取消的 collector 只作不可变失败证据，任何 v2 产物不得写入或复用该根。

balanced all12 数据以新版 noise policy 重建时，提交器及 wrapper 日志前缀使用
`k1-balanced-all12-v4`，必须使用全新发布根。旧 observation ID 不得改名冒充新版 policy；
数据与收据格式没有变化，因此保留其 schema，输入 source/plan 哈希绑定实际新发布。

Phase-A 保留用户把全部活跃参数固定住的合法查询。此类目标没有连续自由度，因此不进入 local-MDN
密度损失或 mixture-median RMS；也不能以零误差样本稀释 memorization 指标。gate 必须分别记录
known-truth 总数、至少一个可学习目标的数量、全固定目标数量和实际 varying coordinate 数。全固定目标仍按
canonical local `0.5` 保存，并由物理端点精确解码与 authoritative forward 验证，而不是从数据集中删除。
Phase-A 的可学习目标按固定顺序、固定大小的循环 mini-batch 优化，初始与最终指标则用同样的有界 batch
遍历全部可学习目标并按目标数聚合。这样在 16 GiB GPU 上不会把 `recipes × q-points × width` 的全部中间
激活同时驻留显存；batch size 属于结果哈希覆盖的配置，不能在运行期间隐式改变。
Phase-A 的冻结优化预算为 18,000 个 Adam update，初始学习率 `3e-3`，按 update 数作确定性 cosine decay
到 `3e-5`；初始值、末值、schedule 名称、update 数与 batch size 都进入结果哈希。该选择来自同一冻结
512-recipe 数据上的非验收从零训练对照：常数 `3e-3` 在 18,000 updates 的 median local RMS 仍约为
`1.01e-2`，cosine 到 `1e-4` 为 `7.33e-4`，cosine 到 `3e-5` 为 `1.56e-4`，同时 NLL 改善而不是靠放大
scale 取巧。诊断结果只用于冻结优化配置，正式结论仍必须由全新 artifact chain 复现。v14 的
18,000-step 正式复演被调度到 P100，并在科学结果发布前达到两小时 Slurm 时限；它只能作为完整保留的
资源失败证据。v15 不改变模型、数据、目标函数或训练预算，只把该 GPU gate 固定到已验证的 H200
capacity envelope，并把时限提高到四小时，避免硬件代际差异被误判为科学失败。v15 的 Phase-A
科学门禁随后完整通过，但 Phase-B 的历史证据复核错误地要求 Python 风格的 `0o400` 字符串，而
Phase-A capability 的唯一生产编码一直是零填充 POSIX 形式 `0400`；v16 只把消费端与既有生产契约
对齐并新增契约回归，不改变模型、数据、目标函数、训练预算或权限要求。v16 Phase-A 随后科学通过，
但 Phase-B worker 的 source-ownership 检查错误地把有序 source inventory 的第一项（launcher）当成
当前 worker 文件；v17 改为核对 inventory 中显式命名的 worker 相对路径，并为 Phase-B 使用新的
v4 发布根和日志前缀。该修复只纠正执行身份检查，不改变模型、数据、科学指标或门禁。
v17 的 Phase-A 科学门禁通过，但 Phase-B v4 随后揭示 `st_dev` 被错误地当作跨节点稳定的 Lustre
文件身份：同一只读 plan 在登录节点与 worker 上具有相同 SHA、size、inode、mode、mtime、ctime 和
nlink，却因客户端 mount namespace 不同而报告不同 device number。v18 只把 plan 的跨节点绑定限定为
这些稳定字段，同时在不可变 launcher receipt 中保留登录节点 device；Phase-B 使用新的 v5 发布根、
契约版本和日志前缀。科学模型、数据、训练、推理、指标和门禁均不变。
v18 Phase-A 科学门禁通过，但 Phase-B v5 随后证明同一挂载局部字段也残留在 pinned wrapper、receipt
和 launch-completion 的跨节点比较中，因而在正式科学计算前 fail closed。v19 将所有跨节点审计文件
身份统一规范化为 path、SHA、byte count、mode、inode、mtime、ctime 与 nlink；device 仍由每个节点的
严格本地 fd/path 检查验证，但不写入跨节点 canonical evidence。Phase-B 使用新的 v6 发布根、契约与
日志前缀；科学模型、数据、训练、推理、指标和门禁仍完全不变。
v19 Phase-A 科学门禁通过，Phase-B v6 engineering smoke 也完成，但 formal worker 复核上游 smoke
completion 时发现同一个只读 result 在两个 worker mount namespace 中的 `st_dev` 不同；v19 的统一
规则漏掉了这条跨阶段 result identity。v20 将 Phase-B result completion 中的文件身份也规范化为同一组
跨节点稳定字段，device 继续只在产生或消费它的节点内通过 fd/path 与 pre/post stat 严格核验；Phase-B
使用新的 v7 发布根、契约、completion schema 与日志前缀。科学模型、数据、训练、推理、指标和门禁不变。
v20 Phase-A 科学门禁通过，但 Phase-B v7 的 parent replay 在 AMD 数据生成节点与 Intel evaluation worker
之间出现由平台 `libm` 引起的 binary64 ULP 漂移；固定 seed 的历史采样器因此不能充当跨 CPU 字节身份。
v21 将由不可变 grouped-dataset SHA 与 manifest 绑定的 canonical recipe 作为跨 worker 权威输入，严格重验
顶层和嵌套字段集合、canonical JSON、SHA、seed lineage、query/amplitude contract、branch presence、物理范围
与 float32 local-target wire；持久化 float64 truth 继续用于 authoritative exact-forward。seeded replay 仍保留为
生成节点内部审计，但不再作为跨 CPU 门禁。Phase-B 使用新的 v8 发布根及 result/completion/launch schema；
科学模型、训练、候选搜索、exact-forward 指标与验收阈值均不变。
v21 的 engineering smoke 通过，但 formal 覆盖到一个合法的 query-fixed coupled width：历史持久化真值与
当前节点重算出的固定端点只差若干 binary64 ULP，旧 decoder 却把它交给连续区间 encoder 而误拒绝。
v22 对没有连续自由度的坐标先要求持久化真值与 authoritative decoded endpoint 在声明的 float32 target wire
精度下完全一致，再仅用 decoded endpoint 完成 codec round-trip；所有 varying 坐标仍使用持久化真值通过完整
encode 校验，超出 wire 等价范围的 fixed truth 仍 fail closed。Phase-B 发布根提升为 v9，并同步提升
result/completion/launch schema；模型、数据、训练、搜索、exact-forward 指标与阈值均不变。
v22 Phase-A 与 Phase-B engineering smoke 均通过，但 formal 首次覆盖到一个所有 active 坐标都由 query
固定的合法 parent；旧 evaluator 错误地要求每个 parent 至少有一个连续自由度，因而在科学循环前 fail closed。
v23 保留 fully-fixed parent 并继续执行完整候选、refinement 与 exact-forward 审计，但将 local-MDN RMS 的
统计总体严格限定为至少有一个 varying coordinate 的 learnable parent；fully-fixed parent 的 local RMS 必须
为空，不能以零误差稀释门禁。正式结果同时核对 clean、learnable、fully-fixed、varying-coordinate、local
metric 与 exact-forward 的分母，且至少要求一个 learnable parent；exact-forward 指标仍覆盖全部 clean
parent。Phase-B 发布根提升为 v10，并同步提升 result/completion/launch schema；科学模型、数据、训练、搜索、
exact-forward 指标与既有阈值不变。
v23 Phase-A 与 Phase-B engineering smoke 均通过；formal 完整遍历 512 个 parent 后，只在一个合法的
极小强度样本上失败。该样本的有效粒子幅度约为 `5.94e-302`，逐点相对误差权重使加权设计矩阵元素仍为
有限 binary64，但旧列二范数通过直接平方求和而溢出；同时远离观测尺度的通用多面体 witness 在缩放坐标中
也会溢出。v24 用最大元素归一化后再求 L2 范数，并仅在通用 witness 不可表示时使用非负最小二乘得到的、
经原始 coupled GUI polytope 严格复核的数据尺度初值。这只是同一加权最小二乘问题的数值稳定表示，不改变
正向公式、可行域、候选、预算或验收阈值。Phase-B 发布根提升为 v11，并同步提升 solver、result、completion
与 launch schema；旧失败链和诊断保持不可变。
v24 Phase-A 与 Phase-B engineering smoke 均通过；formal 完整遍历 512 个 parent 后，local-MDN、完整性、
物理、范围、幅度与预算门禁全部通过，但受约束幅度求解器在 25 个 fully-fixed parent 和若干 learnable
parent 上从远离数据尺度的 contract midpoint 启动，SLSQP 错误报告约束不相容并回退到该 midpoint。v25
先求非负正交域上的全局最小二乘解；只有该解再经完整 coupled coefficient polytope 严格验证时才直接接受。
由于正式可行域是非负正交域的子集，落在该子集内的正交域全局最优也必然是正式问题的全局最优；否则仍走
既有受约束路径并 fail closed。full-512 非验收 worker replay 在不改变模型、数据、候选、正向公式、预算或
门禁的前提下得到 exact-compatible rate 1.0、exact natural-log RMSE p90 `2.37e-10`，且 16,384 次
refinement 全部成功。Phase-B 发布根提升为 v12，并同步提升 solver、result、completion 与 launch schema；
旧 v11 formal failure 及 r8-r10 诊断保持不可变。
v25 的 Maxwell regression 在生产 SciPy 上揭示两个只影响数值契约回归的问题：同一个近零强度 NNLS
证书在不同 SciPy 版本上的绝对残差都很小，但旧测试错误要求逐点 `2e-10` 相对一致；另一个边界最优在
闭区间上端外侧相差一个 binary64 ULP，幅度 polish 的外盒预检却使用严格比较，和既有 polytope 容差契约
不一致。v26 将前一回归改为仍远严于正式门禁、但跨生产 SciPy 稳定的 `5e-7` 相对检查；对后一情形只把
处于既有 `AmplitudeBounds.contains` 容差内的上端舍入值夹回闭端点。非零下端越界及任何实质上端越界仍
fail closed，coupled polytope 和最终 GUI audit 仍必须全部通过。Phase-B 发布根提升为 v13，并同步提升
result/completion/launch schema；v25 根、日志及取消链保持不可变。
v26 Phase-A 与单分支 Phase-B 均通过后，balanced all-K1 数据生产首次覆盖到若干合法的 coupled
`D`/cylinder query；旧 `varying_mask` 只用接近闭区间边界的探针判断连续自由度，而这些探针可能在
hard-core spacing 的三角可行域外，导致数据 worker 在生成物理样本前 fail closed。v27 先保留原边界探针；
仅当它因物理不可行而失败时，才用固定的 `0.25/0.75` 内部支持探针重试，内部探针仍失败则继续 fail
closed。该判断只识别既有 query 可行域内是否存在连续自由度，不改变 query、物理约束、target、模型、
exact-forward 或任何验收阈值。非验收 worker r15 已遍历冻结的 17,280 个 all-K1 train/tuning parent：
12 分支各 1,440，88 个轴使用安全 fallback，零 recipe/mask 失败，且全部输入 pre/post SHA 一致。
Phase-A 发布根提升为 v27；Phase-B 发布根提升为 v14，并同步提升 result、completion 与 launch schema。
v27 的 Maxwell regression 随后在生产 SciPy 上返回一个合法的闭区间上端解：第一项 GUI `Int`
为 `0.4000000000000001`，其 coupled polytope、显式 `k` witness 与正式 amplitude audit 均通过，但旧测试
直接使用无容差的 `<=0.4`，与 `ClosedInterval.contains` 的唯一数值契约不一致。v28 只把该回归改为调用
既有闭区间契约；公式、系数、可行域、范围端点、正式科学门禁与运行时输出均不改变。旧 v27 根、日志和
自动取消链保持不可变；Phase-A 发布根提升为 v28，Phase-B 发布根提升为 v15，并同步提升 result、
completion 与 launch schema。

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
Phase-A 模型目录发布时先把每个文件冻结为只读，再在仍为 job-private 的可写 staging 目录中移动目录项；
发布后目标目录必须为 `0500`，成员必须为 `0400`、`nlink=1`，并且 stage completion 仍最后发布。不能先把
staging 目录本身改为 `0500` 后再尝试删除其中目录项，因为 Linux/Lustre 会按目录写权限拒绝该操作。

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
