# GISAXS Posterior V8 V5.2 跨电脑交接

交接日期：2026-09-09。远端状态最后实际检查于 2026-09-07；迁移后必须重新检查，不能把历史状态当作实时状态。本文不包含凭据。

## 给新电脑上的助手

请接续此任务，而不是重新设计或从头训练。先阅读仓库 AGENTS.md 和唯一权威科学文档 `docs/architecture/multisolution-inversion.md`，检查 git status/diff 并保护全部已有改动。用户已授权范围内自主选择科学/规模参数、版本化实现、归档和 Slurm 提交。用户后来因额度暂停了自动任务；只有收到继续指令后才恢复计算。不要擅自恢复定时任务。

目标：完成 GISAXS 一键多解反演，输入真实 Cut_Data 和用户物理范围，输出多个相容候选、参数、重建曲线、误差和校准后的不确定性，集成 GUI；顺序为 full-K1 → K2 → all34 → 校准/真实数据/GUI → 论文级验收。单 Sphere 成功不代表整个项目完成。

## 硬约束

- 大数据、模型、checkpoint、缓存、日志只写 `/data/dust/user/zhaiyufe/`。
- 旧模型、版本根、日志、失败证据和用户数据不可删除、覆盖或重新用于晋级。现有已发布 balanced 数据允许作为新流水线的只读输入；不得复用 Phase-A 模型权重。
- 绝不修改或取消用户作业 24373458–24373464。
- 登录节点只做轻量检查、文件管理和 Slurm；生成、训练、批量哈希/重放和评估在 worker。
- 只剩运行/排队作业时不忙轮询。30 分钟自动任务 `maxwell-gisaxs` 已暂停。
- 不保存或输出密码/OTP，不复制旧凭据。出现 OTP 或认证拒绝时停下请求新的 OTP，不重用历史验证码。
- 不擅自 commit、push、stash、clean、restore 或重置工作树。
- 不降低 branch recall@4≥0.95、any-compatible≥0.95、reference recall@N16/B4096≥0.85，以及 exact-forward、物理、范围、幅度、duplicates、预算门禁。

## 迁移代码：仅 clone 不够

仓库远端：`git@github.com:zyffcc/gisaxs_gui.git`。交接时分支 main，存在大量 tracked 修改及 untracked 新模块、测试。仅 clone/pull 会漏掉尚未提交的工作。

优先安全复制当前整个代码工作目录（包括 `.git` 和未跟踪的代码/测试），但不要复制 SSH 私钥、凭据、环境缓存、数据或模型。不要把旧电脑的绝对路径当成新电脑路径。本文不是代码归档，不能单独替代工作目录迁移。

如果无法复制工作目录，可使用下述远端冻结源码作为恢复基线；但它没有最后的本地 v3 修复，必须按下文补回并重新验证。新机器登录能力需要重新建立，旧电脑的 multiplex socket 不能迁移。

## 已完成的科学工作

1. Phase-A v26：24438621–24438626 全通过。H200，width128/6 blocks/batch32，18000 Adam updates，cosine 3e-3→3e-5；目标 MDN/coverage/Top4 alignment 权重 1/100/1。训练样本记忆门禁 median local RMS=0.00015558343147858977，不是独立测试集泛化成绩。
2. 单 Sphere Phase-B v13：24438729/24438730 全通过。512 parents（481 learnable、31 fully-fixed、809 varying coordinates），16384/16384 精修，零失败。exact compatible=1，natural-log RMSE p90=2.4422022204118997e-10。只授权 K1 Sphere 单分支，误差为合成样本物理精修后的结果。
3. balanced all12 数据 v3：train=1152/branch、tuning=288/branch、views=(0,1,2)，master seeds=2844333258/3110101137。总 train13824、tuning3456、60 shards。作业24444579/24444580 完成。
4. Phase-C holdout identity v1：13824 clean，24445195/24445196 完成。train/tuning/holdout recipe/group 六项交集均零；身份隔离证据不授权训练或验收。
5. IID calibration v2：160020 独立样本，60 strata，各2667，作业24447800/24447801 完成。该产物是阈值校准，不代表完整模型验收。
6. all-K1 full-search-supervision 的计划/worker/collector/严格发布与启动代码已实现；尚未完成正式搜索数据生产、新模型训练和 Phase-C 验收。

## 远端路径

以下 BASE 为 `/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2`。

- balanced：BASE/`k1_balanced_all12_dataset_dag_v3`
- holdout：BASE/`k1_phase_c_holdout_dataset_dag_v1`
- calibration：BASE/`k1_iid_calibration_dag_v2/artifacts/k1-iid-compatibility-v2.json`
- Sobol schedule：BASE/`k1_exact_search_schedule_dag_v1/artifacts/k1-paper-full-b4096-local-sobol-v1.gvd5`
- Sphere 结果：BASE/`k1_phase_b_v5_2_dag_v13`
- 专用环境：`/data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2`。

关键 file SHA256（仍须从正式 completion/receipt 重放验证）：

- train/tuning receipt：0273bd1a0d8095dc063f98191aded35f3e167fae0f050aad25a389926bbec9ea
- 三方 receipt：ce7243ecec09e8b21cdc211e5f0d358ef20d7e5514a64a62fc9080de8d87e155
- calibration：4acfa94b11b009c49886d8072f15db3e70854b5523c0b338ad22b3281338787b
- Sobol schedule：ff777bb8e3270ab67f6a0d560cb17b373502df8f608a571c8d911e20680a1cd8

## 最后冻结源码与尚未归档的修复

最后远端冻结 archive SHA256：
`a3b4227ccef9a0d6319ffef731efb557df2ab5c3eb98143fcab40c9e54ca12bf`

archive：`/data/dust/user/zhaiyufe/source-archives/gisaxs-posterior-v8-r2-a3b4227ccef9a0d6319ffef731efb557df2ab5c3eb98143fcab40c9e54ca12bf.tar`

snapshot：`/data/dust/user/zhaiyufe/source-snapshots/gisaxs-posterior-v8-r2-a3b4227ccef9a0d6319ffef731efb557df2ab5c3eb98143fcab40c9e54ca12bf`

manifest SHA=0d45b58c7baffc061b9f7cecbca1bd5a387735ab45d5d8ee8353dfbc7081f7b0；tree SHA=dc49cffadfb97343e13e2630980a14ae9a098fe09c1f73463304b150c22706e0。1157 files，11980800 bytes，已验证只读和无 symlink。

对应 Mac reference：`/data/dust/user/zhaiyufe/reference-manifests/sobol-cross-platform-r2-37791b11095eac3ca95d315e71964ccc75a19a71e20228aea57383608415b8e5.json`。

计划生成 v1 作业24453902 因 CLI 参数到 config 字段映射错误失败，已在上面冻结源码修复。随后 v2 作业 **24455508 FAILED 1:0**，22 秒，节点 max-ferrari020；在首次路径检查因 PurePosixPath 没有 resolve 方法失败。尚无 candidate plan 或 plan-author completion，未进入正式搜索。

失败根 BASE/`k1_balanced_full_search_plan_authoring_v2` 必须保留。搜索根 `k1_balanced_full_search_dag_v2` 最后未创建，迁移后仍需检查，不能假定不存在。

最后本地 v3 修复（未重新归档/提交）：

- `author_k1_balanced_full_search_plan_v5.py` 中 `_under_root` 改为 `root = Path(allowed_root).resolve(strict=True)`。
- author VERSION 改为 `posterior_v8_worker_only_publication_replay_concrete_root_completion_last_v3`，计划载荷 schema 未变。
- `test_posterior_v8_k1_balanced_full_search_plan_author_v5.py` 对 Path/PurePosixPath 两类根参数化验证只读排他发布。
- `slurm/v5_k1_balanced_full_search_plan_author_cpu.sbatch` 默认日志前缀变 v3。
- 唯一科学文档记录原因和边界。
- 最后仅此 focused 文件 5 passed、相关 ruff、git diff --check 通过。完整测试尚未针对 v3 重跑。上一冻结版本完整测试为1336 passed/3 skipped/52 subtests，Sobol24 passed；不能把它冒充 v3 完整验证。

## 恢复后的具体执行顺序

1. 核对迁移后的代码和上述 v3 改动，读取相关源码，不直接复制旧启动命令。重新建立 FS 登录，遇 OTP 再向用户请求。用户指定的入口为 `zhaiyufe@max-fs-display.desy.de`。
2. 旧电脑曾使用 `/tmp/codex_maxwell_fs_v1_20260907.sock`，master17722/keepalive19397；这些只作历史记录，在新电脑不能使用其 PID 或声称保活仍健康。避免重复建立常驻 keepalive；不删除旧连接。
3. 为 v3 增加足够的入口/路径回归，运行 focused/full/Sobol/ruff/compileall/全部 sbatch bash-n/diff-check。科学定义和门禁不变。
4. 新 source 双归档、新 reference、新 content-addressed snapshot；旧 archive/snapshot 只读保留。用全新 authoring 根和日志提交 worker，held-readback 后 release。
5. worker 生成 candidate plan 后严格核验0400/nlink1、self/file hashes、输入 pre/post、60 shards、source identity 和 completion-last。
6. 从同一冻结 snapshot 运行 `launch_k1_balanced_full_search_dag_v5.py` dry-run，然后正式提交60-task array + afterok collector；必须使用全新搜索根、held-readback/reverse-release。计划 authoring 的重放不能在登录节点执行。
7. 搜索结果通过后补模型专属 tuning exact-budget adapter、formal training authorization、从零 multiseed full-K1 训练、Phase-C live producer/writer/replay。现有 identity-only gate 固定不授权梯度，不得旁路。
8. full-K1 通过后才推进 K2、all34、真实数据和 GUI；不得宣称论文级全部完成。

核心代码位于 `utils/ML_Fitting_1D_GISAXS/PosteriorV8/`，关注 `author_k1_balanced_full_search_plan_v5.py`、`k1_balanced_full_search_*_v5.py`、`launch_k1_balanced_full_search_dag_v5.py`、`k1_training_chain_*_v5.py`、`tuning_checkpoint_runtime_v5.py` 和 `formal_production_training_promotion_v5.py`。

## 建议新任务的首条消息

“请阅读 GISAXS_HANDOFF_20260909.md 和 AGENTS.md，核对迁移后的未提交代码，接续 GISAXS Posterior V8 V5.2 工作。先恢复安全 SSH 和 v3 计划生成修复的验证，再按交接顺序推进。保留旧产物，不降低门禁，所有重计算走 Maxwell worker。不要自动恢复30分钟定时任务，除非我明确要求。”
