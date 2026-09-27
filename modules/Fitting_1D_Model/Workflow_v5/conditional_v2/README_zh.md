# 条件预测v2实验包

这是完整组分和两个resolution值已知时的预测工作流。相同组分可以返回多套参数。`neural`为纯网络，`four-step`额外做四步自动微分Gauss–Newton数值校正。未知组分识别、全部可能解及校准后验概率尚未解决。

输入：q单位nm⁻¹，强度采用训练体系的原始标度，不自动缩放；NPZ含q/observed/sigma，可含mask，或文本三列q/intensity/sigma。两列文本需显式提供--relative-noise假设。sigma-res为0.007–0.013 nm⁻¹，nu-res为5–10。组分为sphere/random_cylinder/vertical_cylinder，可重复，总数1–4。

在此目录中使用已有GUI Python环境运行：

```powershell
& 'D:\conda\envs\GUI\python.exe' predict.py --input measured.txt --output new_output --components sphere --sigma-res 0.010 --nu-res 7 --mode four-step
```

上述数值仅演示格式，不能当作实验真值。输出目录必须不存在。`solutions.json`含按观测误差排序和去重的候选，`forward_candidates.npz`保留全部原始参数和正演曲线。长度单位nm；分布宽度包含相对标准差及换算的绝对宽度。混合weight不是候选概率、质量分数或体积分数。

真实数据没有clean真值，输出只报告观测残差，不把合成数据的clean logRMSE<0.05直接当作真实曲线验收线。首次调用会加载、编译；程序调用时复用ConditionalFastPredictor实例。GPU数值部分耗时不等于完整响应时间。本地当前环境使用CPU。

本版本已通过独立128条测试和完整入口CPU/GPU一致性检查：纯神经84/128达到clean logRMSE<0.05，加入固定四步校正后114/128。完整GPU后续调用中位1.81秒、P90 3.83秒，首次30.95秒；不代表CPU速度。适用条件与局限见MODEL_CARD_zh.md、VALIDATION_zh.md和机器可读VALIDATION.json。原正式模型未替换。目录沿用candidate名称以保留来源。

`check_example.py`仅使用随包的一个TRAIN示例检验两种模式与独立目录加载，不构成新样本准确率。

验证图：[独立128汇总](validation_figures/independent128_summary.png)、[每个K固定首条曲线对照](validation_figures/native_q_fixed_examples.png)。后者沿用事前CPU重放样本，未按拟合好坏挑选；全曲线平均误差小不意味着每个q点都无偏差。
