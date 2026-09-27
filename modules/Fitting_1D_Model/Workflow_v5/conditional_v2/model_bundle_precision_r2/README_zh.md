# precision-r2：完整校正默认＋可选条件快速模式

默认仍使用原完整校正，未知组分仍计算分类器前12组合：

```bash
python source/precision_predict.py --bundle . --input example_input.npz --output prediction
```

已知完整组分时，可使用已确认的按需停止模式：

```bash
python source/adaptive_predict.py --bundle . --input example_input.npz --components 1 1 1 2 --output fast_prediction
```

快速模式保留指定组合的六个参数起点，每个候选单独检查噪声感知质量，不会因为已有一套好解而丢掉其它头。它仍包含数值校正；停止策略固定为early_strict，--threshold .03/.05只改变输出质量等级标记。当前快速模式仅支持完整指定components；未知组分请用默认入口。

新确认数据、质量与速度对照、噪声限制见ADAPTIVE_REPORT_zh.md。默认完整方案报告REPORT_zh.md描述上一轮独立测试，不能把两批测试混为同一批。原有单位约定继续适用；真实实验切线尚需强度和噪声对接，没有宣称真实数据验证完成。

---

# 1D多组分多参数候选：precision-r1

默认入口是 `source/precision_predict.py`。模型使用原r3网络的六个起点，再做有限次信赖域数值校正。它包含精修，不能称为神经网络直接输出或完整后验采样。

在已配置TensorFlow/SciPy的兼容环境中，进入本目录运行未知组分预测：

```bash
python source/precision_predict.py --bundle . --input example_input.npz --output prediction
```

如果已知完整组分（示例输入的组分如下），使用：

```bash
python source/precision_predict.py --bundle . --input example_input.npz --components 1 1 1 2 --output conditional_prediction
```

类型1=球，2=随机取向圆柱，3=竖直圆柱；必须给完整1–4组分多重集合，允许重复。未知组分默认返回分类器前12种组合，每组保留至多六套数值不等价参数；已知组分只返回指定组合。没有在模型中加入第五类形状。

输入NPZ包含q、observed、sigma和可选mask。q单位nm^-1，强度与sigma遵守V5相对强度约定；每条有效点8–1000，q递增、强度与sigma为正。不会自动换算或重新归一化实验强度。输出solutions.json和forward_candidates.npz包含物理参数、曲线、多解及噪声相关质量标记。--threshold .03或.05指定期望clean误差等级，不能直接解释为observed误差截断。

新独立128条测试：给定C，所选解96.09%达到clean logRMSE<.05，池内97.66%存在合格解，93.75%存在至少两套不同合格参数。平均条件预测约2.24秒/条（H200）；未知12组合完整例约34.3秒。完整方法、难例、噪声规则及局限见REPORT_zh.md。

曲线拟合良好不保证恢复真实参数。候选数量有限，不保证穷尽所有解；分类分数和参数头不作为校准概率。质量未通过的候选会明确标记，不能把全部返回项都当成合格解。小权重不等于对应组分已被可靠辨认。

Maxwell已验证Python3.11/TensorFlow2.15/SciPy1.14环境；本地另做了NumPy双精度独立正演核验。包内旧source/predict_1d.py提供基础网络类，默认使用上面的precision入口。base_model_config.json及继承的旧验证文件仅描述基础网络。推理不依赖原始训练集、离线教师或实验缓存。
