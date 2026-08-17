# Format Converter 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| step 1 source actions、`input_tree`、dataset selector | `Input / ParameterSection` | 原 signal、source model 和 dataset 选择不变 |
| step 2 tools、`selection_table` | `Configure / ParameterSection` | include、filter、sort、remove 不变 |
| frame mode/range/custom/Every N | `Configure / AdvancedSection` | 默认折叠；值、默认 All、frame parsing 不变 |
| First/Middle/Last labels 与 statistics | `Preview / PlotPanel` | 原 preview worker、orientation 和 statistics 不变 |
| output format、destination、naming | `Configure output / ParameterSection` | format 与命名规则不变 |
| dtype、metadata、container | `Advanced output options / AdvancedSection` | 默认折叠；serialization 和默认值不变 |
| `output_summary`、Review & Convert | `Run, Results & Export / ParameterSection` | 原确认 dialog 和 conversion use case 不变 |
| conversion title/detail/progress/pause/cancel | shared `JobStatus` | 原 worker pause/cancel 信号不变 |
| succeeded/failed、Open folder、View report | conversion result area | 原 report path 和按钮不变 |

没有重命名业务控件，没有改变 ViewModel command、输出格式、metadata JSON、frame selection
或 dtype conversion。

## 手动验收清单

- [ ] Add files、Add folder、Use current file 均可添加输入；
- [ ] NXS 多 dataset 选择仍能更新 frame/shape；
- [ ] Select all/none、filter、sort、remove 正常；
- [ ] 展开 Advanced 后 All、Current、Range、Custom、Every N 结果与原来一致；
- [ ] First/Middle/Last preview 的方向、dtype 和统计信息正确；
- [ ] TIFF、CBF、HDF5、NumPy 可选性与输入类型兼容；
- [ ] destination、命名样例、collision suffix 和 output estimate 正确；
- [ ] Advanced 折叠/展开不重置 dtype、metadata 或 container 值；
- [ ] Review & Convert 摘要的 input/output/frame count 正确；
- [ ] conversion 的 progress、Pause/Resume、Cancel 正常；
- [ ] 成功后 Open output folder、View report 可用；
- [ ] 关闭运行中的 progress dialog 仍受原安全限制。
