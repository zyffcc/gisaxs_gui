# 参数范围设置说明

## 概述

已将所有fitParticleWidget相关参数控件和全局拟合参数控件的数值范围扩展到实数域，支持更广泛的参数输入。

## 新的参数范围设置

### 数值范围
- **最小值**: -1×10¹⁰ (-10,000,000,000)
- **最大值**: +1×10¹⁰ (+10,000,000,000)
- **小数精度**: 6位小数
- **单步递增**: 0.1 (大部分参数), 0.01 (sigma_res)

### 影响的控件

#### 粒子参数控件 (每个粒子1-3)
**球形参数**:
- `fitParticleSphereIntValue_X` - 强度
- `fitParticleSphereRValue_X` - 半径
- `fitParticleSphereSigmaRValue_X` - 半径标准差
- `fitParticleSphereDValue_X` - 直径
- `fitParticleSphereSigmaDValue_X` - 直径标准差
- `fitParticleSphereBGValue_X` - 背景

**圆柱形参数**:
- `fitParticleCylinderIntValue_X` - 强度
- `fitParticleCylinderRValue_X` - 半径
- `fitParticleCylinderSigmaRValue_X` - 半径标准差
- `fitParticleCylinderhValue_X` - 高度
- `fitParticleCylinderSigmahValue_X` - 高度标准差
- `fitParticleCylinderDValue_X` - 直径
- `fitParticleCylinderSigmaDValue_X` - 直径标准差
- `fitParticleCylinderBGValue_X` - 背景

#### 全局拟合参数控件
- `fitSigmaResValue` - Sigma分辨率
- `fitKValue` - K值

## 支持的数值示例

### 正常科学数值
```
1.0          # 标准值
10.5         # 小数值
100.123456   # 高精度小数
```

### 大数值
```
1000000.0    # 一百万
1.23e6       # 科学记数法: 1,230,000
-5000000.0   # 负五百万
```

### 小数值
```
0.000001     # 微小正值
-0.000001    # 微小负值
1.23e-6      # 科学记数法: 0.00000123
```

### 高精度数值
```
3.141592653589793   # 高精度π
2.718281828459045   # 高精度e
```

## 功能特性

### 自动初始化
- 在控制器初始化时自动设置所有参数控件的范围
- 不需要手动配置每个控件

### 兼容性
- 与现有的参数保存/加载系统完全兼容
- 支持JSON序列化/反序列化
- 保持现有的UI交互逻辑

### 用户体验
- 鼠标滚轮支持快速调整数值
- 键盘输入支持科学记数法
- 单步调整针对不同参数类型优化

## 实现细节

### 代码位置
- `controllers/fitting_controller.py` - `_setup_parameter_ranges()` 方法
- 在粒子形状连接器初始化时自动调用

### 设置逻辑
```python
# 遍历所有粒子参数控件
for widget_id in self.particle_shape_configs.keys():
    # 设置球形和圆柱形参数控件
    for param_key, widget_name in parameter_mapping.items():
        if hasattr(self.ui, widget_name):
            widget = getattr(self.ui, widget_name)
            widget.setRange(-1e10, 1e10)  # 设置范围
            widget.setDecimals(6)         # 设置精度
            widget.setSingleStep(0.1)     # 设置步长
```

### 错误处理
- 控件不存在时安全跳过
- 设置完成后记录日志信息
- 统计成功设置的控件数量

## 测试验证

可以使用以下脚本测试参数范围功能:
```bash
python test_parameter_ranges.py
```

测试内容包括:
- 各种数值范围的参数设置
- 保存/加载极端数值
- 边界条件测试
- 精度保持验证

## 注意事项

1. **精度限制**: 虽然支持6位小数，但实际精度受Python float类型限制
2. **用户界面**: 在输入极大或极小数值时，建议使用科学记数法
3. **物理意义**: 虽然数学上支持负值，但某些物理参数(如半径)在实际应用中应为正值
4. **性能**: 大数值不会影响计算性能，但可能影响显示格式

## 扩展性

系统设计允许轻松调整:
- 修改数值范围上下限
- 调整小数精度
- 为不同参数类型设置不同的步长
- 添加新的参数控件自动应用相同设置