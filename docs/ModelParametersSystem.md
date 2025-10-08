# 模型参数管理系统说明

## 概述

新的模型参数管理系统专门用于存储和管理GISAXS GUI应用程序中的所有模型相关参数，包括粒子参数和全局拟合参数。

## 主要组件

### 1. `config/model_parameters.json`
存储所有模型参数的JSON配置文件，包括：
- **粒子参数**：每个粒子的形状、启用状态和具体参数（球形、圆柱形）
- **全局拟合参数**：`sigma_res`和`k_value`等全局拟合参数
- **元数据**：版本信息、创建日期等

### 2. `config/model_parameters_manager.py`
模型参数管理器类，提供以下功能：
- 参数的加载、保存和访问
- 粒子参数管理（形状、具体参数值）
- 全局参数管理
- 参数变更信号发射

### 3. 在`controllers/fitting_controller.py`中的集成
- 自动初始化：根据保存的参数设置UI控件状态
- 实时保存：参数变更时自动保存到文件
- 便利方法：提供高级API来管理参数

## 核心功能

### 粒子参数管理
```python
# 设置粒子形状
fitting_controller.set_particle_shape(1, 'Sphere')  # 设置粒子1为球形

# 获取粒子形状
shape = fitting_controller.get_particle_shape(1)

# 获取所有粒子状态
status = fitting_controller.get_particles_status()
```

### 全局参数管理
```python
# 设置全局参数
fitting_controller.set_global_parameter('sigma_res', 0.15)
fitting_controller.set_global_parameter('k_value', 1.2)

# 获取全局参数
sigma_res = fitting_controller.get_global_parameter('sigma_res')
k_value = fitting_controller.get_global_parameter('k_value')
```

### 参数导入导出
```python
# 导出参数到文件
fitting_controller.export_particle_parameters('backup.json')

# 从文件导入参数
fitting_controller.import_particle_parameters('backup.json')
```

## 初始化逻辑

1. **启动时判断**：根据保存的`fitParticleShapeCombox_1/2/3`状态初始化对应页面
2. **参数加载**：从`model_parameters.json`加载保存的参数值到UI控件
3. **信号连接**：连接UI控件变更信号到参数保存功能
4. **自动保存**：参数变更后延迟500ms自动保存（避免频繁I/O）

## UI控件映射

### 粒子参数控件
每个粒子（1、2、3）对应的控件：
- **球形参数**：`fitParticleSphereIntValue_X`、`fitParticleSphereRValue_X`等
- **圆柱形参数**：`fitParticleCylinderIntValue_X`、`fitParticleCylinderRValue_X`等

### 全局参数控件
- **Sigma Res**：`fitSigmaResValue`
- **K Value**：`fitKValue`

## 文件结构
```
config/
├── model_parameters.json          # 参数存储文件
├── model_parameters_manager.py    # 参数管理器
└── __init__.py                    # 模块导入

controllers/
└── fitting_controller.py         # 集成模型参数管理

test_model_params.py              # 测试脚本
example_model_params_usage.py     # 使用示例
```

## 使用场景

1. **用户操作**：用户在GUI中修改参数，自动保存到配置文件
2. **程序设置**：程序代码中批量设置参数
3. **会话恢复**：应用启动时恢复上次的参数设置
4. **参数备份**：导出参数配置用于备份或分享
5. **批量操作**：重置所有参数或特定类型参数

## 扩展性

系统设计考虑了未来扩展：
- 易于添加新的粒子形状
- 支持多个模块（fitting、gisaxs_predict、classification）
- 灵活的参数结构
- 版本化的配置格式

## 注意事项

1. 参数变更会自动保存，无需手动保存
2. 初始化时会自动设置UI控件状态
3. 系统使用延迟保存机制避免频繁I/O操作
4. 所有参数操作都有错误处理和日志记录