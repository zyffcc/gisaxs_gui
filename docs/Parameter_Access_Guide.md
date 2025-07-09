# GISAXS软件参数访问系统使用指南

## 概述

现在您可以在GISAXS软件的**任何地方**轻松调用整个软件的参数！我们创建了一个完整的参数访问系统，让您在编写物理函数时能够方便地获取所需的参数。

## 核心文件

1. **`core/global_params.py`** - 全局参数管理器（单例模式）
2. **`utils/parameter_access.py`** - 参数访问工具和便捷函数
3. **`physics/calculation_demo.py`** - 物理计算函数示例

## 基本使用方法

### 1. 导入参数访问函数

```python
# 在任何Python文件中导入
from utils.parameter_access import (
    get_all_software_params,           # 获取所有软件参数
    get_physics_params_for_calculation, # 获取物理计算参数
    get_param_by_path,                 # 通过路径获取单个参数
    get_wavelength,                    # 快速获取波长
    get_particle_size,                 # 快速获取粒子尺寸
    get_scattering_setup,              # 获取散射设置
    get_sample_info                    # 获取样品信息
)
```

### 2. 获取所有软件参数

```python
def my_physics_function():
    # 获取整个软件的所有参数
    all_params = get_all_software_params()
    
    # all_params 包含所有模块的参数：
    # 'beam', 'detector', 'sample', 'preprocessing', 
    # 'trainset', 'fitting', 'classification', 'gisaxs_predict', 'system'
    
    beam_params = all_params['beam']
    sample_params = all_params['sample']
    # ... 使用参数进行计算
```

### 3. 获取物理计算专用参数

```python
def calculate_scattering():
    # 获取物理计算所需的核心参数
    physics_params = get_physics_params_for_calculation()
    
    # physics_params 包含：'beam', 'detector', 'sample', 'system'
    wavelength = physics_params['beam']['wavelength']
    detector_distance = physics_params['detector']['distance']
    particle_size = physics_params['sample']['particle_size']
    # ... 进行物理计算
```

### 4. 通过路径获取单个参数

```python
def my_calculation():
    # 使用点号分隔的路径获取嵌套参数
    wavelength = get_param_by_path('beam.wavelength')
    grazing_angle = get_param_by_path('beam.grazing_angle')
    qr_min = get_param_by_path('preprocessing.focus_region.qr_min')
    
    # 支持默认值
    custom_param = get_param_by_path('beam.custom_setting', default=1.0)
```

### 5. 使用便捷的快速访问函数

```python
def simple_calculation():
    # 直接获取常用参数
    wavelength = get_wavelength()        # 波长 (nm)
    angle = get_grazing_angle()          # 掠射角 (degrees)
    distance = get_detector_distance()   # 探测器距离 (mm)
    size = get_particle_size()           # 粒子尺寸 (nm)
    material = get_material()            # 材料
    substrate = get_substrate()          # 基底
```

### 6. 获取参数组

```python
def advanced_calculation():
    # 获取散射几何参数组
    geometry = get_scattering_setup()
    # 包含：wavelength, grazing_angle, detector_distance, 
    #       beam_center_x, beam_center_y, pixel_size_x, pixel_size_y, etc.
    
    # 获取样品结构参数组  
    sample = get_sample_info()
    # 包含：particle_shape, particle_size, material, substrate, 
    #       thickness, roughness, density, etc.
    
    # 获取计算设置参数组
    calc_settings = get_calc_settings()
    # 包含：calculation_method, approximation, max_iterations, etc.
```

## 实际应用示例

### 示例1：计算散射矢量q范围

```python
from utils.parameter_access import get_scattering_setup, get_wavelength
import numpy as np

def calculate_q_range():
    """计算散射矢量q的范围"""
    # 获取所需参数
    wavelength = get_wavelength()  # nm
    geometry = get_scattering_setup()
    
    detector_distance = geometry['detector_distance']  # mm
    pixel_size_x = geometry['pixel_size_x']  # μm
    beam_center_x = geometry['beam_center_x']  # pixels
    
    # 计算波矢量
    k0 = 2 * np.pi / wavelength
    
    # 计算q范围
    # ... 物理计算逻辑
    
    return q_range
```

### 示例2：形状因子计算

```python
from utils.parameter_access import get_sample_info, get_param_by_path

def calculate_form_factor():
    """计算粒子形状因子"""
    # 获取样品参数
    sample = get_sample_info()
    
    particle_shape = sample['particle_shape']
    particle_size = sample['particle_size']
    size_distribution = sample['size_distribution']
    
    # 根据粒子形状选择计算方法
    if particle_shape == 'Sphere':
        # 球形粒子计算
        radius = particle_size / 2
        # ... 计算球形粒子形状因子
    elif particle_shape == 'Cylinder':
        # 圆柱粒子计算
        # ... 
    
    return form_factor
```

### 示例3：完整的GISAXS模拟

```python
from utils.parameter_access import (
    get_physics_params_for_calculation,
    validate_params_for_physics
)

def gisaxs_simulation():
    """完整的GISAXS模拟计算"""
    
    # 1. 验证参数
    validation = validate_params_for_physics()
    if not validation['is_valid']:
        print("参数验证失败:", validation['errors'])
        return None
    
    # 2. 获取所有物理参数
    params = get_physics_params_for_calculation()
    
    # 3. 提取需要的参数
    beam = params['beam']
    detector = params['detector'] 
    sample = params['sample']
    system = params['system']
    
    # 4. 进行物理计算
    # ... 散射计算逻辑
    
    # 5. 返回结果
    return simulation_result
```

## 参数模块说明

### beam (光束参数)
- `wavelength`: 波长 (nm)
- `grazing_angle`: 掠射角 (degrees)
- `beam_size_x/y`: 光束尺寸 (mm)
- `flux`: 光子通量 (photons/s)
- `polarization`: 偏振

### detector (探测器参数)
- `distance`: 探测器距离 (mm)
- `pixel_size_x/y`: 像素尺寸 (μm)
- `beam_center_x/y`: 光束中心 (pixels)
- `nbins_x/y`: 像素数
- `exposure_time`: 曝光时间 (s)

### sample (样品参数)
- `particle_shape`: 粒子形状
- `particle_size`: 粒子尺寸 (nm)
- `material`: 材料
- `substrate`: 基底
- `thickness`: 厚度 (nm)
- `density`: 粒子密度

### system (系统参数)
- `calculation_method`: 计算方法 ('DWBA', 'Born', etc.)
- `approximation`: 近似方法
- `max_iterations`: 最大迭代次数
- `convergence_threshold`: 收敛阈值

## 高级功能

### 1. 参数验证

```python
from utils.parameter_access import validate_params_for_physics

def safe_calculation():
    validation = validate_params_for_physics()
    
    if validation['is_valid']:
        # 参数验证通过，可以安全计算
        proceed_with_calculation()
    else:
        # 处理验证错误
        print("错误:", validation['errors'])
        print("警告:", validation['warnings'])
        print("缺失参数:", validation['missing_parameters'])
```

### 2. 批量获取参数

```python
from utils.parameter_access import params

def get_multiple_params():
    # 批量获取多个参数
    param_paths = [
        'beam.wavelength',
        'beam.grazing_angle', 
        'sample.particle_size',
        'detector.distance'
    ]
    
    values = params.get_multiple_parameters(param_paths)
    # values = {'beam.wavelength': 0.1, 'beam.grazing_angle': 0.4, ...}
```

### 3. 导出计算专用格式

```python
from utils.parameter_access import export_params_for_physics

def prepare_for_calculation():
    # 获取字典格式
    params_dict = export_params_for_physics('dict')
    
    # 获取扁平化格式（便于传递给C/Fortran函数）
    params_flat = export_params_for_physics('flat')
    # {'beam_wavelength': 0.1, 'beam_grazing_angle': 0.4, ...}
    
    # 获取JSON格式
    params_json = export_params_for_physics('json')
```

## 在现有代码中集成

### 如果您已有物理函数

只需在函数开头添加参数获取代码：

```python
# 原有函数
def my_existing_function(wavelength, particle_size, ...):
    # ... 原有计算逻辑

# 新版本 - 自动获取参数
def my_existing_function():
    # 自动获取参数
    wavelength = get_wavelength()
    particle_size = get_particle_size()
    # ... 其他参数
    
    # ... 原有计算逻辑保持不变
```

## 运行演示

启动主程序后，会自动运行参数访问演示：

```bash
python main.py
```

您会看到类似输出：
```
=== 初始化GISAXS参数系统 ===
使用内置默认参数
✓ 控制器注册完成

=== 参数访问演示 ===
软件参数模块: ['beam', 'detector', 'sample', 'preprocessing', 'trainset', 'fitting', 'classification', 'gisaxs_predict', 'system']
物理参数模块: ['beam', 'detector', 'sample', 'system']
当前波长: 0.1 nm
当前粒子尺寸: 10.0 nm
✓ 物理参数验证通过

=== 物理计算演示 ===
--- 计算散射矢量q范围 ---
使用参数: λ=0.1nm, 距离=2000mm, 角度=0.4°
✓ 物理计算演示完成
```

## 总结

现在您可以：

1. **在任何地方**导入参数访问函数
2. **轻松获取**整个软件的任何参数
3. **验证参数**的完整性和合理性
4. **专注于物理计算**，无需担心参数管理
5. **扩展性强**，可以随时添加新的参数访问方法

这个系统让您在编写物理函数时完全不用担心参数来源，只需要调用相应的函数即可获得所需的参数！
