"""
物理计算函数示例 - 演示如何在任何地方调用整个软件的参数
这个文件可以在项目的任何地方被导入和使用
"""

import numpy as np
from typing import Dict, Any, Tuple
import math

# 导入参数访问系统
from utils.parameter_access import (
    get_all_software_params,
    get_physics_params_for_calculation, 
    get_param_by_path,
    get_scattering_setup,
    get_sample_info,
    get_calc_settings,
    get_wavelength,
    get_grazing_angle,
    get_detector_distance,
    get_particle_size,
    get_material,
    validate_params_for_physics
)


def calculate_q_range() -> Dict[str, Any]:
    """
    计算散射矢量q的范围
    演示如何在物理函数中调用软件参数
    
    Returns:
        包含q_x, q_y, q_z范围的字典
    """
    print("=== 计算散射矢量q范围 ===")
    
    # 方法1: 直接获取所需参数
    wavelength = get_wavelength()  # nm
    detector_distance = get_detector_distance()  # mm
    grazing_angle = get_grazing_angle()  # degrees
    
    # 方法2: 获取散射设置参数组
    geometry = get_scattering_setup()
    pixel_size_x = geometry['pixel_size_x']  # μm
    pixel_size_y = geometry['pixel_size_y']  # μm
    beam_center_x = geometry['beam_center_x']  # pixels
    beam_center_y = geometry['beam_center_y']  # pixels
    nbins_x = geometry['nbins_x']
    nbins_y = geometry['nbins_y']
    
    print(f"使用参数: λ={wavelength}nm, 距离={detector_distance}mm, 角度={grazing_angle}°")
    
    # 计算波矢量
    k0 = 2 * np.pi / wavelength  # 1/nm
    
    # 转换单位
    detector_distance_nm = detector_distance * 1e6  # mm -> nm
    pixel_size_x_nm = pixel_size_x * 1e3  # μm -> nm
    pixel_size_y_nm = pixel_size_y * 1e3  # μm -> nm
    
    # 计算q范围
    # q_x范围
    x_max = (nbins_x - beam_center_x) * pixel_size_x_nm
    x_min = -beam_center_x * pixel_size_x_nm
    q_x_max = k0 * x_max / detector_distance_nm
    q_x_min = k0 * x_min / detector_distance_nm
    
    # q_y范围 (假设探测器在y方向)
    y_max = (nbins_y - beam_center_y) * pixel_size_y_nm
    y_min = -beam_center_y * pixel_size_y_nm
    q_y_max = k0 * y_max / detector_distance_nm
    q_y_min = k0 * y_min / detector_distance_nm
    
    # q_z范围
    grazing_rad = np.radians(grazing_angle)
    q_z_min = k0 * np.sin(grazing_rad)
    
    # 最大散射角估计
    theta_max = np.arctan(max(abs(x_max), abs(y_max)) / detector_distance_nm)
    q_z_max = k0 * (np.sin(grazing_rad + theta_max) + np.sin(grazing_rad))
    
    result = {
        'q_x_range': (q_x_min, q_x_max),
        'q_y_range': (q_y_min, q_y_max), 
        'q_z_range': (q_z_min, q_z_max),
        'q_x_center': 0,
        'q_y_center': 0,
        'wavelength_used': wavelength,
        'detector_distance_used': detector_distance,
        'grazing_angle_used': grazing_angle
    }
    
    print(f"计算结果:")
    print(f"  q_x范围: {q_x_min:.4f} 到 {q_x_max:.4f} nm⁻¹")
    print(f"  q_y范围: {q_y_min:.4f} 到 {q_y_max:.4f} nm⁻¹")
    print(f"  q_z范围: {q_z_min:.4f} 到 {q_z_max:.4f} nm⁻¹")
    
    return result


def calculate_form_factor() -> Dict[str, Any]:
    """
    计算粒子的形状因子
    演示如何获取样品参数进行物理计算
    
    Returns:
        形状因子计算结果
    """
    print("\n=== 计算粒子形状因子 ===")
    
    # 方法1: 获取单个参数
    particle_size = get_particle_size()  # nm
    material = get_material()
    
    # 方法2: 获取完整样品信息
    sample_info = get_sample_info()
    particle_shape = sample_info['particle_shape']
    size_distribution = sample_info['size_distribution']
    density = sample_info['density']
    
    print(f"计算 {particle_shape} 粒子的形状因子")
    print(f"材料: {material}, 尺寸: {particle_size}nm, 分布: {size_distribution}")
    
    # 简化的球形粒子形状因子计算
    if particle_shape.lower() == 'sphere':
        # 获取q范围用于计算
        q_data = calculate_q_range()
        q_x_range = q_data['q_x_range']
        q_z_range = q_data['q_z_range']
        
        # 创建q网格
        q_points = 50
        q_x = np.linspace(q_x_range[0], q_x_range[1], q_points)
        q_z = np.linspace(q_z_range[0], q_z_range[1], q_points)
        Q_x, Q_z = np.meshgrid(q_x, q_z)
        
        # 计算总散射矢量
        Q_total = np.sqrt(Q_x**2 + Q_z**2)
        
        # 球形粒子形状因子: F(q) = 3[sin(qR) - qR*cos(qR)]/(qR)³
        R = particle_size / 2  # 半径
        qR = Q_total * R
        
        # 避免除零
        qR_safe = np.where(qR == 0, 1e-10, qR)
        form_factor = 3 * (np.sin(qR_safe) - qR_safe * np.cos(qR_safe)) / (qR_safe**3)
        
        # 在qR=0处的极限值
        form_factor = np.where(qR == 0, 1.0, form_factor)
        
        # 考虑尺寸分布的影响（简化为高斯分布）
        if size_distribution > 0:
            sigma_R = R * size_distribution
            size_broadening = np.exp(-(Q_total * sigma_R)**2 / 2)
            form_factor *= size_broadening
        
        result = {
            'form_factor': form_factor,
            'q_x_grid': Q_x,
            'q_z_grid': Q_z,
            'particle_radius': R,
            'size_distribution': size_distribution,
            'material': material,
            'max_form_factor': np.max(form_factor),
            'min_form_factor': np.min(form_factor)
        }
        
        print(f"形状因子计算完成:")
        print(f"  粒子半径: {R:.2f} nm")
        print(f"  最大形状因子: {np.max(form_factor):.4f}")
        print(f"  最小形状因子: {np.min(form_factor):.4f}")
        
        return result
    
    else:
        print(f"暂不支持 {particle_shape} 粒子的计算")
        return {'error': f'Unsupported particle shape: {particle_shape}'}


def estimate_scattering_intensity() -> Dict[str, Any]:
    """
    估算散射强度
    演示如何综合使用多个模块的参数
    
    Returns:
        散射强度估算结果
    """
    print("\n=== 估算散射强度 ===")
    
    # 获取所有物理参数
    physics_params = get_physics_params_for_calculation()
    
    # 获取光束参数
    beam_params = physics_params['beam']
    flux = beam_params.get('flux', 1e12)  # photons/s
    beam_size_x = beam_params.get('beam_size_x', 0.1)  # mm
    beam_size_y = beam_params.get('beam_size_y', 0.1)  # mm
    
    # 获取探测器参数
    detector_params = physics_params['detector']
    exposure_time = detector_params.get('exposure_time', 1.0)  # s
    
    # 获取样品参数
    sample_params = physics_params['sample']
    density = sample_params.get('density', 0.5)
    thickness = sample_params.get('thickness', 100.0)  # nm
    
    print(f"光束参数: 通量={flux:.1e} photons/s, 尺寸={beam_size_x}×{beam_size_y} mm²")
    print(f"探测器: 曝光时间={exposure_time} s")
    print(f"样品: 密度={density}, 厚度={thickness} nm")
    
    # 计算形状因子
    form_factor_data = calculate_form_factor()
    
    if 'error' not in form_factor_data:
        form_factor = form_factor_data['form_factor']
        
        # 简化的散射强度计算
        # I(q) ∝ |F(q)|² × N × σ × Φ
        
        # 粒子数密度估计
        particle_volume = (4/3) * np.pi * (get_particle_size()/2)**3  # nm³
        sample_volume = beam_size_x * beam_size_y * thickness * 1e12  # nm³ (mm² * nm)
        N_particles = density * sample_volume / particle_volume
        
        # 散射截面（简化）
        wavelength = get_wavelength()
        sigma_thomson = 7.94e-26  # Thomson scattering cross section (cm²)
        
        # 相对强度
        intensity = (form_factor**2) * N_particles * sigma_thomson * flux * exposure_time
        
        result = {
            'intensity': intensity,
            'q_x_grid': form_factor_data['q_x_grid'],
            'q_z_grid': form_factor_data['q_z_grid'],
            'N_particles': N_particles,
            'max_intensity': np.max(intensity),
            'total_counts': np.sum(intensity),
            'parameters_used': {
                'flux': flux,
                'exposure_time': exposure_time,
                'density': density,
                'thickness': thickness,
                'particle_size': get_particle_size(),
                'wavelength': wavelength
            }
        }
        
        print(f"散射强度估算完成:")
        print(f"  粒子数: {N_particles:.1e}")
        print(f"  最大强度: {np.max(intensity):.1e} counts")
        print(f"  总计数: {np.sum(intensity):.1e} counts")
        
        return result
    
    else:
        return form_factor_data


def run_physics_calculation_demo():
    """
    运行物理计算演示
    展示如何在任何地方调用软件参数进行物理计算
    """
    print("=" * 60)
    print("物理计算演示 - 展示参数访问系统")
    print("=" * 60)
    
    # 1. 验证参数
    validation = validate_params_for_physics()
    if not validation['is_valid']:
        print("⚠ 参数验证失败:")
        for error in validation['errors']:
            print(f"  - {error}")
        return
    
    print("✓ 参数验证通过")
    
    # 2. 计算q范围
    q_data = calculate_q_range()
    
    # 3. 计算形状因子
    form_factor_data = calculate_form_factor()
    
    # 4. 估算散射强度
    intensity_data = estimate_scattering_intensity()
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("计算总结:")
    print("=" * 60)
    
    if 'error' not in intensity_data:
        print(f"✓ 成功计算了 {get_material()} {get_param_by_path('sample.particle_shape')} 粒子的散射")
        print(f"✓ 粒子尺寸: {get_particle_size()} nm")
        print(f"✓ 使用波长: {get_wavelength()} nm")
        print(f"✓ 预期最大强度: {intensity_data['max_intensity']:.1e} counts")
        
        # 保存结果到全局参数（可选）
        result_summary = {
            'calculation_type': 'gisaxs_simulation',
            'timestamp': 'current',
            'q_range': q_data,
            'max_intensity': float(intensity_data['max_intensity']),
            'total_counts': float(intensity_data['total_counts'])
        }
        
        # 可以将结果存储到全局参数系统
        from utils.parameter_access import params
        # params.set_parameter_by_path('system.last_calculation', result_summary)
        
        print("✓ 计算完成，结果已生成")
    else:
        print("⚠ 计算过程中出现错误")
    
    return {
        'q_data': q_data,
        'form_factor_data': form_factor_data,
        'intensity_data': intensity_data
    }


# 便捷函数：可以在任何地方调用
def get_current_physics_state() -> Dict[str, Any]:
    """
    获取当前物理状态的快照
    这个函数可以在软件的任何地方调用
    
    Returns:
        当前物理状态字典
    """
    return {
        'wavelength': get_wavelength(),
        'grazing_angle': get_grazing_angle(),
        'detector_distance': get_detector_distance(),
        'particle_size': get_particle_size(),
        'material': get_material(),
        'substrate': get_param_by_path('sample.substrate'),
        'calculation_method': get_param_by_path('system.calculation_method'),
        'all_beam_params': get_param_by_path('beam'),
        'all_sample_params': get_param_by_path('sample')
    }


def quick_parameter_check():
    """
    快速参数检查 - 可以在任何地方调用
    """
    print("\n=== 快速参数检查 ===")
    state = get_current_physics_state()
    
    for key, value in state.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} 个参数")
        else:
            print(f"{key}: {value}")
    
    return state


if __name__ == "__main__":
    # 如果直接运行这个文件，执行演示
    print("这是一个物理计算演示模块")
    print("请在主程序中导入并使用这些函数")
    
    # 示例使用方法
    print("\n示例使用方法:")
    print("from physics.calculation_demo import run_physics_calculation_demo")
    print("result = run_physics_calculation_demo()")
