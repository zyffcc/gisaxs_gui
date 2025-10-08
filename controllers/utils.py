"""
工具函数模块 - 包含常用的工具函数和计算方法
"""

import numpy as np
import json
import os
from datetime import datetime

class GISAXSUtils:
    """GISAXS相关的工具函数"""
    
    @staticmethod
    def wavelength_to_energy(wavelength_nm):
        """将波长(nm)转换为能量(keV)"""
        return 1.24 / wavelength_nm
    
    @staticmethod
    def energy_to_wavelength(energy_kev):
        """将能量(keV)转换为波长(nm)"""
        return 1.24 / energy_kev
    
    @staticmethod
    def calculate_q_values(detector_params, beam_params):
        """计算q值"""
        distance_mm = detector_params['distance']
        pixel_size_x = detector_params['pixel_size_x'] / 1000  # 转换为mm
        pixel_size_y = detector_params['pixel_size_y'] / 1000  # 转换为mm
        wavelength_nm = beam_params['wavelength']
        
        # 计算q空间转换因子
        q_per_pixel_x = 4 * np.pi * np.sin(np.arctan(pixel_size_x / distance_mm) / 2) / (wavelength_nm * 1e-9)
        q_per_pixel_y = 4 * np.pi * np.sin(np.arctan(pixel_size_y / distance_mm) / 2) / (wavelength_nm * 1e-9)
        
        return q_per_pixel_x, q_per_pixel_y
    
    @staticmethod
    def pixel_to_q(pixel_x, pixel_y, detector_params, beam_params):
        """将像素坐标转换为q坐标"""
        distance_mm = detector_params['distance']
        pixel_size_x = detector_params['pixel_size_x'] / 1000
        pixel_size_y = detector_params['pixel_size_y'] / 1000
        beam_center_x = detector_params['beam_center_x']
        beam_center_y = detector_params['beam_center_y']
        wavelength_nm = beam_params['wavelength']
        
        # 计算相对于光束中心的距离
        dx = (pixel_x - beam_center_x) * pixel_size_x
        dy = (pixel_y - beam_center_y) * pixel_size_y
        
        # 计算散射角
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan(r / distance_mm)
        
        # 计算q值
        q = 4 * np.pi * np.sin(theta / 2) / (wavelength_nm * 1e-9)
        
        # 计算qx, qy分量
        qx = q * dx / r if r > 0 else 0
        qy = q * dy / r if r > 0 else 0
        
        return qx / 1e9, qy / 1e9  # 转换为nm^-1
    
    @staticmethod
    def calculate_critical_angle(material_delta):
        """计算临界角"""
        return np.sqrt(2 * material_delta) * 1000  # 转换为mrad
    
    @staticmethod
    def form_factor_sphere(q, radius):
        """球形粒子的形状因子"""
        qr = q * radius
        if qr == 0:
            return 1.0
        return 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
    
    @staticmethod
    def size_distribution_gaussian(r, r_mean, sigma):
        """高斯尺寸分布"""
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - r_mean) / sigma)**2)
    
    @staticmethod
    def size_distribution_lognormal(r, r_mean, sigma):
        """对数正态尺寸分布"""
        return (1 / (r * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(r) - np.log(r_mean)) / sigma)**2)

class FileUtils:
    """文件操作工具函数"""
    
    @staticmethod
    def save_json(data, file_path, indent=4):
        """保存JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {e}")
            return False
    
    @staticmethod
    def load_json(file_path):
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return None
    
    @staticmethod
    def create_timestamped_filename(base_name, extension=".json"):
        """创建带时间戳的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}{extension}"
    
    @staticmethod
    def ensure_directory(directory_path):
        """确保目录存在"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录失败: {e}")
            return False

class ValidationUtils:
    """参数验证工具函数"""
    
    @staticmethod
    def validate_range(value, min_val, max_val, name="参数"):
        """验证数值范围"""
        if value < min_val or value > max_val:
            return False, f"{name}必须在{min_val}-{max_val}范围内"
        return True, "有效"
    
    @staticmethod
    def validate_positive(value, name="参数"):
        """验证正数"""
        if value <= 0:
            return False, f"{name}必须大于0"
        return True, "有效"
    
    @staticmethod
    def validate_file_path(file_path, check_exists=True):
        """验证文件路径"""
        if not file_path:
            return False, "文件路径不能为空"
        
        if check_exists and not os.path.exists(file_path):
            return False, "文件不存在"
        
        return True, "有效"
    
    @staticmethod
    def validate_directory_path(dir_path, check_writable=True):
        """验证目录路径"""
        if not dir_path:
            return False, "目录路径不能为空"
        
        if not os.path.exists(dir_path):
            return False, "目录不存在"
        
        if check_writable and not os.access(dir_path, os.W_OK):
            return False, "目录没有写入权限"
        
        return True, "有效"

class MathUtils:
    """数学计算工具函数"""
    
    @staticmethod
    def interpolate_1d(x, y, x_new):
        """一维插值"""
        return np.interp(x_new, x, y)
    
    @staticmethod
    def smooth_data(data, window_size=5):
        """数据平滑"""
        if window_size < 3:
            return data
        
        # 使用移动平均
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    @staticmethod
    def normalize_data(data, method='minmax'):
        """数据归一化"""
        if method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            else:
                return data
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return (data - mean) / std
            else:
                return data - mean
        else:
            return data
    
    @staticmethod
    def add_gaussian_noise(data, snr_db):
        """添加高斯噪声"""
        # 计算信号功率
        signal_power = np.mean(data**2)
        
        # 从SNR计算噪声功率
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 生成噪声
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        
        return data + noise

class PlotUtils:
    """绘图工具函数"""
    
    @staticmethod
    def setup_gisaxs_plot_style():
        """设置GISAXS绘图样式"""
        try:
            import matplotlib.pyplot as plt
            import warnings
            
            # 配置字体和警告
            plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
            
            plt.style.use('seaborn-v0_8')  # 或其他可用样式
            plt.rcParams['figure.figsize'] = (10, 8)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['legend.fontsize'] = 12
            
            return True
        except ImportError:
            print("matplotlib未安装，无法设置绘图样式")
            return False
    
    @staticmethod
    def create_2d_plot(data, title="GISAXS Pattern", xlabel="qx", ylabel="qy"):
        """创建2D图"""
        try:
            import matplotlib.pyplot as plt
            import warnings
            
            # 配置字体和警告（确保每次调用都有正确配置）
            plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
            
            # 垂直翻转图像数据以修正显示方向
            data = np.flipud(data)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, origin='lower', aspect='auto', cmap='hot')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            plt.colorbar(im, ax=ax, label='Intensity')
            plt.tight_layout()
            
            return fig, ax
        except ImportError:
            print("matplotlib未安装，无法创建绘图")
            return None, None

# 导出主要工具类
__all__ = [
    'GISAXSUtils',
    'FileUtils', 
    'ValidationUtils',
    'MathUtils',
    'PlotUtils'
]
