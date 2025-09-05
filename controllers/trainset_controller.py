"""
训练集生成控制器 - 管理GISAXS训练集的生成过程，包含所有相关的子模块参数
"""

import os
import json
import time
import threading
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMessageBox, QFileDialog

# 导入全局参数管理器
from core.global_params import GlobalParameterManager


class TrainsetController(QObject):
    """训练集生成控制器，包含光束、探测器、样品和预处理参数管理"""
    
    # 信号定义
    parameters_changed = pyqtSignal(str, dict)
    generation_started = pyqtSignal()
    generation_finished = pyqtSignal()
    generation_error = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 获取全局参数管理器实例
        self.global_params = GlobalParameterManager()
        
        # 训练集生成相关参数
        self.generation_params = {
            'save_path': '',
            'filename': 'trainset',
            'total_number': 1000,
            'save_interval': 100,
            'current_index': 0
        }
        
        # 子模块参数
        self.beam_params = self._init_beam_params()
        self.detector_params = self._init_detector_params()
        self.sample_params = self._init_sample_params()
        self.preprocessing_params = self._init_preprocessing_params()
        
        # 从全局参数管理器同步参数到控制器
        self._sync_from_global_params()
        
        # 初始化探测器预设
        self._load_detector_presets()
        
        # 生成控制
        self.is_generating = False
        self.generation_thread = None
        self.generation_timer = QTimer()
        self.generation_timer.timeout.connect(self._update_generation_progress)
        
        # 设置连接
        self._setup_connections()
    
    def _init_beam_params(self):
        """初始化光束参数"""
        return {
            'wavelength': 0.1,  # nm
            'grazing_angle': 0.4,  # degrees
        }
        
    def _init_detector_params(self):
        """初始化探测器参数"""
        # 从Trainset模块专用参数读取当前值，而不是使用硬编码默认值
        return {
            'preset': self.global_params.get_parameter('trainset', 'detector.preset', 'Pilatus 2M'),
            'distance': self.global_params.get_parameter('trainset', 'detector.distance', 2000),  # mm
            'nbins_x': self.global_params.get_parameter('trainset', 'detector.nbins_x', 1475),
            'nbins_y': self.global_params.get_parameter('trainset', 'detector.nbins_y', 1475),
            'pixel_size_x': self.global_params.get_parameter('trainset', 'detector.pixel_size_x', 172),  # μm
            'pixel_size_y': self.global_params.get_parameter('trainset', 'detector.pixel_size_y', 172),  # μm
            'beam_center_x': self.global_params.get_parameter('trainset', 'detector.beam_center_x', 737),  # bin
            'beam_center_y': self.global_params.get_parameter('trainset', 'detector.beam_center_y', 737),  # bin
        }
        
    def _init_sample_params(self):
        """初始化样品参数"""
        return {
            'particle_shape': 'Sphere',
            'particle_size': 10.0,  # nm
            'size_distribution': 0.1,
            'material': 'Gold',
            'substrate': 'Silicon'
        }
        
    def _init_preprocessing_params(self):
        """初始化预处理参数"""
        return {
            'focus_region': {
                'type': 'q',
                'qr_min': 0.01,
                'qr_max': 3.0,
                'qz_min': 0.01,
                'qz_max': 3.0,
            },
            'noising': {
                'type': 'Gaussian',
                'snr_min': 80,
                'snr_max': 130,
            },
            'others': {
                'crop_edge': True,
                'add_mask': True,
                'normalize': True,
                'logarization': True,
            }
        }

    def _setup_connections(self):
        """设置信号连接"""
        # 训练集生成相关连接
        if hasattr(self.ui, 'trainsetGenerateFileNameValue'):
            self.ui.trainsetGenerateFileNameValue.textChanged.connect(self._on_generation_params_changed)
        
        if hasattr(self.ui, 'trainsetGenerateTrainsetNumberValue'):
            self.ui.trainsetGenerateTrainsetNumberValue.textChanged.connect(self._on_trainset_number_changed)
        
        if hasattr(self.ui, 'trainsetGenerateSaveEveryValue'):
            self.ui.trainsetGenerateSaveEveryValue.textChanged.connect(self._on_generation_params_changed)
        
        # 按钮连接
        if hasattr(self.ui, 'trainsetGenerateSelectFolderButton'):
            self.ui.trainsetGenerateSelectFolderButton.clicked.connect(self._select_save_folder)
        if hasattr(self.ui, 'trainsetGenerateRunButton'):
            self.ui.trainsetGenerateRunButton.clicked.connect(self._start_generation)
        if hasattr(self.ui, 'trainsetGenerateStopButton'):
            self.ui.trainsetGenerateStopButton.clicked.connect(self._stop_generation)
            self.ui.trainsetGenerateStopButton.setEnabled(False)
            
        # 光束参数连接
        self._setup_beam_connections()
        
        # 探测器参数连接
        self._setup_detector_connections()
        
        # 样品参数连接
        self._setup_sample_connections()
        
        # 预处理参数连接
        self._setup_preprocessing_connections()
        
    def _setup_beam_connections(self):
        """设置光束参数连接"""
        if hasattr(self.ui, 'wavelengthValue'):
            self.ui.wavelengthValue.textChanged.connect(self._on_beam_params_changed)
        if hasattr(self.ui, 'angleValue'):
            self.ui.angleValue.textChanged.connect(self._on_beam_params_changed)
            
    def _setup_detector_connections(self):
        """设置探测器参数连接"""
        if hasattr(self.ui, 'detectorPresetCombox'):
            self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_detector_preset_changed)
        
        # 非关键参数 - 不会触发自动切换到User-defined
        if hasattr(self.ui, 'distanceValue'):
            self.ui.distanceValue.textChanged.connect(self._on_detector_params_changed)
        if hasattr(self.ui, 'beamCenterXValue'):
            self.ui.beamCenterXValue.textChanged.connect(self._on_detector_params_changed)
        if hasattr(self.ui, 'beamCenterYValue'):
            self.ui.beamCenterYValue.textChanged.connect(self._on_detector_params_changed)
        
        # 关键参数 - 会触发自动切换到User-defined的逻辑
        if hasattr(self.ui, 'NbinsXValue'):
            self.ui.NbinsXValue.textChanged.connect(self._on_detector_critical_params_changed)
        if hasattr(self.ui, 'NbinsYValue'):
            self.ui.NbinsYValue.textChanged.connect(self._on_detector_critical_params_changed)
        if hasattr(self.ui, 'pixelSizeXValue'):
            self.ui.pixelSizeXValue.textChanged.connect(self._on_detector_critical_params_changed)
        if hasattr(self.ui, 'pixelSizeYValue'):
            self.ui.pixelSizeYValue.textChanged.connect(self._on_detector_critical_params_changed)
            
    def _setup_sample_connections(self):
        """设置样品参数连接"""
        if hasattr(self.ui, 'particleShapeInitValue'):
            self.ui.particleShapeInitValue.currentTextChanged.connect(self._on_sample_params_changed)
        if hasattr(self.ui, 'particleSizeValue'):
            self.ui.particleSizeValue.textChanged.connect(self._on_sample_params_changed)
        if hasattr(self.ui, 'materialCombox'):
            self.ui.materialCombox.currentTextChanged.connect(self._on_sample_params_changed)
            
    def _setup_preprocessing_connections(self):
        """设置预处理参数连接"""
        if hasattr(self.ui, 'focusRegionTypeCombox'):
            self.ui.focusRegionTypeCombox.currentTextChanged.connect(self._on_preprocessing_params_changed)
        if hasattr(self.ui, 'noisingTypeCombox'):
            self.ui.noisingTypeCombox.currentTextChanged.connect(self._on_preprocessing_params_changed)
        if hasattr(self.ui, 'OthersCropEdgeCheckBox'):
            self.ui.OthersCropEdgeCheckBox.toggled.connect(self._on_preprocessing_params_changed)
    
    def _sync_from_global_params(self):
        """从全局参数管理器同步参数到控制器"""
        try:
            # 同步光束参数
            beam_params = self.global_params.get_module_parameters('beam')
            if beam_params:
                self.beam_params.update(beam_params)
            
            # 同步探测器参数
            detector_params = self.global_params.get_module_parameters('detector')
            if detector_params:
                self.detector_params.update(detector_params)
            
            # 同步样品参数
            sample_params = self.global_params.get_module_parameters('sample')
            if sample_params:
                self.sample_params.update(sample_params)
            
            # 同步预处理参数
            preprocessing_params = self.global_params.get_module_parameters('preprocessing')
            if preprocessing_params:
                self.preprocessing_params.update(preprocessing_params)
                
            # 更新UI显示
            self._update_ui_from_params()
            
        except Exception as e:
            print(f"从全局参数管理器同步参数失败: {e}")
    
    def _update_ui_from_params(self):
        """根据当前参数更新UI显示"""
        try:
            # 更新光束参数UI
            if hasattr(self.ui, 'wavelengthValue'):
                self.ui.wavelengthValue.setText(str(self.beam_params.get('wavelength', 0.1)))
            if hasattr(self.ui, 'angleValue'):
                self.ui.angleValue.setText(str(self.beam_params.get('grazing_angle', 0.4)))
            
            # 更新探测器参数UI
            if hasattr(self.ui, 'detectorDistanceValue'):
                self.ui.detectorDistanceValue.setText(str(self.detector_params.get('distance', 2000)))
            if hasattr(self.ui, 'detectorPixelXValue'):
                self.ui.detectorPixelXValue.setText(str(self.detector_params.get('pixel_size_x', 172)))
            if hasattr(self.ui, 'detectorPixelYValue'):
                self.ui.detectorPixelYValue.setText(str(self.detector_params.get('pixel_size_y', 172)))
            
            # 更新样品参数UI
            if hasattr(self.ui, 'particleShapeInitValue'):
                particle_shape = self.sample_params.get('particle_shape', 'Sphere')
                index = self.ui.particleShapeInitValue.findText(particle_shape)
                if index >= 0:
                    self.ui.particleShapeInitValue.setCurrentIndex(index)
                # 切换到对应的页面
                self._switch_particle_page(particle_shape)
                    
            if hasattr(self.ui, 'particleSizeValue'):
                self.ui.particleSizeValue.setText(str(self.sample_params.get('particle_size', 10.0)))
                
            if hasattr(self.ui, 'materialCombox'):
                material = self.sample_params.get('material', 'Gold')
                index = self.ui.materialCombox.findText(material)
                if index >= 0:
                    self.ui.materialCombox.setCurrentIndex(index)
            
            # 更新预处理参数UI
            if hasattr(self.ui, 'focusRegionTypeCombox'):
                focus_type = self.preprocessing_params.get('focus_region', {}).get('type', 'q')
                index = self.ui.focusRegionTypeCombox.findText(focus_type)
                if index >= 0:
                    self.ui.focusRegionTypeCombox.setCurrentIndex(index)
                    
            if hasattr(self.ui, 'noisingTypeCombox'):
                noising_type = self.preprocessing_params.get('noising', {}).get('type', 'Gaussian')
                index = self.ui.noisingTypeCombox.findText(noising_type)
                if index >= 0:
                    self.ui.noisingTypeCombox.setCurrentIndex(index)
                    
            if hasattr(self.ui, 'OthersCropEdgeCheckBox'):
                crop_edge = self.preprocessing_params.get('others', {}).get('crop_edge', True)
                self.ui.OthersCropEdgeCheckBox.setChecked(crop_edge)
                
        except Exception as e:
            print(f"更新UI显示失败: {e}")

    def initialize(self):
        """初始化训练集生成参数"""
        # 设置默认参数
        default_params = {
            'generation': {
                'file_name': 'trainset',
                'save_path': '',
                'trainset_number': 1000,
                'save_every': 100
            },
            'beam': self.beam_params,
            'detector': self.detector_params,
            'sample': self.sample_params,
            'preprocessing': self.preprocessing_params
        }
        self.set_parameters(default_params)
        self._update_ui_state()
        
        # 初始化时设置默认粒子形状页面
        if hasattr(self.ui, 'particleShapeInitValue'):
            default_shape = self.ui.particleShapeInitValue.currentText() or 'Sphere'
            self._switch_particle_page(default_shape)
    
    def _select_save_folder(self):
        """选择保存文件夹"""
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        
        folder_path = QFileDialog.getExistingDirectory(
            main_window,
            "选择训练集保存文件夹",
            os.path.expanduser("~")
        )
        
        if folder_path:
            self.ui.trainsetGenerateSavePathValue.setText(folder_path)
            # 同步到全局参数管理器
            self.global_params.set_parameter('trainset', 'save_path', folder_path)
            self._emit_parameters_changed()
    
    def _on_trainset_number_changed(self):
        """训练集数量改变处理"""
        try:
            number = int(self.ui.trainsetGenerateTrainsetNumberValue.text())
            # 同步到全局参数管理器
            self.global_params.set_parameter('trainset', 'trainset_number', number)
            # 更新预计时间显示等
            self._update_generation_estimation()
        except ValueError:
            pass
        
        self._emit_parameters_changed()
    
    def _on_generation_params_changed(self):
        """训练集生成参数改变处理"""
        try:
            # 同步文件名
            if hasattr(self.ui, 'trainsetGenerateFileNameValue'):
                filename = self.ui.trainsetGenerateFileNameValue.text()
                self.global_params.set_parameter('trainset', 'file_name', filename)
            
            # 同步保存路径
            if hasattr(self.ui, 'trainsetGenerateSavePathValue'):
                save_path = self.ui.trainsetGenerateSavePathValue.text()
                self.global_params.set_parameter('trainset', 'save_path', save_path)
            
            # 同步保存间隔
            if hasattr(self.ui, 'trainsetGenerateSaveEveryValue'):
                save_every = int(self.ui.trainsetGenerateSaveEveryValue.text())
                self.global_params.set_parameter('trainset', 'save_every', save_every)
                
        except ValueError:
            pass
        
        self._emit_parameters_changed()
    
    def _update_generation_estimation(self):
        """更新生成时间估算"""
        try:
            trainset_number = int(self.ui.trainsetGenerateTrainsetNumberValue.text())
            # 假设每个样本需要0.1秒生成
            estimated_time = trainset_number * 0.1
            
            # 更新界面显示（如果有相关的标签的话）
            if estimated_time < 60:
                time_str = f"预计时间: {estimated_time:.1f} 秒"
            else:
                time_str = f"预计时间: {estimated_time/60:.1f} 分钟"
                
            self.status_updated.emit(time_str)
            
        except ValueError:
            pass
    
    def _start_generation(self):
        """开始训练集生成"""
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        
        # 验证参数
        is_valid, error_message = self.validate_parameters()
        if not is_valid:
            QMessageBox.warning(main_window, "参数错误", error_message)
            return
        
        # 获取所有参数
        all_parameters = self._get_all_system_parameters()
        if not all_parameters:
            QMessageBox.warning(main_window, "参数错误", "无法获取系统参数，请检查各模块设置")
            return
        
        # 确认开始生成
        reply = QMessageBox.question(
            main_window,
            "确认生成",
            f"即将生成 {self.ui.trainsetGenerateTrainsetNumberValue.text()} 个训练样本。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 开始生成
        self.is_generating = True
        self.stop_requested = False
        self.current_progress = 0
        
        # 更新UI状态
        self._update_ui_state()
        
        # 启动生成线程
        self.generation_thread = threading.Thread(target=self._generation_worker, args=(all_parameters,))
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        self.generation_started.emit()
    
    def _stop_generation(self):
        """停止训练集生成"""
        if self.is_generating:
            self.stop_requested = True
            self.status_updated.emit("正在停止生成...")
    
    def _generation_worker(self, parameters):
        """训练集生成工作线程"""
        try:
            trainset_params = self.get_parameters()
            trainset_number = trainset_params['trainset_number']
            save_every = trainset_params['save_every']
            batch_size = trainset_params.get('batch_size', 10)
            
            self.total_samples = trainset_number
            
            # 创建保存目录
            save_path = self._create_save_directory(trainset_params)
            
            # 保存参数配置
            self._save_generation_config(save_path, parameters)
            
            # 生成样本
            generated_count = 0
            batch_data = []
            
            for i in range(trainset_number):
                if self.stop_requested:
                    break
                
                # 生成单个样本
                sample_data = self._generate_single_sample(parameters, i)
                batch_data.append(sample_data)
                generated_count += 1
                
                # 批量保存
                if len(batch_data) >= batch_size or generated_count % save_every == 0:
                    self._save_batch_data(save_path, batch_data, generated_count)
                    batch_data = []
                
                # 更新进度
                progress = int((generated_count / trainset_number) * 100)
                if progress != self.current_progress:
                    self.current_progress = progress
                    self.progress_updated.emit(progress)
                
                # 状态更新
                if generated_count % 10 == 0:
                    self.status_updated.emit(f"已生成 {generated_count}/{trainset_number} 个样本")
                
                # 模拟计算时间
                time.sleep(0.01)
            
            # 保存剩余数据
            if batch_data:
                self._save_batch_data(save_path, batch_data, generated_count)
            
            # 生成完成
            self._on_generation_completed(generated_count, trainset_number)
            
        except Exception as e:
            self.generation_error.emit(f"生成过程中发生错误: {str(e)}")
        finally:
            self.is_generating = False
            self._update_ui_state()
    
    def _generate_single_sample(self, parameters, sample_index):
        """生成单个训练样本"""
        # 这里实现实际的GISAXS模拟计算
        # 根据参数生成散射图案
        
        # 示例：生成随机数据（实际应该是物理模拟）
        sample_data = {
            'index': sample_index,
            'timestamp': datetime.now().isoformat(),
            'parameters': self._randomize_parameters(parameters),
            'scattering_pattern': self._simulate_scattering_pattern(parameters),
        }
        
        return sample_data
    
    def _randomize_parameters(self, base_parameters):
        """随机化参数（在指定范围内）"""
        # 实现参数随机化逻辑
        randomized = base_parameters.copy()
        
        # 示例：随机化一些参数
        if 'sample' in randomized and 'sphere' in randomized['sample']:
            sphere_params = randomized['sample']['sphere']
            r_min = sphere_params.get('r_min', 0)
            r_max = sphere_params.get('r_max', 20)
            
            # 在范围内随机选择半径
            randomized['sample']['sphere']['r_current'] = np.random.uniform(r_min, r_max)
        
        return randomized
    
    def _simulate_scattering_pattern(self, parameters):
        """模拟散射图案"""
        # 这里应该实现实际的GISAXS物理模拟
        # 暂时返回随机数据作为示例
        
        detector_params = parameters.get('detector', {})
        nbins_x = detector_params.get('nbins_x', 100)
        nbins_y = detector_params.get('nbins_y', 100)
        
        # 生成模拟散射图案
        pattern = np.random.exponential(1.0, (nbins_y, nbins_x))
        
        return pattern.tolist()  # 转换为可序列化的格式
    
    def _create_save_directory(self, trainset_params):
        """创建保存目录"""
        base_path = trainset_params['save_path']
        file_name = trainset_params['file_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = os.path.join(base_path, f"{file_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir
    
    def _save_generation_config(self, save_path, parameters):
        """保存生成配置"""
        config_file = os.path.join(save_path, "generation_config.json")
        
        config_data = {
            'generation_time': datetime.now().isoformat(),
            'parameters': parameters,
            'software_version': '1.0.0',  # 可以从配置文件读取
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
    
    def _save_batch_data(self, save_path, batch_data, current_count):
        """保存批量数据"""
        batch_file = os.path.join(save_path, f"batch_{current_count:06d}.json")
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        # 创建索引文件
        self._update_index_file(save_path, batch_file, len(batch_data))
    
    def _update_index_file(self, save_path, batch_file, batch_size):
        """更新索引文件"""
        index_file = os.path.join(save_path, "trainset_index.json")
        
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            index_data = {
                'total_samples': 0,
                'batch_files': [],
                'creation_time': datetime.now().isoformat()
            }
        
        index_data['batch_files'].append({
            'file': os.path.basename(batch_file),
            'samples': batch_size,
            'timestamp': datetime.now().isoformat()
        })
        index_data['total_samples'] += batch_size
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=4, ensure_ascii=False)
    
    def _on_generation_completed(self, generated_count, total_requested):
        """生成完成处理"""
        if self.stop_requested:
            self.status_updated.emit(f"生成已停止，共生成 {generated_count} 个样本")
        else:
            self.status_updated.emit(f"生成完成！共生成 {generated_count} 个样本")
        
        self.progress_updated.emit(100)
        self.generation_finished.emit()
    
    def _get_all_system_parameters(self):
        """获取所有系统参数"""
        try:
            # 从主控制器获取所有参数
            if hasattr(self.parent, 'get_all_parameters'):
                return self.parent.get_all_parameters()
            else:
                # 如果没有主控制器，则返回空参数
                return {}
        except Exception as e:
            print(f"获取系统参数错误: {e}")
            return {}
    
    def get_parameters(self):
        """获取所有参数（包含所有子模块）"""
        try:
            generation_params = {
                'file_name': self.ui.trainsetGenerateFileNameValue.text() if hasattr(self.ui, 'trainsetGenerateFileNameValue') else '',
                'save_path': self.ui.trainsetGenerateSavePathValue.text() if hasattr(self.ui, 'trainsetGenerateSavePathValue') else '',
                'trainset_number': int(self.ui.trainsetGenerateTrainsetNumberValue.text()) if hasattr(self.ui, 'trainsetGenerateTrainsetNumberValue') else 1000,
                'save_every': int(self.ui.trainsetGenerateSaveEveryValue.text()) if hasattr(self.ui, 'trainsetGenerateSaveEveryValue') else 100,
            }
        except (ValueError, AttributeError):
            generation_params = self.generation_params.copy()
            
        return {
            'generation': generation_params,
            'beam': self.beam_params.copy(),
            'detector': self.detector_params.copy(),
            'sample': self.sample_params.copy(),
            'preprocessing': self.preprocessing_params.copy()
        }
    
    def set_parameters(self, parameters):
        """设置所有参数（包含所有子模块）"""
        # 设置训练集生成参数
        if 'generation' in parameters:
            gen_params = parameters['generation']
            if 'file_name' in gen_params and hasattr(self.ui, 'trainsetGenerateFileNameValue'):
                self.ui.trainsetGenerateFileNameValue.setText(gen_params['file_name'])
            if 'save_path' in gen_params and hasattr(self.ui, 'trainsetGenerateSavePathValue'):
                self.ui.trainsetGenerateSavePathValue.setText(gen_params['save_path'])
            if 'trainset_number' in gen_params and hasattr(self.ui, 'trainsetGenerateTrainsetNumberValue'):
                self.ui.trainsetGenerateTrainsetNumberValue.setText(str(gen_params['trainset_number']))
            if 'save_every' in gen_params and hasattr(self.ui, 'trainsetGenerateSaveEveryValue'):
                self.ui.trainsetGenerateSaveEveryValue.setText(str(gen_params['save_every']))
                
        # 设置光束参数
        if 'beam' in parameters:
            self.beam_params.update(parameters['beam'])
            if hasattr(self.ui, 'wavelengthValue'):
                self.ui.wavelengthValue.setText(str(self.beam_params['wavelength']))
            if hasattr(self.ui, 'angleValue'):
                self.ui.angleValue.setText(str(self.beam_params['grazing_angle']))
                
        # 设置探测器参数
        if 'detector' in parameters:
            self.detector_params.update(parameters['detector'])
            self._update_detector_ui()
            
        # 设置样品参数
        if 'sample' in parameters:
            self.sample_params.update(parameters['sample'])
            if hasattr(self.ui, 'particleShapeInitValue'):
                self.ui.particleShapeInitValue.setCurrentText(self.sample_params['particle_shape'])
            if hasattr(self.ui, 'particleSizeValue'):
                self.ui.particleSizeValue.setText(str(self.sample_params['particle_size']))
                
        # 设置预处理参数
        if 'preprocessing' in parameters:
            self.preprocessing_params.update(parameters['preprocessing'])
            if hasattr(self.ui, 'focusRegionTypeCombox'):
                self.ui.focusRegionTypeCombox.setCurrentText(self.preprocessing_params['focus_region']['type'])
            if hasattr(self.ui, 'noisingTypeCombox'):
                self.ui.noisingTypeCombox.setCurrentText(self.preprocessing_params['noising']['type'])
        
        if 'trainset_number' in parameters:
            self.ui.trainsetGenerateTrainsetNumberValue.setText(str(parameters['trainset_number']))
        
        if 'save_every' in parameters:
            self.ui.trainsetGenerateSaveEveryValue.setText(str(parameters['save_every']))
        
        self._emit_parameters_changed()
    
    def validate_parameters(self):
        """验证训练集生成参数"""
        try:
            params = self.get_parameters()
            
            # 验证文件名
            file_name = params.get('file_name', '').strip()
            if not file_name:
                return False, "文件名不能为空"
            
            # 验证保存路径
            save_path = params.get('save_path', '').strip()
            if not save_path:
                return False, "保存路径不能为空"
            
            if not os.path.exists(save_path):
                return False, "保存路径不存在"
            
            if not os.access(save_path, os.W_OK):
                return False, "保存路径没有写入权限"
            
            # 验证训练集数量
            trainset_number = params.get('trainset_number', 0)
            if trainset_number <= 0 or trainset_number > 1000000:
                return False, "训练集数量必须在1-1000000范围内"
            
            # 验证保存间隔
            save_every = params.get('save_every', 0)
            if save_every <= 0 or save_every > trainset_number:
                return False, "保存间隔必须大于0且不超过训练集总数"
            
            return True, "训练集生成参数有效"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def reset_to_defaults(self):
        """重置所有参数到默认值"""
        # 重置子模块参数
        self.beam_params = self._init_beam_params()
        self.detector_params = self._init_detector_params()
        self.sample_params = self._init_sample_params()
        self.preprocessing_params = self._init_preprocessing_params()
        
        # 重置训练集生成参数
        default_generation_params = {
            'generation': {
                'file_name': 'trainset',
                'save_path': '',
                'trainset_number': 1000,
                'save_every': 100
            },
            'beam': self.beam_params,
            'detector': self.detector_params,
            'sample': self.sample_params,
            'preprocessing': self.preprocessing_params
        }
        
        self.set_parameters(default_generation_params)
        self.status_updated.emit("训练集参数已重置为默认值")
    
    def _update_ui_state(self):
        """更新UI状态"""
        self.ui.trainsetGenerateRunButton.setEnabled(not self.is_generating)
        self.ui.trainsetGenerateStopButton.setEnabled(self.is_generating)
        
        # 禁用/启用参数输入
        input_widgets = [
            self.ui.trainsetGenerateFileNameValue,
            self.ui.trainsetGenerateSavePathValue,
            self.ui.trainsetGenerateTrainsetNumberValue,
            self.ui.trainsetGenerateSaveEveryValue,
            self.ui.trainsetGenerateSelectFolderButton,
        ]
        
        for widget in input_widgets:
            widget.setEnabled(not self.is_generating)
    
    def _update_generation_progress(self):
        """更新生成进度（定时器回调）"""
        if self.is_generating and hasattr(self, 'current_progress'):
            # 这个方法会被定时器调用，用于定期更新UI
            # 实际的进度更新是在生成线程中完成的
            pass
    
    def _emit_parameters_changed(self):
        """发出参数改变信号"""
        parameters = self.get_parameters()
        self.parameters_changed.emit("训练集参数", parameters)
    
    def get_generation_status(self):
        """获取生成状态信息"""
        return {
            'is_generating': self.is_generating,
            'current_progress': self.current_progress,
            'total_samples': self.total_samples,
            'stop_requested': self.stop_requested,
        }
    
    def _on_beam_params_changed(self):
        """光束参数改变处理"""
        try:
            if hasattr(self.ui, 'wavelengthValue'):
                self.beam_params['wavelength'] = float(self.ui.wavelengthValue.text())
                # 同步到全局参数管理器
                self.global_params.set_parameter('beam', 'wavelength', self.beam_params['wavelength'])
                
            if hasattr(self.ui, 'angleValue'):
                self.beam_params['grazing_angle'] = float(self.ui.angleValue.text())
                # 同步到全局参数管理器
                self.global_params.set_parameter('beam', 'grazing_angle', self.beam_params['grazing_angle'])
                
            self._emit_parameters_changed()
        except ValueError:
            pass
    
    def _load_detector_presets(self):
        """从配置文件加载探测器预设并更新ComboBox"""
        try:
            import os
            detector_config_file = os.path.join('config', 'detectors.json')
            
            if os.path.exists(detector_config_file):
                with open(detector_config_file, 'r', encoding='utf-8') as f:
                    self.detector_presets = json.load(f)
                print(f"✓ 已加载探测器配置: {len(self.detector_presets)} 个预设")
            else:
                # 如果文件不存在，使用默认配置
                self.detector_presets = {
                    "Pilatus 2M": {
                        "nbins_x": 1475, "nbins_y": 1679,
                        "pixel_size_x": 172, "pixel_size_y": 172,
                        "beam_center_x": 737, "beam_center_y": 839
                    },
                    "Pilatus 1M": {
                        "nbins_x": 981, "nbins_y": 1043,
                        "pixel_size_x": 172, "pixel_size_y": 172,
                        "beam_center_x": 490, "beam_center_y": 521
                    }
                }
                print("⚠ 探测器配置文件不存在，使用默认配置")
            
            # 更新ComboBox
            self._update_detector_preset_combobox()
            
        except Exception as e:
            print(f"加载探测器配置失败: {e}")
            self.detector_presets = {}
    
    def _update_detector_preset_combobox(self):
        """更新探测器预设ComboBox"""
        if not hasattr(self.ui, 'detectorPresetCombox'):
            return
            
        try:
            # 暂时断开信号（安全方式）
            try:
                self.ui.detectorPresetCombox.currentTextChanged.disconnect()
            except TypeError:
                # 如果没有连接信号，会抛出TypeError，这是正常的
                pass
            
            # 清空现有选项
            self.ui.detectorPresetCombox.clear()
            
            # 添加从配置文件加载的预设
            for preset_name in self.detector_presets.keys():
                self.ui.detectorPresetCombox.addItem(preset_name)
                print(f"  添加预设: {preset_name}")
            
            # 最后添加User-defined选项
            self.ui.detectorPresetCombox.addItem("User-defined")
            print(f"  添加选项: User-defined")
            
            # 设置默认选择第一个预设
            if self.detector_presets:
                self.ui.detectorPresetCombox.setCurrentIndex(0)
                first_preset = list(self.detector_presets.keys())[0]
                print(f"✓ 默认探测器预设: {first_preset}")
                
                # 加载第一个预设的参数
                self._load_preset_parameters(first_preset)
            
            # 重新连接信号
            self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_detector_preset_changed)
            
        except Exception as e:
            print(f"更新探测器预设ComboBox失败: {e}")
            # 重新连接信号（确保不会丢失连接）
            try:
                self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_detector_preset_changed)
            except:
                pass
            
    def _on_detector_preset_changed(self, preset_name):
        """探测器预设改变处理"""
        try:
            if preset_name == 'User-defined':
                # 用户自定义模式，不更改当前参数值，只更新预设标记
                self.detector_params['preset'] = preset_name
                self.global_params.set_parameter('trainset.detector', 'preset', preset_name)
                print("✓ 切换到用户自定义模式")
                return
            
            # 检查是否是有效预设
            if preset_name in self.detector_presets:
                preset_config = self.detector_presets[preset_name]
                
                # 更新detector_params（合并所有参数）
                self.detector_params.update(preset_config)
                self.detector_params['preset'] = preset_name
                
                # 更新UI显示（会自动断开和重连信号）
                self._update_detector_ui()
                
                # 同步到全局参数管理器
                self.global_params.set_parameter('trainset.detector', 'preset', preset_name)
                for key, value in preset_config.items():
                    # 跳过描述字段
                    if key != 'description':
                        self.global_params.set_parameter('trainset.detector', key, value)
                
                self._emit_parameters_changed()
                print(f"✓ 探测器预设已切换到: {preset_name}")
            else:
                print(f"⚠ 未知的探测器预设: {preset_name}")
            
        except Exception as e:
            print(f"探测器预设切换失败: {e}")
            
    def _on_detector_params_changed(self):
        """探测器非关键参数改变处理（距离、光束中心）"""
        try:
            if hasattr(self.ui, 'distanceValue'):
                self.detector_params['distance'] = float(self.ui.distanceValue.text())
                # 同步到Trainset模块专用参数
                self.global_params.set_parameter('trainset', 'detector.distance', self.detector_params['distance'])
                
            if hasattr(self.ui, 'beamCenterXValue'):
                self.detector_params['beam_center_x'] = float(self.ui.beamCenterXValue.text())
                # 同步到Trainset模块专用参数
                self.global_params.set_parameter('trainset', 'detector.beam_center_x', self.detector_params['beam_center_x'])
                
            if hasattr(self.ui, 'beamCenterYValue'):
                self.detector_params['beam_center_y'] = float(self.ui.beamCenterYValue.text())
                # 同步到Trainset模块专用参数
                self.global_params.set_parameter('trainset', 'detector.beam_center_y', self.detector_params['beam_center_y'])
                
            self._emit_parameters_changed()
        except ValueError:
            pass
    
    def _on_detector_critical_params_changed(self):
        """探测器关键参数改变处理（Nbins和像素大小），会触发自动切换到User-defined"""
        try:
            # 更新参数值
            if hasattr(self.ui, 'NbinsXValue'):
                self.detector_params['nbins_x'] = int(self.ui.NbinsXValue.text())
                self.global_params.set_parameter('trainset.detector', 'nbins_x', self.detector_params['nbins_x'])
                
            if hasattr(self.ui, 'NbinsYValue'):
                self.detector_params['nbins_y'] = int(self.ui.NbinsYValue.text())
                self.global_params.set_parameter('trainset.detector', 'nbins_y', self.detector_params['nbins_y'])
                
            if hasattr(self.ui, 'pixelSizeXValue'):
                self.detector_params['pixel_size_x'] = float(self.ui.pixelSizeXValue.text())
                self.global_params.set_parameter('trainset.detector', 'pixel_size_x', self.detector_params['pixel_size_x'])
                
            if hasattr(self.ui, 'pixelSizeYValue'):
                self.detector_params['pixel_size_y'] = float(self.ui.pixelSizeYValue.text())
                self.global_params.set_parameter('trainset.detector', 'pixel_size_y', self.detector_params['pixel_size_y'])
            
            # 检查是否需要切换到User-defined
            self._check_and_switch_to_user_defined()
                
            self._emit_parameters_changed()
        except ValueError:
            pass
    
    def _check_and_switch_to_user_defined(self):
        """检查当前参数是否与预设匹配，如果不匹配则切换到User-defined"""
        current_nbins_x = self.detector_params.get('nbins_x')
        current_nbins_y = self.detector_params.get('nbins_y')
        current_pixel_x = self.detector_params.get('pixel_size_x')
        current_pixel_y = self.detector_params.get('pixel_size_y')
        
        # 获取当前预设
        current_preset = self.detector_params.get('preset', '')
        
        # 如果已经是User-defined，不需要检查
        if current_preset == 'User-defined':
            return
        
        # 检查当前参数是否与任何预设完全匹配
        for preset_name, preset_config in self.detector_presets.items():
            if (current_nbins_x == preset_config.get('nbins_x') and
                current_nbins_y == preset_config.get('nbins_y') and
                current_pixel_x == preset_config.get('pixel_size_x') and
                current_pixel_y == preset_config.get('pixel_size_y')):
                
                # 匹配某个预设，如果当前预设不是这个，则切换
                if current_preset != preset_name:
                    self._switch_to_preset(preset_name)
                return
        
        # 不匹配任何预设，切换到User-defined
        self._switch_to_user_defined()
    
    def _on_detector_nbins_changed(self):
        """Nbins参数改变处理，自动切换到User-defined模式"""
        try:
            # 获取当前Nbins值
            current_nbins_x = None
            current_nbins_y = None
            
            if hasattr(self.ui, 'NbinsXValue'):
                current_nbins_x = int(self.ui.NbinsXValue.text())
                self.detector_params['nbins_x'] = current_nbins_x
                self.global_params.set_parameter('trainset.detector', 'nbins_x', current_nbins_x)
                
            if hasattr(self.ui, 'NbinsYValue'):
                current_nbins_y = int(self.ui.NbinsYValue.text())
                self.detector_params['nbins_y'] = current_nbins_y
                self.global_params.set_parameter('trainset.detector', 'nbins_y', current_nbins_y)
            
            # 检查当前的Nbins值是否与预设值匹配
            if self._should_switch_to_user_defined(current_nbins_x, current_nbins_y):
                self._switch_to_user_defined()
                
            self._emit_parameters_changed()
        except ValueError:
            pass
    
    def _should_switch_to_user_defined(self, nbins_x, nbins_y):
        """检查是否应该切换到User-defined模式（基于Nbins值）"""
        # 获取当前预设
        current_preset = self.detector_params.get('preset', '')
        
        # 如果已经是User-defined，不需要切换
        if current_preset == 'User-defined':
            return False
            
        # 检查当前Nbins值是否与任何预设匹配
        for preset_name, preset_config in self.detector_presets.items():
            if (nbins_x == preset_config.get('nbins_x') and 
                nbins_y == preset_config.get('nbins_y')):
                # 如果匹配，但当前预设不是这个，则切换预设
                if current_preset != preset_name:
                    self._switch_to_preset(preset_name)
                return False
        
        # 如果不匹配任何预设，需要切换到User-defined
        return True
    
    def _switch_to_user_defined(self):
        """切换到User-defined模式"""
        if hasattr(self.ui, 'detectorPresetCombox'):
            # 暂时断开信号连接，避免循环触发
            self.ui.detectorPresetCombox.currentTextChanged.disconnect()
            
            # 设置为User-defined
            index = self.ui.detectorPresetCombox.findText('User-defined')
            if index >= 0:
                self.ui.detectorPresetCombox.setCurrentIndex(index)
                self.detector_params['preset'] = 'User-defined'
                self.global_params.set_parameter('trainset.detector', 'preset', 'User-defined')
                print("✓ 由于Nbins值不匹配预设，自动切换到User-defined模式")
            
            # 重新连接信号
            self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_detector_preset_changed)
    
    def _switch_to_preset(self, preset_name):
        """切换到指定的预设"""
        if hasattr(self.ui, 'detectorPresetCombox'):
            # 暂时断开信号连接，避免循环触发
            self.ui.detectorPresetCombox.currentTextChanged.disconnect()
            
            # 设置预设
            index = self.ui.detectorPresetCombox.findText(preset_name)
            if index >= 0:
                self.ui.detectorPresetCombox.setCurrentIndex(index)
                self.detector_params['preset'] = preset_name
                self.global_params.set_parameter('trainset.detector', 'preset', preset_name)
                print(f"✓ 由于Nbins值匹配，自动切换到预设: {preset_name}")
            
            # 重新连接信号
            self.ui.detectorPresetCombox.currentTextChanged.connect(self._on_detector_preset_changed)
            
    def _on_sample_params_changed(self):
        """样品参数改变处理"""
        try:
            if hasattr(self.ui, 'particleShapeInitValue'):
                self.sample_params['particle_shape'] = self.ui.particleShapeInitValue.currentText()
                # 同步到全局参数管理器
                self.global_params.set_parameter('sample', 'particle_shape', self.sample_params['particle_shape'])
                
                # 根据粒子形状切换页面
                self._switch_particle_page(self.sample_params['particle_shape'])
                
            if hasattr(self.ui, 'particleSizeValue'):
                self.sample_params['particle_size'] = float(self.ui.particleSizeValue.text())
                # 同步到全局参数管理器
                self.global_params.set_parameter('sample', 'particle_size', self.sample_params['particle_size'])
                
            if hasattr(self.ui, 'materialCombox'):
                self.sample_params['material'] = self.ui.materialCombox.currentText()
                # 同步到全局参数管理器
                self.global_params.set_parameter('sample', 'material', self.sample_params['material'])
                
            self._emit_parameters_changed()
        except ValueError:
            pass
    
    def _switch_particle_page(self, particle_shape):
        """根据粒子形状切换对应的页面"""
        if not hasattr(self.ui, 'sampleParametersParticleStackedWidget'):
            return
            
        try:
            # 定义形状与页面索引的映射
            shape_to_index = {
                'Sphere': 0,
                'Ellipsoid': 1,
                'Cylinder': 0,  # 如果没有Cylinder页面，暂时使用Sphere页面
                'None': 0
            }
            
            # 获取对应的页面索引
            page_index = shape_to_index.get(particle_shape, 0)
            
            # 切换到对应页面
            self.ui.sampleParametersParticleStackedWidget.setCurrentIndex(page_index)
            
            print(f"✓ 切换粒子形状页面: {particle_shape} -> 页面索引 {page_index}")
            
        except Exception as e:
            print(f"切换粒子形状页面失败: {e}")
            
    def _on_preprocessing_params_changed(self):
        """预处理参数改变处理"""
        if hasattr(self.ui, 'focusRegionTypeCombox'):
            self.preprocessing_params['focus_region']['type'] = self.ui.focusRegionTypeCombox.currentText()
            # 同步到全局参数管理器
            self.global_params.set_parameter('preprocessing', 'focus_region', self.preprocessing_params['focus_region'])
            
        if hasattr(self.ui, 'noisingTypeCombox'):
            self.preprocessing_params['noising']['type'] = self.ui.noisingTypeCombox.currentText()
            # 同步到全局参数管理器
            self.global_params.set_parameter('preprocessing', 'noising', self.preprocessing_params['noising'])
            
        if hasattr(self.ui, 'OthersCropEdgeCheckBox'):
            self.preprocessing_params['others']['crop_edge'] = self.ui.OthersCropEdgeCheckBox.isChecked()
            # 同步到全局参数管理器
            self.global_params.set_parameter('preprocessing', 'others', self.preprocessing_params['others'])
            
        self._emit_parameters_changed()
        
    def _update_detector_ui(self):
        """更新探测器UI显示，暂时断开信号连接避免循环触发"""
        try:
            # 断开关键参数的信号连接
            critical_widgets = []
            if hasattr(self.ui, 'NbinsXValue'):
                self.ui.NbinsXValue.textChanged.disconnect()
                critical_widgets.append(('NbinsXValue', self._on_detector_critical_params_changed))
            if hasattr(self.ui, 'NbinsYValue'):
                self.ui.NbinsYValue.textChanged.disconnect()
                critical_widgets.append(('NbinsYValue', self._on_detector_critical_params_changed))
            if hasattr(self.ui, 'pixelSizeXValue'):
                self.ui.pixelSizeXValue.textChanged.disconnect()
                critical_widgets.append(('pixelSizeXValue', self._on_detector_critical_params_changed))
            if hasattr(self.ui, 'pixelSizeYValue'):
                self.ui.pixelSizeYValue.textChanged.disconnect()
                critical_widgets.append(('pixelSizeYValue', self._on_detector_critical_params_changed))
            
            # 更新UI控件值
            if hasattr(self.ui, 'NbinsXValue'):
                self.ui.NbinsXValue.setText(str(self.detector_params.get('nbins_x', 1475)))
            if hasattr(self.ui, 'NbinsYValue'):
                self.ui.NbinsYValue.setText(str(self.detector_params.get('nbins_y', 1679)))
            if hasattr(self.ui, 'pixelSizeXValue'):
                self.ui.pixelSizeXValue.setText(str(self.detector_params.get('pixel_size_x', 172)))
            if hasattr(self.ui, 'pixelSizeYValue'):
                self.ui.pixelSizeYValue.setText(str(self.detector_params.get('pixel_size_y', 172)))
            if hasattr(self.ui, 'distanceValue'):
                self.ui.distanceValue.setText(str(self.detector_params.get('distance', 2000)))
            if hasattr(self.ui, 'beamCenterXValue'):
                self.ui.beamCenterXValue.setText(str(self.detector_params.get('beam_center_x', 737)))
            if hasattr(self.ui, 'beamCenterYValue'):
                self.ui.beamCenterYValue.setText(str(self.detector_params.get('beam_center_y', 839)))
            
            # 重新连接关键参数的信号
            for widget_name, handler in critical_widgets:
                widget = getattr(self.ui, widget_name)
                widget.textChanged.connect(handler)
                
        except Exception as e:
            print(f"更新探测器UI失败: {e}")
            # 确保信号连接不会丢失
            self._setup_detector_connections()
    
    def _update_generation_progress(self):
        """更新生成进度（定时器回调）"""
        if self.is_generating and hasattr(self, 'current_progress'):
            # 这个方法会被定时器调用，用于定期更新UI
            # 实际的进度更新是在生成线程中完成的
            pass
