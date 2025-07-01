"""
训练集生成控制器 - 管理GISAXS训练集的生成过程
"""

import os
import json
import time
import threading
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMessageBox, QFileDialog


class TrainsetController(QObject):
    """训练集生成控制器"""
    
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
        
        # 生成状态
        self.is_generating = False
        self.generation_thread = None
        self.stop_requested = False
        
        # 默认参数
        self.default_parameters = {
            'file_name': 'gisaxs_trainset',
            'save_path': '',
            'trainset_number': 1000,
            'save_every': 100,
            'batch_size': 10,
        }
        
        # 进度跟踪
        self.current_progress = 0
        self.total_samples = 0
        
        # 设置信号连接
        self._setup_connections()
    
    def _setup_connections(self):
        """设置信号连接"""
        # 文件名和路径
        self.ui.trainsetGenerateFileNameValue.textChanged.connect(self._emit_parameters_changed)
        
        # 训练集数量
        self.ui.trainsetGenerateTrainsetNumberValue.textChanged.connect(self._on_trainset_number_changed)
        
        # 保存间隔
        self.ui.trainsetGenerateSaveEveryValue.textChanged.connect(self._emit_parameters_changed)
        
        # 按钮
        self.ui.trainsetGenerateSelectFolderButton.clicked.connect(self._select_save_folder)
        self.ui.trainsetGenerateRunButton.clicked.connect(self._start_generation)
        self.ui.trainsetGenerateStopButton.clicked.connect(self._stop_generation)
        
        # 初始状态设置
        self.ui.trainsetGenerateStopButton.setEnabled(False)
    
    def initialize(self):
        """初始化训练集生成参数"""
        self.set_parameters(self.default_parameters)
        self._update_ui_state()
    
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
            self._emit_parameters_changed()
    
    def _on_trainset_number_changed(self):
        """训练集数量改变处理"""
        try:
            number = int(self.ui.trainsetGenerateTrainsetNumberValue.text())
            # 更新预计时间显示等
            self._update_generation_estimation()
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
        """获取当前训练集生成参数"""
        try:
            parameters = {
                'file_name': self.ui.trainsetGenerateFileNameValue.text(),
                'save_path': self.ui.trainsetGenerateSavePathValue.text(),
                'trainset_number': int(self.ui.trainsetGenerateTrainsetNumberValue.text()),
                'save_every': int(self.ui.trainsetGenerateSaveEveryValue.text()),
            }
            return parameters
        except (ValueError, AttributeError):
            return self.default_parameters.copy()
    
    def set_parameters(self, parameters):
        """设置训练集生成参数"""
        if 'file_name' in parameters:
            self.ui.trainsetGenerateFileNameValue.setText(parameters['file_name'])
        
        if 'save_path' in parameters:
            self.ui.trainsetGenerateSavePathValue.setText(parameters['save_path'])
        
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
        """重置为默认参数"""
        self.set_parameters(self.default_parameters)
    
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
