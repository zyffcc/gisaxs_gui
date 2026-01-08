"""
菜单管理器 - 负责创建和管理主窗口的菜单系统
"""

from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, QObject
from core.global_params import global_params


class MenuManager(QObject):
    """菜单管理器，负责创建和管理主窗口的菜单"""
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
    def setup_menus(self):
        """设置所有菜单"""
        self.create_parameters_menu()
        print("The menu system has been initialized.")
    
    def create_parameters_menu(self):
        """创建参数菜单"""
        try:
            # 获取或创建菜单栏
            menubar = self.main_window.menuBar()
            
            # 查找或创建Parameters菜单
            parameters_menu = None
            for action in menubar.actions():
                if action.text() == '参数(&P)' or action.text() == 'Parameters':
                    parameters_menu = action.menu()
                    break
            
            if parameters_menu is None:
                parameters_menu = menubar.addMenu('参数(&P)')
            
            # 添加Reset菜单项
            if not hasattr(self.main_window, 'actionReset'):
                self.main_window.actionReset = QAction('重置参数(&R)', self.main_window)
                self.main_window.actionReset.setShortcut('Ctrl+R')
                self.main_window.actionReset.setStatusTip('重置所有参数为初始默认值')
                self.main_window.actionReset.triggered.connect(self.reset_parameters)
                parameters_menu.addAction(self.main_window.actionReset)
            
            # 添加保存参数菜单项
            if not hasattr(self.main_window, 'actionSaveParams'):
                self.main_window.actionSaveParams = QAction('保存参数(&S)', self.main_window)
                self.main_window.actionSaveParams.setShortcut('Ctrl+S')
                self.main_window.actionSaveParams.setStatusTip('立即保存当前参数')
                self.main_window.actionSaveParams.triggered.connect(self.save_parameters)
                parameters_menu.addAction(self.main_window.actionSaveParams)
            
            # 添加加载参数菜单项
            if not hasattr(self.main_window, 'actionLoadParams'):
                self.main_window.actionLoadParams = QAction('加载参数(&L)', self.main_window)
                self.main_window.actionLoadParams.setShortcut('Ctrl+L')
                self.main_window.actionLoadParams.setStatusTip('从文件加载参数')
                self.main_window.actionLoadParams.triggered.connect(self.load_parameters)
                parameters_menu.addAction(self.main_window.actionLoadParams)
            
            print("✓ Parameter menu created")
            
        except Exception as e:
            print(f"Failed to create parameter menu: {e}")
    
    def reset_parameters(self):
        """重置所有参数为初始默认值"""
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self.main_window, 
                'Confirm Reset', 
                'Are you sure you want to reset all parameters to their initial default values?\nThis will overwrite your current settings.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 重置参数
                global_params.reset_to_initial_parameters()
                
                # 显示成功消息
                QMessageBox.information(
                    self.main_window, 
                    'Reset complete', 
                    'All parameters have been reset to their initial default values!'
                )
                
                print("✓ User manually reset parameters")
            else:
                print("User canceled parameter reset")
                
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                'Reset Failed', 
                f'Failed to reset parameters: {str(e)}'
            )
            print(f"Failed to reset parameters: {e}")
    
    def save_parameters(self):
        """手动保存参数"""
        try:
            # 打开文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                'Save Parameters File',
                'config/my_parameters.json',
                'JSON Files (*.json);;All Files (*)'
            )
            
            if file_path:
                global_params.save_parameters(file_path)
                QMessageBox.information(
                    self.main_window, 
                    'Saved', 
                    f'Parameters have been saved to: {file_path}'
                )
                print(f"✓ User manually saved parameters to: {file_path}")
            
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                'Save Failed', 
                f'Failed to save parameters: {str(e)}'
            )
            print(f"Failed to save parameters: {e}")
    
    def load_parameters(self):
        """手动加载参数"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                'Load Parameters File',
                'config/',
                'JSON Files (*.json);;All Files (*)'
            )
            
            if file_path:
                global_params.load_parameters(file_path)
                QMessageBox.information(
                    self.main_window, 
                    'Loaded', 
                    f'Parameters have been loaded from: {file_path}'
                )
                print(f"✓ User manually loaded parameters from: {file_path}")
            
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                'Load Failed', 
                f'Failed to load parameters: {str(e)}'
            )
            print(f"Failed to load parameters: {e}")
