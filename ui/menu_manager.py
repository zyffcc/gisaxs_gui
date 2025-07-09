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
        print("✓ 菜单系统已初始化")
    
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
            
            print("✓ 参数菜单已创建")
            
        except Exception as e:
            print(f"创建参数菜单失败: {e}")
    
    def reset_parameters(self):
        """重置所有参数为初始默认值"""
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self.main_window, 
                '确认重置', 
                '确定要重置所有参数为初始默认值吗？\n这将覆盖您的当前设置。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 重置参数
                global_params.reset_to_initial_parameters()
                
                # 显示成功消息
                QMessageBox.information(
                    self.main_window, 
                    '重置完成', 
                    '所有参数已重置为初始默认值！'
                )
                
                print("✓ 用户手动重置参数完成")
            else:
                print("用户取消了参数重置")
                
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                '重置失败', 
                f'参数重置失败：{str(e)}'
            )
            print(f"参数重置失败: {e}")
    
    def save_parameters(self):
        """手动保存参数"""
        try:
            # 打开文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                '保存参数文件',
                'config/my_parameters.json',
                'JSON文件 (*.json);;所有文件 (*)'
            )
            
            if file_path:
                global_params.save_parameters(file_path)
                QMessageBox.information(
                    self.main_window, 
                    '保存成功', 
                    f'参数已保存到：{file_path}'
                )
                print(f"✓ 用户手动保存参数到: {file_path}")
            
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                '保存失败', 
                f'参数保存失败：{str(e)}'
            )
            print(f"参数保存失败: {e}")
    
    def load_parameters(self):
        """手动加载参数"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                '加载参数文件',
                'config/',
                'JSON文件 (*.json);;所有文件 (*)'
            )
            
            if file_path:
                global_params.load_parameters(file_path)
                QMessageBox.information(
                    self.main_window, 
                    '加载成功', 
                    f'参数已从文件加载：{file_path}'
                )
                print(f"✓ 用户手动加载参数从: {file_path}")
            
        except Exception as e:
            QMessageBox.warning(
                self.main_window, 
                '加载失败', 
                f'参数加载失败：{str(e)}'
            )
            print(f"参数加载失败: {e}")
