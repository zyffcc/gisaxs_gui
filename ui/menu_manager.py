"""
Menu Manager - responsible for creating and managing the main window menu system
"""

from pathlib import Path
import tempfile

from PyQt5.QtGui import QDesktopServices, QTextDocument
from PyQt5.QtWidgets import (
    QMenuBar,
    QMenu,
    QAction,
    QMessageBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QObject, QUrl
from core.global_params import global_params
from ui.app_assets import app_colored_logo_pixmap, app_icon
from utils.path_utils import normalize_path


APP_VERSION = "GIMaP v0.0.2 Alpha (Pre-release)"
GITHUB_URL = "https://github.com/zyffcc/gisaxs_gui"


class MenuManager(QObject):
    """Menu manager, responsible for creating and managing the main window menus"""
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
    def setup_menus(self):
        """Set up all menus"""
        self.create_parameters_menu()
        self.create_tools_menu()
        self.create_help_menu()
        print("The menu system has been initialized.")

    def create_tools_menu(self):
        """Create independent analysis tools without changing the active page."""
        try:
            menubar = self.main_window.menuBar()
            tools_menu = None
            for action in menubar.actions():
                if action.text().replace("&", "") == "Tools":
                    tools_menu = action.menu()
                    break
            if tools_menu is None:
                tools_menu = menubar.addMenu("Tools (&T)")
            if not hasattr(self.main_window, "actionGeometryCalibration"):
                action = QAction("Geometry Calibration...", self.main_window)
                action.setShortcut("Ctrl+Shift+G")
                action.setStatusTip("Calibrate beam center and detector distance from a standard image")
                action.triggered.connect(self.open_geometry_calibration)
                tools_menu.addAction(action)
                self.main_window.actionGeometryCalibration = action
        except Exception as exc:
            print(f"Failed to create Tools menu: {exc}")

    def open_geometry_calibration(self):
        """Show one modeless calibration dialog and preserve the current page."""
        try:
            dialog = getattr(self, "_geometry_calibration_dialog", None)
            if dialog is None:
                from ui.geometry_calibration_dialog import GeometryCalibrationDialog
                dialog = GeometryCalibrationDialog(self.main_window)
                dialog.setAttribute(Qt.WA_DeleteOnClose, True)
                dialog.destroyed.connect(lambda: setattr(self, "_geometry_calibration_dialog", None))
                self._geometry_calibration_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Geometry Calibration", f"The calibration tool could not be opened:\n{exc}")
    
    def create_parameters_menu(self):
        """Create Parameters menu"""
        try:
            # 获取或创建菜单栏
            menubar = self.main_window.menuBar()
            
            # 查找或创建Parameters菜单
            parameters_menu = None
            for action in menubar.actions():
                if action.text() in ('参数(&P)', 'Parameters', 'Parameters (&P)'):
                    # Normalize menu text to English
                    if action.text() != 'Parameters (&P)':
                        action.setText('Parameters (&P)')
                    parameters_menu = action.menu()
                    break
            
            if parameters_menu is None:
                parameters_menu = menubar.addMenu('Parameters (&P)')
            
            # 添加Reset菜单项
            if not hasattr(self.main_window, 'actionReset'):
                self.main_window.actionReset = QAction('Reset Parameters (&R)', self.main_window)
                self.main_window.actionReset.setShortcut('Ctrl+R')
                self.main_window.actionReset.setStatusTip('Reset all parameters to their initial default values')
                self.main_window.actionReset.triggered.connect(self.reset_parameters)
                parameters_menu.addAction(self.main_window.actionReset)
            
            # 添加保存参数菜单项
            if not hasattr(self.main_window, 'actionSaveParams'):
                self.main_window.actionSaveParams = QAction('Save Parameters (&S)', self.main_window)
                self.main_window.actionSaveParams.setShortcut('Ctrl+S')
                self.main_window.actionSaveParams.setStatusTip('Save the current parameters immediately')
                self.main_window.actionSaveParams.triggered.connect(self.save_parameters)
                parameters_menu.addAction(self.main_window.actionSaveParams)
            
            # 添加加载参数菜单项
            if not hasattr(self.main_window, 'actionLoadParams'):
                self.main_window.actionLoadParams = QAction('Load Parameters (&L)', self.main_window)
                self.main_window.actionLoadParams.setShortcut('Ctrl+L')
                self.main_window.actionLoadParams.setStatusTip('Load parameters from a file')
                self.main_window.actionLoadParams.triggered.connect(self.load_parameters)
                parameters_menu.addAction(self.main_window.actionLoadParams)

            parameters_menu.addSeparator()

            if not hasattr(self.main_window, 'actionSaveFittingParams'):
                self.main_window.actionSaveFittingParams = QAction('Save Fitting Parameters...', self.main_window)
                self.main_window.actionSaveFittingParams.setStatusTip('Save only Cut/Fitting parameters, including particle model parameters')
                self.main_window.actionSaveFittingParams.triggered.connect(self.save_fitting_parameters)
                parameters_menu.addAction(self.main_window.actionSaveFittingParams)

            if not hasattr(self.main_window, 'actionLoadFittingParams'):
                self.main_window.actionLoadFittingParams = QAction('Load Fitting Parameters...', self.main_window)
                self.main_window.actionLoadFittingParams.setStatusTip('Load only Cut/Fitting parameters')
                self.main_window.actionLoadFittingParams.triggered.connect(self.load_fitting_parameters)
                parameters_menu.addAction(self.main_window.actionLoadFittingParams)

            if not hasattr(self.main_window, 'actionOpenAIFittingWorkspace'):
                self.main_window.actionOpenAIFittingWorkspace = QAction('Open AI Fitting Workspace...', self.main_window)
                self.main_window.actionOpenAIFittingWorkspace.setStatusTip('Open the detached AI fitting workspace')
                self.main_window.actionOpenAIFittingWorkspace.triggered.connect(self.open_ai_fitting_workspace)
                parameters_menu.addAction(self.main_window.actionOpenAIFittingWorkspace)
            
            print("✓ Parameter menu created")
            
        except Exception as e:
            print(f"Failed to create parameter menu: {e}")

    def create_help_menu(self):
        """Create Help menu with version and documentation links."""
        try:
            menubar = self.main_window.menuBar()

            help_menu = None
            for action in menubar.actions():
                if action.text() in ('Help', 'Help (&H)', '&Help'):
                    if action.text() != 'Help (&H)':
                        action.setText('Help (&H)')
                    help_menu = action.menu()
                    break

            if help_menu is None:
                help_menu = menubar.addMenu('Help (&H)')

            if not hasattr(self.main_window, 'actionOpenUserManual'):
                self.main_window.actionOpenUserManual = QAction('User Manual...', self.main_window)
                self.main_window.actionOpenUserManual.setIcon(app_icon())
                self.main_window.actionOpenUserManual.setStatusTip('Open the local GIMaP user manual')
                self.main_window.actionOpenUserManual.triggered.connect(self.open_user_manual)
                help_menu.addAction(self.main_window.actionOpenUserManual)

            if not hasattr(self.main_window, 'actionOpenGitHub'):
                self.main_window.actionOpenGitHub = QAction('GitHub Repository...', self.main_window)
                self.main_window.actionOpenGitHub.setStatusTip('Open the GIMaP GitHub repository')
                self.main_window.actionOpenGitHub.triggered.connect(self.open_github_repository)
                help_menu.addAction(self.main_window.actionOpenGitHub)

            help_menu.addSeparator()

            if not hasattr(self.main_window, 'actionAboutGIMaP'):
                self.main_window.actionAboutGIMaP = QAction('About GIMaP...', self.main_window)
                self.main_window.actionAboutGIMaP.setIcon(app_icon())
                self.main_window.actionAboutGIMaP.setStatusTip('Show GIMaP version and project information')
                self.main_window.actionAboutGIMaP.triggered.connect(self.show_about_dialog)
                help_menu.addAction(self.main_window.actionAboutGIMaP)

            print("Help menu created")

        except Exception as e:
            print(f"Failed to create help menu: {e}")
    
    def reset_parameters(self):
        """Reset all parameters to their initial default values"""
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
        """Manually save parameters"""
        try:
            # 打开文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                'Save Parameters File',
                'config/my_parameters.json',
                'JSON Files (*.json);;All Files (*)'
            )
            
            if file_path:
                file_path = normalize_path(file_path)
                main_controller = getattr(self.main_window, 'main_controller', None)
                if main_controller is not None and hasattr(main_controller, 'save_parameters_to_file'):
                    ok = main_controller.save_parameters_to_file(file_path)
                    if not ok:
                        raise RuntimeError("Main controller failed to save parameters")
                else:
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
        """Manually load parameters"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                'Load Parameters File',
                'config/',
                'JSON Files (*.json);;All Files (*)'
            )
            
            if file_path:
                file_path = normalize_path(file_path)
                main_controller = getattr(self.main_window, 'main_controller', None)
                if main_controller is not None and hasattr(main_controller, 'load_parameters_from_file'):
                    ok = main_controller.load_parameters_from_file(file_path)
                    if not ok:
                        raise RuntimeError("Main controller failed to load parameters")
                else:
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

    def _fitting_controller(self):
        main_controller = getattr(self.main_window, 'main_controller', None)
        return getattr(main_controller, 'fitting_controller', None)

    def save_fitting_parameters(self):
        controller = self._fitting_controller()
        if controller is None:
            QMessageBox.warning(self.main_window, 'Fitting Parameters', 'Fitting controller is not available yet.')
            return
        controller.save_fitting_parameters_dialog()

    def load_fitting_parameters(self):
        controller = self._fitting_controller()
        if controller is None:
            QMessageBox.warning(self.main_window, 'Fitting Parameters', 'Fitting controller is not available yet.')
            return
        controller.load_fitting_parameters_dialog()

    def open_ai_fitting_workspace(self):
        controller = self._fitting_controller()
        if controller is None:
            QMessageBox.warning(self.main_window, 'AI Fitting', 'Fitting controller is not available yet.')
            return
        controller.open_ai_fitting_workspace()

    def open_user_manual(self):
        """Open the local user manual in the system default browser."""
        manual_path = Path(__file__).resolve().parents[1] / 'docs' / 'User_Manual.md'
        if not manual_path.exists():
            QMessageBox.warning(
                self.main_window,
                'User Manual',
                f'User manual was not found:\n{manual_path}'
            )
            return

        try:
            manual_text = manual_path.read_text(encoding='utf-8')
            document = QTextDocument()
            if hasattr(document, 'setMarkdown'):
                document.setMarkdown(manual_text)
            else:
                document.setPlainText(manual_text)

            html_path = Path(tempfile.gettempdir()) / 'GIMaP_User_Manual.html'
            html_path.write_text(
                (
                    '<!doctype html>\n'
                    '<html><head><meta charset="utf-8">\n'
                    '<title>GIMaP User Manual</title>\n'
                    '<style>\n'
                    'body { max-width: 980px; margin: 32px auto; padding: 0 24px; '
                    'line-height: 1.58; color: #1f2933; }\n'
                    'code, pre { background: #f4f6f8; border-radius: 4px; }\n'
                    'code { padding: 1px 4px; }\n'
                    'pre { padding: 12px; overflow-x: auto; }\n'
                    'h1, h2, h3 { color: #102a43; }\n'
                    'a { color: #0b63ce; }\n'
                    '</style></head><body>\n'
                    f'{document.toHtml()}\n'
                    '</body></html>\n'
                ),
                encoding='utf-8',
            )
        except Exception as exc:
            QMessageBox.warning(
                self.main_window,
                'User Manual',
                f'Failed to prepare user manual:\n{exc}'
            )
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

    def open_github_repository(self):
        """Open the project repository in the default browser."""
        QDesktopServices.openUrl(QUrl(GITHUB_URL))

    def show_about_dialog(self):
        """Show version, repository, and documentation information."""
        dialog = QMessageBox(self.main_window)
        dialog.setWindowTitle('About GIMaP')
        dialog.setWindowIcon(app_icon())
        dialog.setTextFormat(Qt.RichText)
        logo = app_colored_logo_pixmap(96, 96)
        if not logo.isNull():
            dialog.setIconPixmap(logo)
        dialog.setText(
            (
                f'<b style="font-size: 18px;">GIMaP</b><br>'
                f'<span style="color: #475569;">{APP_VERSION}</span><br><br>'
                f'GIMaP is a desktop application for GISAXS/GIWAXS data '
                f'visualization, fitting, and machine-learning-assisted workflows.<br><br>'
                f'GitHub: <a href="{GITHUB_URL}">{GITHUB_URL}</a><br>'
                f'User Manual: docs/User_Manual.md'
            )
        )
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec_()
