"""Composition boundary for application menus and feature tool dialogs."""

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
from src.gimap.app.presentation.assets import app_colored_logo_pixmap, app_icon
from src.gimap.app.ports import SettingsRepository
from src.gimap.shared.file_paths import normalize_path


APP_VERSION = "GIMaP v0.0.2 Alpha (Pre-release)"
GITHUB_URL = "https://github.com/zyffcc/gisaxs_gui"


class MenuManager(QObject):
    """Menu manager, responsible for creating and managing the main window menus"""

    def __init__(
        self,
        main_window,
        *,
        settings: SettingsRepository,
        calibration_dialog_factory=None,
        format_converter_dialog_factory=None,
    ):
        super().__init__()
        self.main_window = main_window
        self.settings = settings
        self.calibration_dialog_factory = calibration_dialog_factory
        self.format_converter_dialog_factory = format_converter_dialog_factory

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
                action.setStatusTip(
                    "Calibrate beam center and detector distance from a standard image"
                )
                action.triggered.connect(self.open_geometry_calibration)
                tools_menu.addAction(action)
                self.main_window.actionGeometryCalibration = action

            tools_menu.addSeparator()
            if not hasattr(self.main_window, "actionFormatConverter"):
                action = QAction("Format Converter...", self.main_window)
                action.setShortcut("Ctrl+Shift+C")
                action.setStatusTip("Convert NXS, CBF, and TIFF detector images")
                action.triggered.connect(lambda: self.open_format_converter(include_current=True))
                tools_menu.addAction(action)
                self.main_window.actionFormatConverter = action

                shortcuts = tools_menu.addMenu("Format Converter shortcuts")
                current_action = shortcuts.addAction("Convert current file")
                current_action.setStatusTip(
                    "Add the currently opened file to a new conversion task"
                )
                current_action.triggered.connect(
                    lambda: self.open_format_converter(include_current=True)
                )
                open_action = shortcuts.addAction("Open converter...")
                open_action.setStatusTip("Open an empty format conversion task")
                open_action.triggered.connect(
                    lambda: self.open_format_converter(include_current=False)
                )
                self.main_window.actionConvertCurrentFile = current_action
                self.main_window.actionOpenFormatConverter = open_action
        except Exception as exc:
            print(f"Failed to create Tools menu: {exc}")

    def open_geometry_calibration(self):
        """Show one modeless calibration dialog and preserve the current page."""
        try:
            dialog = getattr(self, "_geometry_calibration_dialog", None)
            if dialog is None:
                if self.calibration_dialog_factory is None:
                    raise RuntimeError("Geometry calibration dialog is not configured")
                dialog = self.calibration_dialog_factory(self.main_window)
                dialog.setAttribute(Qt.WA_DeleteOnClose, True)
                dialog.destroyed.connect(
                    lambda: setattr(self, "_geometry_calibration_dialog", None)
                )
                self._geometry_calibration_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:
            QMessageBox.warning(
                self.main_window,
                "Geometry Calibration",
                f"The calibration tool could not be opened:\n{exc}",
            )

    def _current_detector_file(self) -> str:
        """Return the file belonging to the active image page when possible."""
        components = getattr(self.main_window, "components", None)
        waxs_page = getattr(components, "waxs_page", None)
        current_index = None
        try:
            current_index = self.main_window.mainWindowWidget.currentIndex()
        except Exception:
            pass
        waxs_index = getattr(self.main_window, "waxsPageIndex", None)
        if waxs_page is not None and current_index == waxs_index:
            path = getattr(waxs_page, "current_file", "")
            if path:
                return str(path)

        binding = self._fitting_binding()
        if binding is not None:
            path = binding.get_imported_file()
            if path:
                return str(path)
        if waxs_page is not None:
            return str(getattr(waxs_page, "current_file", "") or "")
        return ""

    def open_format_converter(self, *, include_current: bool = True):
        """Open one modeless converter; adding a current file never starts work."""
        try:
            dialog = getattr(self, "_format_converter_dialog", None)
            if dialog is None:
                current_file = self._current_detector_file() if include_current else ""
                if self.format_converter_dialog_factory is None:
                    raise RuntimeError("Format converter dialog is not configured")
                dialog = self.format_converter_dialog_factory(
                    self.main_window,
                    current_file=current_file,
                )
                dialog.destroyed.connect(lambda: setattr(self, "_format_converter_dialog", None))
                self._format_converter_dialog = dialog
            elif include_current:
                current_file = self._current_detector_file()
                if current_file:
                    dialog.current_file = current_file
                    dialog.current_button.setEnabled(True)
                    dialog.add_paths([current_file])
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:
            QMessageBox.warning(
                self.main_window, "Format Converter", f"The converter could not be opened:\n{exc}"
            )

    def create_parameters_menu(self):
        """Create Parameters menu"""
        try:
            # 获取或创建菜单栏
            menubar = self.main_window.menuBar()

            # 查找或创建Parameters菜单
            parameters_menu = None
            for action in menubar.actions():
                if action.text() in ("参数(&P)", "Parameters", "Parameters (&P)"):
                    # Normalize menu text to English
                    if action.text() != "Parameters (&P)":
                        action.setText("Parameters (&P)")
                    parameters_menu = action.menu()
                    break

            if parameters_menu is None:
                parameters_menu = menubar.addMenu("Parameters (&P)")

            # 添加Reset菜单项
            if not hasattr(self.main_window, "actionReset"):
                self.main_window.actionReset = QAction("Reset Parameters (&R)", self.main_window)
                self.main_window.actionReset.setShortcut("Ctrl+R")
                self.main_window.actionReset.setStatusTip(
                    "Reset all parameters to their initial default values"
                )
                self.main_window.actionReset.triggered.connect(self.reset_parameters)
                parameters_menu.addAction(self.main_window.actionReset)

            # 添加保存参数菜单项
            if not hasattr(self.main_window, "actionSaveParams"):
                self.main_window.actionSaveParams = QAction(
                    "Save Parameters (&S)", self.main_window
                )
                self.main_window.actionSaveParams.setShortcut("Ctrl+S")
                self.main_window.actionSaveParams.setStatusTip(
                    "Save the current parameters immediately"
                )
                self.main_window.actionSaveParams.triggered.connect(self.save_parameters)
                parameters_menu.addAction(self.main_window.actionSaveParams)

            # 添加加载参数菜单项
            if not hasattr(self.main_window, "actionLoadParams"):
                self.main_window.actionLoadParams = QAction(
                    "Load Parameters (&L)", self.main_window
                )
                self.main_window.actionLoadParams.setShortcut("Ctrl+L")
                self.main_window.actionLoadParams.setStatusTip("Load parameters from a file")
                self.main_window.actionLoadParams.triggered.connect(self.load_parameters)
                parameters_menu.addAction(self.main_window.actionLoadParams)

            parameters_menu.addSeparator()

            if not hasattr(self.main_window, "actionSaveFittingParams"):
                self.main_window.actionSaveFittingParams = QAction(
                    "Save Fitting Parameters...", self.main_window
                )
                self.main_window.actionSaveFittingParams.setStatusTip(
                    "Save only Cut/Fitting parameters, including particle model parameters"
                )
                self.main_window.actionSaveFittingParams.triggered.connect(
                    self.save_fitting_parameters
                )
                parameters_menu.addAction(self.main_window.actionSaveFittingParams)

            if not hasattr(self.main_window, "actionLoadFittingParams"):
                self.main_window.actionLoadFittingParams = QAction(
                    "Load Fitting Parameters...", self.main_window
                )
                self.main_window.actionLoadFittingParams.setStatusTip(
                    "Load only Cut/Fitting parameters"
                )
                self.main_window.actionLoadFittingParams.triggered.connect(
                    self.load_fitting_parameters
                )
                parameters_menu.addAction(self.main_window.actionLoadFittingParams)

            if not hasattr(self.main_window, "actionOpenAIFittingWorkspace"):
                self.main_window.actionOpenAIFittingWorkspace = QAction(
                    "Open AI Fitting Workspace...", self.main_window
                )
                self.main_window.actionOpenAIFittingWorkspace.setStatusTip(
                    "Open the detached AI fitting workspace"
                )
                self.main_window.actionOpenAIFittingWorkspace.triggered.connect(
                    self.open_ai_fitting_workspace
                )
                parameters_menu.addAction(self.main_window.actionOpenAIFittingWorkspace)

            print("Parameter menu created")

        except Exception as e:
            print(f"Failed to create parameter menu: {e}")

    def create_help_menu(self):
        """Create Help menu with version and documentation links."""
        try:
            menubar = self.main_window.menuBar()

            help_menu = None
            for action in menubar.actions():
                if action.text() in ("Help", "Help (&H)", "&Help"):
                    if action.text() != "Help (&H)":
                        action.setText("Help (&H)")
                    help_menu = action.menu()
                    break

            if help_menu is None:
                help_menu = menubar.addMenu("Help (&H)")

            if not hasattr(self.main_window, "actionOpenUserManual"):
                self.main_window.actionOpenUserManual = QAction("User Manual...", self.main_window)
                self.main_window.actionOpenUserManual.setIcon(app_icon())
                self.main_window.actionOpenUserManual.setStatusTip(
                    "Open the local GIMaP user manual"
                )
                self.main_window.actionOpenUserManual.triggered.connect(self.open_user_manual)
                help_menu.addAction(self.main_window.actionOpenUserManual)

            if not hasattr(self.main_window, "actionOpenGitHub"):
                self.main_window.actionOpenGitHub = QAction(
                    "GitHub Repository...", self.main_window
                )
                self.main_window.actionOpenGitHub.setStatusTip("Open the GIMaP GitHub repository")
                self.main_window.actionOpenGitHub.triggered.connect(self.open_github_repository)
                help_menu.addAction(self.main_window.actionOpenGitHub)

            help_menu.addSeparator()

            if not hasattr(self.main_window, "actionAboutGIMaP"):
                self.main_window.actionAboutGIMaP = QAction("About GIMaP...", self.main_window)
                self.main_window.actionAboutGIMaP.setIcon(app_icon())
                self.main_window.actionAboutGIMaP.setStatusTip(
                    "Show GIMaP version and project information"
                )
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
                "Confirm Reset",
                "Are you sure you want to reset all parameters to their initial default values?\nThis will overwrite your current settings.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                # 重置参数
                self.settings.reset()

                # 显示成功消息
                QMessageBox.information(
                    self.main_window,
                    "Reset complete",
                    "All parameters have been reset to their initial default values!",
                )

                print("User manually reset parameters")
            else:
                print("User canceled parameter reset")

        except Exception as e:
            QMessageBox.warning(
                self.main_window, "Reset Failed", f"Failed to reset parameters: {str(e)}"
            )
            print(f"Failed to reset parameters: {e}")

    def save_parameters(self):
        """Manually save parameters"""
        try:
            # 打开文件保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Parameters File",
                "config/my_parameters.json",
                "JSON Files (*.json);;All Files (*)",
            )

            if file_path:
                file_path = normalize_path(file_path)
                runtime = getattr(self.main_window, "runtime", None)
                if runtime is not None and hasattr(
                    runtime, "save_parameters_to_file"
                ):
                    ok = runtime.save_parameters_to_file(file_path)
                    if not ok:
                        raise RuntimeError("Application runtime failed to save parameters")
                else:
                    raise RuntimeError("Application runtime is not available")
                QMessageBox.information(
                    self.main_window, "Saved", f"Parameters have been saved to: {file_path}"
                )
                print(f"User manually saved parameters to: {file_path}")

        except Exception as e:
            QMessageBox.warning(
                self.main_window, "Save Failed", f"Failed to save parameters: {str(e)}"
            )
            print(f"Failed to save parameters: {e}")

    def load_parameters(self):
        """Manually load parameters"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Load Parameters File",
                "config/",
                "JSON Files (*.json);;All Files (*)",
            )

            if file_path:
                file_path = normalize_path(file_path)
                runtime = getattr(self.main_window, "runtime", None)
                if runtime is not None and hasattr(
                    runtime, "load_parameters_from_file"
                ):
                    ok = runtime.load_parameters_from_file(file_path)
                    if not ok:
                        raise RuntimeError("Application runtime failed to load parameters")
                else:
                    raise RuntimeError("Application runtime is not available")
                QMessageBox.information(
                    self.main_window, "Loaded", f"Parameters have been loaded from: {file_path}"
                )
                print(f"User manually loaded parameters from: {file_path}")

        except Exception as e:
            QMessageBox.warning(
                self.main_window, "Load Failed", f"Failed to load parameters: {str(e)}"
            )
            print(f"Failed to load parameters: {e}")

    def _fitting_binding(self):
        runtime = getattr(self.main_window, "runtime", None)
        return getattr(runtime, "fitting", None)

    def save_fitting_parameters(self):
        binding = self._fitting_binding()
        if binding is None:
            QMessageBox.warning(
                self.main_window, "Fitting Parameters", "Fitting workspace is not available yet."
            )
            return
        binding.save_fitting_parameters_dialog()

    def load_fitting_parameters(self):
        binding = self._fitting_binding()
        if binding is None:
            QMessageBox.warning(
                self.main_window, "Fitting Parameters", "Fitting workspace is not available yet."
            )
            return
        binding.load_fitting_parameters_dialog()

    def open_ai_fitting_workspace(self):
        binding = self._fitting_binding()
        if binding is None:
            QMessageBox.warning(
                self.main_window, "AI Fitting", "Fitting workspace is not available yet."
            )
            return
        binding.open_ai_fitting_workspace()

    def open_user_manual(self):
        """Open the local user manual in the system default browser."""
        manual_path = Path(__file__).resolve().parents[1] / "docs" / "User_Manual.md"
        if not manual_path.exists():
            QMessageBox.warning(
                self.main_window, "User Manual", f"User manual was not found:\n{manual_path}"
            )
            return

        try:
            manual_text = manual_path.read_text(encoding="utf-8")
            document = QTextDocument()
            if hasattr(document, "setMarkdown"):
                document.setMarkdown(manual_text)
            else:
                document.setPlainText(manual_text)

            html_path = Path(tempfile.gettempdir()) / "GIMaP_User_Manual.html"
            html_path.write_text(
                (
                    "<!doctype html>\n"
                    '<html><head><meta charset="utf-8">\n'
                    "<title>GIMaP User Manual</title>\n"
                    "<style>\n"
                    "body { max-width: 980px; margin: 32px auto; padding: 0 24px; "
                    "line-height: 1.58; color: #1f2933; }\n"
                    "code, pre { background: #f4f6f8; border-radius: 4px; }\n"
                    "code { padding: 1px 4px; }\n"
                    "pre { padding: 12px; overflow-x: auto; }\n"
                    "h1, h2, h3 { color: #102a43; }\n"
                    "a { color: #0b63ce; }\n"
                    "</style></head><body>\n"
                    f"{document.toHtml()}\n"
                    "</body></html>\n"
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.warning(
                self.main_window, "User Manual", f"Failed to prepare user manual:\n{exc}"
            )
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

    def open_github_repository(self):
        """Open the project repository in the default browser."""
        QDesktopServices.openUrl(QUrl(GITHUB_URL))

    def show_about_dialog(self):
        """Show version, repository, and documentation information."""
        dialog = QMessageBox(self.main_window)
        dialog.setWindowTitle("About GIMaP")
        dialog.setWindowIcon(app_icon())
        dialog.setTextFormat(Qt.RichText)
        logo = app_colored_logo_pixmap(96, 96)
        if not logo.isNull():
            dialog.setIconPixmap(logo)
        dialog.setText(
            (
                f'<b style="font-size: 18px;">GIMaP</b><br>'
                f'<span style="color: #475569;">{APP_VERSION}</span><br><br>'
                f"GIMaP is a desktop application for GISAXS/GIWAXS data "
                f"visualization, fitting, and machine-learning-assisted workflows.<br><br>"
                f'GitHub: <a href="{GITHUB_URL}">{GITHUB_URL}</a><br>'
                f"User Manual: docs/User_Manual.md"
            )
        )
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec_()
