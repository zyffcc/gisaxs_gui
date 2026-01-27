import sys
import cv2
import os
import numpy as np
import tempfile
"""Ensure Matplotlib uses a Qt-compatible backend when running standalone.
This avoids WXAgg conflicts if MPLBACKEND is set in the environment."""
try:
    import matplotlib as _mpl
    _mpl.use('Qt5Agg', force=True)
except Exception:
    pass
from scipy.interpolate import make_interp_spline
import glob
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, \
    QLineEdit, QVBoxLayout, QSizePolicy, QGridLayout, QWidget, QRadioButton, QButtonGroup, \
    QFileSystemModel, QTreeView, QHBoxLayout, QSplitter, QDesktopWidget, QMessageBox, QComboBox, \
    QFrame, QCheckBox, QProgressBar, QMenu, QMenuBar, QAction, QTextEdit, QDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QTransform, QMovie
from PyQt5.QtCore import QSize, Qt, QRect, QPoint, QDir, QTimer, QCoreApplication, QEventLoop,\
    QSettings, QThread, pyqtSignal, QResource
from matplotlib import cm
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
import h5py
from pathlib import Path
import re
import json
try:
    import fabio
except ImportError:
    fabio = None

def load_image_matrix(
    file_path,
    frame_idx: int = 0,
    dataset_path: str = "/entry/instrument/detector/data",
    dist_path: str = "/entry/instrument/detector/translation/distance",
    mask_path: str = "/entry/instrument/detector/pixel_mask",
    lmbda_x: int = 516,
    lmbda_y: int = 1556,
):
    """
    Load a 2D intensity matrix from either an image file (tif/tiff/png/jpg/etc.)
    or an P03-style NXs module file (and its m-series siblings), returning the
    final stitched image matrix as float32 with NaNs for empty regions.

    - Image files: uses cv2 to read (any depth), returns the raw matrix.
    - NXs files: detects the exact m-series prefix based on `file_path`, reads
      each module first/selected frame, applies P03 translation, composes onto
      a canvas, and returns the transposed + vertically flipped matrix
      (`grid_show`) consistent with typical GISAXS display.

    Parameters
    - file_path: str or Path, path to image or .nxs file
    - frame_idx: int, which frame to read if dataset is 3D (default: 0)
    - dataset_path, dist_path, mask_path: HDF5 dataset paths
    - lmbda_x, lmbda_y: expected module dimensions (x=516, y=1556)

    Returns
    - np.ndarray (float32), 2D matrix. For NXs, NaNs indicate padding/empty.
    """

    p = Path(file_path)
    ext = p.suffix.lower()

    # --- Branch: image file ---
    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    if ext in image_exts:
        im = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH)
        if im is None:
            # fallback
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError(f"Failed to read image file: {p}")
        # Ensure 2D grayscale for downstream processing
        if im.ndim == 3 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        return im.astype(np.float32)

    # --- Branch: ctxti text file ---
    if ext == ".ctxti":
        arr = np.loadtxt(str(p))
        if arr.ndim == 1:
            arr = np.atleast_2d(arr)
        return arr.astype(np.float32)

    # --- Branch: EDF file ---
    if ext == ".edf":
        if fabio is None:
            raise ImportError("fabio is required to read .edf files. Install with 'pip install fabio'.")
        try:
            edf = fabio.open(str(p))
            im = edf.data
        except Exception as e:
            raise ValueError(f"Failed to read EDF file: {p}: {e}")
        if im is None:
            raise ValueError(f"Empty EDF data in: {p}")
        # Ensure float32
        return np.asarray(im, dtype=np.float32)

    # --- Branch: CBF file ---
    if ext == ".cbf":
        if fabio is None:
            raise ImportError("fabio is required to read .cbf files. Install with 'pip install fabio'.")
        try:
            cbf = fabio.open(str(p))
            im = cbf.data
        except Exception as e:
            raise ValueError(f"Failed to read CBF file: {p}: {e}")
        if im is None:
            raise ValueError(f"Empty CBF data in: {p}")
        return np.asarray(im, dtype=np.float32)

    # --- Branch: NXs file ---
    if ext != ".nxs":
        raise ValueError(f"Unsupported file type: {p.suffix}. Only image, .ctxti, .edf or .nxs supported.")

    base_name = p.name
    folder = p.parent

    # Detect suffix like "_m09.nxs" or "m09.nxs" and derive exact prefix before the m-part
    m_match = re.search(r"_m(\d+)\.nxs$", base_name, re.IGNORECASE)
    sep = ""
    prefix = None
    if m_match:
        sep = "_"
        prefix = base_name[:m_match.start()]  # before "_mNN.nxs"
    else:
        m2 = re.search(r"m(\d+)\.nxs$", base_name, re.IGNORECASE)
        if m2:
            sep = ""
            prefix = base_name[:m2.start()]   # before "mNN.nxs"

    ordered_paths = []
    if prefix is not None:
        glob_pat = f"{prefix}{sep}m*.nxs"
        series = []
        for fp in folder.glob(glob_pat):
            mm = re.search(r"m(\d+)\.nxs$", fp.name, re.IGNORECASE)
            if mm:
                series.append((int(mm.group(1)), fp))
        if series:
            series.sort(key=lambda x: x[0])
            ordered_paths = [pp for _, pp in series]

    # Fallback: no series detected -> use the single file
    if not ordered_paths:
        ordered_paths = [p]

    # Read modules
    modules = []  # each: {img, trans_x, trans_y, mask}
    for mp in ordered_paths:
        with h5py.File(str(mp), "r") as f:
            dset = f[dataset_path]
            # 3D: (frames, H, W) or 2D: (H, W)
            if dset.ndim == 3:
                img = dset[frame_idx].astype(np.float32)
            elif dset.ndim == 2:
                img = dset[()].astype(np.float32)
            else:
                raise ValueError(f"Unexpected dataset ndim {dset.ndim} in {mp.name}")

            # translations (P03 style): trans_x = dist[1], trans_y = dist[0]
            if dist_path in f:
                dist = list(f[dist_path])
                trans_x = int(dist[1])
                trans_y = int(dist[0])
            else:
                trans_x = 0
                trans_y = 0

            # optional mask
            msk = f[mask_path][()] if mask_path in f else None

        # Orient image to (x,y) = (516,1556)
        if img.shape == (lmbda_y, lmbda_x):
            img_xy = img.T  # (1556,516) -> (516,1556)
            msk_xy = msk.T if msk is not None and msk.shape == img.shape else msk
        elif img.shape == (lmbda_x, lmbda_y):
            img_xy = img
            msk_xy = msk
        else:
            raise ValueError(
                f"{mp.name}: unexpected module shape {img.shape}. "
                f"Expected {(lmbda_y, lmbda_x)} or {(lmbda_x, lmbda_y)}."
            )

        modules.append(dict(img=img_xy, trans_x=trans_x, trans_y=trans_y, mask=msk_xy))

    # Shift to non-negative coordinates if any negative translations
    min_tx = min(m["trans_x"] for m in modules)
    min_ty = min(m["trans_y"] for m in modules)
    shift_x = -min_tx if min_tx < 0 else 0
    shift_y = -min_ty if min_ty < 0 else 0
    for m in modules:
        m["sx"] = m["trans_x"] + shift_x
        m["sy"] = m["trans_y"] + shift_y

    # Canvas size (x-first): pos = module_size + trans
    size_x = 0
    size_y = 0
    for m in modules:
        pos_x = lmbda_x + m["sx"]
        pos_y = lmbda_y + m["sy"]
        size_x = int(pos_x) if pos_x > size_x else int(size_x)
        size_y = int(pos_y) if pos_y > size_y else int(size_y)

    # Compose
    grid = np.full((size_x, size_y), np.nan, dtype=np.float32)
    for m in modules:
        x0, y0 = int(m["sx"]), int(m["sy"])
        grid[x0:x0 + lmbda_x, y0:y0 + lmbda_y] = m["img"]

    # Display-friendly orientation: transpose then flip vertically
    grid_show = grid.T  # (size_y, size_x)
    grid_show = np.flipud(grid_show)

    return grid_show

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.image_layout = ImageLayout(self)
        self.image_widget = ImageWidget(self)
        self.parameter = Parameter(self, image_widget=self.image_widget)
        # self.parameter = Parameter(self)
        self.image_layout = ImageLayout(self, parameter=self.parameter, image_widget = self.image_widget)
        self.image_widget.set_image_layout(self.image_layout, self.parameter)
        self.parameter.set_image_layout(self.image_layout)
        self.dirtree = FileExplorer(self.image_layout)

        # 创建BatchProcessor实例对象
        self.batch_processor = BatchProcessor(self.image_widget,self.image_layout)
        self.image_widget.set_batch_processor(self.batch_processor)
        self.image_layout.set_batch_processor(self.batch_processor)

        # 设置左侧部件和右侧部件
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.dirtree)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_layout, 6)
        # 添加第一条水平方向的横线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line1)
        right_layout.addWidget(self.parameter, 1)
        # 添加第二条水平方向的横线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line2)
        right_layout.addWidget(self.batch_processor,1)
        right_widget.setLayout(right_layout)

        # 创建QSplitter对象并添加left_widget和right_widget
        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # 设置初始大小比例为1:4
        splitter.setSizes([2,4])
        # splitter.setStretchFactor(0, 3)  # 设置左部件的拉伸因子为1
        # splitter.setStretchFactor(1, 4)  # 设置右部件的拉伸因子为4

        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dirtree.setMinimumWidth(300)
        self.image_layout.setMinimumWidth(400)

        # 设置整体布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

        # 设置主窗口的标题（英文）
        self.setWindowTitle('In-situ Data Processing')

        # 创建菜单项（英文）
        help_action = QAction('User Guide', self)
        help_action.triggered.connect(self.show_help)
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)

        # 创建菜单（英文）
        help_menu = QMenu('Help', self)
        help_menu.addAction(help_action)
        about_menu = QMenu('About', self)
        about_menu.addAction(about_action)

        # 创建菜单栏并添加菜单
        menu_bar = QMenuBar(self)
        menu_bar.addMenu(help_menu)
        menu_bar.addMenu(about_menu)

        # 将菜单栏设置为窗口的菜单栏
        self.setMenuBar(menu_bar)

        self.readme_content = """
                User Guide

                This tool provides 2D scattering image processing, 1D integration, and in-situ batch processing.

            1) 2D Image Processing
                - Import images by double-clicking .tif on the left, using the file chooser, or drag-and-drop.
                - Adjust Colorbar_min and Colorbar_max. Default colormap is jet.
                - Coordinate display: default is pixel. Use "Cut" to switch to Qxy/Qz.
                - Export: choose an output folder, then click "Export JPG".
                - Cut view range: set Qr_min, Qr_max, Qz_min, Qz_max. Set -121 for no limit.
                - Mask bad pixels/gaps: use Mask_min and Mask_max to zero out values outside the range.

            2) Integrate 2D to 1D
                - Import image as above (original or cut view).
                - Experiment parameters: incidence angle (°), center-X/Y (pixel), distance (mm), pixel-X/Y (µm), wavelength (Å).
          Parameters can be calibrated via Fit2D (http://ftp.esrf.eu/pub/expg/FIT2D/).
                - Select ROI: click the ROI button and pick four points (start angle, end angle, inner radius, outer radius), or input manually.
                - Integration method: radial (q) or azimuthal (angle). Step size default: 500.
                - Axes options: default log-q (Å⁻¹) with smoothing; also supports pixel, 2Theta (Cu Kα 1.54 Å), unsmoothed, and linear Y.
                - Export 1D: choose folder, then export as txt.

            3) In-situ Batch Processing
                - Choose the in-situ folder.
                - Filename pattern, e.g., Cl*.tif for Cl0001.tif, Cl0002.tif, ...
                - Batch export: 2D images, 1D curves, and background subtraction.
                    Adjust Colorbar and Mask; choose cut or original.
                - Set experiment parameters and ROI (same as section 2).
                - Background subtraction parameters: base image index; X-range start/end.
                - Export results: outputs include output and output_subbg. First column is X; subsequent columns are intensities.
                    Preview with the in-situ heatmap; Origin heatmap module recommended.
                - Import processed in-situ: load output.txt to preview heatmap.

                Changelog
                - Remembers inputs after closing.
                - Added pause for in-situ processing.
                - Added flip button (vertical flip); note: not applied to ROI selection image.
                - Optimized background subtraction parameters.
                - Heatmap preview uses interpolation to improve display with fewer points.
                - Added support for image formats beyond tif/jpg.
                """


    def show_help(self):
        # 显示帮助信息
        help_dialog = HelpDialog(self.readme_content, self)
        help_dialog.exec_()

    def show_about(self):
        QMessageBox.about(self, 'About', 'Copyright (c) Yufeng Zhai'
                                         '\nVersion v1.0'
                                         '\nDate 2023-05-03')

    def closeEvent(self, event):
        # Save current settings
        settings = QSettings('mycompany', 'myapp')
        settings.setValue('Angle_incidence', self.parameter.Angle_incidence.text())
        settings.setValue('x_Center', self.parameter.x_Center.text())
        settings.setValue('y_Center', self.parameter.y_Center.text())
        settings.setValue('distance', self.parameter.distance.text())
        settings.setValue('pixel_x', self.parameter.pixel_x.text())
        settings.setValue('pixel_y', self.parameter.pixel_y.text())
        settings.setValue('lamda', self.parameter.lamda.text())
        settings.setValue('textbox_min',self.image_layout.textbox_min.text())
        settings.setValue('textbox_max', self.image_layout.textbox_max.text())
        settings.setValue('Qr_min', self.parameter.Qr_min.text())
        settings.setValue('Qr_max', self.parameter.Qr_max.text())
        settings.setValue('Qz_min', self.parameter.Qz_min.text())
        settings.setValue('Qz_max', self.parameter.Qz_max.text())
        settings.setValue('threshold_min', self.parameter.threshold_min.text())
        settings.setValue('threshold_max', self.parameter.threshold_max.text())
        settings.setValue('numbin', self.parameter.numbin.text())

        super().closeEvent(event)

    def close_loading(self):
        self.label.close()
        self.show()

class HelpDialog(QDialog):
    def __init__(self, content, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Help")
        self.resize(600, 400)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(content)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

class ImageWidget(QWidget):
    def __init__(self, parent=None, file_name=None, textbox_min=None, textbox_max=None, Angle_incidence=None, x_Center=None,
                  y_Center=None, distance=None, pixel_x=None, pixel_y=None, lamda=None, threshold_min=None, threshold_max=None):
        super().__init__(parent)

        #导出图像用的
        self.fig = None

        self.setAcceptDrops(True)  # 允许接受拖放事件
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.size_label = QLabel(self)
        self.size_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.size_label.setMinimumSize(1, 1)
        self.size_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.size_label)
        self.setLayout(self.layout)

        self.file_name = file_name
        self.textbox_min = textbox_min
        self.textbox_max = textbox_max
        self.Angle_incidence = Angle_incidence
        self.x_Center = x_Center
        self.y_Center = y_Center
        self.distance = distance
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.lamda = lamda
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        # 窗口大小变化时连接resizeEvent事件
        self.resizeEvent = self.on_resize
        # 设置初始缩放级别
        self.scale_factor = 1.0
        # 在滚轮事件中记录时间和鼠标位置
        self.last_wheel_time = 0
        self.last_wheel_pos = QPoint(0, 0)
        # 记录上一次鼠标位置
        self.last_pos = None
        # 记录图像的偏移量
        self.image_offset = QPoint(0, 0)


        #初始化定时器，提高改变窗口大小时调用Cut的流畅度
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.on_resize_timeout)

        # 类变量，用于存储图像窗口的引用
        self.image_fig = None

        # 初始化区域参数
        self.numbin = 1000

        # 设置当前窗口状态
        self.windowstate = 0

        # 当前帧索引（用于 NXS 文件），0-based；默认 0
        self.frame_idx = 0

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and os.access(url.toLocalFile(), os.R_OK):
                event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        # 通过 ImageLayout 统一更新，这样会自动刷新帧号选择器等状态
        if hasattr(self, 'image_layout') and self.image_layout is not None:
            self.image_layout.update_image(file_path)
        else:
            self.file_name = file_path
            self.update_image()

    def update_image(self):
        if self.file_name:
            # 读取图像（支持 NXS 帧）并使用 Matplotlib 直接按 vmin/vmax 显示（不做 0..255 规范化）
            cb_min = float(self.textbox_min.text())
            cb_max = float(self.textbox_max.text())
            im = load_image_matrix(self.file_name, frame_idx=self.frame_idx)

            # 掩蔽坏点/超阈值区域，避免影响色域
            bad_mask = (im >= self.threshold_max) | (im < self.threshold_min) | np.isnan(im)
            A = np.ma.masked_array(im, mask=bad_mask)
            # 记录当前视图对应的原始数据（未翻转），供 ROI/积分使用
            self.last_normal_image_raw = A.filled(np.nan)

            # 翻转（仅显示层面）；load_image_matrix 已按显示方向翻转，这里跟随 UI 的 Flip
            if self.image_layout.flip.isChecked():
                A = np.flipud(A)

            # 用 Matplotlib 画到临时画布并转换为 QLabel 的像素图
            fig, ax = plt.subplots()
            imshow_obj = ax.imshow(A, cmap='jet', vmin=cb_min, vmax=cb_max, aspect='equal')
            fig.colorbar(imshow_obj)
            ax.set_xticks([]); ax.set_yticks([])

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close(fig)

            color_values = cv2.imread(temp_file.name, cv2.IMREAD_COLOR)

            height, width = color_values.shape[:2]
            window_height, window_width = self.label.height(), self.label.width()
            if window_height <= 1 or window_width <= 1:
                temp_file.close(); os.unlink(temp_file.name)
                return
            scale = min(window_height / height, window_width / width)
            resized = cv2.resize(color_values, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            pixmap = self.to_qimage(resized)
            self.label.setPixmap(pixmap)
            self.size_label.setText(f'pixels: {im.shape[1]} x {im.shape[0]} file: {os.path.basename(self.file_name)}')

            temp_file.close()
            os.unlink(temp_file.name)

            self.windowstate = 1
            self.current_view_is_cut = False

    def to_qimage(self, img): #转化为Qpixmap
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        height, width, channels = img.shape
        bytes_per_line = channels * width
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def Cut(self):
        #初始化参数
        if self.file_name:
            Angle_incidence = float(self.Angle_incidence)
            x_Center = float(self.x_Center)
            y_Center = float(self.y_Center)
            distance = float(self.distance)
            pixel_x = float(self.pixel_x)
            pixel_y = float(self.pixel_y)
            lamda = float(self.lamda)

            # 颜色条范围
            cb_min = float(self.textbox_min.text())
            cb_max = float(self.textbox_max.text())

            threshold_min = float(self.threshold_min)
            threshold_max = float(self.threshold_max)

            # 统一读取：图片或 NXS（支持 frame_idx），使用原始强度
            im = load_image_matrix(self.file_name, frame_idx=self.frame_idx).astype(float)

            # 掩蔽坏点/超阈值区域
            bad_mask = (im > threshold_max) | (im < threshold_min) | np.isnan(im)
            A = np.ma.masked_array(im, mask=bad_mask)
            # 显示翻转仅用于输出图像，原始阵列保持未翻转以便 ROI/积分一致

            # 参数设置
            sz = np.shape(A)
            sz_1 = sz[1]
            sz_2 = sz[0]
            # 原始算法在 Q 映射中将中心 Y 转换为自下而上的坐标系
            y_Center = sz_2 - y_Center
            Qr, Qz = np.meshgrid(np.arange(1, sz_1 + 1), np.arange(1, sz_2 + 1))

            # pixel
            Qr = Qr - x_Center
            Qz = (sz_2 - y_Center) - Qz
            # distance
            Qr = Qr * pixel_x * 1e-6
            Qz = Qz * pixel_y * 1e-6
            # Theta
            Qxx = Qr
            Qr = np.arctan(Qr / (distance * 1e-3)) / 2
            Qz = np.arctan(Qz / np.sqrt((distance * 1e-3) ** 2 + Qxx ** 2))
            # Theta = np.arctan(np.sqrt(Qr ** 2 * Qz ** 2) / (distance * 1e-3))

            Theta_f = Qr
            Alpha_f = Qz
            Alpha_i = Angle_incidence * np.pi / 180  # 入射角度

            Qx = 2 * np.pi / lamda * (np.cos(2 * Theta_f) * np.cos(Alpha_f) - np.cos(Alpha_i))
            Qy = 2 * np.pi / lamda * (np.sin(2 * Theta_f) * np.cos(Alpha_f))
            Qz = 2 * np.pi / lamda * (np.sin(Alpha_f) + np.sin(Alpha_i))

            # q 单位：Angstrom
            Qr = np.sign(Qy) * np.sqrt(Qx ** 2 + Qy ** 2)
            Qz = Qz
            # Qr[Qy_temp < 0] = np.nan
            diff_Qy = np.diff(np.sign(Qy), axis=1)
            indices = np.where(diff_Qy != 0)

            # 在 diff_Qy 中找到变号的区域，并将对应的 A 数组中的值设置为 NaN
            A = A.astype(float)
            A[indices[0], indices[1]] = np.nan
            A[indices[0], indices[1] + 1] = np.where(Qy[indices[0], indices[1] + 1] > 0, np.nan,
                                                     A[indices[0], indices[1] + 1])
            A[indices[0], indices[1] - 1] = np.where((indices[1] > 0) & (Qy[indices[0], indices[1] - 1] < 0), np.nan,
                                                     A[indices[0], indices[1] - 1])

            # 记录 cut 后用于积分/ROI 的原始阵列（包含 NaN）以及对应的 Qr/Qz 网格
            self.last_cut_image_raw = np.array(A, copy=True)
            try:
                self.last_cut_qr = np.array(Qr, copy=True)
                self.last_cut_qz = np.array(Qz, copy=True)
            except Exception:
                self.last_cut_qr = None
                self.last_cut_qz = None

            # 创建掩码数组（原始坐标系）
            A_masked = np.ma.masked_where(np.isnan(A), A)

            # 绘制 pcolor 图像（直接用 vmin/vmax 控制色域）；按 Flip 设置决定显示是否上下颠倒

            self.fig, ax = plt.subplots()
            if self.image_layout.flip.isChecked():
                # Flip 仅影响显示：反转 Y 轴
                pcolor = ax.pcolormesh(Qr, Qz, A_masked, cmap='jet', shading='auto', vmin=cb_min, vmax=cb_max)
                ax.invert_yaxis()
            else:
                pcolor = ax.pcolormesh(Qr, Qz, A_masked, cmap='jet', shading='auto', vmin=cb_min, vmax=cb_max)
            self.fig.colorbar(pcolor)
            ax.set_xlabel('Qr')
            ax.set_ylabel('Qz')
            ax.set_aspect('equal')
            # 设置横纵坐标显示范围
            if float(self.parameter.Qr_min.text()) == -121:
                Qr_min = None
            else:
                Qr_min = float(self.parameter.Qr_min.text())

            if float(self.parameter.Qr_max.text()) == -121:
                Qr_max = None
            else:
                Qr_max = float(self.parameter.Qr_max.text())

            if float(self.parameter.Qz_min.text()) == -121:
                Qz_min = None
            else:
                Qz_min = float(self.parameter.Qz_min.text())

            if float(self.parameter.Qz_max.text()) == -121:
                Qz_max = None
            else:
                Qz_max = float(self.parameter.Qz_max.text())

            ax.set_xlim(Qr_min, Qr_max)
            ax.set_ylim(Qz_min, Qz_max)

            # 保存图像为临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            self.fig.savefig(temp_file.name, dpi=300)
            plt.close(self.fig)  # 关闭绘图窗口

            # 读取临时文件并保持颜色
            color_values = cv2.imread(temp_file.name, cv2.IMREAD_COLOR)

            # 缩放图像以适应窗口
            height, width = color_values.shape[:2]
            window_height, window_width = self.label.height(), self.label.width()
            if window_height <= 1 or window_width <= 1:
                return
            scale = min(window_height / height, window_width / width)
            resized = cv2.resize(color_values, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            # 显示图像
            pixmap = self.to_qimage(resized)
            self.label.setPixmap(pixmap)
            self.size_label.setText(f'pixels：{im.shape[1]} x {im.shape[0]} file_name: {os.path.basename(self.file_name)}')

            # 删除临时文件
            temp_file.close()
            os.unlink(temp_file.name)

            # 记录 cut 后用于积分/ROI 的原始阵列（未翻转）
            self.last_cut_image_raw = np.array(A, copy=True)
            self.windowstate = 2
            self.current_view_is_cut = True

    def update_parameters(self, parameter):

        self.Angle_incidence = parameter.Angle_incidence_value
        self.x_Center = parameter.x_Center_value
        self.y_Center = parameter.y_Center_value
        self.distance = parameter.distance_value
        self.pixel_x = parameter.pixel_x_value
        self.pixel_y = parameter.pixel_y_value
        self.lamda = parameter.lamda_value
        self.threshold_min = parameter.threshold_min_value
        self.threshold_max = parameter.threshold_max_value
        self.numbin = parameter.numbin_value

    def on_resize(self, event):
        # 在 1D 显示时避免自动刷新覆盖
        if getattr(self, 'windowstate', 0) == 3:
            event.accept()
            return
            pcolor = ax.pcolormesh(Qr, Qz, A_masked, cmap='jet', shading='auto', vmin=cb_min, vmax=cb_max)
    def on_resize_timeout(self):
        # 在 1D 显示时避免自动刷新覆盖
        if getattr(self, 'windowstate', 0) == 3:
            return
            # 根据 Flip 状态设置显示方向（与原始代码一致：未 Flip 时倒置显示）
            if not self.image_layout.flip.isChecked():
                plt.gca().invert_yaxis()
            # 记录 cut 后用于积分/ROI 的原始阵列（未翻转）
    # def wheelEvent(self, event):
    #     # 获取当前的鼠标位置
    #     mouse_pos = event.pos()
    #
    #     # 计算鼠标位置与标签的偏移量
    #     offset = mouse_pos - self.label.pos()
    #
    #     # 计算缩放因子和缩放中心点
    #     zoom_in_factor = 1.25
    #     zoom_out_factor = 1 / zoom_in_factor
    #     zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
    #     center = QPoint(offset.x() * zoom_factor, offset.y() * zoom_factor)
    #
    #     # 改变图像缩放级别
    #     self.scale_factor *= zoom_factor
    #     if self.scale_factor < 0.1:
    #         self.scale_factor = 0.1
    #
    #     # 改变图像缩放级别并设置缩放中心点
    #     transform = QTransform().translate(center.x(), center.y()).scale(zoom_factor, zoom_factor).translate(
    #         -center.x(), -center.y())
    #     # transform = QTransform().scale(zoom_factor, zoom_factor)
    #     self.label.setPixmap(self.label.pixmap().transformed(transform))

    def mousePressEvent(self, event):
        # 记录鼠标按下时的位置
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        # 如果鼠标左键被按下，移动图片
        if event.buttons() == Qt.LeftButton:
            # 计算鼠标移动距离
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()

            # 更新图像的偏移量
            self.image_offset += delta

            # 更新图像
            # self.update_image()
            pixmap_offset = self.image_offset - self.label.rect().topLeft()
            self.label.move(pixmap_offset)

    def mouseReleaseEvent(self, event):
        # 鼠标释放时清空记录的上一次鼠标位置
        self.last_pos = None

    def set_image_layout(self, image_layout, parameter):
        self.image_layout = image_layout
        self.parameter = parameter

    def set_batch_processor(self, batch_processor):
        self.batch_processor = batch_processor

    def update_batch_processor_filename(self):
        self.file_name = self.batch_processor.filename

    def int_region(self, cb_min, cb_max, x_center, y_center):

        # 根据当前视图选择源数据（原始坐标系，不做翻转）：cut 视图优先
        base = getattr(self, 'last_cut_image_raw', None) if getattr(self, 'current_view_is_cut', False) else getattr(self, 'last_normal_image_raw', None)
        if base is None:
            base = load_image_matrix(self.file_name)
        # 掩蔽坏点 + NaN
        bad_mask = (base >= self.threshold_max) | (base < self.threshold_min) | np.isnan(base)
        A = np.ma.masked_array(base, mask=bad_mask)
        # 按 UI 的 Flip 设置进行显示翻转，仅影响显示
        A_disp = np.flipud(A) if getattr(self.image_layout, 'flip', None) and self.image_layout.flip.isChecked() else A

        # fig, ax = plt.subplots()
        # ax.imshow(im_norm)

        # 创建图像窗口
        if self.image_fig is None or not plt.fignum_exists(self.image_fig.number):
            self.image_fig, axx = plt.subplots()
            axx.imshow(A_disp, cmap='jet', vmin=cb_min, vmax=cb_max)

        else:
            self.image_fig.clf()
            axx = self.image_fig.add_subplot(111)
            axx.imshow(A_disp, cmap='jet', vmin=cb_min, vmax=cb_max)
            plt.draw()

        # 右键设置中心：在图上点击右键后，将 Center X/Y 设置为点击位置，并同步到参数框
        def _on_right_click(event):
            nonlocal x_center, y_center
            if event.button == 3 and event.inaxes is axx and event.xdata is not None and event.ydata is not None:
                # 显示坐标（已 flipud），转换到原始行坐标
                x_clicked = float(event.xdata)
                y_clicked = float(event.ydata)
                if getattr(self.image_layout, 'flip', None) and self.image_layout.flip.isChecked():
                    y_raw = float(A_disp.shape[0]) - y_clicked
                else:
                    y_raw = y_clicked
                # 更新本次 ROI 选择中心
                x_center = x_clicked
                y_center = y_raw
                # 同步到 UI 与模型
                try:
                    if hasattr(self, 'parameter') and self.parameter is not None:
                        self.parameter.x_Center.setText(str(round(x_center, 2)))
                        self.parameter.y_Center.setText(str(round(y_center, 2)))
                    self.x_Center = x_center
                    self.y_Center = y_center
                except Exception:
                    pass
                # 在图上标记新的中心
                axx.plot([x_clicked], [y_clicked], marker='x', color='yellow', markersize=8, mew=2)
                plt.draw()

        cid = self.image_fig.canvas.mpl_connect('button_press_event', _on_right_click)

        # 用于存储鼠标点击位置
        points = []

        # 辅助函数：安全获取一次点击；若用户取消/窗口关闭/无点返回，则返回 None
        def _safe_ginput_once():
            try:
                pts = plt.ginput(1, timeout=0)  # 与 Qt 集成下，不设超时，交给事件循环
            except Exception:
                return None
            if not pts:
                return None
            return pts[0]

        start_angle = end_angle = inner_radius = outer_radius = None
        # 显示坐标中的中心（用于绘制），原始坐标中的中心（用于计算）
        if getattr(self.image_layout, 'flip', None) and self.image_layout.flip.isChecked():
            y_center_disp = float(A_disp.shape[0]) - float(y_center)
        else:
            y_center_disp = float(y_center)

        for i in range(4):
            # 获取鼠标点击位置（允许用户取消，不视为错误）
            point = _safe_ginput_once()
            if point is None:
                # 取消选择：不弹出报错，直接返回 None 让调用方静默退出
                return None

            x_d, y_d = int(round(point[0])), int(round(point[1]))
            # 转换到原始坐标系
            if getattr(self.image_layout, 'flip', None) and self.image_layout.flip.isChecked():
                y_r = int(round(A_disp.shape[0] - y_d))
            else:
                y_r = y_d
            points.append((x_r := x_d, y_r))

            if i == 0 or i == 1:
                # 绘制直线（使用显示坐标中心）
                axx.add_line(Line2D([x_d, int(round(x_center))], [y_d, int(round(y_center_disp))], color='red'))
                plt.draw()  # 强制刷新图像
            else:
                # 计算起始和终止角度
                start_angle = np.arctan2(points[0][1] - float(y_center), points[0][0] - float(x_center))
                end_angle = np.arctan2(points[1][1] - float(y_center), points[1][0] - float(x_center))
                if i == 2:
                    # 计算内半径
                    inner_radius = np.sqrt((points[2][0] - float(x_center)) ** 2 + (points[2][1] - float(y_center)) ** 2)
                    # 绘制扇形区域
                    wedge = Wedge((float(x_center), float(y_center_disp)), inner_radius, math.degrees(start_angle),
                                  math.degrees(end_angle),
                                  width=2)
                    axx.add_patch(wedge)
                    plt.draw()  # 强制刷新图像
                if i == 3:
                    outer_radius = np.sqrt((points[3][0] - float(x_center)) ** 2 + (points[3][1] - float(y_center)) ** 2)
                    # 绘制扇形区域
                    wedge = Wedge((float(x_center), float(y_center_disp)), outer_radius, math.degrees(start_angle),
                                  math.degrees(end_angle),
                                  width=outer_radius - (inner_radius if inner_radius is not None else 0))
                    wedge.set_alpha(0.5)
                    axx.add_patch(wedge)
                    plt.draw()  # 强制刷新图像

        # # 提示用户是否确认选择区域
        # msg_box = QMessageBox()
        # msg_box.setText("是否确认选择区域？")
        # msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # msg_box.setDefaultButton(QMessageBox.Yes)
        # 关闭图像窗口并返回结果
        # plt.close(fig)
        # ret = msg_box.exec_()
        return np.asarray(A_disp.filled(np.nan)), start_angle, end_angle, inner_radius, outer_radius

    # 将笛卡尔坐标系下的图像转换为极坐标系下的图像
    def cart2pol(self, image, center):
        # 计算图像中每个像素点的极坐标值
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        x = x - center[0]
        y = y - center[1]
        r, theta = cv2.cartToPolar(x, y)

        # 插值得到极坐标系下的图像
        polar_image = cv2.remap(image, theta, r, cv2.INTER_LINEAR)

        return polar_image

    # 点击积分按钮调用此函数
    def radial_integral(self, image, center, start_angle, end_angle, inner_radius, outer_radius, num_bins):
        """
        计算选定的扇形区域的径向积分和角向积分
        :param image: 待处理的图像
        :param center: 中心像素位置，tuple类型，(x, y)
        :param start_angle: 起始方位角，单位为度，0度表示x轴正方向，逆时针旋转为正
        :param end_angle: 结束方位角，单位为度，0度表示x轴正方向，逆时针旋转为正
        :param inner_radius: 扇形区域的内径
        :param outer_radius: 扇形区域的外径
        :param num_bins: 径向积分的点数
        :return: (radial_profile, angular_profile)，径向积分和角向积分
        """
        # 与坐标网格一致：直接使用传入的 image 作为掩膜依据，避免与翻转阵列不一致
        im = np.array(image, copy=True)
        start_angle = math.radians(start_angle)
        end_angle = math.radians(end_angle)
        
        # 构造一个极坐标网格
        height, width = image.shape[:2]
        y, x = np.ogrid[:height, :width]
        x = x.astype(np.float64) - float(center[0])
        y = y.astype(np.float64) - float(center[1])
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)

        # 不做额外翻转，确保与 r/theta 网格一致
        # 确定扇形区域的布尔掩码
        mask = (r >= inner_radius) & (r <= outer_radius) & (theta >= start_angle) & (theta <= end_angle)

        if start_angle >= end_angle:
            mask = (r >= inner_radius) & (r <= outer_radius) & ((theta >= start_angle) | (theta <= end_angle))
            end_angle = end_angle + 2 * np.pi
        # 结合阈值与有效数据（排除 NaN），确保空区不计入统计
        mask = mask & np.isfinite(im) & (im >= self.threshold_min) & (im <= self.threshold_max)
        # print(im)
  
        # plt.imshow(mask.astype(np.float64), cmap='gray')
        # plt.show()
        # 计算径向积分
        rbin_edges = np.linspace(inner_radius, outer_radius, num_bins + 1)
        rbin_centers = 0.5 * (rbin_edges[1:] + rbin_edges[:-1])

        # 同时计算计数，用于将空 bin 标记为 NaN，实现绘图断开
        counts_r, _ = np.histogram(r[mask], bins=rbin_edges)
        radial_profile, _ = np.histogram(r[mask], bins=rbin_edges, weights=image[mask].astype(np.float64))
        radial_profile = radial_profile.astype(np.float64) / np.diff(rbin_edges)
        radial_profile[counts_r == 0] = np.nan

        # 计算角向积分
        thetabin_edges = np.linspace(start_angle, end_angle, num_bins + 1)
        thetabin_centers_radians = 0.5 * (thetabin_edges[1:] + thetabin_edges[:-1])
        thetabin_centers_degrees = np.degrees(thetabin_centers_radians)
        counts_t, _ = np.histogram(theta[mask], bins=thetabin_edges)
        angular_profile, _ = np.histogram(theta[mask], bins=thetabin_edges, weights=image[mask].astype(np.float64))
        angular_profile = angular_profile.astype(np.float64) / np.diff(thetabin_edges)
        angular_profile[counts_t == 0] = np.nan

        print("Radial and angular integration completed.")
        # 定义滑动窗口的大小
        window_size = 5
        # 定义滑动窗口的权重
        window = np.ones(window_size) / window_size
        # 对angular_profile进行滑动平均
        # 若存在缺口（NaN），避免跨缺口平滑，保持断开
        if np.any(np.isnan(angular_profile)):
            smoothed_angular_profile = angular_profile.copy()
        else:
            smoothed_angular_profile = np.convolve(angular_profile, window, mode='same')
        if np.any(np.isnan(radial_profile)):
            smoothed_radial_profile = radial_profile.copy()
        else:
            smoothed_radial_profile = np.convolve(radial_profile, window, mode='same')

        # 绘制图像
        self.fig, ax = plt.subplots()
        index = self.image_layout.comboBox.currentIndex()
        distance = float(self.distance) * 1e-3
        pixel_x = float(self.pixel_x) * 1e-6
        pixel_y = float(self.pixel_y) * 1e-6
        lamda = float(self.lamda)

        pixel = (pixel_x + pixel_y)/2
        theta = np.arctan(rbin_centers * pixel / distance) / 2
        q = 4 * np.pi * np.sin(theta) / lamda
        twoTheta = np.arcsin(q * 1.54 / 4 / np.pi) * 180 / np.pi * 2

        if self.image_layout.comboBox2.currentIndex() == 0:
            if self.image_layout.radioButtonRadial.isChecked():
                if index == 0 :
                    ax.semilogy(q, smoothed_radial_profile)
                    ax.set_xlabel('q')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return q, smoothed_radial_profile
                if index == 1:
                    ax.semilogy(twoTheta, smoothed_radial_profile)
                    ax.set_xlabel('2Theta')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return twoTheta, smoothed_radial_profile
                if index == 2:
                    ax.semilogy(rbin_centers, smoothed_radial_profile)
                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return rbin_centers, smoothed_radial_profile
                if index == 3:
                    ax.semilogy(q, radial_profile)
                    ax.set_xlabel('q')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return q, radial_profile
                if index == 4:
                    ax.semilogy(twoTheta, radial_profile)
                    ax.set_xlabel('2Theta')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return twoTheta, radial_profile
                if index == 5:
                    ax.semilogy(rbin_centers, radial_profile)
                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Radial Profile')
                    # return rbin_centers, radial_profile
            if self.image_layout.radioButtonAngular.isChecked():
                if index == 0:
                    ax.semilogy(thetabin_centers_degrees, smoothed_angular_profile)
                    ax.set_xlabel('Theta')
                    ax.set_ylabel('Intensity (Log Scale)')
                    ax.set_title('Azimuth Profile')
                    # return thetabin_centers_degrees, smoothed_angular_profile
        if self.image_layout.comboBox2.currentIndex() == 1:
            if self.image_layout.radioButtonRadial.isChecked():
                if index == 0 :
                    ax.plot(q, smoothed_radial_profile)
                    ax.set_xlabel('q')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return q, smoothed_radial_profile
                if index == 1:
                    ax.plot(twoTheta, smoothed_radial_profile)
                    ax.set_xlabel('2Theta')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return twoTheta, smoothed_radial_profile
                if index == 2:
                    ax.plot(rbin_centers, smoothed_radial_profile)
                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return rbin_centers, smoothed_radial_profile
                if index == 3:
                    ax.plot(q, radial_profile)
                    ax.set_xlabel('q')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return q, radial_profile
                if index == 4:
                    ax.plot(twoTheta, radial_profile)
                    ax.set_xlabel('2Theta')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return twoTheta, radial_profile
                if index == 5:
                    ax.plot(rbin_centers, radial_profile)
                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Radial Profile')
                    # return rbin_centers, radial_profile
            if self.image_layout.radioButtonAngular.isChecked():
                if index == 0:
                    ax.plot(thetabin_centers_degrees, smoothed_angular_profile)
                    ax.set_xlabel('Theta')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Azimuth Profile')
                    # return thetabin_centers_degrees, smoothed_angular_profile

        # 保存图像为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.fig.savefig(temp_file.name, dpi=300)
        plt.close(self.fig)  # 关闭绘图窗口

        # 读取临时文件并保持颜色
        color_values = cv2.imread(temp_file.name, cv2.IMREAD_COLOR)

        # 缩放图像以适应窗口
        height, width = color_values.shape[:2]
        window_height, window_width = self.label.height(), self.label.width()
        if window_height <= 1 or window_width <= 1:
            return
        scale = min(window_height / height, window_width / width)
        resized = cv2.resize(color_values, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        # 显示图像
        pixmap = self.to_qimage(resized)
        self.label.setPixmap(pixmap)
        self.size_label.setText(f'1D image — file: {os.path.basename(self.file_name)}')

        # 删除临时文件
        temp_file.close()
        os.unlink(temp_file.name)

        self.windowstate = 3

        if self.image_layout.comboBox2.currentIndex() == 0:
            if self.image_layout.radioButtonRadial.isChecked():
                if index == 0 :
                    return q, smoothed_radial_profile
                if index == 1:
                    return twoTheta, smoothed_radial_profile
                if index == 2:
                    return rbin_centers, smoothed_radial_profile
                if index == 3:
                    return q, radial_profile
                if index == 4:
                    return twoTheta, radial_profile
                if index == 5:
                    return rbin_centers, radial_profile
            if self.image_layout.radioButtonAngular.isChecked():
                if index == 0:
                    return thetabin_centers_degrees, smoothed_angular_profile

    def azimuth_profile_from_cut(self, qr, qz, I, qmin, qmax, n_chi=360, mask=None, mode="mean"):
        """
        Azimuthal profile in Q-space for cut-mode.
        qr, qz, I: 2D arrays (same shape) from cut result
        qmin, qmax: q-range to integrate
        n_chi: number of chi bins
        mask: boolean array, True for valid pixels (optional)
        mode: "sum" for integration, "mean" for average intensity
        Returns (centers_deg, prof, cnt)
        """
        qr = np.asarray(qr)
        qz = np.asarray(qz)
        I  = np.asarray(I)

        q = np.sqrt(qr**2 + qz**2)
        # Chi definition aligned to pixel ROI: 0 along +Qr (right),
        # upper half (-180..0), lower half (0..180)
        chi = -np.degrees(np.arctan2(qz, qr))
        chi = ((chi + 180.0) % 360.0) - 180.0

        # Normalize input image to plain ndarray, fill masked with NaN
        if np.ma.isMaskedArray(I):
            I = I.filled(np.nan)
        valid = np.isfinite(q) & np.isfinite(chi) & np.isfinite(I)
        valid &= (q >= qmin) & (q <= qmax)
        if mask is not None:
            valid &= mask

        chi_v = chi[valid].ravel()
        I_v   = I[valid].ravel()

        # Histogram across full chi range [-180, 180]
        edges = np.linspace(-180.0, 180.0, int(n_chi) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # If no valid pixels, return empty arrays to avoid flat lines
        if chi_v.size == 0:
            return centers, np.zeros_like(centers), np.zeros_like(centers, dtype=int)
        sum_I, _ = np.histogram(chi_v, bins=edges, weights=I_v)
        cnt, _   = np.histogram(chi_v, bins=edges)

        if mode == "sum":
            prof = sum_I
        elif mode == "mean":
            prof = sum_I / np.maximum(cnt, 1)
        else:
            raise ValueError("mode must be 'sum' or 'mean'")

        # Mark empty bins as NaN to avoid misleading flat zero segments
        prof = np.asarray(prof, dtype=float)
        prof[cnt == 0] = np.nan

        return centers, prof, cnt
        if self.image_layout.comboBox2.currentIndex() == 1:
            if self.image_layout.radioButtonRadial.isChecked():
                if index == 0 :
                    return q, smoothed_radial_profile
                if index == 1:
                    return twoTheta, smoothed_radial_profile
                if index == 2:
                    return rbin_centers, smoothed_radial_profile
                if index == 3:
                    return q, radial_profile
                if index == 4:
                    return twoTheta, radial_profile
                if index == 5:
                    return rbin_centers, radial_profile
            if self.image_layout.radioButtonAngular.isChecked():
                if index == 0:
                    return thetabin_centers_degrees, smoothed_angular_profile

    # 点击积分按钮调用此函数
    def calculate_integral(self):
        try:
            if self.file_name:
                cb_min = float(self.textbox_min.text())
                cb_max = float(self.textbox_max.text())
                # 根据 Original/Cut 选项选择源数据：Original 强制使用原始图像；Cut 使用切换后的 Q 映射图
                if getattr(self.image_layout, 'rb1', None) and self.image_layout.rb1.isChecked():
                    base = getattr(self, 'last_normal_image_raw', None)
                elif getattr(self.image_layout, 'rb2', None) and self.image_layout.rb2.isChecked():
                    base = getattr(self, 'last_cut_image_raw', None)
                else:
                    base = getattr(self, 'last_normal_image_raw', None)
                if base is None:
                    base = load_image_matrix(self.file_name, frame_idx=self.frame_idx)
                # Normalize to ndarray and keep NaN for invalid pixels
                image = np.array(base, copy=True)
                if np.ma.isMaskedArray(image):
                    image = image.filled(np.nan)
                # 保留 NaN，仅对有限值做 vmin/vmax 截断
                finite_mask = np.isfinite(image)
                image[finite_mask & (image > cb_max)] = cb_max
                image[finite_mask & (image < cb_min)] = cb_min

                # 获取所有参数值
                center = [float(self.x_Center), float(self.y_Center)]
                start_angle = float(self.image_layout.textbox_startAngle.text())
                end_angle = float(self.image_layout.textbox_endAngle.text())
                inner_radius = float(self.image_layout.textbox_innerRadius.text())
                outer_radius = float(self.image_layout.textbox_outerRadius.text())
                try:
                    num_bins = max(1, int(self.numbin))
                except (TypeError, ValueError):
                    num_bins = 500
                print(f"Calculating integral with center={center}, start_angle={start_angle}, "
                      f"end_angle={end_angle}, inner_radius={inner_radius}, outer_radius={outer_radius}, num_bins={num_bins}")
                # 分支：径向或方位角积分
                if self.image_layout.radioButtonRadial.isChecked():
                    x, y = self.radial_integral(image, center, start_angle, end_angle, inner_radius, outer_radius, num_bins)
                else:
                    # Azimuthal integration
                    if getattr(self.image_layout, 'rb2', None) and self.image_layout.rb2.isChecked():
                        # Cut 模式：在 Q 空间按 q 范围积分
                        qr = getattr(self, 'last_cut_qr', None)
                        qz = getattr(self, 'last_cut_qz', None)
                        if qr is None or qz is None:
                            # 回退：若未缓存 Q 网格，则临时计算一次（不刷新 UI）
                            try:
                                # 触发一次 Cut 以生成 Q 网格（不改变窗口显示时机）
                                self.Cut()
                                qr = getattr(self, 'last_cut_qr', None)
                                qz = getattr(self, 'last_cut_qz', None)
                            except Exception:
                                qr = None; qz = None
                        if qr is None or qz is None:
                            # 无法进行 Q 空间积分，回退到像素空间角向
                            x, y = self.radial_integral(image, center, start_angle, end_angle, inner_radius, outer_radius, num_bins)
                        else:
                            # Q-space chi (azimuth) and q-magnitude
                            q_mag = np.sqrt(qr**2 + qz**2)
                            # Align chi to pixel ROI convention:
                            # 0 along +Qr, upper half (-180..0), lower half (0..180)
                            chi = -np.degrees(np.arctan2(qz, qr))
                            # Normalize chi and input angles into [-180, 180)
                            chi = ((chi + 180.0) % 360.0) - 180.0
                            cs = ((float(start_angle) + 180.0) % 360.0) - 180.0
                            ce = ((float(end_angle) + 180.0) % 360.0) - 180.0
                            # Build chi-sector mask in Q-space (degrees, -180..180)
                            if cs <= ce:
                                mask_chi = (chi >= cs) & (chi <= ce)
                            else:
                                # sector crosses the -180/180 boundary
                                mask_chi = (chi >= cs) | (chi <= ce)
                            # Pixel radial selection for ROI thickness only (do not use tt for angle)
                            h, w = image.shape[:2]
                            yy, xx = np.ogrid[:h, :w]
                            xx = xx.astype(np.float64) - float(center[0])
                            yy = yy.astype(np.float64) - float(center[1])
                            rr = np.hypot(xx, yy)
                            mask_rr = (rr >= float(inner_radius)) & (rr <= float(outer_radius))
                            # ROI mask used to estimate q-range robustly via percentiles
                            valid_img = np.isfinite(image)
                            valid_q = np.isfinite(q_mag)
                            roi_mask = mask_rr & mask_chi & valid_q & valid_img
                            if np.any(roi_mask):
                                # Robust qmin/qmax from ROI distribution
                                qmin = float(np.nanpercentile(q_mag[roi_mask], 5))
                                qmax = float(np.nanpercentile(q_mag[roi_mask], 95))
                                if not np.isfinite(qmin) or not np.isfinite(qmax) or qmin == qmax:
                                    qmin = float(np.nanmin(q_mag[roi_mask]))
                                    qmax = float(np.nanmax(q_mag[roi_mask]))
                            else:
                                # Fallback: global finite q range
                                finite_all = np.isfinite(q_mag)
                                qmin = float(np.nanpercentile(q_mag[finite_all], 5))
                                qmax = float(np.nanpercentile(q_mag[finite_all], 95))
                            if qmin > qmax:
                                qmin, qmax = qmax, qmin
                            # Final mask in Q-space (consistency with integration space)
                            mask_q = (q_mag >= qmin) & (q_mag <= qmax) & mask_chi & np.isfinite(image)
                            # Include intensity threshold limits
                            mask_q &= (image >= float(self.threshold_min)) & (image <= float(self.threshold_max))
                            centers_deg, prof, cnt = self.azimuth_profile_from_cut(qr, qz, image, qmin, qmax, n_chi=num_bins, mask=mask_q, mode="mean")

                        # 绘制并回传
                        self.fig, ax = plt.subplots()
                        if self.image_layout.comboBox2.currentIndex() == 0:
                            ax.semilogy(centers_deg, prof)
                        else:
                            ax.plot(centers_deg, prof)
                        ax.set_xlabel('Theta')
                        ax.set_ylabel('Intensity' + (' (Log Scale)' if self.image_layout.comboBox2.currentIndex() == 0 else ''))
                        ax.set_title('Azimuth Profile (Q-space)')

                        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        self.fig.savefig(temp_file.name, dpi=300)
                        plt.close(self.fig)
                        color_values = cv2.imread(temp_file.name, cv2.IMREAD_COLOR)
                        window_height, window_width = self.label.height(), self.label.width()
                        height, width = color_values.shape[:2]
                        if window_height > 1 and window_width > 1:
                            scale = min(window_height / height, window_width / width)
                            resized = cv2.resize(color_values, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                            pixmap = self.to_qimage(resized)
                            self.label.setPixmap(pixmap)
                            self.size_label.setText(f'1D image — file: {os.path.basename(self.file_name)}')
                        temp_file.close(); os.unlink(temp_file.name)
                        self.windowstate = 3
                        x, y = centers_deg, prof
                    else:
                        # Original 模式：像素空间的角向积分（与现有逻辑一致）
                        x, y = self.radial_integral(image, center, start_angle, end_angle, inner_radius, outer_radius, num_bins)
                print("Integral calculation completed.")
                # mask = (x >= float(self.batch_processor.background_min.text())) & (x <= float(self.batch_processor.background_max.text()))
                # x_selected = x[mask]
                # y_selected = y[mask]

                return x, y
        except:
            return

class ImageLayout(QWidget):
    def __init__(self, parent = None, file_name = None, parameter = None, image_widget = None):
        super().__init__(parent)

        self.file_name = file_name
        self.parameter = parameter
        settings = QSettings('mycompany', 'myapp')

        # 添加按钮和文本框控件
        self.button = QPushButton('Select File', self)
        self.button.setFixedWidth(200)
        self.button.setFixedHeight(30)
        self.button.clicked.connect(self.select_file)

        # 帧号输入（仅 NXS 可用）
        self.label_no = QLabel('No.')
        self.label_no.setFixedWidth(30)
        self.textbox_frameNo = QLineEdit(self)
        self.textbox_frameNo.setFixedWidth(100)
        self.textbox_frameNo.setFixedHeight(20)
        self.textbox_frameNo.setEnabled(False)  # 默认禁用
        self.textbox_frameNo.setPlaceholderText('1-N')
        self.textbox_frameNo.setToolTip('Select frame number (enabled for NXS)')
        self.textbox_frameNo.editingFinished.connect(self.on_frame_no_changed)
        self.frame_count = 1

        try:
            value = float(settings.value('textbox_min', '0'))
        except ValueError:
            value = 0
        self.textbox_min = QLineEdit(str(value))
        self.textbox_min.setFixedWidth(100)
        self.textbox_min.setFixedHeight(20)
        # self.textbox_min.setText('0')

        try:
            value = float(settings.value('textbox_max', '800'))
        except ValueError:
            value = 800
        self.textbox_max = QLineEdit(str(value))
        self.textbox_max.setFixedWidth(100)
        self.textbox_max.setFixedHeight(20)
        # self.textbox_max.setText('800')

        self.button_output = QPushButton('Export Image (JPG)', self)
        self.button_output.setFixedWidth(200)
        self.button_output.setFixedHeight(30)

        self.button_outputdir = QPushButton('Select Output Folder', self)
        self.button_outputdir.setFixedWidth(200)
        self.button_outputdir.setFixedHeight(30)
        self.textbox_outputdir = QLineEdit(self)
        self.textbox_outputdir.setFixedWidth(400)
        self.textbox_outputdir.setFixedHeight(20)
        self.textbox_outputdir.setText(os.getcwd())
        self.update_output_folder()

        self.rb1 = QRadioButton('Original')
        self.rb2 = QRadioButton('Cut')
        self.flip = QRadioButton('Flip')

        self.button_intRegion = QPushButton('Select ROI',self)
        self.button_integer = QPushButton('Integrate',self)
        self.textbox_startAngle = QLineEdit(self)
        self.textbox_startAngle.setFixedWidth(200)
        self.textbox_startAngle.setFixedHeight(20)
        self.textbox_startAngle.setPlaceholderText('start angle')
        self.textbox_endAngle = QLineEdit(self)
        self.textbox_endAngle.setFixedWidth(200)
        self.textbox_endAngle.setFixedHeight(20)
        self.textbox_endAngle.setPlaceholderText('end angle')
        self.textbox_innerRadius = QLineEdit(self)
        self.textbox_innerRadius.setFixedWidth(200)
        self.textbox_innerRadius.setFixedHeight(20)
        self.textbox_innerRadius.setPlaceholderText('inner radius')
        self.textbox_outerRadius = QLineEdit(self)
        self.textbox_outerRadius.setFixedWidth(200)
        self.textbox_outerRadius.setFixedHeight(20)
        self.textbox_outerRadius.setPlaceholderText('outer radius')
        self.export_1D = QPushButton("Export 1D (txt)", self)

        # 创建下拉菜单和按钮
        self.comboBox = QComboBox()
        self.comboBox.addItems(['q', '2theta', 'pixel', 'q (unsmoothed)', '2theta (unsmoothed)', 'pixel (unsmoothed)'])
        self.comboBox2 = QComboBox()
        self.comboBox2.addItems(['Log', 'Linear'])
        self.comboBox2.setCurrentIndex(1) #默认Linear
        self.radioButtonRadial = QRadioButton('Radial Integration')
        self.radioButtonAngular = QRadioButton('Azimuthal Integration')
        self.radioButtonRadial.setChecked(True)  # 默认选中径向积分

        self.buttonGroup1 = QButtonGroup()
        self.buttonGroup1.addButton(self.radioButtonRadial)
        self.buttonGroup1.addButton(self.radioButtonAngular)
        self.buttonGroup1.setExclusive(True)

        # 创建一个QButtonGroup，并将2个QRadioButton添加到组中
        self.group = QButtonGroup()
        self.group.addButton(self.rb1)
        self.group.addButton(self.rb2)
        # 设置rb1为默认选中状态
        self.rb1.setChecked(True)
        # 设置自动互斥
        self.rb1.setAutoExclusive(True)
        self.rb2.setAutoExclusive(True)

        # 点击按钮改变显示形式
        self.rb1.toggled.connect(lambda: self.on_radiobutton_toggled(self.rb1, self.image_widget.update_image))
        self.rb2.toggled.connect(lambda: self.on_radiobutton_toggled(self.rb2, self.image_widget.Cut))

        self.image_widget = image_widget
        self.image_widget.textbox_min=self.textbox_min
        self.image_widget.textbox_max = self.textbox_max
        self.image_widget.Angle_incidence=parameter.Angle_incidence_value
        self.image_widget.x_Center=parameter.x_Center_value
        self.image_widget.y_Center = parameter.y_Center_value
        self.image_widget.distance=parameter.distance_value
        self.image_widget.pixel_x=parameter.pixel_x_value
        self.image_widget.pixel_y=parameter.pixel_y_value
        self.image_widget.lamda=parameter.lamda_value
        self.image_widget.threshold_min = parameter.threshold_min_value
        self.image_widget.threshold_max = parameter.threshold_max_value
        self.image_widget.numbin = parameter.numbin_value


        # 连接colorbar
        self.textbox_min.editingFinished.connect(self.update_image_finished)
        self.textbox_max.editingFinished.connect(self.update_image_finished)
        self.button_output.clicked.connect(self.export_image)
        self.button_outputdir.clicked.connect(self.select_outputdir)
        self.textbox_outputdir.editingFinished.connect(self.update_output_folder)
        self.button_intRegion.clicked.connect(self.on_intRegion_button_clicked)
        self.textbox_startAngle.editingFinished.connect(self.update_rigionValues)
        self.textbox_endAngle.editingFinished.connect(self.update_rigionValues)
        self.textbox_innerRadius.editingFinished.connect(self.update_rigionValues)
        self.textbox_outerRadius.editingFinished.connect(self.update_rigionValues)
        self.button_integer.clicked.connect(self.image_widget.calculate_integral)
        self.radioButtonRadial.toggled.connect(self.on_radio_button_toggled)
        self.radioButtonAngular.toggled.connect(self.on_radio_button_toggled)
        self.export_1D.clicked.connect(self.export_integral_data)
        self.flip.toggled.connect(self.update_image_finished)

        self.image_widget.setStyleSheet("border: 2px solid #808080; border-radius: 5px;")
        # 批处理时的文件名后缀（如帧号），为空则不追加
        self.batch_suffix = ""
        # 创建布局

        radio_buttons_layout = QHBoxLayout()
        radio_buttons_layout.addWidget(self.rb1)
        radio_buttons_layout.addWidget(self.rb2)

        # Second-row container: Save/Import parameters aligned left
        self.button_saveParams = QPushButton('Save Parameters', self)
        self.button_importParams = QPushButton('Import Parameters', self)
        params_row = QWidget()
        params_layout = QHBoxLayout(params_row)
        params_layout.setContentsMargins(0,0,0,0)
        params_layout.addWidget(self.button_saveParams)
        params_layout.addWidget(self.button_importParams)
        params_layout.addStretch(1)
        # Connect signals after creation
        self.button_saveParams.clicked.connect(self.save_parameters)
        self.button_importParams.clicked.connect(self.import_parameters)

        layout = QGridLayout(self)

        # 顶部两列容器：左（文件名标签+选择按钮），右（No. + 输入框）
        w_file = QWidget()
        h1 = QHBoxLayout(w_file)
        h1.setContentsMargins(0,0,0,0)
        h1.addWidget(QLabel('File name:'))
        h1.addWidget(self.button)

        w_no = QWidget()
        h2 = QHBoxLayout(w_no)
        h2.setContentsMargins(0,0,0,0)
        h2.addWidget(self.label_no)
        h2.addWidget(self.textbox_frameNo)

        layout.addWidget(w_file, 0, 0)
        layout.addWidget(w_no, 0, 1)
        layout.addWidget(params_row, 1, 0, 1, 2)
        layout.addWidget(QLabel('Colorbar_min:'), 2, 0)
        layout.addWidget(self.textbox_min, 2, 1)
        layout.addWidget(QLabel('Colorbar_max:'), 3, 0)
        layout.addWidget(self.textbox_max, 3, 1)
        layout.addWidget(self.button_output, 4, 0)
        layout.addWidget(self.image_widget, 0, 2, 12, 1)
        layout.addLayout(radio_buttons_layout, 5, 0)
        layout.addWidget(self.flip, 5, 1)
        layout.addWidget(self.button_intRegion, 6, 0)
        layout.addWidget(self.button_integer, 6, 1)
        layout.addWidget(self.textbox_startAngle,7, 0)
        layout.addWidget(self.textbox_endAngle,7,1)
        layout.addWidget(self.textbox_innerRadius,8,0)
        layout.addWidget(self.textbox_outerRadius,8,1)
        layout.addWidget(QLabel('X-axis unit:'), 9, 0)
        layout.addWidget(self.comboBox, 9, 1)
        layout.addWidget(QLabel('Y-axis scale:'), 10, 0)
        layout.addWidget(self.comboBox2, 10, 1)
        layout.addWidget(self.radioButtonRadial, 11, 0)
        layout.addWidget(self.radioButtonAngular, 11, 1)
        layout.addWidget(self.export_1D, 4, 1)
        layout.addWidget(self.button_outputdir, 12, 0)
        layout.addWidget(self.textbox_outputdir, 12, 1, 1, 2)

        # 设置原位数据处理窗台码
        self.insitustate = 0

    def _detect_nxs_frame_count(self, file_name: str) -> int:
        count = 1
        try:
            with h5py.File(file_name, 'r') as f:
                if '/entry/instrument/detector/data' in f:
                    dset = f['/entry/instrument/detector/data']
                    if dset.ndim == 3:
                        count = int(dset.shape[0])
        except Exception:
            pass
        return count

    def _update_frame_selector_for(self, file_name: str):
        suffix = Path(file_name).suffix.lower()
        if suffix == '.nxs':
            self.frame_count = self._detect_nxs_frame_count(file_name)
            self.textbox_frameNo.setEnabled(True)
            self.textbox_frameNo.setPlaceholderText(f'1-{self.frame_count}')
            self.textbox_frameNo.setToolTip(f'Frame No. range: 1 to {self.frame_count}')
            # 默认到 1
            if not self.textbox_frameNo.text():
                self.textbox_frameNo.setText('1')
            try:
                val = int(float(self.textbox_frameNo.text()))
            except Exception:
                val = 1
            if val < 1 or val > self.frame_count:
                val = 1
            self.image_widget.frame_idx = val - 1
        else:
            self.textbox_frameNo.setEnabled(False)
            self.textbox_frameNo.setPlaceholderText('1-N')
            self.textbox_frameNo.setToolTip('Select frame number (enabled for NXS)')
            self.image_widget.frame_idx = 0

    def on_radio_button_toggled(self):
        if self.radioButtonRadial.isChecked():
            self.comboBox.clear()
            self.comboBox.addItems(['q', '2theta', 'pixel', 'q (unsmoothed)', '2theta (unsmoothed)', 'pixel (unsmoothed)'])
        elif self.radioButtonAngular.isChecked():
            self.comboBox.clear()
            self.comboBox.addItems(['Theta'])

    def update_image_finished(self):
        # 在 1D 显示时避免自动刷新覆盖
        if getattr(self.image_widget, 'windowstate', 0) == 3:
            return
        if self.rb1.isChecked():
            self.image_widget.update_image()
        if self.rb2.isChecked():
            self.image_widget.Cut()

    def on_radiobutton_toggled(self, button, func):
        if button.isChecked():
            func()

    def select_file(self):
        # 打开文件选择器对话框
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Select File', '',
            'Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;CBF (*.cbf);;EDF (*.edf);;NXS (*.nxs);;All Files (*)',
            options=options)
        if file_name:
            try:
                self.file_name = file_name
                # 更新图像
                self.update_image(self.file_name)
                # 更新帧选择器（针对 NXS）
                self._update_frame_selector_for(self.file_name)
                if not self.textbox_startAngle.text():
                    self.textbox_startAngle.setText('-180')
                if not self.textbox_endAngle.text():
                    self.textbox_endAngle.setText('180')
                if not self.textbox_innerRadius.text():
                    self.textbox_innerRadius.setText('0')
                if not self.textbox_outerRadius.text():
                    self.textbox_outerRadius.setText('1000')
            except:
                QMessageBox.warning(self, "Error",
                                    "The selected file cannot be read. Please select a valid TIFF or JPG file.")

    def update_rigionValues(self):
        try:
            # 读取文本框的值
            start_angle = float(self.textbox_startAngle.text())
            end_angle = float(self.textbox_endAngle.text())
            inner_radius = float(self.textbox_innerRadius.text())
            outer_radius = float(self.textbox_outerRadius.text())
            # 将值更新到image_widget对象中
            self.image_widget.startAngle = start_angle
            self.image_widget.endAngle = end_angle
            self.image_widget.innerRadius = inner_radius
            self.image_widget.outerRadius = outer_radius

        except ValueError:
            return

    def update_image(self, file_name):
        # 更新ImageWidget的file_name属性

        # self.image_widget.file_name = self.file_name
        self.image_widget.file_name = file_name
        # 根据文件类型更新帧选择器状态
        self._update_frame_selector_for(file_name)
        if self.rb1.isChecked():
            # 调用ImageWidget的update_image()方法
            self.image_widget.update_image()
        if self.rb2.isChecked():
            # 调用ImageWidget的Cut()方法
            self.image_widget.Cut()
        # 确保 ROI 文本框有默认值（兼容拖拽加载等路径）
        try:
            if not self.textbox_startAngle.text():
                self.textbox_startAngle.setText('-180')
            if not self.textbox_endAngle.text():
                self.textbox_endAngle.setText('180')
            if not self.textbox_innerRadius.text():
                self.textbox_innerRadius.setText('0')
            if not self.textbox_outerRadius.text():
                # 用图像尺寸的半对角线作为较大的默认外半径
                # 若当前还没有图像，退化为 1000
                try:
                    base = getattr(self.image_widget, 'last_cut_image_raw', None) if getattr(self.image_widget, 'current_view_is_cut', False) else getattr(self.image_widget, 'last_normal_image_raw', None)
                    if base is None and self.image_widget.file_name:
                        base = load_image_matrix(self.image_widget.file_name, frame_idx=self.image_widget.frame_idx)
                    if base is not None:
                        h, w = base.shape[:2]
                        default_or = int((h**2 + w**2) ** 0.5 / 2)
                    else:
                        default_or = 1000
                except Exception:
                    default_or = 1000
                self.textbox_outerRadius.setText(str(default_or))
        except Exception:
            pass

    def on_frame_no_changed(self):
        # 仅在启用时响应
        if not self.textbox_frameNo.isEnabled():
            return
        try:
            val = int(float(self.textbox_frameNo.text()))
        except Exception:
            val = 1
        # 限制到范围 1..frame_count
        if val < 1:
            val = 1
        if val > self.frame_count:
            val = self.frame_count
        # 回写规范化值
        self.textbox_frameNo.setText(str(val))
        # 更新 ImageWidget 的帧索引（0-based）并刷新显示
        self.image_widget.frame_idx = val - 1
        # 避免在 1D 显示（windowstate==3）时被自动刷新覆盖
        if getattr(self.image_widget, 'windowstate', 0) == 3:
            return
        if self.rb1.isChecked():
            self.image_widget.update_image()
        elif self.rb2.isChecked():
            self.image_widget.Cut()

    def select_outputdir(self):
        # 获取用户选择的导出目录路径
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder_path:
            self.output_folder = folder_path
            # 更新导出文件夹的路径文本框
            self.textbox_outputdir.setText(folder_path)

    def on_intRegion_button_clicked(self):
        if self.image_widget.file_name:
            cb_min = float(self.textbox_min.text())
            cb_max = float(self.textbox_max.text())
            x_center = self.image_widget.x_Center
            y_center = self.image_widget.y_Center
            try:
                result = self.image_widget.int_region(cb_min, cb_max, x_center, y_center)
                # 用户取消 ROI 选择时，返回 None，静默退出，不弹框
                if not result:
                    return
                im_norm, start_angle, end_angle, inner_radius, outer_radius = result
                if start_angle is not None and end_angle is not None and inner_radius is not None and outer_radius is not None:
                    self.textbox_startAngle.setText(str(round(math.degrees(start_angle), 2)))
                    self.textbox_endAngle.setText(str(round(math.degrees(end_angle), 2)))
                    self.textbox_innerRadius.setText(str(round(inner_radius, 2)))
                    self.textbox_outerRadius.setText(str(round(outer_radius, 2)))
                    self.update_rigionValues()

            except ValueError as ve:
                print("Error:", ve)
                QMessageBox.warning(self, "Error", "Invalid input values. Please check and try again.")
            except Exception as e:
                print("Error:", e)
                # 降低不必要的弹窗干扰：记录日志，避免频繁弹窗
                QMessageBox.warning(self, "Error", "ROI selection failed. Please try again.")
    # def export_image(self):
    #     if self.output_folder:
    #         # 获取用户选择的文件名并拼接文件路径
    #         file_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.file_name))[0] + '.jpg')
    #         # 获取当前图像的QPixmap对象
    #         pixmap = self.image_widget.label.pixmap()
    #         # 将QPixmap对象保存为jpg格式的文件
    #         pixmap.save(file_path, 'jpg')
    #     else:
    #         # 如果导出文件夹路径未设置，弹出提示信息
    #         QMessageBox.warning(self, '提示', '请先选择导出文件夹！')

    def export_image(self):
        # 获取用户选择的文件名
        self.update_output_folder()
        file_name = self.file_name

        if not file_name:
            return

        # 获取用户选择的文件名并拼接文件路径
        if self.insitustate == 0:
            file_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.file_name))[0] + '.jpg')
        if self.insitustate == 1:
            # 创建 image 文件夹
            folder_name = os.path.splitext(os.path.basename(file_name))[0]
            image_folder_path = os.path.join(self.output_folder, 'image')
            os.makedirs(image_folder_path, exist_ok=True)
            file_path = os.path.join(image_folder_path, folder_name + '.jpg')

        # 批处理时追加后缀（如帧号）
        if getattr(self, 'batch_suffix', ""):
            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_{self.batch_suffix}{ext}"

        if self.image_widget.windowstate == 3: #判断当前图窗是否为一维图像
            # file_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.file_name))[0] + '.jpg')
            self.image_widget.fig.savefig(file_path, dpi=300)
            return
        if self.rb1.isChecked():
            # 使用 Matplotlib 直接导出（不做 255 规范化）
            cb_min = float(self.textbox_min.text())
            cb_max = float(self.textbox_max.text())
            im = load_image_matrix(file_name, frame_idx=self.image_widget.frame_idx)
            bad_mask = (im >= self.image_widget.threshold_max) | (im < self.image_widget.threshold_min) | np.isnan(im)
            A = np.ma.masked_array(im, mask=bad_mask)
            if self.flip.isChecked():
                A = np.flipud(A)

            fig, ax = plt.subplots()
            img = ax.imshow(A, cmap='jet', vmin=cb_min, vmax=cb_max, aspect='equal')
            fig.colorbar(img)
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        if self.rb2.isChecked():

            # file_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.file_name))[0] + '.jpg')
            self.image_widget.fig.savefig(file_path, dpi=300)

    def export_integral_data(self):
        try:
            x, y = self.image_widget.calculate_integral()
            if x is not None and y is not None:
                file_name = os.path.join(self.output_folder,
                                         os.path.splitext(os.path.basename(self.file_name))[0] + '.txt')
                print(file_name)
                with open(file_name, 'a') as f:
                    for i in range(len(x)):
                        f.write(f"{x[i]}\t{y[i]}\n")
                QMessageBox.information(self, "Export Success", "Integral data has been exported successfully!")
        except:
            QMessageBox.warning(self, "Warning", "Failed to export data", QMessageBox.Ok)

    def update_output_folder(self):
        folder_path = self.textbox_outputdir.text()
        if folder_path:
            self.output_folder = folder_path

    def set_file_name(self, file_name):
        self.file_name = file_name

    def set_batch_processor(self, batch_processor):
        self.batch_processor = batch_processor

    def update_batch_processor_filename(self):
        self.file_name = self.batch_processor.filename

    def save_parameters(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Parameters', '', 'JSON Files (*.json);;All Files (*)', options=options)
            if not file_path:
                return
            data = {
                'imageLayout': {
                    'mode': 'Cut' if self.rb2.isChecked() else 'Original',
                    'flip': bool(self.flip.isChecked()),
                    'colorbar_min': self.textbox_min.text(),
                    'colorbar_max': self.textbox_max.text(),
                    'roi': {
                        'startAngle': self.textbox_startAngle.text(),
                        'endAngle': self.textbox_endAngle.text(),
                        'innerRadius': self.textbox_innerRadius.text(),
                        'outerRadius': self.textbox_outerRadius.text(),
                    },
                    'x_axis_unit_index': int(self.comboBox.currentIndex()),
                    'y_axis_scale_index': int(self.comboBox2.currentIndex()),
                    'integration_mode': 'Azimuthal' if self.radioButtonAngular.isChecked() else 'Radial',
                    'output_folder': self.textbox_outputdir.text(),
                    'frame_no': self.textbox_frameNo.text(),
                },
                'parameter': {
                    'Angle_incidence': self.parameter.Angle_incidence.text(),
                    'x_Center': self.parameter.x_Center.text(),
                    'y_Center': self.parameter.y_Center.text(),
                    'distance': self.parameter.distance.text(),
                    'pixel_x': self.parameter.pixel_x.text(),
                    'pixel_y': self.parameter.pixel_y.text(),
                    'lamda': self.parameter.lamda.text(),
                    'Qr_min': self.parameter.Qr_min.text(),
                    'Qr_max': self.parameter.Qr_max.text(),
                    'Qz_min': self.parameter.Qz_min.text(),
                    'Qz_max': self.parameter.Qz_max.text(),
                    'threshold_min': self.parameter.threshold_min.text(),
                    'threshold_max': self.parameter.threshold_max.text(),
                    'numbin': self.parameter.numbin.text(),
                },
                'batch': {
                    'folder_path': self.batch_processor.folder_path_label.text() if hasattr(self, 'batch_processor') else '',
                    'pattern': self.batch_processor.pattern_input.text() if hasattr(self, 'batch_processor') else '',
                    'export_images': bool(self.batch_processor.export_image_check.isChecked()) if hasattr(self, 'batch_processor') else False,
                    'export_curves': bool(self.batch_processor.export_curve_check.isChecked()) if hasattr(self, 'batch_processor') else False,
                    'background_removal': bool(self.batch_processor.background_removal_check.isChecked()) if hasattr(self, 'batch_processor') else False,
                    'background_init_img': self.batch_processor.background_init_img.text() if hasattr(self, 'batch_processor') else '',
                    'background_min': self.batch_processor.background_min.text() if hasattr(self, 'batch_processor') else '',
                    'background_max': self.batch_processor.background_max.text() if hasattr(self, 'batch_processor') else '',
                    'insitu_file': self.batch_processor.insitu_txt_label.text() if hasattr(self, 'batch_processor') else '',
                }
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, 'Saved', 'Parameters saved successfully.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to save parameters: {e}')

    def import_parameters(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getOpenFileName(self, 'Import Parameters', '', 'JSON Files (*.json);;All Files (*)', options=options)
            if not file_path:
                return
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            il = data.get('imageLayout', {})
            # Mode
            mode = il.get('mode', 'Original')
            if mode == 'Cut':
                self.rb2.setChecked(True)
            else:
                self.rb1.setChecked(True)
            # Flip
            self.flip.setChecked(bool(il.get('flip', False)))
            # Colorbar
            self.textbox_min.setText(str(il.get('colorbar_min', self.textbox_min.text())))
            self.textbox_max.setText(str(il.get('colorbar_max', self.textbox_max.text())))
            # ROI
            roi = il.get('roi', {})
            self.textbox_startAngle.setText(str(roi.get('startAngle', self.textbox_startAngle.text())))
            self.textbox_endAngle.setText(str(roi.get('endAngle', self.textbox_endAngle.text())))
            self.textbox_innerRadius.setText(str(roi.get('innerRadius', self.textbox_innerRadius.text())))
            self.textbox_outerRadius.setText(str(roi.get('outerRadius', self.textbox_outerRadius.text())))
            self.update_rigionValues()
            # Axes
            xa_idx = int(il.get('x_axis_unit_index', self.comboBox.currentIndex()))
            ya_idx = int(il.get('y_axis_scale_index', self.comboBox2.currentIndex()))
            self.comboBox.setCurrentIndex(xa_idx)
            self.comboBox2.setCurrentIndex(ya_idx)
            # Integration mode
            integ_mode = il.get('integration_mode', 'Radial')
            if integ_mode == 'Azimuthal':
                self.radioButtonAngular.setChecked(True)
            else:
                self.radioButtonRadial.setChecked(True)
            # Output folder and frame
            self.textbox_outputdir.setText(il.get('output_folder', self.textbox_outputdir.text()))
            self.update_output_folder()
            self.textbox_frameNo.setText(il.get('frame_no', self.textbox_frameNo.text()))
            self.on_frame_no_changed()

            # Parameter panel
            prm = data.get('parameter', {})
            def _set(param_widget, key):
                val = prm.get(key)
                if val is not None:
                    getattr(self.parameter, key).setText(str(val))
            for k in ['Angle_incidence','x_Center','y_Center','distance','pixel_x','pixel_y','lamda',
                      'Qr_min','Qr_max','Qz_min','Qz_max','threshold_min','threshold_max','numbin']:
                _set(self.parameter, k)
            # Push into image_widget
            self.parameter.update_image_widget()

            # Batch panel
            bt = data.get('batch', {})
            if hasattr(self, 'batch_processor'):
                self.batch_processor.folder_path_label.setText(bt.get('folder_path', self.batch_processor.folder_path_label.text()))
                self.batch_processor.pattern_input.setText(bt.get('pattern', self.batch_processor.pattern_input.text()))
                self.batch_processor.export_image_check.setChecked(bool(bt.get('export_images', self.batch_processor.export_image_check.isChecked())))
                self.batch_processor.export_curve_check.setChecked(bool(bt.get('export_curves', self.batch_processor.export_curve_check.isChecked())))
                self.batch_processor.background_removal_check.setChecked(bool(bt.get('background_removal', self.batch_processor.background_removal_check.isChecked())))
                self.batch_processor.background_init_img.setText(bt.get('background_init_img', self.batch_processor.background_init_img.text()))
                self.batch_processor.background_min.setText(bt.get('background_min', self.batch_processor.background_min.text()))
                self.batch_processor.background_max.setText(bt.get('background_max', self.batch_processor.background_max.text()))
                if bt.get('insitu_file'):
                    self.batch_processor.insitu_txt_label.setText(bt.get('insitu_file'))

            # Refresh image view
            self.update_image_finished()
            QMessageBox.information(self, 'Imported', 'Parameters imported successfully.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to import parameters: {e}')

class Parameter(QWidget):
    def __init__(self, parent=None, image_widget = None):
        super().__init__(parent)
        self.init_ui()
        self.image_widget = image_widget

    def init_ui(self):
        # 创建文本框并初始化
        settings = QSettings('mycompany', 'myapp')
        self.Angle_incidence = QLineEdit(settings.value('Angle_incidence', '0.5'))
        self.x_Center = QLineEdit(settings.value('x_Center', '0'))
        self.y_Center = QLineEdit(settings.value('y_Center', '0'))
        self.distance = QLineEdit(settings.value('distance', '300'))
        self.pixel_x = QLineEdit(settings.value('pixel_x', '73.2'))
        self.pixel_y = QLineEdit(settings.value('pixel_y', '73.2'))
        self.lamda = QLineEdit(settings.value('lamda', '1.24'))
        self.Qr_min = QLineEdit(settings.value('Qr_min', '-121'))
        self.Qr_max = QLineEdit(settings.value('Qr_max', '-121'))
        self.Qz_min = QLineEdit(settings.value('Qz_min', '-121'))
        self.Qz_max = QLineEdit(settings.value('Qz_max', '-121'))
        self.threshold_min = QLineEdit(settings.value('threshold_min', '0'))
        self.threshold_max = QLineEdit(settings.value('threshold_max', '1000000'))
        self.numbin = QLineEdit(settings.value('numbin', '500'))

        self.Angle_incidence = QLineEdit(self)
        self.Angle_incidence.setText(self.checkFloatValue(settings.value('Angle_incidence', '0.5')))
        self.x_Center = QLineEdit(self)
        self.x_Center.setText(self.checkFloatValue(settings.value('x_Center', '0')))
        self.y_Center = QLineEdit(self)
        self.y_Center.setText(self.checkFloatValue(settings.value('y_Center', '0')))
        self.distance = QLineEdit(self)
        self.distance.setText(self.checkFloatValue(settings.value('distance', '300')))
        self.pixel_x = QLineEdit(self)
        self.pixel_x.setText(self.checkFloatValue(settings.value('pixel_x', '73.2')))
        self.pixel_y = QLineEdit(self)
        self.pixel_y.setText(self.checkFloatValue(settings.value('pixel_y', '73.2')))
        self.lamda = QLineEdit(self)
        self.lamda.setText(self.checkFloatValue(settings.value('lamda', '1.24')))
        self.Qr_min = QLineEdit(self)
        self.Qr_min.setText(self.checkFloatValue(settings.value('Qr_min', '-121')))
        self.Qr_max = QLineEdit(self)
        self.Qr_max.setText(self.checkFloatValue(settings.value('Qr_max', '-121')))
        self.Qz_min = QLineEdit(self)
        self.Qz_min.setText(self.checkFloatValue(settings.value('Qz_min', '-121')))
        self.Qz_max = QLineEdit(self)
        self.Qz_max.setText(self.checkFloatValue(settings.value('Qz_max', '-121')))
        self.threshold_min = QLineEdit(self)
        self.threshold_min.setText(self.checkFloatValue(settings.value('threshold_min', '0')))
        self.threshold_max = QLineEdit(self)
        self.threshold_max.setText(self.checkFloatValue(settings.value('threshold_max', '1000000')))
        self.numbin = QLineEdit(self)
        self.numbin.setText(self.checkFloatValue(settings.value('numbin', '500')))

        # 将各个参数设为类属性
        self.Angle_incidence_value = float(self.Angle_incidence.text())
        self.x_Center_value = float(self.x_Center.text())
        self.y_Center_value = float(self.y_Center.text())
        self.distance_value = float(self.distance.text())
        self.pixel_x_value = float(self.pixel_x.text())
        self.pixel_y_value = float(self.pixel_y.text())
        self.lamda_value = float(self.lamda.text())
        self.Qr_min_value = float(self.Qr_min.text())
        self.Qr_max_value = float(self.Qr_max.text())
        self.Qz_min_value = float(self.Qz_min.text())
        self.Qz_max_value = float(self.Qz_max.text())
        self.threshold_min_value = float(self.threshold_min.text())
        self.threshold_max_value = float(self.threshold_max.text())
        try:
            self.numbin_value = int(float(self.numbin.text()))
        except ValueError:
            self.numbin_value = 500
            self.numbin.setText(str(self.numbin_value))

        # 绑定文本框的输入与类属性
        self.Angle_incidence.editingFinished.connect(
            lambda: self.update_value('Angle_incidence', self.Angle_incidence.text())
        )
        self.x_Center.editingFinished.connect(
            lambda: self.update_value('x_Center', self.x_Center.text())
        )
        self.y_Center.editingFinished.connect(
            lambda: self.update_value('y_Center', self.y_Center.text())
        )
        self.distance.editingFinished.connect(
            lambda: self.update_value('distance', self.distance.text())
        )
        self.pixel_x.editingFinished.connect(
            lambda: self.update_value('pixel_x', self.pixel_x.text())
        )
        self.pixel_y.editingFinished.connect(
            lambda: self.update_value('pixel_y', self.pixel_y.text())
        )
        self.lamda.editingFinished.connect(
            lambda: self.update_value('lamda', self.lamda.text())
        )
        self.Qr_min.editingFinished.connect(
            lambda: self.update_value('Qr_min', self.Qr_min.text())
        )
        self.Qr_max.editingFinished.connect(
            lambda: self.update_value('Qr_max', self.Qr_max.text())
        )
        self.Qz_min.editingFinished.connect(
            lambda: self.update_value('Qz_min', self.Qz_min.text())
        )
        self.Qz_max.editingFinished.connect(
            lambda: self.update_value('Qz_max', self.Qz_max.text())
        )
        self.threshold_min.editingFinished.connect(
            lambda: self.update_value('threshold_min', self.threshold_min.text())
        )
        self.threshold_max.editingFinished.connect(
            lambda: self.update_value('threshold_max', self.threshold_max.text())
        )
        self.numbin.editingFinished.connect(
            lambda: self.update_value('numbin', self.numbin.text())
        )

        self.Angle_incidence.editingFinished.connect(self.update_image_widget_finished)
        self.x_Center.editingFinished.connect(self.update_image_widget_finished)
        self.y_Center.editingFinished.connect(self.update_image_widget_finished)
        self.distance.editingFinished.connect(self.update_image_widget_finished)
        self.pixel_x.editingFinished.connect(self.update_image_widget_finished)
        self.pixel_y.editingFinished.connect(self.update_image_widget_finished)
        self.lamda.editingFinished.connect(self.update_image_widget_finished)
        self.Qr_min.editingFinished.connect(self.update_image_widget_finished)
        self.Qr_max.editingFinished.connect(self.update_image_widget_finished)
        self.Qz_min.editingFinished.connect(self.update_image_widget_finished)
        self.Qz_max.editingFinished.connect(self.update_image_widget_finished)
        self.threshold_min.editingFinished.connect(self.update_image_widget_finished)
        self.threshold_max.editingFinished.connect(self.update_image_widget_finished)
        # self.numbin.editingFinished.connect(self.update_image_widget)

        # 创建布局
        layout = QGridLayout(self)

        layout.addWidget(QLabel('Incidence angle (°):'), 0, 0)
        layout.addWidget(self.Angle_incidence, 0, 1)
        layout.addWidget(QLabel('Center X (pixel):'), 0, 2)
        layout.addWidget(self.x_Center, 0, 3)
        layout.addWidget(QLabel('Center Y (pixel):'), 0, 4)
        layout.addWidget(self.y_Center, 0, 5)
        layout.addWidget(QLabel('Distance (mm):'), 0, 6)
        layout.addWidget(self.distance, 0, 7)
        layout.addWidget(QLabel('Pixel X (µm):'), 1, 0)
        layout.addWidget(self.pixel_x, 1, 1)
        layout.addWidget(QLabel('Pixel Y (µm):'), 1, 2)
        layout.addWidget(self.pixel_y, 1, 3)
        layout.addWidget(QLabel('Wavelength (Å):'), 1, 4)
        layout.addWidget(self.lamda, 1, 5)
        layout.addWidget(QLabel('Cut Qr_min:'), 2, 0)
        layout.addWidget(self.Qr_min, 2, 1)
        layout.addWidget(QLabel('Cut Qr_max:'), 2, 2)
        layout.addWidget(self.Qr_max, 2, 3)
        layout.addWidget(QLabel('Cut Qz_min:'), 2, 4)
        layout.addWidget(self.Qz_min, 2, 5)
        layout.addWidget(QLabel('Cut Qz_max:'), 2, 6)
        layout.addWidget(self.Qz_max, 2, 7)
        layout.addWidget(QLabel('Mask_min'), 3, 0)
        layout.addWidget(self.threshold_min, 3, 1)
        layout.addWidget(QLabel('Mask_max'), 3, 2)
        layout.addWidget(self.threshold_max, 3, 3)
        layout.addWidget(QLabel('1D points:'), 3, 4)
        layout.addWidget(self.numbin, 3, 5)

    def checkFloatValue(self, value):
        try:
            float_value = float(value)
            return str(float_value)
        except ValueError:
            return '0.0'

    def closeEvent(self, event):
        # Save current settings
        settings = QSettings('mycompany', 'myapp')
        settings.setValue('Angle_incidence', self.Angle_incidence.text())
        settings.setValue('x_Center', self.x_Center.text())
        settings.setValue('y_Center', self.y_Center.text())
        settings.setValue('distance', self.distance.text())
        settings.setValue('pixel_x', self.pixel_x.text())
        settings.setValue('pixel_y', self.pixel_y.text())
        settings.setValue('lamda', self.lamda.text())

        event.accept()

    def update_image_widget_finished(self):
        if self.image_widget.windowstate == 3:
            self.image_widget.calculate_integral()
            return
        if self.image_layout.rb1.isChecked():
            self.image_widget.update_image()
        elif self.image_layout.rb2.isChecked():
            self.image_widget.Cut()

    def update_image_widget(self):
        try:
            self.Angle_incidence_value = float(self.Angle_incidence.text())
            self.x_Center_value = float(self.x_Center.text())
            self.y_Center_value = float(self.y_Center.text())
            self.distance_value = float(self.distance.text())
            self.pixel_x_value = float(self.pixel_x.text())
            self.pixel_y_value = float(self.pixel_y.text())
            self.lamda_value = float(self.lamda.text())
            self.Qr_min_value = float(self.Qr_min.text())
            self.Qr_max_value = float(self.Qr_max.text())
            self.Qz_min_value = float(self.Qz_min.text())
            self.Qz_max_value = float(self.Qz_max.text())
            self.threshold_min_value = float(self.threshold_min.text())
            self.threshold_max_value = float(self.threshold_max.text())
            self.numbin_value = max(1, int(float(self.numbin.text())))

            self.image_widget.update_parameters(self)
        except:
            return



    def _get_float_or_default(self, input_str):
        try:
            value = float(input_str)
        except ValueError:
            value = -121
        return value
    def set_image_layout(self, image_layout):
        self.image_layout = image_layout

    def update_value(self, key, text):
        try:
            if key == 'numbin':
                value = int(float(text)) if text != '' else 0
                if value < 1:
                    value = 1
            else:
                value = float(text) if text != '' else 0.0
        except ValueError:
            value = getattr(self, key + '_value', 0.0)
        setattr(self, key + '_value', value)
        self.update_image_widget()

class BatchProcessor(QWidget):
    def __init__(self, image_widget, image_layout):
        super().__init__()

        self.image_widget = image_widget
        self.image_layout = image_layout

        # 创建控件
        self.folder_label = QLabel("Select Folder:")
        self.folder_path_label = QLabel()
        self.folder_select_button = QPushButton("Browse")

        self.pattern_label = QLabel("Filename pattern:")
        self.pattern_input = QLineEdit()
        self.pattern_input.setText('*.nxs')
        self.process_button = QPushButton("Batch Process")
        self.hotmap_button = QPushButton("In-situ Heatmap Preview")

        self.export_image_check = QCheckBox("Export images")
        self.export_curve_check = QCheckBox("Export 1D curves")
        self.background_removal_check = QCheckBox("Background subtraction")
        self.background_init_img = QLineEdit()
        self.background_init_img.setPlaceholderText('init_img')
        self.background_init_img.setText('1')
        self.background_init_img.setFixedWidth(100)
        self.background_min = QLineEdit()
        self.background_min.setPlaceholderText('1D_min')
        self.background_min.setFixedWidth(100)
        self.background_max = QLineEdit()
        self.background_max.setPlaceholderText('1D_max')
        self.background_max.setFixedWidth(100)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.insitu_txt_button = QPushButton('Import in-situ file')
        self.insitu_txt_button.setFixedWidth(220)
        self.insitu_txt_label = QLabel()

        self.stop_button = QPushButton('Stop')

        # 设置布局
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_path_label)
        folder_layout.addWidget(self.folder_select_button)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(self.pattern_label)
        pattern_layout.addWidget(self.pattern_input)

        check_layout = QHBoxLayout()
        check_layout.addWidget(QLabel("Select export types:"))
        check_layout.addWidget(self.export_image_check)
        check_layout.addWidget(self.export_curve_check)
        check_layout.addWidget(self.background_removal_check)
        check_layout.addWidget(QLabel("Background parameters:"))
        check_layout.addWidget(self.background_init_img)
        check_layout.addWidget(self.background_min)
        check_layout.addWidget(self.background_max)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.hotmap_button)
        button_layout.addWidget(self.progress_bar)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.insitu_txt_button)
        input_layout.addWidget(self.insitu_txt_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(folder_layout)
        main_layout.addLayout(pattern_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(check_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(input_layout)




        self.setLayout(main_layout)
        self.output_matrix = None
        # 连接信号槽
        self.folder_select_button.clicked.connect(self.select_folder)
        self.process_button.clicked.connect(self.batch_process)
        self.hotmap_button.clicked.connect(self.hotmap_plot)
        self.insitu_txt_button.clicked.connect(self.insitu_input)
        self.stop_button.clicked.connect(self.stop_loop)
        self.background_init_img.textChanged.connect(self.update_bg_init_param)

    def update_bg_init_param(self, text):
        try:
            self.background_init_img_value = int(text)
        except ValueError:
            self.background_init_img_value = 1

    def insitu_input(self):
        # 显示文件对话框，以选择输入文件
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)

        # 如果选择了文件，则读取该文件并将其解析为二维数组
        if file_name:
            # 加载文本文件，并将数据存储在一个numpy数组中
            data = np.loadtxt(file_name)
            self.output_matrix = data
            self.insitu_txt_label.setText(file_name)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_label.setText(folder_path)

    def batch_process(self):

        # Reset the stop flag
        self.reset_stop_flag()

        folder_path = self.folder_path_label.text()
        if not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Warning", "Please select a valid folder.")
            return

        pattern_str = self.pattern_input.text()
        if not pattern_str:
            QMessageBox.warning(self, "Warning", "Please specify a filename pattern.")
            return

        # 寻找符合通配符模式的文件
        file_list = sorted(glob.glob(folder_path + "/*" + pattern_str))
        if not file_list:
            QMessageBox.warning(self, "Warning", "No matching files found.")
            return

        total_files = len(file_list)
        # 开启原位处理状态码
        self.image_layout.insitustate = 1

        # 扣背景
        if self.background_removal_check.isChecked():
            self.x_bg = None
            # 首先需要绘制一维图片
            index = int(self.background_init_img.text()) - 1
            self.filename = file_list[index]
            x, y = self.export_integral_data()
            try:
                self.xmin = float(self.background_min.text())
            except ValueError:
                self.xmin = None
            try:
                self.xmax = float(self.background_max.text())
            except ValueError:
                self.xmax = None
            remover = BackgroundRemover(x, y, self.xmin, self.xmax)
            # remover.interactive_plot()
            self.x_bg = remover.remove_background()

            # 循环询问是否需要重新计算背景
            while True:
                reply = QMessageBox.question(self, '确认', '是否确定此背景曲线？',
                                                       QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    break
                else:
                    # plt.close('all')
                    # remover = BackgroundRemover(x, y, self.xmin, self.xmax)
                    # self.x_bg = remover.remove_background()
                    return


        output = []
        output_bk = []
        header_cols = []  # names for each Y column

        # 如果包含 .nxs，则基于帧维度处理（第一维）
        nxs_files = [f for f in file_list if f.lower().endswith('.nxs')]
        if nxs_files:
            rep = nxs_files[0]
            # 检测帧数
            frame_count = self.image_layout._detect_nxs_frame_count(rep)
            if frame_count < 1:
                frame_count = 1

            for fi in range(frame_count):

                if self.stop_flag:
                    self.progress_bar.setValue(0)
                    QMessageBox.warning(self, 'Warning', 'The process was stopped by the user.')
                    return

                # 设置代表文件与帧索引
                self.filename = rep
                self.image_widget.file_name = rep
                self.image_widget.frame_idx = fi
                # 设置批处理文件名后缀（帧号 1-based）
                self.image_layout.batch_suffix = f"f{fi+1:04d}"

                # 一维导出
                if self.export_curve_check.isChecked():
                    try:
                        x, y = self.export_integral_data()
                        if fi == 0:
                            output.append(x)
                        output.append(y)
                        # Add header name for this frame
                        base_name = os.path.splitext(os.path.basename(rep))[0]
                        suffix = getattr(self.image_layout, 'batch_suffix', f"f{fi+1:04d}")
                        header_cols.append(f"{base_name}_{suffix}")
                    except:
                        QMessageBox.warning(self, "Warning", "Integration aborted!", QMessageBox.Ok)
                        self.image_layout.insitustate = 0
                        self.image_layout.batch_suffix = ""
                        return

                # 二维导出
                if self.export_image_check.isChecked():
                    if self.image_layout.rb2.isChecked():
                        self.image_widget.Cut()
                    if self.image_layout.rb1.isChecked():
                        self.image_widget.update_image()
                    self.image_layout.export_image()

                # 进度（按帧）
                progress = (fi + 1) / frame_count * 100
                self.progress_bar.setValue(int(round(progress)))
                QCoreApplication.processEvents()

                plt.close('all')
                fig, ax = plt.subplots()
                if self.background_removal_check.isChecked() and self.x_bg is not None:
                    x, y = self.export_integral_data()
                    idx = [np.abs(x - xi).argmin() for xi in self.x_bg]
                    y_bg = y[idx]
                    interp_spline = make_interp_spline(self.x_bg, y_bg, k=2)
                    y_bg_interp = interp_spline(x)
                    y_corrected = y - y_bg_interp
                    if fi == 0:
                        output_bk.append(x)
                    output_bk.append(y_corrected)
                    ax.clear()
                    ax.plot(x, y_corrected)
                    ax.set_title(f'Frame {fi + 1}')
                    fig.canvas.draw()
                    plt.pause(0.001)

            # 清理后缀
            self.image_layout.batch_suffix = ""

        else:
            # 遍历符合条件的文件并进行处理（普通图片按文件顺序）
            for i, filepath in enumerate(file_list):

                if self.stop_flag:
                    self.progress_bar.setValue(0)
                    QMessageBox.warning(self, 'Warning', 'The process was stopped by the user.')
                    return

                filename = os.path.basename(filepath)

                # 设置image_widget的filename并调用Cut()、update_image()和export_image()
                self.filename = filepath
                self.image_widget.update_batch_processor_filename()
                self.image_layout.update_batch_processor_filename()

                # 如果一维被勾选上
                if self.export_curve_check.isChecked():
                    try:
                        x, y = self.export_integral_data()
                        if i == 0:
                            output.append(x)
                        output.append(y)
                        # Add header name for this file
                        header_cols.append(os.path.splitext(os.path.basename(filepath))[0])

                    except:
                        QMessageBox.warning(self, "Warning", "Integration aborted!", QMessageBox.Ok)
                        self.image_layout.insitustate = 0
                        return

                # 如果二维导出被勾选上
                if self.export_image_check.isChecked():
                    if self.image_layout.rb2.isChecked():
                        self.image_widget.Cut()
                    if self.image_layout.rb1.isChecked():
                        self.image_widget.update_image()
                    self.image_layout.export_image()


                # 更新进度条
                progress = (i + 1) / total_files * 100
                self.progress_bar.setValue(int(round(progress)))
                # 强制处理未处理的事件，以便更新界面
                QCoreApplication.processEvents()

                plt.close('all')
                fig, ax = plt.subplots()
                # 扣背底循环
                if self.background_removal_check.isChecked() and self.x_bg is not None:
                    x, y = self.export_integral_data()
                    # 搜索self.x_bg在x中对应的索引
                    idx = [np.abs(x - xi).argmin() for xi in self.x_bg]
                    # 获取对应的y值
                    y_bg = y[idx]
                    # 使用样条插值
                    interp_spline = make_interp_spline(self.x_bg, y_bg, k=2)
                    y_bg_interp = interp_spline(x)
                    # 扣除背景曲线，得到扣除背景后的曲线
                    y_corrected = y - y_bg_interp
                    # 导出数据
                    if i == 0:
                        output_bk.append(x)
                    output_bk.append(y_corrected)

                    # 清空Axes并绘制新的数据
                    ax.clear()
                    ax.plot(x, y_corrected)
                    ax.set_title('Iteration %d' % (i + 1))
                    # 刷新画布
                    fig.canvas.draw()
                    # 保证窗口能够响应事件
                    plt.pause(0.001)


        # 显示窗口
        plt.show()

        # 如果一维被勾选上，导出txt数据
        if self.export_curve_check.isChecked():
            # 定义 1D 文件夹
            image_folder_path = os.path.join(self.image_layout.output_folder, '1D')
            file_path = os.path.join(image_folder_path, 'output.txt')

            # 转换output为numpy矩阵
            output_matrix = np.column_stack(output)
            # 保存矩阵为txt文件
            header_line = "q/chi " + " ".join(header_cols) if header_cols else "q/chi"
            np.savetxt(file_path, output_matrix, fmt='%.6f', delimiter=' ', header=header_line, comments='# ')
            self.output_matrix = output_matrix
            self.insitu_txt_label.setText(file_path)
        self.image_layout.insitustate = 0
        # QMessageBox.information(None, "完成", "已完成！")
        # 如果一维被勾选上，导出txt数据
        if self.background_removal_check.isChecked():
            # 定义 1D 文件夹
            image_folder_path = os.path.join(self.image_layout.output_folder, '1D')
            file_path = os.path.join(image_folder_path, 'output_subBk.txt')

            # 转换output为numpy矩阵
            output_bk_matrix = np.column_stack(output_bk)
            # 保存矩阵为txt文件
            header_line_bk = "q/chi " + " ".join(header_cols) if header_cols else "q/chi"
            np.savetxt(file_path, output_bk_matrix, fmt='%.6f', delimiter=' ', header=header_line_bk, comments='# ')
            self.output_matrix_bk = output_bk_matrix
            # self.insitu_txt_label.setText(file_path)

        self.image_layout.insitustate = 0

        plt.close()

    # def on_click(self, event):
    #    The code is dedicated to the beloved Sherry, as a token of affection from Yufeng. 2023-04-29
    #     # 如果鼠标左键被点击
    #     if event.button == 1:
    #         # 获取鼠标点击的x坐标和y坐标
    #         x = event.xdata
    #         y = event.ydata
    #         if x is not None and y is not None:
    #             # 在该位置绘制一个点
    #             self.ax.plot(x, y, 'ro')
    #
    #             # 将鼠标点击的x坐标和y坐标添加到列表中
    #             self.clicks.append((x, y))
    #
    #             # 记录下鼠标点击的x坐标和对应的y值
    #             print('x =', x, 'y =', y)
    #
    #         # 刷新图形
    #         self.fig.canvas.draw_idle()
    #     # 如果鼠标右键被点击
    #     elif event.button == 3:
    #         # 对所有鼠标点击的x和y坐标进行样条插值
    #         xs = [click[0] for click in self.clicks]
    #         ys = [click[1] for click in self.clicks]
    #         spl = make_interp_spline(xs, ys, k=2)
    #         ys_interp = spl(xs)
    #
    #         # 在整个x范围内绘制样条插值曲线
    #         xmin, xmax = self.x.min(), self.x.max()
    #         xs_interp = np.linspace(xmin, xmax, 100)
    #         ys_interp = spl(xs_interp)
    #         self.ax.plot(xs_interp, ys_interp)
    #
    #         # 刷新图形
    #         self.fig.canvas.draw_idle()
    #     # 如果鼠标中键被点击
    #     elif event.button == 2:
    #         # 获取鼠标点击的x坐标和y坐标
    #         x = event.xdata
    #         y = event.ydata
    #         if x is not None and y is not None:
    #             # 在该位置移除一个点
    #             idx = None
    #             for i, click in enumerate(self.clicks):
    #                 if np.abs(click[0] - x) < 0.01 and np.abs(click[1] - y) < 0.01:
    #                     idx = i
    #                     break
    #             if idx is not None:
    #                 del self.clicks[idx]
    #                 self.ax.lines.pop(-1)
    #                 for click in self.clicks:
    #                     self.ax.plot(click[0], click[1], 'ro')
    #                 print('Removed point at x =', x, 'y =', y)
    #
    #             # 刷新图形
    #             self.fig.canvas.draw_idle()

    def hotmap_plot(self):
        try:
            if self.output_matrix.any():
                output_matrix = self.output_matrix
                # 提取x轴和y轴数据
                x = output_matrix[:, 0]
                y = output_matrix[:, 1:]

                # 创建一个新的Figure对象和Axes对象
                fig, ax = plt.subplots()
                # 创建一个新的矩阵，只包含y轴的数据
                y_matrix = y.T
                # 绘制热图
                im = ax.imshow(y_matrix, aspect='auto', cmap='jet', origin='lower',
                               extent=[x.min(), x.max(), 1, len(y_matrix)],
                               interpolation='bilinear')  # 添加双线性插值
                cbar = fig.colorbar(im, ax=ax)

                # 设置x轴和y轴的标签
                ax.set_xlabel('X-axis label')
                ax.set_ylabel('Y-axis label')
                cbar.set_label('Intensity')

                # 显示热图
                plt.show()
            else:
                QMessageBox.warning(self, "Warning", "Please batch-process 1D curves or import in-situ file first.", QMessageBox.Ok)
        except:
            QMessageBox.warning(self, "Warning", "Please batch-process 1D curves or import in-situ file first.", QMessageBox.Ok)

    def export_integral_data(self):
        try:
            # 创建 1D 文件夹
            self.image_layout.file_name = self.filename
            self.image_widget.file_name = self.filename
            # Invalidate cached images/grids so each batch item reloads correctly
            if hasattr(self.image_widget, 'last_normal_image_raw'):
                self.image_widget.last_normal_image_raw = None
            if hasattr(self.image_widget, 'last_cut_image_raw'):
                self.image_widget.last_cut_image_raw = None
            if hasattr(self.image_widget, 'last_cut_qr'):
                self.image_widget.last_cut_qr = None
            if hasattr(self.image_widget, 'last_cut_qz'):
                self.image_widget.last_cut_qz = None
            # Align view flag to current mode so calculate_integral chooses correctly
            try:
                self.image_widget.current_view_is_cut = bool(self.image_layout.rb2.isChecked())
            except Exception:
                self.image_widget.current_view_is_cut = False
            folder_name = os.path.splitext(os.path.basename(self.image_layout.file_name))[0]
            image_folder_path = os.path.join(self.image_layout.output_folder, '1D')
            os.makedirs(image_folder_path, exist_ok=True)
            file_path = os.path.join(image_folder_path, folder_name + '.jpg')
            # 批处理时追加后缀（如帧号）
            if getattr(self.image_layout, 'batch_suffix', ""):
                base, ext = os.path.splitext(file_path)
                file_path = f"{base}_{self.image_layout.batch_suffix}{ext}"

            x, y = self.image_widget.calculate_integral()
            self.image_widget.fig.savefig(file_path, dpi=300) #导出一维图片的jpg格式

            # mask = (x >= float(self.batch_processor.background_min.text())) & (
            #             x <= float(self.batch_processor.background_max.text()))
            # x_selected = x[mask]
            # y_selected = y[mask]
            return x, y

        except:
            return None, None

    def stop_loop(self):
        self.stop_flag = True

    def reset_stop_flag(self):
        self.stop_flag = False

        # Process events to ensure the UI is updated
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec_()

class FileExplorer(QWidget):
    def __init__(self, image_layout, parent=None):
        super().__init__(parent)

        # 创建QFileSystemModel和QTreeView对象
        self.model = QFileSystemModel()
        self.model.setRootPath('')
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        # self.tree.setRootIndex(self.model.index(QDir.currentPath()))

        # 隐藏不需要的列和标题栏，禁用排序
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setHeaderHidden(True)
        self.tree.setSortingEnabled(False)

        # 获取当前目录的QModelIndex对象
        root_index = self.model.index(QDir.currentPath())

        # 设置树视图的根项为当前目录
        # self.tree.setRootIndex(root_index)

        # 遍历从根目录到当前目录的所有路径并展开
        index = root_index
        while index.isValid():
            self.tree.expand(index)
            index = index.parent()

        # 创建一个QLabel对象显示文件路径
        self.file_path = QLabel()

        # 将QTreeView和QLabel添加到QWidget上
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        layout.addWidget(self.file_path)
        self.setLayout(layout)

        # 创建一个 ImageLayout 实例
        self.image_layout = image_layout

        # 连接selectionChanged信号和槽函数
        self.tree.selectionModel().selectionChanged.connect(self.on_selection_changed)
        # 连接QTreeView的双击信号和选择文件的槽函数
        self.tree.doubleClicked.connect(self.on_tree_double_clicked)

    def on_selection_changed(self):
        # 获取当前选中的文件路径并设置到QLabel上
        index = self.tree.currentIndex()
        file_path = self.model.filePath(index)
        self.file_path.setText(file_path)
        # 检查是否选中了一个文件并且文件的扩展名为.tif或.jpg
        # if not self.model.isDir(index) and file_path.lower().endswith(('.tif', '.jpg')):
        #     # 实例化 ImageLayout 类并调用 update_image() 方法
        #     image_layout = ImageLayout(file_name = file_path)
        #     image_layout.update_image()

    def on_tree_double_clicked(self, index):
        # 获取当前双击的文件路径
        file_path = self.model.filePath(index)
        # 检查是否选中了一个文件并且文件的扩展名为.tif或.jpg
        if not self.model.isDir(index) and file_path.lower().endswith(('.tif', '.jpg' ,)):
            self.image_layout.set_file_name(file_path)
            self.image_layout.update_image(file_path)

            if not self.image_layout.textbox_startAngle.text():
                self.image_layout.textbox_startAngle.setText('-180')
            if not self.image_layout.textbox_endAngle.text():
                self.image_layout.textbox_endAngle.setText('180')
            if not self.image_layout.textbox_innerRadius.text():
                self.image_layout.textbox_innerRadius.setText('0')
            if not self.image_layout.textbox_outerRadius.text():
                self.image_layout.textbox_outerRadius.setText('1000')

        else:
            try:
                self.image_layout.set_file_name(file_path)
                self.image_layout.update_image(file_path)

                if not self.image_layout.textbox_startAngle.text():
                    self.image_layout.textbox_startAngle.setText('-180')
                if not self.image_layout.textbox_endAngle.text():
                    self.image_layout.textbox_endAngle.setText('180')
                if not self.image_layout.textbox_innerRadius.text():
                    self.image_layout.textbox_innerRadius.setText('0')
                if not self.image_layout.textbox_outerRadius.text():
                    self.image_layout.textbox_outerRadius.setText('1000')

            except:
                QMessageBox.warning(self, "Error",
                                    "The selected file cannot be read. Please select a valid TIFF or JPG file.")


class BackgroundRemover:
    def __init__(self, x, y, xmin=None, xmax=None):
        self.x = x
        self.y = y
        self.xmin = xmin
        self.xmax = xmax
        self.background_points = []
        self.background_dots = []  # 存储所有的红点
        self.fig, self.ax = plt.subplots()

        # 绑定事件处理器
        # self.cid_button_press = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        # self.cid_button_release = self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        # # self.cid_mouse_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.background_points = list(set(self.background_points))
        self.dragging_point = None

    def validate_input(self):
        try:
            if not self.file_name:
                return

            # 读色域范围
            cb_min = float(self.textbox_min.text())
            cb_max = float(self.textbox_max.text())

            # 选择当前视图源数据（cut 优先）；若无缓存则现读
            base = getattr(self, 'last_cut_image_raw', None) if getattr(self, 'current_view_is_cut', False) else getattr(self, 'last_normal_image_raw', None)
            if base is None:
                base = load_image_matrix(self.file_name, frame_idx=self.frame_idx)
            image = np.array(base, copy=True)
            # 仅裁剪有限值，保留 NaN 作为空区
            finite_mask = np.isfinite(image)
            image[finite_mask & (image > cb_max)] = cb_max
            image[finite_mask & (image < cb_min)] = cb_min

            # 保障 ROI 文本框有值；若为空则填入合理默认
            sa_txt = (self.image_layout.textbox_startAngle.text() or '-180')
            ea_txt = (self.image_layout.textbox_endAngle.text() or '180')
            ir_txt = (self.image_layout.textbox_innerRadius.text() or '0')
            or_txt = self.image_layout.textbox_outerRadius.text()
            if not or_txt:
                try:
                    h, w = image.shape[:2]
                    or_txt = str(int((h**2 + w**2) ** 0.5 / 2))
                except Exception:
                    or_txt = '1000'

            # 解析 ROI 参数
            start_angle = float(sa_txt)
            end_angle = float(ea_txt)
            inner_radius = float(ir_txt)
            outer_radius = float(or_txt)

            # 积分点数
            try:
                num_bins = int(self.numbin)
            except Exception:
                num_bins = 500

            # 中心坐标
            center = [float(self.x_Center), float(self.y_Center)]

            # 执行积分并显示 1D 曲线
            x, y = self.radial_integral(image, center, start_angle, end_angle, inner_radius, outer_radius, num_bins)
            return x, y
        except Exception as e:
            # 避免静默失败，打印调试信息但不弹窗
            print('[Integrate] Failed:', repr(e))
            return
    def on_right_click(self, event):
        if event.inaxes != self.ax:
            return
        if len(self.background_points) <= 2:
            print("Cannot remove initial points.")
            return
        x = event.xdata
        # Get the nearest y value to the clicked x value
        idx = np.abs(self.x - x).argmin()
        nearest_idx = np.argmin([np.abs(p[0] - x) for p in self.background_points])
        nearest_point = self.background_points[nearest_idx]
        min_distance = np.sqrt((x - nearest_point[0]) ** 2 + (self.y[idx] - nearest_point[1]) ** 2)
        distance_threshold = 20  # Custom threshold, you can adjust as needed
        if min_distance > distance_threshold:
            print("No points within the threshold.")
            return
        # Remove the nearest point from the background points list
        del self.background_points[nearest_idx]
        # Remove the corresponding dot from the plot
        self.background_dots[nearest_idx].remove()
        del self.background_dots[nearest_idx]
        # Remove the corresponding line from the plot
        self.ax.lines.pop(nearest_idx)
        # Redraw the remaining background points
        self.ax.legend()
        plt.draw()
        self.update_background()

    # def on_button_press(self, event):
    #     if event.button == 1:  # 左键
    #         self.dragging_point = self.find_nearest_point(event)
    #         if self.dragging_point is not None:
    #             self.dragging_line = self.find_line(self.dragging_point)
    # #
    # def on_button_release(self, event):
    #     if event.button == 1:  # 左键
    #         self.dragging_point = None
    #         self.dragging_line = None
    #
    # def on_mouse_move(self, event):
    #     if not self.dragging_point:
    #         return
    #
    #     if event.inaxes != self.ax:
    #         return
    #
    #     new_x, new_y = event.xdata, self.y[np.abs(self.x - event.xdata).argmin()]
    #     line = self.find_line(self.dragging_point)
    #     if line is not None:
    #         line.set_data([new_x], [new_y])
    #         plt.draw()
    #
    #         # Update the background point and update the background curve
    #         point_idx = self.background_points.index(self.dragging_point)
    #         self.background_points[point_idx] = (new_x, new_y)
    #         self.update_background()
    #
    #         self.dragging_point = (new_x, new_y)

    def find_nearest_point(self, event):
        if event.inaxes != self.ax:
            return None

        x = event.xdata
        # Get the nearest y value to the clicked x value
        idx = np.abs(self.x - x).argmin()
        nearest_point = self.background_points[np.argmin([np.abs(p[0] - x) for p in self.background_points])]
        min_distance = np.sqrt((x - nearest_point[0]) ** 2 + (self.y[idx] - nearest_point[1]) ** 2)

        distance_threshold = 50  # Custom threshold

        if min_distance > distance_threshold:
            return None

        return nearest_point

    def find_line(self, point):
        for line in self.ax.lines:
            if np.isclose(line.get_xdata()[0], point[0]) and np.isclose(line.get_ydata()[0], point[1]):
                return line
        return None

    def update_background_point(self, point_idx, new_x, new_y):
        self.background_points[point_idx] = (new_x, new_y)
        self.update_background()

    def update_background(self):
        self.background_points = list(set(self.background_points))
        self.background_points.sort(key=lambda p: p[0])
        x_bg, y_bg = zip(*self.background_points)
        x_bg = np.array(x_bg)
        y_bg = np.array(y_bg)
        if len(x_bg) < 3:
            print("At least three background points are required.")
            return

        if len(set(x_bg)) != len(x_bg):
            print("Duplicate x coordinates are not allowed.")
            return

        interp_spline = make_interp_spline(x_bg, y_bg, k=2)
        xnew = np.linspace(self.x[0], self.x[-1], len(self.x))
        ynew = interp_spline(xnew)
        if hasattr(self, 'background_line'):
            self.background_line.set_data(xnew, ynew)
        else:
            self.background_line, = self.ax.plot(xnew, ynew, 'r-', label='Background')
        self.ax.legend()
        plt.draw()

    def on_key_press(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    def interactive_plot(self):
        self.plot_initial_data()
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_left_click(
            event) if event.button == 1 else self.on_right_click(event))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def remove_background(self):
        self.plot_initial_data()
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_left_click(
            event) if event.button == 1 else self.on_right_click(event))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        while plt.get_fignums():
            plt.pause(0.1)

        if not self.background_points:
            print("No background points selected.")
            return None

        # Call the fit_background() function to perform the background fitting
        try:
            xnew, ynew = self.fit_background()
            ynew = self.y-ynew
            plt.plot(xnew, ynew, label="Result")
            plt.legend()
            plt.show()

            return self.x_bg

        except:
            self.x_bg = None
            return self.x_bg

    def fit_background(self):
        self.background_points.sort(key=lambda p: p[0])
        x_bg, y_bg = zip(*self.background_points)
        x_bg = np.array(x_bg)
        y_bg = np.array(y_bg)
        self.x_bg = x_bg
        interp_spline = make_interp_spline(x_bg, y_bg, k=2)
        xnew = np.linspace(self.x[0], self.x[-1], len(self.x))
        ynew = interp_spline(xnew)
        return xnew, ynew

class SplashScreen(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        label = QLabel(self)
        layout.addWidget(label, 0, Qt.AlignCenter)

        # 使用QResource获取GIF文件
        gif_path = ":/logo.gif"

        # 设置启动画面的背景GIF
        movie = QMovie(gif_path)
        label.setMovie(movie)
        movie.start()

if __name__ == '__main__':
    # 注册资源
    qrc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources.qrc')
    QResource.registerResource(qrc_path)

    app = QApplication(sys.argv)

    # 创建启动画面
    splash = SplashScreen()
    splash.show()

    # 使用QEventLoop在创建MainWindow实例期间处理事件循环
    loop = QEventLoop()
    QTimer.singleShot(0, loop.quit)
    loop.exec_()

    # 创建主窗口
    window = MainWindow()

    # 获取当前屏幕的大小
    screen_size = QDesktopWidget().availableGeometry().size()
    # 设置主窗口的大小为屏幕大小的 0.8 倍
    window.resize(int(screen_size.width() * 0.8), int(screen_size.height() * 0.8))
    # 将主窗口居中显示
    window.move(int(screen_size.width() * 0.1), int(screen_size.height() * 0.1))

    # 延迟显示主窗口并关闭启动画面
    def show_main_window():
        window.show()
        splash.close()

    QTimer.singleShot(1000, show_main_window)  # 1000毫秒后执行show_main_window()

    sys.exit(app.exec_())


