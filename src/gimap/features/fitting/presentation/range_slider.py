# utils/widgets/qtrangeslider.py
from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
import math

class QRangeSlider(QSlider):
    # 原有整型信号（保留兼容）
    rangeChanged = pyqtSignal(int, int)
    # 新增浮点信号（两位小数）
    rangeChangedF = pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        """兼容 QRangeSlider(parent) / QRangeSlider(orientation, parent)"""
        orientation = kwargs.pop("orientation", Qt.Horizontal)
        parent = kwargs.pop("parent", None)
        if args:
            if isinstance(args[0], QWidget):
                parent = args[0]
            else:
                orientation = args[0]
                if len(args) > 1 and isinstance(args[1], QWidget):
                    parent = args[1]

        if parent is not None and (not args or isinstance(args[0], QWidget)):
            super().__init__(parent)
            self.setOrientation(orientation)
        else:
            super().__init__(orientation, parent)

        # 小数相关
        self._decimals = 2
        self._scale = 10 ** self._decimals

        # 内部整型状态
        self._min_value = self.minimum()
        self._max_value = self.maximum()

        self._pressed_control = None
        self._hovered_control = None
        self._last_moved = None           # 记录上次移动的是哪一端（"min"/"max"）
        self._interaction_active = False  # 标记当前是否处于用户交互拖动中
        self.setMouseTracking(True)
        self.setTickPosition(QSlider.NoTicks)

        # 默认范围（浮点）
        self.setRangeF(0.00, 100.00)
        self.setMinValueF(20.00)
        self.setMaxValueF(80.00)

        # 外观参数（可按需暴露为属性）
        self._track_thickness = 6  # 轨道厚度
        self._handle_radius = 8    # 手柄半径
        self._handle_radius_hover = 10  # 悬停/按下半径
        self._shadow_radius = 13   # 轻微外圈光晕

        # 颜色：根据当前调色板推断明暗模式，设置现代配色
        self._recompute_colors()

        # 点击安全区（像素）：点击手柄附近即视为选择该手柄，避免误选另一端
        self._safe_click_px = 10

    # ---------- 小数接口 ----------
    def decimals(self):
        return self._decimals

    def setDecimals(self, d: int):
        """修改小数位并保持当前浮点值不变"""
        d = max(0, int(d))
        if d == self._decimals:
            return
        minF, maxF = self.minValueF(), self.maxValueF()
        self._decimals = d
        self._scale = 10 ** self._decimals
        # 重新映射整型范围与位置
        self.setRangeF(self.minimumF(), self.maximumF())
        self.setMinValueF(minF)
        self.setMaxValueF(maxF)

    def minimumF(self):
        return self._i2f(self.minimum())

    def maximumF(self):
        return self._i2f(self.maximum())

    def setRangeF(self, minF: float, maxF: float):
        """设置浮点范围"""
        imin = self._f2i(minF)
        imax = self._f2i(maxF)
        if imax < imin:
            imin, imax = imax, imin
        super().setMinimum(imin)
        super().setMaximum(imax)
        # 步长（可选）：让单步为 0.01
        super().setSingleStep(max(1, int(self._scale / (10 ** self._decimals))))  # 一般为 1
        # 保持当前值在新范围内
        self._min_value = max(imin, min(self._min_value, imax))
        self._max_value = max(self._min_value, min(self._max_value, imax))
        # 范围改变通常意味着外部数据变化，重置上次移动记录
        self._last_moved = None

    def minValueF(self):
        return self._i2f(self._min_value)

    def maxValueF(self):
        return self._i2f(self._max_value)

    def setMinValueF(self, v: float):
        self.setMinValue(self._f2i(v))

    def setMaxValueF(self, v: float):
        self.setMaxValue(self._f2i(v))

    # ---------- 原有整型接口（内部用） ----------
    def minValue(self): return int(self._min_value)
    def maxValue(self): return int(self._max_value)

    def setMinValue(self, value: int):
        value = int(value)
        value = max(self.minimum(), min(value, self._max_value))
        if value != self._min_value:
            self._min_value = value
            self.update()
            self.rangeChanged.emit(self._min_value, self._max_value)
            self.rangeChangedF.emit(self._i2f(self._min_value), self._i2f(self._max_value))
            if self._interaction_active:
                self._last_moved = "min"

    def setMaxValue(self, value: int):
        value = int(value)
        value = min(self.maximum(), max(value, self._min_value))
        if value != self._max_value:
            self._max_value = value
            self.update()
            self.rangeChanged.emit(self._min_value, self._max_value)
            self.rangeChangedF.emit(self._i2f(self._min_value), self._i2f(self._max_value))
            if self._interaction_active:
                self._last_moved = "max"

    # ---------- 事件处理 ----------
    def enterEvent(self, event):
        self._hovered_control = None
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered_control = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._interaction_active = True
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        pos = event.pos()
        min_handle = self._handleRect(self._min_value)
        max_handle = self._handleRect(self._max_value)
        if min_handle.contains(pos):
            self._pressed_control = "min"
        elif max_handle.contains(pos):
            self._pressed_control = "max"
        else:
            # 点击轨道：将最近的手柄跳转至点击位置（安全区内优先对应手柄）
            click_value = self._pixelPosToRangeValue(pos)
            d_min = self._distance_to_handle_center(pos, min_handle)
            d_max = self._distance_to_handle_center(pos, max_handle)
            # 安全区优先：若点击在某个手柄的安全区内，则固定选择该手柄
            if d_min <= self._safe_click_px and d_max > self._safe_click_px:
                target = "min"
            elif d_max <= self._safe_click_px and d_min > self._safe_click_px:
                target = "max"
            else:
                # 其他情况：严格选择最近的手柄
                target = "min" if d_min <= d_max else "max"

            if target == "min":
                self.setMinValue(click_value)
            else:
                self.setMaxValue(click_value)
            self._pressed_control = target
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if not self._pressed_control:
            # 更新悬停状态
            if self._handleRect(self._min_value).contains(pos):
                hovered = "min"
            elif self._handleRect(self._max_value).contains(pos):
                hovered = "max"
            else:
                hovered = None
            if hovered != self._hovered_control:
                self._hovered_control = hovered
                self.update()
            return super().mouseMoveEvent(event)
        v = self._pixelPosToRangeValue(pos)
        if self._pressed_control == "min":
            self.setMinValue(v)
        else:
            self.setMaxValue(v)
        self.update()

    def mouseReleaseEvent(self, event):
        self._pressed_control = None
        self._interaction_active = False
        return super().mouseReleaseEvent(event)

    # ---------- 绘制 ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        opt = QStyleOptionSlider(); self.initStyleOption(opt)

        # 基础几何
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        c1 = self._handleRect(self._min_value).center()
        c2 = self._handleRect(self._max_value).center()
        sel = QRect(c1, c2).normalized()

        # 轨道与选中范围的圆角矩形
        if self.orientation() == Qt.Horizontal:
            track_rect = QRect(groove.left(), groove.center().y() - self._track_thickness // 2,
                               groove.width(), self._track_thickness)
            sel_rect = QRect(sel.left(), groove.center().y() - self._track_thickness // 2,
                             sel.width(), self._track_thickness)
        else:
            track_rect = QRect(groove.center().x() - self._track_thickness // 2, groove.top(),
                               self._track_thickness, groove.height())
            sel_rect = QRect(groove.center().x() - self._track_thickness // 2, sel.top(),
                             self._track_thickness, sel.height())

        radius = self._track_thickness / 2

        # 绘制轨道背景
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._col_track_bg))
        p.drawRoundedRect(track_rect, radius, radius)

        # 绘制选中范围（高亮）
        p.setBrush(QBrush(self._col_track_fg))
        p.drawRoundedRect(sel_rect, radius, radius)

        # 绘制手柄（带轻微光晕/悬停放大）
        for name, center in (("min", c1), ("max", c2)):
            hovered = (self._hovered_control == name) or (self._pressed_control == name)
            hr = self._handle_radius_hover if hovered else self._handle_radius

            # 光晕
            if hovered:
                p.setPen(Qt.NoPen)
                halo_color = QColor(self._col_track_fg)
                halo_color.setAlpha(60)
                p.setBrush(halo_color)
                if self.orientation() == Qt.Horizontal:
                    p.drawEllipse(center, self._shadow_radius, self._shadow_radius)
                else:
                    p.drawEllipse(center, self._shadow_radius, self._shadow_radius)

            # 手柄主体
            p.setPen(QPen(self._col_handle_border, 1))
            p.setBrush(QBrush(self._col_handle))
            p.drawEllipse(center, hr, hr)

    # ---------- 几何/换算 ----------
    def _handleRect(self, value: int):
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        opt.sliderPosition = int(value)
        return self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

    def _pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        if self.orientation() == Qt.Horizontal:
            sliderMin = groove.x()
            sliderMax = groove.right() - handle.width() + 1
            x = max(sliderMin, min(pos.x(), sliderMax))
            denom = max(1, (sliderMax - sliderMin))
            ratio = (x - sliderMin) / denom
        else:
            sliderMin = groove.y()
            sliderMax = groove.bottom() - handle.height() + 1
            y = max(sliderMin, min(pos.y(), sliderMax))
            denom = max(1, (sliderMax - sliderMin))
            ratio = (y - sliderMin) / denom
        return int(round(self.minimum() + ratio * (self.maximum() - self.minimum())))

    def _distance_to_handle_center(self, pos, handle_rect) -> float:
        c = handle_rect.center()
        if self.orientation() == Qt.Horizontal:
            return abs(pos.x() - c.x())
        else:
            return abs(pos.y() - c.y())

    def _f2i(self, v: float) -> int:
        return int(round(v * self._scale))

    def _i2f(self, i: int) -> float:
        return round(i / self._scale, self._decimals)

    # ---------- 主题/配色 ----------
    def _is_dark_mode(self) -> bool:
        try:
            base = self.palette().window().color()
            # Qt HSV value/lightness heuristic
            return base.lightness() < 128
        except Exception:
            return False

    def _recompute_colors(self):
        dark = self._is_dark_mode()
        if dark:
            self._col_track_bg = QColor(70, 74, 82)
            self._col_track_fg = QColor(90, 156, 255)
            self._col_handle = QColor(235, 235, 238)
            self._col_handle_border = QColor(85, 90, 100)
        else:
            self._col_track_bg = QColor(230, 234, 242)
            self._col_track_fg = QColor(56, 142, 255)
            self._col_handle = QColor(255, 255, 255)
            self._col_handle_border = QColor(180, 184, 196)

    def setColors(self, *, track_bg: QColor = None, track_fg: QColor = None,
                  handle: QColor = None, handle_border: QColor = None):
        if track_bg is not None: self._col_track_bg = QColor(track_bg)
        if track_fg is not None: self._col_track_fg = QColor(track_fg)
        if handle is not None: self._col_handle = QColor(handle)
        if handle_border is not None: self._col_handle_border = QColor(handle_border)
        self.update()

    def setSizes(self, *, track_thickness: int = None, handle_radius: int = None,
                 handle_radius_hover: int = None, shadow_radius: int = None):
        if track_thickness is not None: self._track_thickness = max(2, int(track_thickness))
        if handle_radius is not None: self._handle_radius = max(4, int(handle_radius))
        if handle_radius_hover is not None: self._handle_radius_hover = max(self._handle_radius, int(handle_radius_hover))
        if shadow_radius is not None: self._shadow_radius = max(self._handle_radius_hover, int(shadow_radius))
        self.update()

    # ---------- 交互状态复位（外部可调用） ----------
    def resetLastMoved(self):
        """重置‘上次移动’记录。建议在外部数据/显示更新后调用。"""
        self._last_moved = None

    def setSafeClickMargin(self, px: int):
        """设置点击安全区（像素）。在该距离内点击更倾向于选择对应手柄。"""
        try:
            self._safe_click_px = max(0, int(px))
        except Exception:
            pass
