# utils/widgets/qtrangeslider.py
from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QPainter
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
        self.setTickPosition(QSlider.NoTicks)

        # 默认范围（浮点）
        self.setRangeF(0.00, 100.00)
        self.setMinValueF(20.00)
        self.setMaxValueF(80.00)

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

    def setMaxValue(self, value: int):
        value = int(value)
        value = min(self.maximum(), max(value, self._min_value))
        if value != self._max_value:
            self._max_value = value
            self.update()
            self.rangeChanged.emit(self._min_value, self._max_value)
            self.rangeChangedF.emit(self._i2f(self._min_value), self._i2f(self._max_value))

    # ---------- 事件处理 ----------
    def mousePressEvent(self, event):
        opt = QStyleOptionSlider(); self.initStyleOption(opt)
        pos = event.pos()
        min_handle = self._handleRect(self._min_value)
        max_handle = self._handleRect(self._max_value)
        if min_handle.contains(pos):
            self._pressed_control = "min"
        elif max_handle.contains(pos):
            self._pressed_control = "max"
        else:
            self._pressed_control = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._pressed_control:
            return super().mouseMoveEvent(event)
        v = self._pixelPosToRangeValue(event.pos())
        if self._pressed_control == "min":
            self.setMinValue(v)
        else:
            self.setMaxValue(v)

    def mouseReleaseEvent(self, event):
        self._pressed_control = None
        return super().mouseReleaseEvent(event)

    # ---------- 绘制 ----------
    def paintEvent(self, _):
        p = QPainter(self); opt = QStyleOptionSlider(); self.initStyleOption(opt)
        # 槽
        opt.subControls = QStyle.SC_SliderGroove
        self.style().drawComplexControl(QStyle.CC_Slider, opt, p, self)
        # 选中范围
        r1 = self._handleRect(self._min_value).center()
        r2 = self._handleRect(self._max_value).center()
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        sel = QRect(r1, r2).normalized()
        if self.orientation() == Qt.Horizontal:
            sel.setTop(groove.center().y() - 3); sel.setBottom(groove.center().y() + 3)
        else:
            sel.setLeft(groove.center().x() - 3); sel.setRight(groove.center().x() + 3)
        p.setPen(Qt.NoPen); p.setBrush(self.palette().highlight()); p.drawRect(sel)
        # 两个手柄
        for v in (self._min_value, self._max_value):
            opt.sliderPosition = int(v)
            opt.subControls = QStyle.SC_SliderHandle
            self.style().drawComplexControl(QStyle.CC_Slider, opt, p, self)

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

    def _f2i(self, v: float) -> int:
        return int(round(v * self._scale))

    def _i2f(self, i: int) -> float:
        return round(i / self._scale, self._decimals)
