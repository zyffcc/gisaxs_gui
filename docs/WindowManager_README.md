# 窗口管理功能使用说明

## 概述

新增的窗口管理功能提供了自适应缩放和智能定位功能，代码结构清晰，易于维护。

## 主要文件

1. **`window_manager.py`** - 窗口管理器，负责所有窗口相关的操作
2. **`window_config.py`** - 窗口配置，集中管理所有配置参数
3. **`main.py`** - 简化的主文件，只负责调用窗口管理器

## 功能特性

### 1. 自适应缩放
- 根据屏幕分辨率自动调整窗口大小
- 基于1920x1080为基准进行缩放计算
- 限制缩放范围在0.7-1.5之间
- 确保窗口不超过屏幕可用区域的90%

### 2. 智能定位
- 在鼠标所在的屏幕上显示窗口
- 自动居中显示
- 支持多屏幕环境

### 3. 自适应字体
- 根据缩放比例调整字体大小
- 保守的字体缩放策略，避免字体过大或过小

## 配置选项

在 `window_config.py` 中可以调整以下设置：

```python
# 默认窗口尺寸
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 700

# 最小尺寸
MIN_WIDTH = 800
MIN_HEIGHT = 600

# 功能开关
ENABLE_ADAPTIVE_SCALING = True          # 启用自适应缩放
ENABLE_MOUSE_SCREEN_POSITIONING = True  # 启用鼠标屏幕定位
ENABLE_ADAPTIVE_FONT = True             # 启用自适应字体
```

## 如何禁用某些功能

如果您想禁用某些功能，只需修改 `window_config.py` 中的配置：

```python
# 禁用自适应缩放，使用固定尺寸
ENABLE_ADAPTIVE_SCALING = False

# 禁用鼠标屏幕定位，在主屏幕显示
ENABLE_MOUSE_SCREEN_POSITIONING = False

# 禁用自适应字体
ENABLE_ADAPTIVE_FONT = False
```

## 维护性设计

1. **分离关注点**: 窗口管理逻辑完全独立于主应用逻辑
2. **配置驱动**: 所有参数都可通过配置文件调整
3. **易于扩展**: 新增功能只需在窗口管理器中添加
4. **向后兼容**: 可以随时禁用新功能回到原始行为

## 在其他窗口中使用

如果您有其他窗口也需要这些功能，只需：

```python
from window_manager import window_manager

# 在您的窗口类中
def setup_window(self):
    window_manager.setup_adaptive_window(self)
    window_manager.apply_adaptive_font(self)
```
