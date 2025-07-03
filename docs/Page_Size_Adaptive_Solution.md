# 页面大小自适应解决方案 (改进版)

## 问题描述
- 第二个页面（GISAXS predict 页面）比第一个页面（训练集构建页面）小
- `QStackedWidget` 默认行为是按照所有页面中最大的尺寸来显示
- 原来的解决方案导致整个窗口跳动，用户体验不佳

## 改进的解决方案

### 设计原则
1. **不修改原始 UI 文件**：所有修改都在 Python 代码中完成
2. **避免窗口跳动**：只调整内容区域，不改变整个窗口大小
3. **代码整洁**：将复杂逻辑分离到专门的工具类中
4. **最小侵入**：对现有代码的改动最小化

### 核心组件

#### 1. 布局工具类 (`utils/layout_utils.py`)
```python
class LayoutUtils:
    """布局工具类，提供简单的布局调整方法"""
    
    @staticmethod
    def setup_adaptive_stacked_widget(stacked_widget):
        """设置自适应的StackedWidget"""
        # 设置大小策略和页面切换监听
    
    @staticmethod
    def smooth_page_transition(stacked_widget, target_index):
        """平滑的页面切换"""
        # 无动画的直接切换，避免跳动

    @staticmethod
    def compress_page_layout(page):
        """压缩页面布局"""
        # 设置最大高度，调整边距和间距
```

#### 2. 主窗口集成 (`main.py`)
```python
def _setup_stacked_widget(self):
    """设置StackedWidget的自适应行为"""
    from utils.layout_utils import LayoutUtils
    LayoutUtils.setup_adaptive_stacked_widget(self.mainWindowWidget)
```

#### 3. 控制器使用 (`controllers/main_controller.py`)
```python
def _switch_to_gisaxs_predict(self):
    """切换到GISAXS预测页面"""
    from utils.layout_utils import LayoutUtils
    LayoutUtils.smooth_page_transition(self.ui.mainWindowWidget, 1)
```

### 主要改进

1. **移除窗口大小调整**：不再调整整个窗口大小，避免跳动
2. **简化代码结构**：将复杂逻辑移到工具类中
3. **保持响应性**：页面仍然可以根据内容自适应
4. **错误处理**：添加了导入失败的回退机制
5. **页面压缩功能**：新增页面压缩功能，优化预测页面显示

### 技术细节

#### 大小策略设置
- **训练集页面**：`QSizePolicy.Expanding` - 可以扩展填充空间
- **预测页面**：`QSizePolicy.Preferred` - 根据内容首选大小

#### 布局约束
- 对预测页面设置 `QLayout.SetMinimumSize` 约束
- 确保布局根据内容计算最小尺寸

#### 切换机制
- 使用短暂延迟（10ms）确保页面完全加载
- 直接切换页面，不使用动画效果

#### 页面压缩
- 切换到预测页面时，自动触发压缩
- 设置页面最大高度限制（800像素）
- 减小布局边距和间距

## 使用方法

1. **正常启动应用程序**：
   ```bash
   python main.py
   ```

2. **页面切换**：
   - 点击左侧按钮切换页面
   - 页面会平滑切换，内容区域自适应大小
   - 窗口本身不会跳动或调整大小

3. **页面压缩**：
   - 切换到预测页面时，页面会自动压缩
   - 无需手动操作

## 优势

1. **用户体验**：消除了窗口跳动问题
2. **代码维护**：逻辑清晰，易于维护和扩展
3. **性能优化**：减少了不必要的窗口大小计算和调整
4. **稳定性**：添加了错误处理和回退机制
5. **界面美观**：压缩后的页面更为美观，消除了多余空白

## 文件结构

```
gisaxs_gui/
├── main.py                    # 主窗口，集成布局工具
├── controllers/
│   └── main_controller.py     # 使用布局工具进行页面切换
├── utils/
│   └── layout_utils.py        # 布局工具类（新增）
└── docs/
    └── Page_Size_Adaptive_Solution.md  # 本文档
```

## 测试

应用程序现在应该：
1. 启动时显示训练集构建页面
2. 点击 GISAXS Predict 按钮平滑切换到预测页面
3. 预测页面根据内容自适应大小
4. 窗口整体保持稳定，不会跳动
5. 预测页面切换后自动压缩，消除不必要的空白区域
