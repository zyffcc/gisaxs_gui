# GISAXS Toolkit 项目结构

## 项目概述
这是一个用于GISAXS数据处理的PyQt5桌面应用程序，采用模块化设计，代码结构清晰明了。

## 目录结构

```
gisaxs_gui/
├── main.py                 # 应用程序入口点
├── config/                 # 配置文件目录
│   ├── __init__.py
│   ├── window_config.py    # 窗口配置
│   ├── display_config.py   # 显示配置
│   ├── ui_config.py        # UI配置
│   ├── detectors.json      # 探测器配置
│   ├── materials.json      # 材料配置
│   ├── display_settings.json  # 显示设置
│   └── user_settings.json  # 用户设置
├── core/                   # 核心逻辑模块
│   ├── __init__.py
│   ├── window_manager.py   # 窗口管理器
│   └── user_settings.py    # 用户设置管理
├── ui/                     # 用户界面模块
│   ├── main_window.py      # 主窗口UI
│   ├── main_window.ui      # Qt Designer文件
│   └── settings_dialog.py  # 设置对话框
├── controllers/            # 控制器模块
│   ├── __init__.py
│   ├── main_controller.py  # 主控制器
│   ├── beam_controller.py  # 光束控制器
│   ├── detector_controller.py  # 探测器控制器
│   ├── sample_controller.py    # 样品控制器
│   ├── preprocessing_controller.py  # 预处理控制器
│   ├── trainset_controller.py      # 训练集控制器
│   └── utils.py            # 控制器工具函数
├── utils/                  # 实用工具模块
│   ├── __init__.py
│   └── styles.py           # 样式管理
├── docs/                   # 文档目录
│   ├── Display_Settings_Guide.md      # 显示设置指南
│   ├── Implementation_Summary.md      # 实现总结
│   ├── WindowManager_README.md        # 窗口管理器说明
│   ├── ProjectStructure.txt           # 项目结构(旧)
│   ├── main_simple.py                 # 简化版main(备份)
│   └── main_window.py                 # 重复文件(备份)
└── __pycache__/            # Python缓存文件
```

## 模块说明

### 1. 根目录
- **main.py**: 应用程序的主入口点，负责启动GUI应用

### 2. config/ - 配置管理
- **window_config.py**: 窗口相关的基础配置
- **display_config.py**: 显示相关配置
- **ui_config.py**: UI界面配置
- ***.json**: 各种JSON格式的配置文件

### 3. core/ - 核心业务逻辑
- **window_manager.py**: 窗口管理器，处理窗口定位、缩放等功能
- **user_settings.py**: 用户设置管理，处理设置的保存和加载

### 4. ui/ - 用户界面
- **main_window.py**: 主窗口UI定义
- **main_window.ui**: Qt Designer设计文件
- **settings_dialog.py**: 设置对话框

### 5. controllers/ - 控制器层
- **main_controller.py**: 主控制器，协调各个子控制器
- **各种*_controller.py**: 负责不同功能模块的业务逻辑

### 6. utils/ - 工具模块
- **styles.py**: 样式管理，提供自适应样式

### 7. docs/ - 文档
- 包含各种项目文档、说明文件和备份文件

## 设计原则

1. **分离关注点**: 每个模块只负责特定的功能
2. **配置外化**: 所有配置都在config目录中集中管理
3. **模块化设计**: 功能模块化，便于维护和扩展
4. **文档完整**: 提供完整的文档说明

## 依赖关系

```
main.py
├── ui.main_window
├── controllers.MainController
└── core.window_manager
    ├── config.window_config
    └── core.user_settings
        └── config.window_config

settings_dialog.py
├── config.window_config
└── core.user_settings

window_manager.py
├── config.window_config
└── core.user_settings
```

## 启动流程

1. **main.py** 启动应用程序
2. 创建 **MainWindow** 实例
3. 初始化 **MainController**
4. 通过 **window_manager** 设置窗口属性
5. 连接菜单信号，可以打开 **settings_dialog**

这种结构使得项目更加清晰，便于维护和扩展。
