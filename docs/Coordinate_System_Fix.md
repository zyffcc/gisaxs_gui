# 鼠标框选功能坐标系说明

## ✅ 最终确认的坐标系统

### 原始图像坐标系
- **原点位置**: 左下角 (0, 0)
- **X轴**: 从左到右递增
- **Y轴**: 从下到上递增

### matplotlib显示坐标系
- 图像在显示时会被`np.flipud()`垂直翻转
- 然后使用`origin='lower'`显示
- **最终结果**: matplotlib坐标与原始图像坐标一致

## ✅ UI控件映射关系（已确认正确）

### Center参数映射
- **gisaxsInputCenterVerticalValue** ← Y坐标（竖直方向）✅
- **gisaxsInputCenterParallelValue** ← X坐标（水平方向）✅

### Cut Line参数映射
- **gisaxsInputCutLineVerticalValue** ← height（竖直方向尺寸）✅
- **gisaxsInputCutLineParallelValue** ← width（水平方向尺寸）✅

## ✅ 坐标转换逻辑（已确认）

### 最终确认的转换
```python
# matplotlib坐标直接对应原始图像坐标
original_y = matplotlib_y  # 无需翻转
```

### 为什么这样转换是正确的
1. **原始图像**: Y=0在左下角
2. **flipud处理**: 图像垂直翻转，Y=0移到右上角
3. **origin='lower'显示**: Y=0又回到左下角
4. **最终结果**: matplotlib坐标系与原始图像坐标系一致

## ✅ 功能特性（全部正常工作）

### 鼠标交互
- ✅ 右键激活/退出框选模式
- ✅ 左键拖拽创建选择矩形
- ✅ ESC键退出框选模式
- ✅ Delete键清除当前选择

### 参数同步
- ✅ 自动更新Cut Line Center参数
- ✅ 自动更新Cut Line Vertical/Parallel尺寸
- ✅ 主窗口和独立窗口同步显示选择区域
- ✅ 数值框支持大于99的值

### 坐标精度
- ✅ Vertical Center正确对应Y轴位置
- ✅ Parallel Center正确对应X轴位置
- ✅ 坐标系原点在左下角（符合GISAXS惯例）

## 使用说明
1. 双击主窗口图像打开独立窗口
2. 右键激活框选模式
3. 拖拽选择感兴趣的区域
4. 参数自动更新到Cut Line控件，坐标完全准确
