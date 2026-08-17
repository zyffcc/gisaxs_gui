# Fitting application

本层包含 framework-neutral use cases 和 application-owned ports，包括文件加载/导出、
手动拟合、AI candidate 与可序列化的 in-situ workflow。不得导入 PyQt 或具体文件格式
实现；文件错误通过结构化结果返回给 presentation。
