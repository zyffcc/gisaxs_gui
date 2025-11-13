# convert_ui.py
"""
GISAXS GUI UI文件转换脚本
转换所有.ui文件为.py文件
"""

import os
import subprocess
import sys

def convert_ui_file(ui_path, py_path):
    """转换单个UI文件

    优先使用系统命令 `pyuic5`，若不可用或失败，则回退到
    `python -m PyQt5.uic.pyuic` 使用当前解释器执行。
    """
    # 确保输出目录存在
    out_dir = os.path.dirname(py_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ui_name = os.path.basename(ui_path)
    py_name = os.path.basename(py_path)

    # 1) 首选: 直接调用 pyuic5
    try:
        cmd = ['pyuic5', '-x', ui_path, '-o', py_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {ui_name} -> {py_name} (pyuic5)")
        return True
    except FileNotFoundError:
        # pyuic5 不存在 -> 走回退方案
        pass
    except subprocess.CalledProcessError as e:
        # pyuic5 存在但执行失败 -> 尝试回退方案
        err = (e.stderr or '').strip()
        print(f"⚠ 使用 pyuic5 转换失败: {ui_name} -> {py_name}: {err}")

    # 2) 回退: 使用 python -m PyQt5.uic.pyuic
    try:
        cmd = [sys.executable, '-m', 'PyQt5.uic.pyuic', ui_path, '-o', py_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {ui_name} -> {py_name} (python -m PyQt5.uic.pyuic)")
        return True
    except subprocess.CalledProcessError as e:
        err = (e.stderr or '').strip()
        print(f"✗ 转换失败 {ui_name}: {err}")
        return False
    except FileNotFoundError:
        # 当前解释器不可用或无法找到 python 命令（极少见于此上下文）
        print("✗ 无法调用 Python 解释器或 PyQt5 未安装，请先安装 PyQt5: pip install PyQt5")
        return False

def main():
    print("🔄 GISAXS UI文件转换")
    print("-" * 30)
    
    # 根据您的实际文件结构定义转换列表
    conversions = [
        ('ui/main_window.ui', 'ui/main_window.py')
    ]
    
    success = 0
    total = 0
    
    for ui_file, py_file in conversions:
        if os.path.exists(ui_file):
            total += 1
            if convert_ui_file(ui_file, py_file):
                success += 1
        else:
            print(f"⚠ 跳过不存在的文件: {ui_file}")
    
    print("-" * 30)
    print(f"📊 完成: {success}/{total} 成功")
    
    if success == total and total > 0:
        print("🎉 所有UI文件转换成功！")

if __name__ == "__main__":
    main()