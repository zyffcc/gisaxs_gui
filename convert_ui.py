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
        # pyuic5 exists but failed -> try fallback
        err = (e.stderr or '').strip()
        print(f"⚠ Conversion with pyuic5 failed: {ui_name} -> {py_name}: {err}")

    # 2) 回退: 使用 python -m PyQt5.uic.pyuic
    try:
        cmd = [sys.executable, '-m', 'PyQt5.uic.pyuic', ui_path, '-o', py_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {ui_name} -> {py_name} (python -m PyQt5.uic.pyuic)")
        return True
    except subprocess.CalledProcessError as e:
        err = (e.stderr or '').strip()
        print(f"✗ Conversion failed {ui_name}: {err}")
        return False
    except FileNotFoundError:
        # Python interpreter unavailable or PyQt5 not installed
        print("✗ Unable to call Python interpreter or PyQt5 is not installed. Please install PyQt5: pip install PyQt5")
        return False

def main():
    print("🔄 GISAXS UI file conversion")
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
            print(f"⚠ Skipping non-existent file: {ui_file}")
    
    print("-" * 30)
    print(f"📊 Done: {success}/{total} succeeded")
    
    if success == total and total > 0:
        print("🎉 All UI files converted successfully!")

if __name__ == "__main__":
    main()