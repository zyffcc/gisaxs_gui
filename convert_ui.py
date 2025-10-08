# convert_ui.py
"""
GISAXS GUI UI文件转换脚本
转换所有.ui文件为.py文件
"""

import os
import subprocess
import sys

def convert_ui_file(ui_path, py_path):
    """转换单个UI文件"""
    try:
        cmd = ['pyuic5', '-x', ui_path, '-o', py_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {os.path.basename(ui_path)} -> {os.path.basename(py_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 转换失败 {os.path.basename(ui_path)}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("✗ 未找到pyuic5命令，请安装: pip install PyQt5-tools")
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