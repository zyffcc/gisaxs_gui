#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理调试代码脚本
自动删除所有Debug print语句，保持代码整洁
"""

import re
import os

def clean_debug_code(file_path):
    """清理文件中的调试代码"""
    print(f"清理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    
    # 删除Debug print语句
    patterns = [
        r'^\s*print\(f?"Debug:.*?\)\s*$',  # Debug print语句
        r'^\s*print\("Debug:.*?"\)\s*$',   # Debug print语句（双引号）
        r'^\s*print\(f"Debug:.*?"\)\s*$',  # Debug print语句（f-string）
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    # 删除多余的空行（超过2行连续空行）
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # 删除文件开头和结尾的多余空行
    content = content.strip() + '\n'
    
    cleaned_lines = len(content.split('\n'))
    removed_lines = original_lines - cleaned_lines
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"完成清理，删除了 {removed_lines} 行调试代码")
    return removed_lines

def main():
    """主函数"""
    # 要清理的文件列表
    files_to_clean = [
        r'd:\PythonCode\gisaxs_gui\controllers\gisaxs_predict_controller.py',
        r'd:\PythonCode\gisaxs_gui\controllers\utils.py',
    ]
    
    total_removed = 0
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            removed = clean_debug_code(file_path)
            total_removed += removed
        else:
            print(f"文件不存在: {file_path}")
    
    print(f"\n清理完成，总共删除了 {total_removed} 行调试代码")

if __name__ == '__main__':
    main()
