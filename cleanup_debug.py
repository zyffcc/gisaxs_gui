#!/usr/bin/env python3
"""
清理脚本：删除所有调试print语句
"""
import os
import re
import sys

def clean_debug_prints(file_path):
    """清理文件中的调试print语句"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 匹配调试print语句的正则表达式
        debug_patterns = [
            r'^\s*print\s*\(\s*[\'"]Debug:.*?\)\s*$',  # print("Debug: ...")
            r'^\s*print\s*\(\s*f[\'"]Debug:.*?\)\s*$',  # print(f"Debug: ...")
            r'^\s*print\s*\(\s*[\'"]debug:.*?\)\s*$',   # print("debug: ...")
            r'^\s*print\s*\(\s*f[\'"]debug:.*?\)\s*$',  # print(f"debug: ...")
            r'^\s*print\s*\(\s*[\'"]TEST:.*?\)\s*$',    # print("TEST: ...")
            r'^\s*print\s*\(\s*f[\'"]TEST:.*?\)\s*$',   # print(f"TEST: ...")
            r'^\s*print\s*\(\s*[\'"]test:.*?\)\s*$',    # print("test: ...")
            r'^\s*print\s*\(\s*f[\'"]test:.*?\)\s*$',   # print(f"test: ...")
        ]
        
        # 逐行处理
        lines = content.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line_num, line in enumerate(lines, 1):
            is_debug_line = False
            for pattern in debug_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    is_debug_line = True
                    removed_count += 1
                    print(f"删除第{line_num}行: {line.strip()}")
                    break
            
            if not is_debug_line:
                cleaned_lines.append(line)
        
        if removed_count > 0:
            # 写回文件
            cleaned_content = '\n'.join(cleaned_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"文件 {file_path} 清理完成，删除了 {removed_count} 行调试代码")
            return True
        else:
            print(f"文件 {file_path} 无需清理")
            return False
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    # 需要清理的文件列表
    files_to_clean = [
        'controllers/gisaxs_predict_controller.py',
        'controllers/utils.py',
        'controllers/main_controller.py',
        'controllers/fitting_controller.py',
        'controllers/classification_controller.py',
        'controllers/trainset_controller.py',
        'core/global_params.py',
    ]
    
    cleaned_files = 0
    for file_path in files_to_clean:
        print(f"\n处理文件: {file_path}")
        if clean_debug_prints(file_path):
            cleaned_files += 1
    
    print(f"\n清理完成！共处理了 {cleaned_files} 个文件")

if __name__ == "__main__":
    main()
