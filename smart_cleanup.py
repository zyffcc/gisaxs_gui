#!/usr/bin/env python3
"""
智能清理脚本：只删除调试print语句，保持代码结构完整
"""
import os
import re

def smart_clean_debug_prints(file_path):
    """智能清理文件中的调试print语句，保持代码结构"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        removed_count = 0
        
        for line_num, line in enumerate(lines, 1):
            # 检查是否是调试print语句
            stripped = line.strip()
            
            # 要删除的调试语句模式
            debug_patterns = [
                r'^print\s*\(\s*f?"Debug:',       # print("Debug: ... 或 print(f"Debug: ...
                r'^print\s*\(\s*f?"debug:',       # print("debug: ... 或 print(f"debug: ...
                r'^print\s*\(\s*f?"TEST:',        # print("TEST: ... 或 print(f"TEST: ...
                r'^print\s*\(\s*f?"test:',        # print("test: ... 或 print(f"test: ...
                r'^print\s*\(\s*"=== .* ==="\)',  # print("=== ... ===")
                r'^print\s*\(\s*f"=== .* ==="\)', # print(f"=== ... ===")
            ]
            
            is_debug_line = False
            for pattern in debug_patterns:
                if re.match(pattern, stripped):
                    is_debug_line = True
                    removed_count += 1
                    print(f"删除第{line_num}行: {stripped}")
                    break
            
            # 保留非调试行
            if not is_debug_line:
                cleaned_lines.append(line)
        
        if removed_count > 0:
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
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
    file_path = 'controllers/gisaxs_predict_controller.py'
    print(f"智能清理文件: {file_path}")
    smart_clean_debug_prints(file_path)

if __name__ == "__main__":
    main()
