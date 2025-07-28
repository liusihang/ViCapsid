#!/usr/bin/env python3
"""
convert_hmms.py

功能：
    遍历给定目录（及其子目录）下的所有 .hmm 文件，
    并使用 hmmconvert 将其转换为二进制格式 (.h3m)。

用法：
    python convert_hmms.py /path/to/target_directory
"""

import argparse
import subprocess
import sys
from pathlib import Path

def convert_hmm_to_h3m(hmm_file: Path, output_file: Path) -> bool:
    """
    将单个 .hmm 文件转换为二进制格式 (.h3m)
    
    参数:
        hmm_file (Path): 输入的 .hmm 文件路径
        output_file (Path): 输出的 .h3m 文件路径
        
    返回:
        bool: 转换成功返回 True，否则返回 False
    """
    try:
        # 打开输出文件，以写入转换后的内容
        with output_file.open('w') as out_f:
            # 调用外部命令 hmmconvert -b <hmm_file>
            result = subprocess.run(
                ['hmmconvert', '-b', str(hmm_file)],
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                print(f"Error converting {hmm_file}:\n{result.stderr}", file=sys.stderr)
                return False
    except Exception as ex:
        print(f"Exception occurred while converting {hmm_file}:\n{ex}", file=sys.stderr)
        return False
    return True

def convert_all_hmm_in_directory(target_dir: Path):
    """
    遍历 target_dir 下的所有 .hmm 文件，并进行转换
    
    参数:
        target_dir (Path): 搜索 .hmm 文件的目标目录
    """
    # rglob 递归查找所有 .hmm 文件
    for hmm_file in target_dir.rglob('*.hmm'):
        # 构造输出文件路径，将扩展名 .hmm 替换为 .h3m
        output_file = hmm_file.with_suffix('.h3m')
        print(f"Converting: {hmm_file}  ->  {output_file}")
        success = convert_hmm_to_h3m(hmm_file, output_file)
        if success:
            print(f"Converted successfully: {output_file}")
        else:
            print(f"Conversion failed for: {hmm_file}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="将给定目录下的所有 .hmm 文件转换为二进制格式 (.h3m)"
    )
    parser.add_argument(
        "target_dir",
        help="包含 .hmm 文件的目标目录路径",
        type=str
    )
    
    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    
    # 检查目标目录是否有效
    if not target_dir.is_dir():
        print(f"Error: {target_dir} 不是一个有效的目录。", file=sys.stderr)
        sys.exit(1)
    
    print(f"开始转换目录 '{target_dir}' 下的所有 .hmm 文件...\n")
    convert_all_hmm_in_directory(target_dir)
    print("\n所有转换操作已完成。")

if __name__ == '__main__':
    main()