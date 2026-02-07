#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集中的图像转换和文件扁平化
1. png文件夹：将bmp转换为png
2. svg文件夹：将所有子文件夹中的svg文件移动到根目录
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_bmp_to_png(base_path):
    """
    将png文件夹下的所有bmp图像转换为png格式
    
    Args:
        base_path: 基础路径，例如 /data/phd23_weiguang_zhang/works/svg/my_zhuan
    """
    png_folder = Path(base_path) / "png"
    
    if not png_folder.exists():
        print(f"警告: png文件夹不存在: {png_folder}")
        return
    
    # 查找所有bmp文件
    bmp_files = list(png_folder.rglob("*.bmp"))
    
    if not bmp_files:
        print("没有找到bmp文件")
        return
    
    print(f"找到 {len(bmp_files)} 个bmp文件，开始转换...")
    
    converted_count = 0
    error_count = 0
    
    for bmp_path in tqdm(bmp_files, desc="转换bmp到png"):
        try:
            # 打开bmp图像
            img = Image.open(bmp_path)
            
            # 生成png文件路径（替换扩展名）
            png_path = bmp_path.with_suffix('.png')
            
            # 保存为png
            img.save(png_path, 'PNG')
            
            # 删除原始bmp文件
            bmp_path.unlink()
            
            converted_count += 1
            
        except Exception as e:
            print(f"\n错误: 转换 {bmp_path} 失败: {e}")
            error_count += 1
    
    print(f"\n转换完成! 成功: {converted_count}, 失败: {error_count}")


def flatten_svg_folder(base_path):
    """
    将svg文件夹下所有子文件夹中的svg文件移动到svg根目录
    处理文件名冲突：如果文件名已存在，则添加数字后缀
    
    Args:
        base_path: 基础路径，例如 /data/phd23_weiguang_zhang/works/svg/my_zhuan
    """
    svg_folder = Path(base_path) / "svg"
    
    if not svg_folder.exists():
        print(f"警告: svg文件夹不存在: {svg_folder}")
        return
    
    # 查找所有子文件夹中的svg文件（不包括根目录的svg文件）
    all_svg_files = []
    for svg_path in svg_folder.rglob("*.svg"):
        # 只处理在子文件夹中的文件
        if svg_path.parent != svg_folder:
            all_svg_files.append(svg_path)
    
    if not all_svg_files:
        print("没有找到需要移动的svg文件（子文件夹中）")
        return
    
    print(f"找到 {len(all_svg_files)} 个svg文件在子文件夹中，开始移动...")
    
    moved_count = 0
    error_count = 0
    
    for svg_path in tqdm(all_svg_files, desc="移动svg文件"):
        try:
            # 目标路径
            target_path = svg_folder / svg_path.name
            
            # 处理文件名冲突
            if target_path.exists():
                # 添加数字后缀
                counter = 1
                stem = svg_path.stem
                suffix = svg_path.suffix
                while target_path.exists():
                    target_path = svg_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            # 移动文件
            shutil.move(str(svg_path), str(target_path))
            moved_count += 1
            
        except Exception as e:
            print(f"\n错误: 移动 {svg_path} 失败: {e}")
            error_count += 1
    
    print(f"\n移动完成! 成功: {moved_count}, 失败: {error_count}")
    
    # 清理空的子文件夹
    print("\n清理空的子文件夹...")
    removed_dirs = 0
    for dirpath, dirnames, filenames in os.walk(svg_folder, topdown=False):
        dir_path = Path(dirpath)
        # 跳过根目录
        if dir_path == svg_folder:
            continue
        # 如果文件夹为空，删除它
        if not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                removed_dirs += 1
            except Exception as e:
                print(f"删除空文件夹 {dir_path} 失败: {e}")
    
    print(f"删除了 {removed_dirs} 个空文件夹")


def main():
    # 数据集基础路径
    base_path = "/data/phd23_weiguang_zhang/works/svg/my_zhuan"
    
    print("=" * 60)
    print("开始处理数据集")
    print("=" * 60)
    
    # 任务1: 转换bmp到png
    print("\n[任务1] 转换bmp图像为png格式")
    print("-" * 60)
    convert_bmp_to_png(base_path)
    
    # 任务2: 扁平化svg文件夹
    print("\n[任务2] 扁平化svg文件夹结构")
    print("-" * 60)
    flatten_svg_folder(base_path)
    
    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
