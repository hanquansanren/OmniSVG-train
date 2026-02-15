#!/usr/bin/env python3
"""
批量重命名文件，删除文件名中的汉字字段
格式转换: unicode_汉字_字库_min.ext -> unicode_字库_min.ext
例如: 626D_扭_FZLiYBZSFU_min.png -> 626D_FZLiYBZSFU_min.png
同时更新CSV文件中的文件名引用
"""
import argparse
import csv
import re
from pathlib import Path
from tqdm import tqdm


def parse_filename(filename: str):
    """
    解析文件名格式: unicode_汉字_字库_min.ext
    
    Args:
        filename: 文件名
        
    Returns:
        tuple: (unicode, 汉字, 字库, min标记, 扩展名) 或 None
    """
    # 移除扩展名
    stem = Path(filename).stem
    ext = Path(filename).suffix
    
    # 尝试匹配格式: unicode_汉字_字库_min
    # unicode应该是4-5位十六进制
    # 汉字是中文字符
    # 字库是英文字母组合
    # min是固定的
    parts = stem.split('_')
    
    if len(parts) >= 4:
        unicode_part = parts[0]
        hanzi = parts[1]
        font = parts[2]
        min_suffix = '_'.join(parts[3:])  # 可能有多个下划线
        
        return unicode_part, hanzi, font, min_suffix, ext
    
    return None


def generate_new_filename(unicode_part: str, font: str, min_suffix: str, ext: str):
    """
    生成新的文件名，删除汉字字段
    
    Args:
        unicode_part: unicode部分
        font: 字库部分
        min_suffix: min后缀部分
        ext: 扩展名
        
    Returns:
        str: 新文件名
    """
    return f"{unicode_part}_{font}_{min_suffix}{ext}"


def rename_files_in_folder(
    folder: Path,
    file_extension: str,
    dry_run: bool = False
):
    """
    重命名指定文件夹中的文件
    
    Args:
        folder: 文件夹路径
        file_extension: 文件扩展名 (如 ".png", ".svg")
        dry_run: 是否只预览
        
    Returns:
        dict: 文件名映射 {旧文件名: 新文件名}
    """
    if not folder.exists():
        print(f"警告: 文件夹不存在: {folder}")
        return {}
    
    print(f"\n正在处理 {file_extension.upper()} 文件夹: {folder}")
    
    # 扫描所有指定扩展名的文件
    files = list(folder.glob(f"*{file_extension}"))
    
    if len(files) == 0:
        print(f"警告: 没有找到任何 {file_extension} 文件")
        return {}
    
    print(f"找到 {len(files):,} 个 {file_extension} 文件")
    
    # 解析并生成重命名映射
    rename_mapping = {}
    files_to_rename = []
    files_skipped = []
    
    for file_path in files:
        filename = file_path.name
        parsed = parse_filename(filename)
        
        if parsed is None:
            files_skipped.append(filename)
            continue
        
        unicode_part, hanzi, font, min_suffix, ext = parsed
        new_filename = generate_new_filename(unicode_part, font, min_suffix, ext)
        
        if filename != new_filename:
            new_path = file_path.parent / new_filename
            files_to_rename.append((file_path, new_path, filename, new_filename))
            rename_mapping[filename] = new_filename
        else:
            files_skipped.append(filename)
    
    print(f"需要重命名的文件: {len(files_to_rename):,} 个")
    print(f"跳过的文件: {len(files_skipped):,} 个")
    
    if len(files_skipped) > 0 and len(files_skipped) <= 5:
        print(f"\n跳过的文件:")
        for f in files_skipped:
            print(f"  {f}")
    
    if len(files_to_rename) == 0:
        print("✅ 没有需要重命名的文件")
        return rename_mapping
    
    # 显示重命名示例
    print(f"\n重命名操作示例（前5个）:")
    for _, _, old_name, new_name in files_to_rename[:5]:
        print(f"  {old_name}")
        print(f"  -> {new_name}")
    
    if dry_run:
        print(f"\n[预览模式] 使用 --no-dry-run 参数来实际执行")
        return rename_mapping
    
    # 执行重命名
    print(f"\n开始重命名文件...")
    success_count = 0
    error_count = 0
    
    for old_path, new_path, old_name, new_name in tqdm(files_to_rename, desc="重命名进度"):
        try:
            if new_path.exists():
                print(f"\n警告: 目标文件已存在，跳过: {new_name}")
                error_count += 1
                continue
            
            old_path.rename(new_path)
            success_count += 1
        except Exception as e:
            print(f"\n错误: 重命名失败 {old_name}: {e}")
            error_count += 1
    
    print(f"\n✅ {file_extension.upper()} 重命名完成!")
    print(f"   成功: {success_count:,} 个文件")
    if error_count > 0:
        print(f"   失败: {error_count:,} 个文件")
    
    return rename_mapping


def generate_mapping_from_csv(
    dataset_folder: Path
):
    """
    从CSV文件中读取旧格式的文件名并生成映射
    用于文件已经被重命名但CSV还未更新的情况
    
    Args:
        dataset_folder: 数据集根目录
        
    Returns:
        dict: 文件名映射（不带扩展名）
    """
    print(f"从CSV文件生成映射...")
    csv_files = list(dataset_folder.glob("*.csv"))
    
    if len(csv_files) == 0:
        return {}
    
    mapping = {}
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            for row in rows:
                for cell in row:
                    cell_stripped = cell.strip()
                    # 尝试解析旧格式的文件名（不带扩展名）
                    if cell_stripped and '_' in cell_stripped:
                        # 临时添加扩展名用于解析
                        temp_filename = cell_stripped + ".tmp"
                        parsed = parse_filename(temp_filename)
                        
                        if parsed is not None:
                            unicode_part, hanzi, font, min_suffix, _ = parsed
                            # 检查是否包含汉字（说明是旧格式）
                            if hanzi and len(hanzi) > 0:
                                # 生成新的文件名（不带扩展名）
                                new_stem = f"{unicode_part}_{font}_{min_suffix}"
                                mapping[cell_stripped] = new_stem
        except Exception as e:
            print(f"  警告: 读取CSV文件出错 {csv_file.name}: {e}")
    
    if len(mapping) > 0:
        print(f"\n映射示例（前5个）:")
        for i, (old, new) in enumerate(list(mapping.items())[:5]):
            print(f"  {old} -> {new}")
    
    return mapping


def update_csv_files(
    dataset_folder: Path,
    rename_mapping: dict,
    dry_run: bool = False
):
    """
    更新CSV文件中的文件名引用
    
    Args:
        dataset_folder: 数据集根目录
        rename_mapping: 文件名映射字典（带扩展名）
        dry_run: 是否只预览
    """
    # 如果没有从文件重命名获得的映射，尝试从CSV生成映射
    if not rename_mapping:
        print("\n文件已重命名，从CSV文件中生成映射...")
        stem_mapping = generate_mapping_from_csv(dataset_folder)
        if not stem_mapping:
            print("无法生成映射，跳过CSV更新")
            return
        print(f"从CSV中找到 {len(stem_mapping)} 个需要更新的文件名")
    else:
        # 创建不带扩展名的映射（CSV中存储的是不带扩展名的文件名）
        stem_mapping = {}
        for old_name, new_name in rename_mapping.items():
            old_stem = Path(old_name).stem  # 去掉扩展名
            new_stem = Path(new_name).stem
            stem_mapping[old_stem] = new_stem
    
    print(f"\n正在更新CSV文件: {dataset_folder}")
    csv_files = list(dataset_folder.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("警告: 没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"  {csv_file.name}")
    
    for csv_file in csv_files:
        print(f"\n处理CSV文件: {csv_file.name}")
        
        try:
            # 读取CSV文件
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) == 0:
                print(f"  跳过空文件")
                continue
            
            # 更新文件名
            updated_count = 0
            example_updates = []
            
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    cell_stripped = cell.strip()
                    # 直接匹配完整的单元格内容
                    if cell_stripped in stem_mapping:
                        old_value = rows[i][j]
                        new_value = stem_mapping[cell_stripped]
                        rows[i][j] = new_value
                        updated_count += 1
                        
                        # 保存前5个示例
                        if len(example_updates) < 5:
                            example_updates.append((old_value, new_value))
            
            print(f"  更新了 {updated_count} 处引用")
            
            if len(example_updates) > 0:
                print(f"  更新示例:")
                for old_val, new_val in example_updates:
                    print(f"    {old_val} -> {new_val}")
            
            if updated_count > 0 and not dry_run:
                # 写回CSV文件
                with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                print(f"  ✅ 已保存更新")
            elif updated_count > 0 and dry_run:
                print(f"  [预览模式] 未实际保存")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="批量重命名文件，删除汉字字段并更新CSV文件"
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/my_zhuan2",
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="只预览不实际执行（默认开启）"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="实际执行重命名操作"
    )
    
    args = parser.parse_args()
    
    dataset_folder = Path(args.dataset_folder)
    
    if not dataset_folder.exists():
        print(f"错误: 数据集文件夹不存在: {dataset_folder}")
        return
    
    print(f"{'='*60}")
    print(f"数据集路径: {dataset_folder}")
    print(f"模式: {'预览模式' if args.dry_run else '实际执行'}")
    print(f"{'='*60}")
    
    # 收集所有文件名映射
    all_rename_mapping = {}
    
    # 重命名PNG文件
    png_folder = dataset_folder / "png"
    png_mapping = rename_files_in_folder(png_folder, ".png", args.dry_run)
    all_rename_mapping.update(png_mapping)
    
    # 重命名SVG文件
    svg_folder = dataset_folder / "svg"
    svg_mapping = rename_files_in_folder(svg_folder, ".svg", args.dry_run)
    all_rename_mapping.update(svg_mapping)
    
    # 更新CSV文件
    update_csv_files(dataset_folder, all_rename_mapping, args.dry_run)
    
    print(f"\n{'='*60}")
    print(f"✅ 全部完成!")
    print(f"   PNG文件映射: {len(png_mapping)} 个")
    print(f"   SVG文件映射: {len(svg_mapping)} 个")
    print(f"   总计: {len(all_rename_mapping)} 个文件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
