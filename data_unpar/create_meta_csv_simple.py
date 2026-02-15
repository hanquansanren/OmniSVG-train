#!/usr/bin/env python3
"""
从数据集文件夹直接生成train_meta.csv和val_meta.csv
只包含id字段，按照9:1的比例划分训练集和验证集
"""
import csv
import random
from pathlib import Path
import argparse
from tqdm import tqdm


def create_meta_csv_from_folder(
    input_folder: str,
    output_dir: str = None,
    train_ratio: float = 0.9,
    random_seed: int = 42,
    file_extension: str = ".svg",
    subfolder: str = "svg"
):
    """
    从文件夹扫描文件并生成CSV格式的train和val元数据文件
    
    Args:
        input_folder: 输入的数据集文件夹路径
        output_dir: 输出目录（默认为输入文件夹）
        train_ratio: 训练集比例（默认0.9，即90%）
        random_seed: 随机种子（保证可复现）
        file_extension: 文件扩展名（默认.svg）
        subfolder: 子文件夹名称（默认svg，如果为None则直接在输入文件夹中扫描）
    """
    input_folder = Path(input_folder)
    
    # 如果指定了子文件夹，则在子文件夹中扫描
    if subfolder:
        scan_folder = input_folder / subfolder
    else:
        scan_folder = input_folder
    
    # 如果没有指定输出目录，默认为输入文件夹
    if output_dir is None:
        output_dir = input_folder
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在扫描文件夹: {scan_folder}")
    
    # 扫描所有指定扩展名的文件
    files = list(scan_folder.glob(f"*{file_extension}"))
    
    if len(files) == 0:
        print(f"警告: 没有找到任何 {file_extension} 文件")
        return
    
    # 提取文件ID（文件名不带扩展名）
    file_ids = [f.stem for f in files]
    
    total_count = len(file_ids)
    print(f"总文件数: {total_count:,}")
    
    # 计算划分数量
    train_count = int(total_count * train_ratio)
    val_count = total_count - train_count
    
    print(f"训练集: {train_count:,} ({train_ratio*100:.1f}%)")
    print(f"验证集: {val_count:,} ({(1-train_ratio)*100:.1f}%)")
    
    # 随机打乱数据
    random.seed(random_seed)
    random.shuffle(file_ids)
    
    # 划分训练集和验证集
    train_ids = file_ids[:train_count]
    val_ids = file_ids[train_count:]
    
    # CSV字段只包含id
    csv_fields = ['id']
    
    # 写入训练集CSV
    train_csv_path = output_dir / 'train_meta.csv'
    print(f"\n生成训练集CSV: {train_csv_path}")
    with open(train_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        
        for file_id in tqdm(train_ids, desc="写入训练集"):
            csv_row = {'id': file_id}
            writer.writerow(csv_row)
    
    # 写入验证集CSV
    val_csv_path = output_dir / 'val_meta.csv'
    print(f"生成验证集CSV: {val_csv_path}")
    with open(val_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        
        for file_id in tqdm(val_ids, desc="写入验证集"):
            csv_row = {'id': file_id}
            writer.writerow(csv_row)
    
    print("\n✅ CSV文件生成完成!")
    print(f"   训练集: {train_csv_path} ({train_count:,} 条记录)")
    print(f"   验证集: {val_csv_path} ({val_count:,} 条记录)")
    
    # 显示示例
    print("\n训练集示例（前5行）:")
    with open(train_csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 6:  # 包括header
                print(f"  {line.rstrip()}")
            else:
                break
    
    print("\n验证集示例（前5行）:")
    with open(val_csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 6:  # 包括header
                print(f"  {line.rstrip()}")
            else:
                break


def main():
    parser = argparse.ArgumentParser(
        description="从文件夹直接生成训练集和验证集CSV文件（只包含id字段）"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/my_zhuan",
        help="输入的数据集文件夹路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为输入文件夹）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认0.9，即90%%）"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子（保证可复现）"
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default=".svg",
        help="文件扩展名（默认.svg）"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="svg",
        help="子文件夹名称（默认svg，如果为None则直接在输入文件夹中扫描）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not Path(args.input_folder).exists():
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        return
    
    create_meta_csv_from_folder(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        file_extension=args.file_extension,
        subfolder=args.subfolder
    )


if __name__ == "__main__":
    main()
