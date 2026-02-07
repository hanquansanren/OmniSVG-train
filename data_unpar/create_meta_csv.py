#!/usr/bin/env python3
"""
从转换后的数据集生成train_meta.csv和val_meta.csv
按照990k:110k的比例划分训练集和验证集
"""
import json
import csv
import random
from pathlib import Path
import argparse
from tqdm import tqdm


def create_meta_csv(
    input_json: str,
    output_dir: str,
    train_ratio: float = 0.9,
    random_seed: int = 42
):
    """
    从JSON元数据生成CSV格式的train和val元数据文件
    
    Args:
        input_json: 输入的JSON元数据文件路径
        output_dir: 输出目录（将在此目录创建train_meta.csv和val_meta.csv）
        train_ratio: 训练集比例（默认0.9，即90%）
        random_seed: 随机种子（保证可复现）
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取JSON文件: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f"总记录数: {total_count:,}")
    
    # 计算划分数量
    train_count = int(total_count * train_ratio)
    val_count = total_count - train_count
    
    print(f"训练集: {train_count:,} ({train_ratio*100:.1f}%)")
    print(f"验证集: {val_count:,} ({(1-train_ratio)*100:.1f}%)")
    
    # 随机打乱数据
    random.seed(random_seed)
    random.shuffle(data)
    
    # 划分训练集和验证集
    train_data = data[:train_count]
    val_data = data[train_count:]
    
    # 生成CSV文件
    csv_fields = ['id', 'desc_en', 'detail', 'keywords', 'len_pix']
    
    # 写入训练集CSV
    train_csv_path = output_dir / 'train_meta.csv'
    print(f"\n生成训练集CSV: {train_csv_path}")
    with open(train_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        
        for record in tqdm(train_data, desc="写入训练集"):
            # 提取ID（从image_path中获取文件名，去掉扩展名）
            image_path = record.get('image_path', '')
            if image_path:
                record_id = Path(image_path).stem
            else:
                record_id = record.get('id', '')
            
            csv_row = {
                'id': record_id,
                'desc_en': record.get('description', ''),
                'detail': record.get('detail', ''),
                'keywords': record.get('keywords', ''),
                'len_pix': record.get('token_len', 0)
            }
            writer.writerow(csv_row)
    
    # 写入验证集CSV
    val_csv_path = output_dir / 'val_meta.csv'
    print(f"生成验证集CSV: {val_csv_path}")
    with open(val_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        
        for record in tqdm(val_data, desc="写入验证集"):
            # 提取ID
            image_path = record.get('image_path', '')
            if image_path:
                record_id = Path(image_path).stem
            else:
                record_id = record.get('id', '')
            
            csv_row = {
                'id': record_id,
                'desc_en': record.get('description', ''),
                'detail': record.get('detail', ''),
                'keywords': record.get('keywords', ''),
                'len_pix': record.get('token_len', 0)
            }
            writer.writerow(csv_row)
    
    print("\n✅ CSV文件生成完成!")
    print(f"   训练集: {train_csv_path} ({train_count:,} 条记录)")
    print(f"   验证集: {val_csv_path} ({val_count:,} 条记录)")
    
    # 显示示例
    print("\n训练集示例（前3行）:")
    with open(train_csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 4:  # 包括header
                print(f"  {line.rstrip()}")
            else:
                break


def main():
    parser = argparse.ArgumentParser(
        description="从JSON元数据生成训练集和验证集CSV文件"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-converted/dataset_metadata.json",
        help="输入的JSON元数据文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-converted",
        help="输出目录"
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
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_json).exists():
        print(f"错误: 输入文件不存在: {args.input_json}")
        return
    
    create_meta_csv(
        input_json=args.input_json,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
