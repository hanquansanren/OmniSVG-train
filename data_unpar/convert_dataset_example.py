#!/usr/bin/env python3
"""
简单示例：转换parquet数据集的前几个文件
"""
import os
from pathlib import Path
import pandas as pd
import json


def convert_sample_data(
    parquet_dir="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon/data",
    output_dir="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample",
    max_files=3,  # 只处理前3个parquet文件
    max_records_per_file=100  # 每个文件只处理前100条记录
):
    """
    转换部分数据作为示例
    """
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_dir / "images"
    svgs_dir = output_dir / "svgs"
    images_dir.mkdir(parents=True, exist_ok=True)
    svgs_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取parquet文件
    parquet_files = sorted(parquet_dir.glob("*.parquet"))[:max_files]
    print(f"将处理 {len(parquet_files)} 个parquet文件")
    
    all_metadata = []
    total_count = 0
    
    for parquet_file in parquet_files:
        print(f"\n处理文件: {parquet_file.name}")
        
        # 读取parquet文件
        df = pd.read_parquet(parquet_file)
        print(f"  文件包含 {len(df)} 条记录，将处理前 {max_records_per_file} 条")
        
        # 只处理前N条记录
        df = df.head(max_records_per_file)
        
        for idx, row in df.iterrows():
            record_id = row['id']
            
            # 保存PNG图片
            if 'image' in row and row['image'] is not None:
                try:
                    image_bytes = row['image']['bytes']
                    image_path = images_dir / f"{record_id}.png"
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                except Exception as e:
                    print(f"  警告: 保存图片失败 {record_id}: {e}")
            
            # 保存SVG文件
            if 'svg' in row and row['svg'] is not None:
                try:
                    svg_content = row['svg']
                    svg_path = svgs_dir / f"{record_id}.svg"
                    with open(svg_path, 'w', encoding='utf-8') as f:
                        f.write(svg_content)
                except Exception as e:
                    print(f"  警告: 保存SVG失败 {record_id}: {e}")
            
            # 收集元数据
            metadata = {
                'id': record_id,
                'description': row.get('description', ''),
                'keywords': row.get('keywords', ''),
                'detail': row.get('detail', ''),
                'token_len': int(row.get('token_len', 0)) if pd.notna(row.get('token_len')) else 0,
                'svg_path': f"svgs/{record_id}.svg",
                'image_path': f"images/{record_id}.png",
            }
            all_metadata.append(metadata)
            total_count += 1
        
        print(f"  完成 {len(df)} 条记录")
    
    # 保存元数据
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 转换完成!")
    print(f"   总记录数: {total_count}")
    print(f"   图片目录: {images_dir}")
    print(f"   SVG目录: {svgs_dir}")
    print(f"   元数据文件: {metadata_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("MMSVG-Icon 数据集转换示例")
    print("=" * 60)
    
    # 转换示例数据
    convert_sample_data()
    
    print("\n" + "=" * 60)
    print("如果要转换全部数据，请使用 convert_parquet_to_original.py")
    print("=" * 60)
