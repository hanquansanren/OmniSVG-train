#!/usr/bin/env python3
"""
将Hugging Face下载的parquet格式数据集转换回原始格式
"""
import os
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse


def convert_parquet_to_original(
    parquet_dir: str,
    output_dir: str,
    save_images: bool = True,
    save_svgs: bool = True,
    save_metadata: bool = True
):
    """
    将parquet文件转换回原始格式
    
    Args:
        parquet_dir: parquet文件所在目录
        output_dir: 输出目录
        save_images: 是否保存PNG图片
        save_svgs: 是否保存SVG文件
        save_metadata: 是否保存元数据JSON文件
    """
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
    
    if save_svgs:
        svgs_dir = output_dir / "svgs"
        svgs_dir.mkdir(parents=True, exist_ok=True)
    
    if save_metadata:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有parquet文件
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 统计信息
    total_records = 0
    all_metadata = []
    
    # 逐个处理parquet文件
    for parquet_file in tqdm(parquet_files, desc="处理parquet文件"):
        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)
            total_records += len(df)
            
            # 处理每一行数据
            for idx, row in tqdm(df.iterrows(), total=len(df), 
                               desc=f"处理 {parquet_file.name}", 
                               leave=False):
                record_id = row['id']
                
                # 保存PNG图片
                if save_images and 'image' in row and row['image'] is not None:
                    try:
                        image_bytes = row['image']['bytes']
                        image_path = images_dir / f"{record_id}.png"
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                    except Exception as e:
                        print(f"保存图片失败 {record_id}: {e}")
                
                # 保存SVG文件
                if save_svgs and 'svg' in row and row['svg'] is not None:
                    try:
                        svg_content = row['svg']
                        svg_path = svgs_dir / f"{record_id}.svg"
                        with open(svg_path, 'w', encoding='utf-8') as f:
                            f.write(svg_content)
                    except Exception as e:
                        print(f"保存SVG失败 {record_id}: {e}")
                
                # 收集元数据
                if save_metadata:
                    metadata = {
                        'id': record_id,
                        'description': row.get('description', ''),
                        'keywords': row.get('keywords', ''),
                        'detail': row.get('detail', ''),
                        'token_len': int(row.get('token_len', 0)) if pd.notna(row.get('token_len')) else 0,
                        'svg_path': f"svgs/{record_id}.svg" if save_svgs else None,
                        'image_path': f"images/{record_id}.png" if save_images else None,
                    }
                    all_metadata.append(metadata)
        
        except Exception as e:
            print(f"处理文件 {parquet_file.name} 时出错: {e}")
            continue
    
    # 保存所有元数据到一个JSON文件
    if save_metadata and all_metadata:
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
        print(f"\n元数据已保存到: {metadata_file}")
    
    # 打印统计信息
    print(f"\n转换完成!")
    print(f"总记录数: {total_records}")
    if save_images:
        print(f"图片保存位置: {images_dir}")
    if save_svgs:
        print(f"SVG保存位置: {svgs_dir}")
    if save_metadata:
        print(f"元数据文件数: {len(all_metadata)}")


def main():
    parser = argparse.ArgumentParser(
        description="将Hugging Face的parquet格式数据集转换回原始格式"
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon/data",
        help="parquet文件所在目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-converted",
        help="输出目录"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="不保存PNG图片"
    )
    parser.add_argument(
        "--no-svgs",
        action="store_true",
        help="不保存SVG文件"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不保存元数据JSON文件"
    )
    
    args = parser.parse_args()
    
    convert_parquet_to_original(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        save_images=not args.no_images,
        save_svgs=not args.no_svgs,
        save_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
