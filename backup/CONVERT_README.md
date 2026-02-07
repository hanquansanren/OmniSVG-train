# Parquet数据集转换工具

将Hugging Face下载的parquet格式MMSVG-Icon数据集转换回原始格式（SVG文件 + PNG图片）。

## 数据集结构

parquet文件包含以下字段：
- `id`: 唯一标识符
- `svg`: SVG代码内容
- `description`: 图标描述
- `keywords`: 关键词
- `detail`: 详细信息
- `image`: PNG图片（二进制数据）
- `token_len`: token长度

## 使用方法

### 方法1: 快速示例（推荐先测试）

运行示例脚本，只转换前几个文件的部分数据：

```bash
python3 convert_dataset_example.py
```

这会转换前3个parquet文件的前100条记录，输出到 `/data/phd23_weiguang_zhang/works/svg/MMSVG-icon-sample/`

### 方法2: 转换全部数据

**注意：完整数据集非常大（约91个parquet文件），转换可能需要较长时间和大量磁盘空间！**

#### 基本使用

```bash
python3 convert_parquet_to_original.py
```

#### 自定义参数

```bash
# 指定输入和输出目录
python3 convert_parquet_to_original.py \
    --parquet_dir /data/phd23_weiguang_zhang/works/svg/MMSVG-icon/data \
    --output_dir /data/phd23_weiguang_zhang/works/svg/MMSVG-icon-converted

# 只保存SVG文件，不保存图片（节省空间）
python3 convert_parquet_to_original.py --no-images

# 只保存图片，不保存SVG文件
python3 convert_parquet_to_original.py --no-svgs

# 只提取文件，不生成元数据JSON
python3 convert_parquet_to_original.py --no-metadata
```

## 输出结构

转换后的目录结构：

```
output_dir/
├── images/              # PNG图片文件
│   ├── fc63b761b0a3a836cc9c558c0d95b7a6.png
│   ├── ...
├── svgs/                # SVG文件
│   ├── fc63b761b0a3a836cc9c558c0d95b7a6.svg
│   ├── ...
└── dataset_metadata.json  # 所有记录的元数据
```

## 元数据JSON格式

`dataset_metadata.json` 包含所有记录的元数据：

```json
[
  {
    "id": "fc63b761b0a3a836cc9c558c0d95b7a6",
    "description": "A black icon depicting...",
    "keywords": "black icon, vertical layout...",
    "detail": "The image consists of...",
    "token_len": 246,
    "svg_path": "svgs/fc63b761b0a3a836cc9c558c0d95b7a6.svg",
    "image_path": "images/fc63b761b0a3a836cc9c558c0d95b7a6.png"
  },
  ...
]
```

## 依赖项

```bash
pip install pandas pyarrow tqdm
```

## 注意事项

1. **磁盘空间**: 完整数据集转换后需要大量磁盘空间，请确保有足够空间
2. **处理时间**: 91个parquet文件包含大量数据，完整转换可能需要较长时间
3. **建议**: 先运行 `convert_dataset_example.py` 测试，确认无误后再转换全部数据

## 常见问题

**Q: 转换需要多长时间？**
A: 取决于数据量和磁盘速度，完整数据集可能需要数小时。建议先用示例脚本测试。

**Q: 如何只转换部分文件？**
A: 修改 `convert_dataset_example.py` 中的 `max_files` 和 `max_records_per_file` 参数。

**Q: 是否可以中断后继续？**
A: 当前脚本不支持断点续传。如需此功能，可以修改脚本添加已处理文件的记录。
