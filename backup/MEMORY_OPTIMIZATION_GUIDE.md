# OmniSVG 显存优化指南

本指南介绍如何降低 OmniSVG 训练时的显存占用。

## 🎯 快速开始

### 方案 1：使用低显存配置文件（推荐）

直接使用预配置的低显存配置：

```bash
# 方法 1：复制低显存配置
cp configs/train_config_low_memory.yaml configs/train_config.yaml

# 方法 2：或者在启动脚本中指定
CUDA_VISIBLE_DEVICES=0 bash debug_run.sh
```

**预期显存节省：约 50-60%**

### 方案 2：手动调整配置

编辑 `configs/train_config.yaml`，按需调整以下参数：

## 📊 显存优化选项详解

### 1. Gradient Checkpointing（梯度检查点）⭐ 推荐

**显存节省：30-40%**  
**速度影响：慢 20%**

```yaml
model:
  use_gradient_checkpointing: true
```

**原理：** 用计算换内存。训练时不保存所有中间激活值，反向传播时重新计算。

**适用场景：** 显存紧张但计算资源充足时的首选方案。

---

### 2. 减小图像分辨率

**显存节省：按比例**  
**精度影响：中等**

```yaml
data:
  target_image_size: 336  # 默认 448
  # 可选值: 448(高质量) -> 384(中等) -> 336(低显存)
```

| 分辨率 | 相对显存 | 图像质量 |
|--------|----------|----------|
| 448    | 100%     | 最佳     |
| 384    | ~70%     | 良好     |
| 336    | ~56%     | 可接受   |

---

### 3. 减小序列长度

**显存节省：按比例**  
**适用性：取决于SVG复杂度**

```yaml
data:
  max_seq_length: 1536  # 默认 2048
  # 可选值: 2048(完整) -> 1536(中等) -> 1024(简单SVG)
```

**注意：** 如果训练数据中有大量复杂 SVG，减小序列长度可能导致截断。

---

### 4. 增大梯度累积步数

**显存节省：有效批次大小不变，实际显存占用降低**  
**速度影响：轻微**

```yaml
training:
  gradient_accumulation_steps: 8  # 默认 4
```

**说明：**
- `batch_size=1, grad_accum=4` ≈ `batch_size=4, grad_accum=1`（效果相同）
- `batch_size=1, grad_accum=8` = 更低显存，相同训练效果

---

### 5. 减小 Batch Size

**显存节省：线性**  
**训练稳定性：可能受影响**

在 `debug_run.sh` 中：

```bash
BATCH_SIZE=1  # 已经是最小值
```

**建议：** 配合增大 `gradient_accumulation_steps` 维持有效批次大小。

---

### 6. 减少 DataLoader Workers

**CPU 内存节省：中等**

```yaml
dataloader:
  num_workers: 4  # 默认 8
```

**权衡：** 数据加载可能成为瓶颈，训练速度略降。

---

## 🔧 配置组合建议

### 极低显存模式（< 12GB）

```yaml
model:
  use_gradient_checkpointing: true

data:
  target_image_size: 336
  max_seq_length: 1024

training:
  gradient_accumulation_steps: 16

dataloader:
  num_workers: 2
```

**预期显存：** 约 8-10GB  
**速度：** 约为标准配置的 40%

---

### 低显存模式（12-16GB）⭐ 推荐

```yaml
model:
  use_gradient_checkpointing: true

data:
  target_image_size: 336
  max_seq_length: 1536

training:
  gradient_accumulation_steps: 8

dataloader:
  num_workers: 4
```

**预期显存：** 约 12-14GB  
**速度：** 约为标准配置的 60%

---

### 中等显存模式（16-24GB）

```yaml
model:
  use_gradient_checkpointing: true

data:
  target_image_size: 384
  max_seq_length: 2048

training:
  gradient_accumulation_steps: 4

dataloader:
  num_workers: 8
```

**预期显存：** 约 18-20GB  
**速度：** 约为标准配置的 80%

---

### 标准模式（24GB+）

使用默认的 `configs/train_config.yaml`

**预期显存：** 约 22-24GB  
**速度：** 100%（基准）

---

## 📈 监控显存使用

训练时监控 GPU 显存：

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat（需安装）
pip install gpustat
watch -n 1 gpustat
```

---

## 💡 其他优化技巧

### 1. 清理 CUDA 缓存

如果显存碎片化，训练前清理：

```python
import torch
torch.cuda.empty_cache()
```

### 2. 使用混合精度训练

已默认启用 `bfloat16`，无需额外配置。

### 3. DeepSpeed ZeRO（多GPU）

如果有多个 GPU，可使用 DeepSpeed ZeRO-2/3 进一步优化：

```bash
# 创建 deepspeed 配置
accelerate config

# 选择 DeepSpeed ZeRO-2 或 ZeRO-3
```

---

## ❓ 常见问题

### Q: Gradient Checkpointing 会影响最终精度吗？

A: 不会。只影响训练速度，不影响最终模型质量。

### Q: 我应该先尝试哪个优化？

A: 按顺序尝试：
1. Gradient Checkpointing（影响最小）
2. 减小图像分辨率到 384
3. 增大梯度累积步数
4. 减小序列长度（如果 SVG 不太复杂）

### Q: 显存还是不够怎么办？

A: 考虑：
1. 使用更小的模型（但本项目只支持 4B/8B）
2. 使用量化训练（需要额外实现）
3. 使用 LoRA/QLoRA 等参数高效微调方法
4. 租用更大显存的云 GPU

---

## 📞 支持

如果遇到问题，请检查：
- GPU 驱动版本
- CUDA 版本兼容性
- PyTorch 版本

祝训练顺利！🚀
