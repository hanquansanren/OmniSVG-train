# 训练配置文件切换指南

## 🎯 功能说明

现在您可以轻松切换不同的训练配置文件，无需手动替换文件。

## 📝 使用方法

### 方法 1：在 `debug_run.sh` 中配置（推荐）

编辑 `debug_run.sh` 文件，找到第 60-66 行：

```bash
# Training config file name
# Options: 
#   - "train_config.yaml" (standard configuration)
#   - "train_config_low_memory.yaml" (optimized for low VRAM)
#   - Or create your own custom config file
TRAIN_CONFIG_FILE="train_config.yaml"
```

**修改配置文件名：**

```bash
# 使用标准配置
TRAIN_CONFIG_FILE="train_config.yaml"

# 或使用低显存配置
TRAIN_CONFIG_FILE="train_config_low_memory.yaml"

# 或使用自定义配置
TRAIN_CONFIG_FILE="my_custom_config.yaml"
```

然后正常运行：

```bash
CUDA_VISIBLE_DEVICES=4 bash debug_run.sh
```

---

### 方法 2：命令行直接指定

不修改 `debug_run.sh`，直接在命令行中指定：

```bash
# 使用标准配置
accelerate launch train.py \
    --model_size 4B \
    --train_config_file train_config.yaml \
    --data_dir ./data

# 使用低显存配置
accelerate launch train.py \
    --model_size 4B \
    --train_config_file train_config_low_memory.yaml \
    --data_dir ./data

# 使用自定义配置
accelerate launch train.py \
    --model_size 4B \
    --train_config_file my_experiment.yaml \
    --data_dir ./data
```

---

## 📂 可用的配置文件

### 1. `train_config.yaml` - 标准配置

**适用场景：** GPU 显存 ≥ 24GB

```yaml
model:
  use_gradient_checkpointing: false
data:
  target_image_size: 448
  max_seq_length: 2048
training:
  gradient_accumulation_steps: 4
```

**特点：**
- ✅ 最快的训练速度
- ✅ 最佳的模型质量
- ❌ 需要大量显存（~22-24GB）

---

### 2. `train_config_low_memory.yaml` - 低显存配置

**适用场景：** GPU 显存 12-16GB

```yaml
model:
  use_gradient_checkpointing: true  # 启用梯度检查点
data:
  target_image_size: 336             # 降低图像分辨率
  max_seq_length: 1536               # 减少序列长度
training:
  gradient_accumulation_steps: 8    # 增大梯度累积
```

**特点：**
- ✅ 显存占用低（~12-14GB）
- ✅ 适合大多数消费级 GPU
- ⚠️ 训练速度约为标准配置的 60%
- ⚠️ 图像分辨率降低可能轻微影响质量

---

## 🔧 创建自定义配置

### 步骤 1：复制模板

```bash
# 从标准配置创建
cp configs/train_config.yaml configs/my_config.yaml

# 或从低显存配置创建
cp configs/train_config_low_memory.yaml configs/my_config.yaml
```

### 步骤 2：编辑配置

```bash
vim configs/my_config.yaml
```

根据需求修改参数。

### 步骤 3：使用配置

在 `debug_run.sh` 中设置：

```bash
TRAIN_CONFIG_FILE="my_config.yaml"
```

---

## 📊 配置文件对比表

| 配置项 | 标准配置 | 低显存配置 | 说明 |
|--------|---------|-----------|------|
| Gradient Checkpointing | ❌ | ✅ | 节省30-40%显存 |
| 图像分辨率 | 448 | 336 | 影响图像质量 |
| 序列长度 | 2048 | 1536 | 影响复杂SVG支持 |
| 梯度累积 | 4 | 8 | 影响训练速度 |
| **预期显存** | **22-24GB** | **12-14GB** | - |
| **训练速度** | **100%** | **60%** | 相对标准配置 |

---

## 💡 实用场景

### 场景 1：快速实验（低显存）

```bash
# debug_run.sh
TRAIN_CONFIG_FILE="train_config_low_memory.yaml"
BATCH_SIZE=1
MAX_SEQ_LENGTH=1024  # 可以进一步降低
```

### 场景 2：正式训练（标准配置）

```bash
# debug_run.sh
TRAIN_CONFIG_FILE="train_config.yaml"
BATCH_SIZE=2
MAX_SEQ_LENGTH=2048
```

### 场景 3：超低显存（< 12GB）

创建 `configs/train_config_ultra_low.yaml`：

```yaml
model:
  use_gradient_checkpointing: true
data:
  target_image_size: 336
  max_seq_length: 1024  # 进一步降低
training:
  gradient_accumulation_steps: 16  # 更大的累积
dataloader:
  num_workers: 2
```

---

## ⚙️ 命令行参数优先级

**优先级从高到低：**

1. 命令行参数（最高优先级）
   ```bash
   --train_config_file my_config.yaml
   ```

2. `debug_run.sh` 中的设置
   ```bash
   TRAIN_CONFIG_FILE="my_config.yaml"
   ```

3. 默认值（如果都不指定）
   ```
   train_config.yaml
   ```

---

## 🔍 验证配置

运行训练时，会显示当前使用的配置文件：

```
============================================================
OmniSVG Training
============================================================
Train Config File: train_config_low_memory.yaml  ← 这里显示
============================================================
```

---

## ❓ 常见问题

### Q: 如何知道当前使用哪个配置文件？

A: 训练开始时会在终端显示 "Train Config File: xxx.yaml"

### Q: 配置文件必须放在 configs/ 目录吗？

A: 是的，所有配置文件都应该放在 `configs/` 目录下。

### Q: 可以在运行时切换配置文件吗？

A: 不可以，需要停止训练后重新运行。

### Q: 命令行参数会覆盖配置文件中的设置吗？

A: 是的，例如 `--max_seq_length` 会覆盖配置文件中的 `max_seq_length`。

### Q: 如何创建针对特定任务的配置？

A: 复制现有配置，修改相关参数。例如：
   - 纯文本到SVG：设置 `initial_text_only_ratio: 1.0`
   - 高质量训练：增大 `target_image_size`

---

## 📝 配置文件示例

### 示例 1：快速原型开发

`configs/train_config_fast_prototype.yaml`：

```yaml
model:
  use_gradient_checkpointing: true
data:
  target_image_size: 336
  max_seq_length: 1024
training:
  gradient_accumulation_steps: 8
  epochs: 10  # 快速训练
logging:
  log_every: 5
  save_every: 1000
  val_every: 1000
```

### 示例 2：高精度训练

`configs/train_config_high_quality.yaml`：

```yaml
model:
  use_gradient_checkpointing: false
data:
  target_image_size: 512  # 更高分辨率
  max_seq_length: 2048
training:
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-6  # 更小的学习率
logging:
  val_every: 2000  # 更频繁的验证
```

---

## 🚀 开始使用

1. **选择配置文件**：根据您的 GPU 显存选择合适的配置
2. **编辑 debug_run.sh**：设置 `TRAIN_CONFIG_FILE` 变量
3. **运行训练**：`CUDA_VISIBLE_DEVICES=4 bash debug_run.sh`
4. **监控训练**：观察显存占用和训练速度

祝训练顺利！🎉
