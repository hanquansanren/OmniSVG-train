# VSCode 调试指南

本文档介绍如何在 VSCode 中调试 OmniSVG 训练过程。

## 🎯 新增的调试配置

已在 `.vscode/launch.json` 中添加了以下训练调试配置：

### 1. Debug: Train (模拟 debug_run.sh) ⭐

**用途：** 模拟 `debug_run.sh` 的标准训练流程，方便在 VSCode 中打断点调试。

**配置参数：**
```json
- Model Size: 4B
- Batch Size: 1
- Max Seq Length: 2048
- Config File: train_config.yaml (标准配置)
- GPU: CUDA_VISIBLE_DEVICES=4
- NCCL: P2P/IB 启用（A6000 模式）
```

**使用场景：**
- 调试训练逻辑
- 检查数据加载流程
- 验证模型前向传播
- 排查训练错误

---

### 2. Debug: Train (低显存配置)

**用途：** 使用低显存配置进行调试。

**配置参数：**
```json
- Model Size: 4B
- Batch Size: 1
- Max Seq Length: 1536 (降低)
- Config File: train_config_low_memory.yaml
- GPU: CUDA_VISIBLE_DEVICES=4
- Gradient Checkpointing: 启用
```

**使用场景：**
- 在有限显存下调试
- 测试低显存优化是否正常工作
- 验证梯度检查点功能

---

## 📝 如何使用

### 方法 1：使用侧边栏（推荐）

1. 点击 VSCode 左侧的 **"运行和调试"** 图标（或按 `Ctrl+Shift+D`）
2. 在顶部下拉菜单中选择：
   - **"Debug: Train (模拟 debug_run.sh)"**
   - 或 **"Debug: Train (低显存配置)"**
3. 点击绿色的 **"开始调试"** 按钮（或按 `F5`）

### 方法 2：使用快捷键

1. 按 `F5` 打开调试配置选择
2. 选择对应的配置
3. 按 `Enter` 开始调试

---

## 🔧 调试技巧

### 设置断点

在以下关键位置设置断点可以更好地理解训练流程：

#### 训练初始化
```python
# train.py 第 641 行 - Accelerator 初始化
accelerator = Accelerator(...)

# train.py 第 737 行 - 模型加载
model = load_model(...)

# train.py 第 721 行 - 数据加载器创建
train_dataloader = create_dataloader(...)
```

#### 训练循环
```python
# train.py 第 820 行 - 训练主循环开始
for epoch in range(start_epoch, config.training.epochs):

# train.py 第 836 行 - 批次处理
for step, batch in enumerate(train_dataloader):

# train.py 第 869 行 - 前向传播
outputs = model(...)

# train.py 第 872 行 - 反向传播
accelerator.backward(loss)
```

#### 数据处理
```python
# train.py 第 445 行 - 混合批次处理
def process_mixed_batch(...)

# train.py 第 390 行 - Collate 函数
def collate_fn(batch):
```

---

## 🎨 自定义调试配置

### 修改 GPU 编号

在 `.vscode/launch.json` 中找到对应配置，修改：

```json
"env": {
    "CUDA_VISIBLE_DEVICES": "0",  // 改为您想用的 GPU
    ...
}
```

### 修改数据目录

```json
"args": [
    ...
    "--data_dir", "/your/custom/data/path",
    ...
]
```

### 修改配置文件

```json
"args": [
    ...
    "--train_config_file", "your_custom_config.yaml",
    ...
]
```

### 添加检查点恢复

```json
"args": [
    ...
    "--resume_from_checkpoint", "/path/to/checkpoint",
]
```

---

## ⚠️ 注意事项

### 1. 与 debug_run.sh 的差异

| 特性 | debug_run.sh | VSCode 调试 |
|------|-------------|------------|
| Accelerate | ✅ 使用 | ❌ 不使用（单进程） |
| 多 GPU | ✅ 支持 | ⚠️ 仅支持单 GPU |
| 断点调试 | ❌ 不支持 | ✅ 完全支持 |
| 生产训练 | ✅ 推荐 | ❌ 仅用于调试 |

**重要：** VSCode 调试配置**不使用** `accelerate launch`，因此：
- ✅ 适合单 GPU 调试和开发
- ❌ 不适合多 GPU 训练
- ❌ 不适合生产环境训练

对于多 GPU 或生产训练，仍然使用：
```bash
CUDA_VISIBLE_DEVICES=4 bash debug_run.sh
```

### 2. 环境变量

调试配置已包含以下环境变量：

```json
"env": {
    "CUDA_VISIBLE_DEVICES": "4",        // GPU 选择
    "NCCL_P2P_DISABLE": "0",           // P2P 启用（A6000）
    "NCCL_IB_DISABLE": "0",            // IB 启用（A6000）
    "TOKENIZERS_PARALLELISM": "false"  // 避免 tokenizer 警告
}
```

如果使用 RTX 4000，改为：
```json
"NCCL_P2P_DISABLE": "1",
"NCCL_IB_DISABLE": "1",
```

### 3. 显存占用

调试时可能会有额外的显存开销：
- VSCode Python 扩展
- 调试器开销
- 符号信息

如果遇到 OOM，可以：
- 使用 "低显存配置" 调试项
- 减小 batch_size
- 减小 max_seq_length

---

## 🐛 常见调试场景

### 场景 1：检查数据加载

```python
# 在 train.py 第 836 行设置断点
for step, batch in enumerate(train_dataloader):
    # 查看 batch 内容
    print(batch)
    
    # 检查批次消息
    batch_messages, pix_seq_lists, batch_task_types = batch
    print(f"Task types: {batch_task_types}")
    
    # 继续执行（F5）或单步调试（F10）
```

### 场景 2：检查模型输出

```python
# 在 train.py 第 869 行设置断点
outputs = model(...)

# 在调试控制台中：
print(outputs.loss)
print(outputs.logits.shape)
```

### 场景 3：检查梯度

```python
# 在 train.py 第 881 行（优化器步骤后）设置断点
optimizer.step()

# 检查梯度范数
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 场景 4：验证配置加载

```python
# 在 train.py 第 1157 行设置断点
config = OmniSVGConfig(...)

# 检查配置
print(f"Train config file: {args.train_config_file}")
print(f"Use gradient checkpointing: {config.training.use_gradient_checkpointing}")
print(f"Max seq length: {config.training.max_seq_length}")
```

---

## 📊 调试控制台技巧

### 实时检查变量

在断点处，可以在 **调试控制台** 中输入：

```python
# 检查设备
print(model.device)

# 检查显存使用
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 检查批次大小
print(len(batch_messages))

# 检查输入形状
print(input_ids.shape, labels.shape)
```

### 条件断点

右键断点，选择 "编辑断点" → "条件"：

```python
# 只在第 100 步停止
step == 100

# 只在损失异常时停止
loss > 10.0 or torch.isnan(loss)

# 只在特定任务类型时停止
'image' in batch_task_types
```

---

## 🚀 快速开始

1. **打开 VSCode**
2. **按 F5**
3. **选择 "Debug: Train (模拟 debug_run.sh)"**
4. **在感兴趣的地方设置断点**
5. **开始调试！**

---

## 📚 相关文档

- **debug_run.sh** - 生产训练脚本
- **CONFIG_FILE_GUIDE.md** - 配置文件使用指南
- **MEMORY_OPTIMIZATION_GUIDE.md** - 显存优化指南

---

## 💡 提示

1. **首次调试**：建议使用小批次（batch_size=1）和短序列（max_seq_length=512）快速验证
2. **性能分析**：使用 VSCode 的 Python Profiler 扩展进行性能分析
3. **日志查看**：调试时建议开启详细日志，修改 `log_every=1`
4. **保存调试配置**：可以复制配置并自定义，创建您自己的调试场景

祝调试顺利！🎉
