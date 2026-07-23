# CXX-GPT

从零实现的 **生成式预训练 Transformer（GPT）语言模型**，使用 **C++17 + CUDA 加速**。本项目是教育性质的项目，旨在通过手动实现来理解 Transformer 内部原理——没有 Python ML 框架，没有自动微分，只有 C++、CUDA 和手动梯度计算。

## 特性

- **完整 GPT 模型** — 仅解码器 Transformer，包含多头因果自注意力、GELU 激活、预归一化 LayerNorm 和嵌入层
- **手动前向/反向** — 每一层都自行实现前向和反向传播；梯度显式流过计算图
- **自定义张量库 (TensorN)** — CPU (OpenBLAS) 和 GPU (CUDA/cuBLAS) 上的 N 维张量运算
- **CUDA 内核** — 为注意力掩码、层归一化、交叉熵损失、嵌入查找、AdamW 优化器等编写了融合内核
- **BPE 分词器** — 用 C++ 实现的完整字节对编码分词器，支持 UTF-8 和中文
- **训练流程** — AdamW 优化器、断点续训（Ctrl+C 中断保存）、流式 JSONL 数据集、内置数值梯度检查
- **交互式 Demo** — 使用 top-k 采样与训练后的模型对话

## 模型对比实验

在 `dataset/train.jsonl`（7377 条对话）和 `dataset/val.jsonl`（868 条）上对 **Softmax GPT** 和 **LinearGPT** 进行了对比训练。

### 配置

| 参数 | Softmax GPT | LinearGPT |
|------|-------------|-----------|
| 嵌入维度 (n_embd) | 128 | 128 |
| 层数 (n_layer) | 6 | 6 |
| 参数量 | 2,842,880 | 2,842,880 |
| 批大小 | 8 | 8 |
| 序列长度 | 64 | 64 |
| 学习率 | 1e-3 | 1e-3 |
| 训练步数 | 2,769 (3 epoch) | 2,769 (3 epoch) |

### 结果

| 指标 | Softmax GPT | LinearGPT |
|------|-------------|-----------|
| 初始损失 | 9.5071 | 9.3921 |
| 最终训练损失 | **1.9185** | 1.9667 |
| 最终验证损失 | **2.1623** | 2.2375 |
| 训练时间 | **74s** | 145s |

**结论：** 在相同参数量下，Softmax GPT 比 LinearGPT 训练速度快约 2 倍，且最终损失略低。

### 为什么 LinearGPT 反而更慢？

LinearGPT 的理论复杂度为 O(n)，Softmax 为 O(n²)，但当前序列长度（S=64）太短，O(n²) 的常数因子极小，GPU 上的矩阵乘可以一次并行完成，而线性注意力的实现存在以下瓶颈：

1. **串行累积** — 前向内核（`linear_cuda_kernels.cu:35`）用一个 `for` 循环逐时间步更新 KV state，S=64 就需要串行迭代 64 次，每次含 4 次 `__syncthreads()`，同步开销巨大
2. **特征映射预处理** — Q 和 K 需先经过 Hadamard + Exp 变换（`phi()`），额外多一次 kernel launch
3. **反向多 kernel** — 反向传播需 3 个独立 kernel（step1 / dQ / dKdV），Softmax 通常使用 fused kernel 一次性完成
4. **因果掩码实现** — 线性注意力通过前缀和（cumsum）实现因果，本质串行；Softmax 直接在注意力矩阵上加掩码，GPU 高度并行

**线性注意力在序列长度 > 1024 时才开始体现 O(n) 优势，短序列下 Softmax 的 GPU 矩阵乘效率更高。**

## 快速开始

### 环境要求

- CMake 3.18+
- Visual Studio 2022 (MSVC)
- CUDA 12+（推荐，但非必需）
- Python 3（用于数据集预编码）

### 构建

```bash
cmake -B build -G "Visual Studio 17 2022" -DTENSORN_ENABLE_CUDA=ON
cmake --build build --config Release
```

二进制文件输出到 `build/bin/`。

### 训练

**Softmax GPT：**
```bash
./build/bin/Release/gpt_train.exe
```
训练约 280 万参数的 GPT 模型，checkpoint 保存为 `checkpoint.bin`，最终模型保存为 `model.bin`。

**LinearGPT（线性注意力）：**
```bash
./build/bin/Release/linear_train.exe
```
训练相同参数量的 LinearGPT 模型，checkpoint 保存为 `linear_checkpoint.bin`，最终模型保存为 `linear_model.bin`。

### 对话

**Softmax GPT：**
```bash
./build/bin/Release/gpt_demo.exe
```

**LinearGPT：**
```bash
./build/bin/Release/linear_demo.exe
```

加载训练好的模型，启动交互式对话。

## 项目结构

```
CXX-GPT/
├── CMakeLists.txt                  # 构建配置
├── src/
│   ├── train.cpp                   # Softmax GPT 训练
│   ├── demo.cpp                    # Softmax GPT 交互式对话
│   ├── dataset.hpp                 # 流式 JSONL 数据集加载器
│   ├── cuda_kernels.cuh/.cu        # CUDA 内核（注意力、LayerNorm、AdamW 等）
│   ├── GPT/gpt.hpp                 # GPT 模型完整定义
│   └── LinearAttention/
│       ├── linear_gpt.hpp          # LinearGPT 模型（Hadamard+Exp 核线性注意力）
│       ├── linear_train.cpp        # LinearGPT 训练
│       ├── linear_demo.cpp         # LinearGPT 交互式对话
│       └── linear_cuda_kernels.cuh/.cu  # 线性注意力 CUDA 内核
├── tokenizer/
│   ├── tokenizer.json              # BPE 分词器模型
│   └── tokenizer_config.json       # 分词器配置
├── dataset/                        # 训练/验证/测试数据（JSONL + 二进制）
│   ├── train.jsonl                 # 训练集（7377 条对话）
│   ├── val.jsonl                   # 验证集（868 条对话）
│   ├── test.jsonl                  # 测试集
│   ├── pretrain_t2t_mini.jsonl     # 预训练数据
│   └── sft_t2t_mini.jsonl          # 微调数据
├── scripts/
│   └── preencode.py                # Python 数据集预编码脚本
└── TensorN/                        # 自定义张量库（子项目）
```

## 架构

| 组件 | 说明 |
|------|------|
| **GPT** | 仅解码器 Transformer，包含 Token + 位置嵌入、N 个 Transformer 块和 lm_head |
| **Block** | 预归一化 LayerNorm → 因果自注意力 → LayerNorm → MLP (GELU) |
| **LinearGPT** | 使用 Hadamard 矩阵 + Exp 特征映射实现线性注意力的变体 |
| **TensorN** | 头文件-only C++ 张量库，支持 CPU (OpenBLAS) 和 GPU (CUDA) 后端 |
| **CUDA 内核** | 为注意力、损失函数、优化器和归一化编写的融合前向/反向内核 |
| **分词器** | BPE 分词器，6400 词汇量，支持 UTF-8 和特殊 token 处理 |
| **数据集** | 流式 JSONL 读取器，支持二进制预编码加速加载 |

## 模型配置

默认配置（约 280 万参数）：
- `vocab_size`：6400
- `block_size`：64
- `n_embd`：128
- `n_layer`：6

## 许可

MIT 许可证 — 详见 [LICENSE](LICENSE)。
