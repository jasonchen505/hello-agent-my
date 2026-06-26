# 增量学习笔记：基于 Happy Agent 项目的新知识点

> 记录在复现和学习过程中，相比前两轮文档新发现的知识点
> 持续更新中...

---

## 目录

1. [Transformer 实现细节](#1-transformer-实现细节)
2. [预训练实战经验](#2-预训练实战经验)
3. [SFT 微调技巧](#3-sft-微调技巧)
4. [DPO/RLHF 后训练](#4-dporlhf-后训练)
5. [Agent 系统工程](#5-agent-系统工程)
6. [踩坑与解决方案](#6-踩坑与解决方案)
7. [面试深度追问点](#7-面试深度追问点)

---

## 1. Transformer 实现细节

### 1.1 Pre-Norm vs Post-Norm

**新发现**: 原始 Transformer 使用 Post-Norm，但现代 LLM 几乎都用 Pre-Norm

```python
# Post-Norm (原始 Transformer)
x = x + self_attn(x)
x = LayerNorm(x)  # 先加后归一化

# Pre-Norm (现代 LLM)
x = x + self_attn(LayerNorm(x))  # 先归一化再加
```

**为什么 Pre-Norm 更好？**
- 梯度流更稳定，不会因为层数加深而爆炸/消失
- 训练更稳定，可以用更大的学习率
- 但有研究表明 Post-Norm 的最终效果可能更好（如果训练稳定的话）

**面试追问点**:
> Q: 既然 Post-Norm 效果可能更好，为什么现在都用 Pre-Norm？
> A: 因为训练稳定性 > 最终效果。训练不稳定的代价太大（时间、算力浪费），Pre-Norm 让训练更容易成功。

---

### 1.2 激活函数演进：ReLU → GELU → SwiGLU

**新发现**: 项目中用的是 ReLU，但现代 LLM 用 SwiGLU

```python
# ReLU (项目中使用)
x = torch.relu(x)

# GELU (BERT, GPT-2)
x = torch.nn.functional.gelu(x)

# SwiGLU (LLaMA, Qwen)
# 门控机制: output = SiLU(xW1) ⊙ (xW2)
class SwiGLU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(self.w1(x)) * self.w2(x)
```

**为什么 SwiGLU 更好？**
- 门控机制让网络可以选择性地传递信息
- SiLU (Swish) 是平滑的，梯度更好
- 实验表明效果优于 ReLU/GELU

**面试追问点**:
> Q: SwiGLU 的参数量比 ReLU 多，怎么处理？
> A: 通常会把 d_ff 从 4*d_model 减小到 8/3*d_model，保持总参数量不变。

---

### 1.3 位置编码的工程实现细节

**新发现**: 位置编码的实现有很多细节会影响效果

```python
# 项目中的实现 (transformer.py:85-111)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # ...
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为buffer
```

**关键细节**:

1. **register_buffer vs Parameter**
   - `register_buffer`: 不是参数，但会随模型移动（如 .to(device)）
   - `Parameter`: 是参数，会被优化器更新
   - 位置编码应该是 buffer（不需要学习）

2. **RoPE 的实现要点**
   ```python
   # RoPE 核心: 旋转矩阵
   def apply_rotary_emb(x, freqs):
       # x: (batch, seq_len, num_heads, d_k)
       # freqs: (seq_len, d_k//2)
       cos = torch.cos(freqs)
       sin = torch.sin(freqs)
       # 旋转操作
       x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
       return x * cos + x_rot * sin
   ```

3. **长度外推问题**
   - 训练长度 1024，推理时输入 2048 会怎样？
   - 正弦编码: 可以但效果下降
   - RoPE: 需要 NTK-aware 插值

---

### 1.4 KV Cache 的实现细节

**新发现**: KV Cache 是推理优化的关键，但实现有很多坑

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, d_k):
        # 预分配内存
        self.k_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, d_k)
        self.v_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, d_k)
        self.seq_len = 0
    
    def update(self, k, v):
        # 追加新的 KV
        self.k_cache[:, :, self.seq_len:self.seq_len+k.size(2)] = k
        self.v_cache[:, :, self.seq_len:self.seq_len+k.size(2)] = v
        self.seq_len += k.size(2)
        
        # 返回完整的 KV
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]
```

**工程问题**:
1. **内存碎片**: 每次生成都要分配新内存 → 预分配 + 复用
2. **动态长度**: 不同请求长度不同 → PagedAttention (vLLM)
3. **多请求并发**: 如何共享 KV Cache → Continuous Batching

**面试追问点**:
> Q: KV Cache 的显存占用怎么算？
> A: `2 * num_layers * num_heads * d_k * seq_len * batch_size * dtype_size`
> 例如 7B 模型，seq_len=2048，batch=1，bf16: 约 2GB

---

## 2. 预训练实战经验

### 2.1 数据处理的坑

**新发现**: 数据处理比想象中重要得多

```python
# 常见问题

问题1: 数据重复
# 相同数据重复出现会导致过拟合
# 解决: MinHash 去重
from datasketch import MinHash, MinHashLSH

def deduplicate(documents, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold)
    for i, doc in enumerate(documents):
        m = MinHash()
        for word in doc.split():
            m.update(word.encode('utf8'))
        if not lsh.query(m):  # 没有相似文档
            lsh.insert(str(i), m)
            yield doc

问题2: 数据质量差
# 包含垃圾内容（广告、乱码、过短文本）
# 解决: 多维度过滤
def quality_filter(text):
    # 长度过滤
    if len(text) < 100:
        return False
    # 重复率过滤
    if len(set(text.split())) / len(text.split()) < 0.3:
        return False
    # 特殊字符过滤
    if text.count('�') > 0:
        return False
    return True

问题3: 数据泄露
# 验证集数据混入训练集
# 解决: 严格的去重和哈希检查
```

**面试追问点**:
> Q: 数据去重用 MinHash 还是 SimHash？
> A: MinHash 更精确但更慢，SimHash 更快但有误判。大规模用 SimHash 初筛 + MinHash 精筛。

---

### 2.2 训练稳定性问题

**新发现**: 训练大模型时经常遇到 loss spike

```python
# Loss Spike 的常见原因

原因1: 学习率太大
# 表现: loss 突然变大
# 解决: 降低学习率、增加 warmup

原因2: 数据质量问题
# 表现: loss 在某些 batch 突然变大
# 解决: 数据清洗、跳过异常 batch

原因3: 梯度爆炸
# 表现: loss 变成 NaN
# 解决: gradient clipping

原因4: 数值不稳定
# 表现: loss 逐渐变大
# 解决: 使用 bf16 而不是 fp16

# 实用的监控代码
def check_training_health(loss, grad_norm, step):
    if torch.isnan(loss):
        raise ValueError(f"Loss is NaN at step {step}")
    if loss > 10 * initial_loss:
        print(f"Warning: Loss spike at step {step}")
    if grad_norm > 1000:
        print(f"Warning: Gradient explosion at step {step}")
```

---

### 2.3 Scaling Law 的实际应用

**新发现**: Scaling Law 不只是理论，实际训练中要用来做决策

```python
# Chinchilla Scaling Law
# 最优配置: tokens ≈ 20 * parameters

# 实际应用
model_configs = {
    "125M": {"tokens": 2.5B, "gpu_hours": 24},      # 20 * 125M = 2.5B
    "1.3B": {"tokens": 26B, "gpu_hours": 240},      # 20 * 1.3B = 26B
    "7B": {"tokens": 140B, "gpu_hours": 2400},      # 20 * 7B = 140B
}

# 8卡3090的选择
# 如果想训练 1.3B 模型，需要 26B tokens
# 3090 训练速度约 1000 tokens/sec/gpu
# 8卡: 8000 tokens/sec
# 时间: 26B / 8000 = 3250000 sec ≈ 37.6 天

# 优化建议
# 1. 使用更小的模型 (125M)
# 2. 减少训练 tokens (但效果会下降)
# 3. 使用开源模型继续训练
```

**面试追问点**:
> Q: 如果给你 8 卡 3090，你会训练多大的模型？
> A: 125M-1.3B。训练 7B 需要约 140B tokens，按 8 卡速度需要 ~38 天，不划算。建议用开源 7B 模型做 SFT/DPO。

---

## 3. SFT 微调技巧

### 3.1 QLoRA 的实现细节

**新发现**: QLoRA 不只是加个 LoRA，有很多工程细节

```python
# QLoRA 的关键组件

1. 4-bit 量化 (NF4)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4，比普通INT4更好
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用bf16
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步压缩
)

2. LoRA 配置
from peft import LoraConfig

lora_config = LoraConfig(
    r=64,  # 秩，越大效果越好但显存越多
    lora_alpha=16,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

3. 训练时的注意事项
# - 梯度累积: 小 batch 累积到大 batch
# - 学习率: 比全参数微调大 10 倍左右
# - Epoch: 通常 1-3 个 epoch
```

---

### 3.2 SFT 数据格式的重要性

**新发现**: 数据格式对效果影响很大

```python
# 不同的数据格式

格式1: Alpaca 格式
{
    "instruction": "请翻译以下句子",
    "input": "Hello world",
    "output": "你好世界"
}

格式2: ShareGPT 格式
{
    "conversations": [
        {"from": "human", "value": "请翻译以下句子: Hello world"},
        {"from": "gpt", "value": "你好世界"}
    ]
}

格式3: System/User/Assistant 格式
{
    "messages": [
        {"role": "system", "content": "你是一个翻译助手"},
        {"role": "user", "content": "请翻译: Hello world"},
        {"role": "assistant", "content": "你好世界"}
    ]
}

# 关键点
1. System prompt 很重要，定义角色和行为
2. 多轮对话要保持上下文一致性
3. 格式要和推理时一致
```

**面试追问点**:
> Q: 为什么 SFT 数据格式这么重要？
> A: 因为模型学习的是"格式+内容"的联合分布。如果训练和推理格式不一致，模型会困惑。

---

### 3.3 数据质量 vs 数量

**新发现**: LIMA 论文的结论在实践中得到验证

```python
# LIMA: Less Is More for Alignment
# 结论: 1000 条高质量数据 > 50000 条低质量数据

# 实践经验
实验1: 
- 数据量: 100K 条低质量数据
- 效果: 指令遵循能力一般，经常跑偏

实验2:
- 数据量: 10K 条高质量数据（人工筛选）
- 效果: 指令遵循能力明显提升

# 高质量数据的标准
1. 指令清晰明确
2. 回复详细准确
3. 格式规范一致
4. 覆盖多种任务类型
5. 没有错误或误导信息

# 数据筛选方法
1. 用 GPT-4 打分 (1-10分)
2. 人工抽样检查
3. 去除重复和相似数据
4. 过滤过短或过长的回复
```

---

## 4. DPO/RLHF 后训练

### 4.1 DPO 的数学推导

**新发现**: DPO 不只是"去掉 RM"，有严格的数学推导

```python
# RLHF 目标
maximize E[r(x,y)] - β * KL[π(y|x) || π_ref(y|x)]

# 最优策略的闭式解
π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)

# 反解奖励函数
r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)

# 代入 Bradley-Terry 模型
P(y_w > y_l) = σ(r(x,y_w) - r(x,y_l))
             = σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))

# DPO 损失函数
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

**面试追问点**:
> Q: DPO 的 β 参数怎么选？
> A: β 控制偏离参考策略的程度。β 太大，模型变化小；β 太小，模型可能过拟合。通常 0.1-0.5。

---

### 4.2 DPO 的实际问题

**新发现**: DPO 在实践中有一些问题需要注意

```python
问题1: 过拟合偏好数据
# 表现: 在训练数据上表现好，但泛化差
# 解决: 
# - 使用更多样化的偏好数据
# - 增大 β
# - 减少训练 epoch

问题2: 长度偏差
# 表现: 模型倾向于生成更长的回答
# 原因: 更长的回答可能有更高的奖励
# 解决:
# - 在奖励中加入长度惩罚
# - 使用 IPO (Identity Preference Optimization)

问题3: 分布偏移
# 表现: 训练和测试分布不一致
# 原因: 偏好数据和实际使用场景不同
# 解决:
# - 在线采样 (Online DPO)
# - 迭代训练

# IPO 的改进
# IPO 不使用 log 比率，而是使用平方差
L_IPO = E[(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) - 1/(2β))²]
```

---

### 4.3 GRPO 的实现细节

**新发现**: GRPO 比 PPO 简单很多，效果也不错

```python
# GRPO (Group Relative Policy Optimization)
# DeepSeek 使用的方法

核心思想:
1. 对同一个问题采样 K 个回答
2. 用规则或模型给每个回答打分
3. 用组内均值和方差标准化奖励
4. 用标准化后的奖励作为优势估计

# 实现
def grpo_loss(policy, ref_policy, prompts, responses, rewards):
    # 计算策略概率
    pi_logprobs = policy.log_prob(prompts, responses)
    ref_logprobs = ref_policy.log_prob(prompts, responses)
    
    # 组内标准化奖励
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # 重要性采样比率
    ratio = torch.exp(pi_logprobs - ref_logprobs)
    
    # PPO 风格的裁剪
    clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    return loss.mean()

# 优势
1. 不需要 Critic Model (节省显存)
2. 组内相对奖励更稳定
3. 适合有明确答案的任务 (数学、代码)
```

**面试追问点**:
> Q: GRPO 和 PPO 的核心区别是什么？
> A: GRPO 用组内相对奖励代替 Critic Model 的价值估计，简化了训练流程，节省了显存。

---

## 5. Agent 系统工程

### 5.1 Function Calling vs ReAct 格式

**新发现**: Function Calling 比 ReAct 格式可靠得多

```python
# ReAct 格式 (项目中使用)
"""
Thought: 我需要查询天气
Action: get_weather(city="杭州")
"""

# 问题:
# 1. 格式不稳定，正则解析容易失败
# 2. 模型可能输出多余内容
# 3. 参数解析复杂

# Function Calling 格式
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        }
    }]
)

# 优势:
# 1. 结构化输出，解析简单
# 2. 模型原生支持，更稳定
# 3. 支持并行工具调用
```

---

### 5.2 Agent 的错误处理

**新发现**: 生产环境的 Agent 需要完善的错误处理

```python
class RobustAgent:
    def __init__(self):
        self.max_retries = 3
        self.max_steps = 10
        self.timeout = 30
    
    def run(self, query):
        for step in range(self.max_steps):
            try:
                # 1. 调用 LLM
                response = self.call_llm_with_retry(query)
                
                # 2. 解析动作
                action = self.parse_action(response)
                
                # 3. 检查是否完成
                if action.type == "finish":
                    return action.result
                
                # 4. 执行工具（带重试和超时）
                observation = self.execute_tool_with_retry(action)
                
                # 5. 更新上下文
                query = self.update_context(query, response, observation)
                
            except MaxRetriesExceeded:
                return "抱歉，我遇到了技术问题，请稍后再试"
            except TimeoutError:
                return "抱歉，处理超时，请简化您的问题"
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return "抱歉，出现了未知错误"
        
        return "抱歉，我无法在有限步骤内完成任务"
    
    def call_llm_with_retry(self, query):
        for attempt in range(self.max_retries):
            try:
                return self.llm.generate(query)
            except RateLimitError:
                time.sleep(2 ** attempt)  # 指数退避
            except APIError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
    
    def execute_tool_with_retry(self, action):
        for attempt in range(self.max_retries):
            try:
                with timeout(self.timeout):
                    return self.tools.execute(action)
            except TimeoutError:
                if attempt == self.max_retries - 1:
                    return "工具执行超时"
            except ToolError as e:
                if attempt == self.max_retries - 1:
                    return f"工具执行失败: {e}"
```

---

### 5.3 Agent 的记忆系统

**新发现**: 记忆系统让 Agent 更像"智能体"

```python
class MemorySystem:
    def __init__(self, embedding_model="BAAI/bge-base-zh"):
        self.embedder = SentenceTransformer(embedding_model)
        self.short_term = []  # 当前对话历史
        self.long_term = []   # 长期记忆
        self.vector_store = None  # 向量数据库
    
    def add_to_short_term(self, message):
        """添加到短期记忆（当前对话）"""
        self.short_term.append(message)
        # 保持窗口大小
        if len(self.short_term) > 10:
            # 摘要压缩
            summary = self.summarize(self.short_term[:5])
            self.long_term.append(summary)
            self.short_term = self.short_term[5:]
    
    def add_to_long_term(self, memory):
        """添加到长期记忆"""
        embedding = self.embedder.encode(memory)
        self.long_term.append({"text": memory, "embedding": embedding})
        # 更新向量索引
        self.update_vector_store()
    
    def retrieve(self, query, k=5):
        """检索相关记忆"""
        query_embedding = self.embedder.encode(query)
        
        # 从短期记忆检索
        short_term_scores = [
            (msg, self.similarity(query_embedding, msg))
            for msg in self.short_term
        ]
        
        # 从长期记忆检索（向量搜索）
        long_term_results = self.vector_store.search(query_embedding, k=k)
        
        # 合并结果
        all_results = short_term_scores + long_term_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in all_results[:k]]
```

---

## 6. 踩坑与解决方案

### 6.1 显存不足的解决方案

```python
# 方案1: Gradient Checkpointing
model.gradient_checkpointing_enable()
# 原理: 不保存中间激活值，反向传播时重新计算
# 效果: 显存减少 60-70%，计算增加 20-30%

# 方案2: 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 方案3: DeepSpeed ZeRO
# ZeRO-1: 优化器状态分片
# ZeRO-2: 优化器状态 + 梯度分片
# ZeRO-3: 优化器状态 + 梯度 + 参数分片

# 方案4: CPU Offload
# 将优化器状态和部分计算卸载到 CPU
```

---

### 6.2 训练速度优化

```python
# 优化1: Flash Attention
from flash_attn import flash_attn_func

# 原始注意力: O(n²) 内存
# Flash Attention: O(n) 内存，更快

# 优化2: 数据打包 (Packing)
# 将多条短序列打包成一条长序列，避免 padding 浪费
def pack_sequences(sequences, max_len):
    packed = []
    current = []
    current_len = 0
    
    for seq in sequences:
        if current_len + len(seq) <= max_len:
            current.append(seq)
            current_len += len(seq)
        else:
            packed.append(current)
            current = [seq]
            current_len = len(seq)
    
    return packed

# 优化3: 梯度累积
# 小 batch 累积到大 batch，减少通信
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 优化4: 编译优化
model = torch.compile(model)  # PyTorch 2.0+
```

---

### 6.3 多卡训练的坑

```python
# 坑1: 数据不一致
# 不同卡的数据不一样，导致梯度不同步
# 解决: 使用 DistributedSampler

from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 坑2: 随机种子不一致
# 不同卡的 dropout 等不一样
# 解决: 设置相同的种子，但不同的 local_rank

def set_seed(seed, local_rank):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed + local_rank)  # 不同卡不同种子

# 坑3: 日志重复
# 每个卡都打印日志，输出混乱
# 解决: 只在主卡打印

if local_rank == 0:
    print(f"Loss: {loss.item()}")

# 坑4: Checkpoint 保存
# 所有卡都保存，浪费空间
# 解决: 只在主卡保存

if local_rank == 0:
    torch.save(model.state_dict(), "model.pt")
```

---

## 7. 面试深度追问点

### 7.1 追问: 为什么选择 DPO 而不是 RLHF？

**深挖方向**:
```
Q1: DPO 的数学推导过程是什么？
Q2: DPO 有什么局限性？
Q3: 什么情况下 RLHF 更好？
Q4: 你们实际训练时 β 参数怎么选的？
Q5: 训练过程中遇到过什么问题？
```

**回答要点**:
```
1. DPO 从 RLHF 目标函数推导出来，消除了显式 RM
2. 局限性: 离线方法，无法在线探索
3. RLHF 更好: 需要在线学习、奖励函数复杂
4. β=0.1，通过实验调参
5. 遇到过拟合问题，通过增加数据多样性解决
```

---

### 7.2 追问: Agent 系统如何保证可靠性？

**深挖方向**:
```
Q1: 工具调用失败怎么办？
Q2: LLM 输出格式不稳定怎么办？
Q3: 如何防止 Agent 进入死循环？
Q4: 如何处理恶意输入（Prompt Injection）？
Q5: 线上出问题了怎么回滚？
```

**回答要点**:
```
1. 重试机制 + 指数退避 + 降级策略
2. Function Calling + 格式验证 + 自修复
3. 最大步数限制 + 死循环检测
4. 输入过滤 + 输出检查 + 沙箱执行
5. 版本管理 + 灰度发布 + 快速回滚
```

---

### 7.3 追问: 如何评估 Agent 的效果？

**深挖方向**:
```
Q1: 用什么指标评估？
Q2: 如何保证评估的客观性？
Q3: A/B 测试怎么做？
Q4: 如何处理评估中的不确定性？
Q5: 评估结果和预期不符怎么办？
```

**回答要点**:
```
1. 任务完成率、工具准确率、响应时间、用户满意度
2. 多人标注 + 一致性检查 + LLM辅助
3. 随机分流 + 分层 + 统计显著性检验
4. 多次运行取平均 + 置信区间
5. 排查数据问题、模型问题、系统问题
```

---

## 持续更新中...

*最后更新: 2026-06-23*
*基于 8x RTX 3090 实验环境*
