# LLM & Agent 算法实习面试准备指南

> 基于 happy_agent 项目的深度解析与面试考察点

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [核心模块深度解析](#2-核心模块深度解析)
3. [LLM 基础知识面试考察点](#3-llm-基础知识面试考察点)
4. [Agent 系统设计面试考察点](#4-agent-系统设计面试考察点)
5. [后训练(Post-Training)技术考察点](#5-后训练post-training技术考察点)
6. [代码实现细节深挖](#6-代码实现细节深挖)
7. [系统设计与工程实践](#7-系统设计与工程实践)
8. [行为面试与项目介绍](#8-行为面试与项目介绍)

---

## 1. 项目架构总览

### 1.1 项目文件结构

```
happy_agent/
├── eliza.py              # 经典规则聊天机器人
├── word_embedding.py     # 词向量演示
├── bpe.py                # BPE分词算法
├── transformer.py        # Transformer完整实现
├── llm_client.py         # LLM API客户端
├── tools.py              # Agent工具系统
├── firstagent.py         # ReAct模式Agent
├── .env                  # 配置文件
└── arxiv_agent/          # 论文知识图谱应用
    └── ar2graph.py
```

### 1.2 技术栈演进路线

```
规则系统(eliza) → 词向量(word_embedding) → 分词(bpe) 
    → 序列模型(transformer) → LLM调用(llm_client) 
    → 工具系统(tools) → Agent系统(firstagent)
```

**面试可考察点**: 请解释这条技术演进路线背后的原因，每一步解决了什么问题？

---

## 2. 核心模块深度解析

### 2.1 ELIZA 聊天机器人 (eliza.py)

```python
# 核心设计：正则表达式模式匹配 + 代词转换
rules = {
    r'I need (.*)': ["Why do you need {0}?", ...],
    r'.* mother .*': ["Tell me more about your mother.", ...],
    r'.*': ["Please tell me more.", ...]  # 通配符兜底
}
```

**关键实现细节**:
- 使用正则表达式捕获组提取用户意图
- 代词转换 (`pronoun_swap`): "I am" → "you are"
- 随机响应模板增加多样性

**面试深挖问题**:
1. ELIZA 的局限性是什么？与现代 LLM 的本质区别？
2. 如何扩展 ELIZA 的规则库？有什么工程挑战？
3. 这种模式匹配方式在什么场景下仍有价值？

### 2.2 词嵌入 (word_embedding.py)

```python
# 核心公式: king - man + woman ≈ queen
result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]
sim = cosine_similarity(result_vec, embeddings["queen"])
```

**面试考察点**:
1. **为什么词嵌入能捕获语义关系？**
   - 分布式假说：词的含义由其上下文决定
   - 训练目标：预测上下文/被上下文预测
   
2. **余弦相似度 vs 欧氏距离？**
   - 余弦相似度衡量方向，不受向量长度影响
   - 欧氏距离衡量绝对距离
   
3. **Word2Vec 的两种模式**:
   - CBOW: 用上下文预测中心词
   - Skip-gram: 用中心词预测上下文

### 2.3 BPE 分词 (bpe.py)

```python
def get_stats(vocab):
    """统计相邻词元对的频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """合并最高频的词元对"""
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    # ...
```

**面试深挖问题**:

1. **BPE 的时间复杂度是多少？**
   - 每次合并需要遍历整个词表，O(V)
   - 总共需要 K 次合并，总复杂度 O(K*V)
   
2. **BPE vs WordPiece vs Unigram 的区别？**
   - BPE: 自底向上合并
   - WordPiece: 基于似然最大化的合并
   - Unigram: 自顶向下删除

3. **为什么 `</w>` 标记很重要？**
   - 标记词边界，区分 "un" + "happy" vs "unhappy"
   - 影响分词的确定性和可逆性

4. **现代 Tokenizer (如 tiktoken) 的优化？**
   - 字节级 BPE (Byte-level BPE)
   - 正则表达式预分词
   - 并行化实现

### 2.4 Transformer 模型 (transformer.py)

这是项目中最核心的模块，实现了完整的 Transformer 架构。

#### 2.4.1 多头注意力机制

```python
class MultiHeadAttention(nn.Module):
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 核心公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
```

**面试必问问题**:

1. **为什么要除以 sqrt(d_k)？**
   - 当 d_k 较大时，QK^T 的方差会很大
   - 大的点积值会导致 softmax 进入梯度饱和区
   - 除以 sqrt(d_k) 使方差保持在 1 左右
   
2. **Multi-Head 的作用？**
   - 不同的 head 可以关注不同类型的关系
   - 语法关系、语义关系、位置关系等
   - 类似 CNN 中多个 filter 捕获不同特征

3. **Mask 的类型和作用？**
   - Padding mask: 忽略填充位置
   - Causal mask: 防止看到未来信息（下三角矩阵）
   - 为什么用 -1e9 而不是 -inf？（数值稳定性）

#### 2.4.2 位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
```

**面试深挖**:

1. **为什么 Transformer 需要位置编码？**
   - 自注意力机制是置换不变的 (permutation invariant)
   - 没有位置信息就无法区分 "猫吃鱼" 和 "鱼吃猫"

2. **正弦位置编码的优点？**
   - 可以外推到更长的序列
   - 相对位置可以通过线性变换表示
   - PE(pos+k) 可以表示为 PE(pos) 的线性函数

3. **RoPE (旋转位置编码) 的改进？**
   - 将位置信息编码到注意力计算中
   - 天然支持相对位置
   - 更好的长序列泛化能力

#### 2.4.3 Encoder-Decoder 架构

```python
class Transformer(nn.Module):
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # padding mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones(...)).bool()  # causal mask
        tgt_mask = tgt_pad_mask & tgt_sub_mask
```

**面试考察点**:

1. **Encoder-Only vs Decoder-Only vs Encoder-Decoder 的适用场景？**
   - Encoder-Only (BERT): 理解任务，分类、NER
   - Decoder-Only (GPT): 生成任务，对话、续写
   - Encoder-Decoder (T5): Seq2Seq，翻译、摘要

2. **为什么现代 LLM 大多是 Decoder-Only？**
   - 统一的自回归训练目标
   - 更容易 scale up
   - In-context learning 能力更强

---

## 3. LLM 基础知识面试考察点

### 3.1 预训练相关

**Q: 解释 LLM 的预训练过程**

```
1. 数据准备: 网页、书籍、代码等海量文本
2. Tokenization: BPE/WordPiece 分词
3. 训练目标: Next Token Prediction (自回归)
4. 训练规模: 数千 GPU，数周到数月
5. 涌现能力: 数学推理、代码生成、指令遵循
```

**Q: 什么是 Scaling Law？**

- Chinchilla Scaling Law: 模型大小和数据量应该同步增长
- 计算最优配置: 约 20 tokens per parameter
- 实际应用: 用更小模型+更多数据往往优于大模型+少数据

**Q: 解释 KV Cache 机制**

```python
# 在自回归生成中，避免重复计算
# 第 t 步只需要计算新 token 的 Q，与历史的 K,V 拼接
# 空间复杂度: O(n * d_model * num_layers)
```

### 3.2 推理优化

**Q: 常见的推理加速方法？**

1. **Flash Attention**: IO-aware 的注意力计算优化
2. **量化**: INT8/INT4 量化，减少显存和计算
3. **投机解码 (Speculative Decoding)**: 小模型草稿 + 大模型验证
4. **连续批处理 (Continuous Batching)**: 动态合并请求

**Q: 什么是 PagedAttention (vLLM)？**

- 借鉴操作系统虚拟内存思想
- KV Cache 分页管理，减少内存碎片
- 支持高效的 beam search 和并行采样

### 3.3 长上下文处理

**Q: 如何处理超长文本？**

1. **RoPE 外推/内插**: 调整位置编码频率
2. **Sliding Window Attention**: 局部注意力窗口
3. **Ring Attention**: 分布式长序列处理
4. **RAG**: 检索增强，减少对长上下文的依赖

---

## 4. Agent 系统设计面试考察点

### 4.1 ReAct 模式详解 (firstagent.py)

```python
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手...

# 行动格式:
Thought: [思考过程和计划]
Action: [工具调用，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当收集到足够信息时，使用 finish(answer="...") 输出最终答案
"""
```

**ReAct 循环核心代码**:

```python
for i in range(5):  # 最大循环次数
    # 1. 构建 Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 2. 调用 LLM
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    
    # 3. 解析 Action
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    action_str = action_match.group(1).strip()
    
    # 4. 执行工具
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    observation = available_tools[tool_name](**kwargs)
    
    # 5. 记录观察
    prompt_history.append(f"Observation: {observation}")
```

**面试深挖问题**:

1. **ReAct 的优势是什么？**
   - 思维链 (CoT) + 工具调用的结合
   - 可解释性强：可以看到推理过程
   - 可纠错：观察结果可以修正错误推理

2. **这个实现有什么问题？如何改进？**
   - 正则解析脆弱：应该使用结构化输出 (JSON)
   - 固定循环次数：应该有终止条件检测
   - 无记忆机制：应该有长期记忆存储
   - 单 Agent：可以扩展为多 Agent 协作

3. **如何处理 Agent 的错误和异常？**
   - 工具调用失败的重试机制
   - 输出格式错误的自修复
   - 死循环检测和打破

### 4.2 工具系统设计 (tools.py)

```python
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def registerTool(self, name: str, description: str, func: callable):
        self.tools[name] = {"description": description, "func": func}
    
    def getAvailableTools(self) -> str:
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
```

**面试考察点**:

1. **工具描述的重要性**
   - LLM 根据描述决定何时使用哪个工具
   - 描述不清会导致工具误用
   - 应该包含参数说明和使用示例

2. **工具调用的标准化**
   - Function Calling vs JSON Mode vs ReAct 格式
   - OpenAI 的 function_call 参数
   - 如何处理复杂参数类型

3. **工具编排 (Tool Orchestration)**
   - 串行 vs 并行工具调用
   - 工具依赖关系处理
   - 工具调用结果的缓存

### 4.3 现代 Agent 框架设计

**Q: 设计一个生产级 Agent 系统需要考虑什么？**

```
1. 对话管理
   - 上下文窗口管理
   - 历史消息压缩/摘要
   - 多轮对话状态追踪

2. 工具系统
   - 动态工具注册
   - 工具权限控制
   - 工具调用审计日志

3. 安全性
   - Prompt 注入防护
   - 工具调用沙箱
   - 输出内容过滤

4. 可观测性
   - 调用链追踪 (Tracing)
   - 性能监控
   - 错误告警

5. 人机协作
   - 人类审批流程
   - 降级策略
   - 反馈收集
```

---

## 5. 后训练(Post-Training)技术考察点

### 5.1 SFT (Supervised Fine-Tuning)

**Q: 什么是 SFT？为什么需要 SFT？**

```
预训练模型 → 学会了语言能力 → 但不会遵循指令
SFT → 用高质量的 (instruction, response) 对训练 → 学会遵循指令
```

**面试深挖**:

1. **SFT 数据质量 vs 数量？**
   - LIMA 论文: 1000 条高质量数据 > 50000 条低质量数据
   - 数据多样性比数量更重要
   - 如何构建高质量 SFT 数据集

2. **SFT 的常见问题？**
   - 灾难性遗忘：如何保留预训练能力
   - 过拟合：小数据集训练的正则化
   - 对齐税 (Alignment Tax)：某些能力下降

3. **LoRA/QLoRA 微调？**
   - 低秩适配: W' = W + BA
   - 减少可训练参数: 从 7B 到几十 M
   - QLoRA: 4-bit 量化 + LoRA

### 5.2 RLHF (Reinforcement Learning from Human Feedback)

**Q: 解释 RLHF 的完整流程**

```
阶段1: SFT → 基础指令遵循能力
阶段2: Reward Model 训练 → 学习人类偏好
阶段3: PPO 优化 → 用强化学习优化策略
```

**面试深挖**:

1. **Reward Model 如何训练？**
   - 人类标注偏好对 (chosen vs rejected)
   - Bradley-Terry 模型
   - 损失函数: -log(σ(r(chosen) - r(rejected)))

2. **PPO 算法的核心思想？**
   - 策略梯度的改进
   - Clipping 机制防止策略更新过大
   - GAE (Generalized Advantage Estimation)

3. **RLHF 的问题？**
   - Reward Hacking: 学会欺骗奖励模型
   - 训练不稳定: 奖励模型和策略的对抗
   - 标注成本高: 需要大量人类偏好数据

### 5.3 DPO (Direct Preference Optimization)

**Q: DPO 如何简化 RLHF？**

```python
# DPO 损失函数
loss = -log(sigmoid(beta * (log_pi(chosen)/pi_ref(chosen) 
                          - log_pi(rejected)/pi_ref(rejected))))
```

**面试考察点**:

1. **DPO 的数学推导**
   - 从 RLHF 目标函数出发
   - 最优策略的闭式解
   - 消除显式的 Reward Model

2. **DPO vs RLHF 的优劣？**
   - DPO: 更简单、更稳定、无需 RM
   - RLHF: 更灵活、可在线学习、理论上更强

3. **DPO 的变体？**
   - IPO: 避免过拟合偏好数据
   - KTO: 只需要正/负标签，不需要配对
   - ORPO: 将 SFT 和对齐合并

### 5.4 GRPO (Group Relative Policy Optimization)

**Q: GRPO 是什么？为什么 DeepSeek 使用它？**

```python
# GRPO 核心: 组内相对奖励
# 对同一个问题采样多个回答
# 用组内均值和方差标准化奖励
advantage = (reward - mean(rewards)) / std(rewards)
```

**面试考察点**:

1. **GRPO vs PPO 的区别？**
   - 无需 Critic Model (节省显存)
   - 组内相对奖励 (更稳定)
   - 更适合数学/代码任务

2. **如何设计奖励函数？**
   - 规则奖励: 答案正确性
   - 模型奖励: 过程奖励模型 (PRM)
   - 混合奖励: 多维度综合

---

## 6. 代码实现细节深挖

### 6.1 LLM 客户端实现 (llm_client.py)

```python
class HelloAgentsLLM:
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,  # 流式输出
        )
        
        # 处理流式响应
        collected_content = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            collected_content.append(content)
        return "".join(collected_content)
```

**面试问题**:

1. **为什么默认 temperature=0？**
   - 确定性输出，便于调试
   - 生产环境中创意任务需要更高温度
   
2. **流式输出的实现原理？**
   - Server-Sent Events (SSE)
   - chunk.choices[0].delta.content
   - 前端如何处理流式渲染

3. **错误处理和重试策略？**
   - 网络超时重试
   - Rate limiting 处理
   - 降级策略

### 6.2 搜索工具实现 (tools.py)

```python
def search(query: str) -> str:
    # 智能解析：优先寻找最直接的答案
    if "answer_box_list" in results:
        return "\n".join(results["answer_box_list"])
    if "answer_box" in results and "answer" in results["answer_box"]:
        return results["answer_box"]["answer"]
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
        return results["knowledge_graph"]["description"]
    if "organic_results" in results and results["organic_results"]:
        snippets = [...]  # 格式化前三个结果
        return "\n\n".join(snippets)
```

**面试考察点**:

1. **搜索结果的优先级设计**
   - 直接答案 > 知识图谱 > 有机结果
   - 不同搜索引擎的 API 差异
   
2. **如何优化搜索查询？**
   - 查询改写 (Query Rewriting)
   - 多轮搜索策略
   - 搜索结果去重和聚合

### 6.3 知识图谱构建 (arxiv_agent/ar2graph.py)

```python
class Ar2Graph:
    def extract_triplets(self):
        system_prompt = """
        从论文摘要中提取核心知识三元组。
        规则：
        1. 输出必须是纯 JSON 格式
        2. 包含 'head', 'relation', 'tail' 三个键
        3. 尽量提取 3-5 个最核心的三元组
        """
        # 使用 JSON Mode 确保输出格式
        response = self.client.chat.completions.create(
            model="glm-4.6",
            response_format={"type": "json_object"}
        )
```

**面试问题**:

1. **如何保证 LLM 输出结构化数据？**
   - JSON Mode: response_format={"type": "json_object"}
   - Function Calling: 定义函数签名
   - 输出解析和自修复

2. **知识图谱的应用场景？**
   - RAG 增强
   - 推荐系统
   - 问答系统

---

## 7. 系统设计与工程实践

### 7.1 设计一个 RAG 系统

**Q: 如何设计一个生产级 RAG 系统？**

```
文档处理层:
├── 文档解析 (PDF, Word, HTML)
├── 文本分块 (Chunking Strategy)
├── 向量化 (Embedding Model)
└── 存储 (Vector DB: Milvus/Pinecone/Weaviate)

检索层:
├── 向量检索 (ANN Search)
├── 关键词检索 (BM25)
├── 混合检索 (Hybrid Search)
└── 重排序 (Reranker)

生成层:
├── Prompt 构建
├── LLM 推理
├── 引用溯源
└── 答案验证
```

**面试深挖**:

1. **Chunking 策略？**
   - 固定长度 vs 语义分块
   - 重叠窗口
   - 递归分块

2. **如何评估 RAG 效果？**
   - 检索评估: Recall@K, MRR
   - 生成评估: Faithfulness, Relevancy
   - 端到端评估: 人工评测

### 7.2 设计一个多 Agent 系统

**Q: 如何设计多 Agent 协作系统？**

```
架构模式:
1. 管道模式: A → B → C (顺序执行)
2. 路由模式: Router → 选择专家 Agent
3. 辩论模式: 多个 Agent 讨论达成共识
4. 层级模式: Manager Agent 协调 Worker Agents

通信机制:
- 共享内存 (Message Pool)
- 消息传递 (Direct Messaging)
- 事件驱动 (Event Bus)

关键挑战:
- 任务分解和分配
- 冲突解决
- 状态同步
- 错误传播
```

### 7.3 Prompt 工程最佳实践

**Q: 如何设计高质量的 Prompt？**

```python
# 结构化 Prompt 设计
system_prompt = """
# Role: 你是一个专业的...

# Context: 
- 背景信息1
- 背景信息2

# Task: 请完成以下任务...

# Constraints:
- 约束条件1
- 约束条件2

# Output Format:
- 输出格式要求

# Examples:
- 示例输入输出
"""
```

**面试考察点**:

1. **Prompt 的关键要素？**
   - 角色定义 (Role)
   - 上下文 (Context)
   - 任务描述 (Task)
   - 约束条件 (Constraints)
   - 输出格式 (Format)
   - 示例 (Examples)

2. **Prompt 优化技巧？**
   - Few-shot vs Zero-shot
   - Chain-of-Thought (CoT)
   - Self-Consistency
   - Tree-of-Thought (ToT)

---

## 8. 行为面试与项目介绍

### 8.1 如何介绍这个项目？

**STAR 法则回答模板**:

```
Situation (背景):
- 学习 LLM 和 Agent 的过程中，我发现理论和实践有差距
- 希望通过动手实现来深入理解底层原理

Task (任务):
- 从零实现一个完整的 Agent 系统
- 涵盖从基础 (Transformer) 到应用 (Agent) 的全链路

Action (行动):
- 实现了 Transformer 核心组件 (Attention, Position Encoding)
- 封装了 LLM 客户端，支持流式输出
- 设计了工具注册和执行系统
- 实现了 ReAct 模式的 Agent 循环
- 构建了 ArXiv 论文知识图谱应用

Result (结果):
- 深入理解了 LLM 和 Agent 的工作原理
- 掌握了从模型调用到系统设计的完整技能栈
- 能够独立设计和实现 Agent 应用
```

### 8.2 常见行为面试问题

**Q: 遇到过什么技术挑战？如何解决的？**

示例回答:
```
在实现 Agent 的工具调用时，我发现 LLM 的输出格式不稳定，
有时候会输出多余的文本，导致正则解析失败。

解决方案:
1. 使用更鲁棒的解析策略 (多层 fallback)
2. 在 System Prompt 中明确格式要求
3. 添加输出验证和自修复机制
4. 考虑使用 Function Calling 替代文本解析
```

**Q: 你对 LLM 未来发展的看法？**

```
技术趋势:
1. 多模态融合: 视觉、语音、代码的统一
2. 长上下文: 从 4K 到 100K+ 的突破
3. 推理能力: 从模式匹配到真正推理
4. 效率优化: 更小模型达到更好效果

应用趋势:
1. Agent 生态: 从对话到行动
2. 垂直领域: 金融、医疗、法律的专业化
3. 人机协作: AI 增强而非替代
```

### 8.3 面试反问环节

**你应该问面试官的问题**:

1. "团队目前在 Agent 方面的主要挑战是什么？"
2. "实习期间会参与什么类型的项目？"
3. "团队如何平衡模型效果和工程效率？"
4. "对于 Agent 的安全性，团队有什么实践经验？"

---

## 附录: 高频面试题目速查

### 基础概念题

| 题目 | 关键点 |
|------|--------|
| Transformer 的注意力机制 | QKV、缩放因子、多头 |
| BPE 分词原理 | 频率合并、词边界 |
| 位置编码的作用 | 正弦/余弦、RoPE |
| KV Cache | 避免重复计算、空间换时间 |
| Flash Attention | IO-aware、分块计算 |

### Agent 相关题

| 题目 | 关键点 |
|------|--------|
| ReAct 模式 | Thought-Action-Observation |
| 工具调用设计 | 描述清晰、参数标准化 |
| 多 Agent 协作 | 通信机制、任务分配 |
| Agent 安全性 | 注入防护、权限控制 |

### 后训练题

| 题目 | 关键点 |
|------|--------|
| SFT 的作用 | 指令遵循、数据质量 |
| RLHF 流程 | RM 训练、PPO 优化 |
| DPO 原理 | 隐式奖励、简化流程 |
| GRPO 特点 | 组内相对奖励、无需 Critic |

### 系统设计题

| 题目 | 关键点 |
|------|--------|
| RAG 系统设计 | 检索+生成、Chunking |
| 长上下文处理 | RoPE 外推、Sliding Window |
| 推理优化 | 量化、连续批处理 |
| Prompt 工程 | 结构化、CoT、Few-shot |

---

## 学习资源推荐

### 论文
1. Attention Is All You Need (Transformer 原论文)
2. BERT/GPT 系列论文
3. ReAct: Synergizing Reasoning and Acting
4. DPO: Direct Preference Optimization
5. Constitutional AI (RLHF 改进)

### 开源项目
1. LangChain / LlamaIndex (Agent 框架)
2. vLLM / TGI (推理引擎)
3. OpenRLHF / TRL (后训练框架)
4. AutoGPT / MetaGPT (Agent 应用)

### 课程
1. Stanford CS324: Advances in Foundation Models
2. DeepLearning.AI: Building Systems with ChatGPT
3. Hugging Face NLP Course

---

*文档生成日期: 2026-06-22*
*基于 happy_agent 项目代码分析*
