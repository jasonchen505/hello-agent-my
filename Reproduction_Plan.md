# Happy Agent 项目全流程复现计划

> 硬件配置: 8x RTX 3090 (24GB each, 192GB total)
> 目标: 完整复现 LLM + Agent + 后训练全流程

---

## 目录

1. [算力评估与资源规划](#1-算力评估与资源规划)
2. [Phase 1: 基础组件复现](#2-phase-1-基础组件复现)
3. [Phase 2: 小模型预训练](#3-phase-2-小模型预训练)
4. [Phase 3: SFT 监督微调](#4-phase-3-sft-监督微调)
5. [Phase 4: RLHF/DPO 后训练](#5-phase-4-rlhfdpo-后训练)
6. [Phase 5: Agent 系统构建](#6-phase-5-agent-系统构建)
7. [Phase 6: 端到端集成与评估](#7-phase-6-端到端集成与评估)
8. [时间规划与里程碑](#8-时间规划与里程碑)
9. [风险与备选方案](#9-风险与备选方案)

---

## 1. 算力评估与资源规划

### 1.1 硬件资源清单

```
GPU: 8x NVIDIA RTX 3090
├── 单卡显存: 24GB GDDR6X
├── 总显存: 192GB
├── FP32 算力: 35.6 TFLOPS
├── FP16 算力: 71 TFLOPS (with Tensor Cores)
└── 内存带宽: 936 GB/s

建议配置:
├── 系统内存: 256GB+ (数据加载)
├── 存储: 2TB+ NVMe SSD (数据集、checkpoint)
└── 网络: NVLink 或高速 PCIe (多卡通信)
```

### 1.2 不同任务的显存需求估算

| 任务 | 模型规模 | 显存需求 | 可行性 | 方案 |
|------|----------|----------|--------|------|
| Transformer 预训练 | 125M | ~8GB | ✅ 单卡 | 直接训练 |
| Transformer 预训练 | 1.3B | ~40GB | ✅ 2-4卡 | DDP/FSDP |
| Transformer 预训练 | 7B | ~160GB | ✅ 8卡 | DeepSpeed ZeRO-3 |
| SFT (全参数) | 7B | ~120GB | ✅ 8卡 | DeepSpeed ZeRO-3 |
| SFT (QLoRA) | 7B | ~24GB | ✅ 单卡 | 4-bit量化 + LoRA |
| DPO (全参数) | 7B | ~180GB | ✅ 8卡 | DeepSpeed ZeRO-3 |
| DPO (LoRA) | 7B | ~48GB | ✅ 2卡 | LoRA + DDP |
| RLHF (PPO) | 7B | ~200GB+ | ⚠️ 紧张 | 需优化 |
| Agent 推理 | 7B | ~16GB | ✅ 单卡 | vLLM |

### 1.3 资源分配策略

```
推荐分配:
├── Phase 1 (基础): 1卡用于实验
├── Phase 2 (预训练): 4-8卡用于训练
├── Phase 3 (SFT): 2-8卡用于微调
├── Phase 4 (后训练): 4-8卡用于RLHF/DPO
├── Phase 5 (Agent): 1-2卡用于推理服务
└── Phase 6 (集成): 按需分配

并行策略:
├── 数据并行 (DDP): batch 分散到多卡
├── 模型并行 (FSDP): 参数分散到多卡
├── 流水线并行: 层间并行 (不推荐，3090无NVLink)
└── 专家并行 (MoE): 专家分散到多卡
```

---

## 2. Phase 1: 基础组件复现

### 目标
深入理解 Transformer、BPE、Attention 的底层实现

### 任务清单

#### 1.1 BPE 分词器实现 (Day 1)
```python
# 文件: bpe.py
# 当前状态: 基础demo已完成
# 扩展任务:

1. 实现完整的 BPE 训练流程
   - 支持大规模语料
   - 添加特殊 token 处理
   - 实现 encode/decode 函数

2. 与 tiktoken 对比
   - 分词效率对比
   - 分词结果对比
   - 理解 byte-level BPE

3. 实验任务:
   - 在中英文混合语料上训练
   - 分析词表大小对分词效果的影响
   - 可视化分词结果
```

#### 1.2 Transformer 核心模块 (Day 2-3)
```python
# 文件: transformer.py
# 当前状态: 完整实现已完成
# 深入任务:

1. 实现变体对比
   - Pre-Norm vs Post-Norm
   - ReLU vs GELU vs SwiGLU
   - 绝对位置编码 vs RoPE

2. 性能优化
   - Flash Attention 实现/调用
   - 混合精度训练
   - 梯度检查点

3. 实验任务:
   - 对比不同位置编码的外推能力
   - 分析注意力模式可视化
   - 测量不同配置的训练速度
```

#### 1.3 训练基础设施 (Day 4-5)
```python
# 新建: training_utils.py

1. 实现训练循环
   - 数据加载器
   - 优化器配置
   - 学习率调度
   - 梯度累积

2. 实现评估框架
   - 困惑度计算
   - 下游任务评估
   - 生成质量评估

3. 实现日志和监控
   - TensorBoard 集成
   - 显存监控
   - 训练速度统计
```

### 预期产出
- [ ] 可运行的 BPE 训练器
- [ ] 优化后的 Transformer 实现
- [ ] 完整的训练工具链
- [ ] 基础实验报告

---

## 3. Phase 2: 小模型预训练

### 目标
从零训练一个可用的语言模型，理解预训练全流程

### 3.1 模型配置

```python
# 配置1: 小模型 (用于快速实验)
small_config = {
    "vocab_size": 32000,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "d_ff": 3072,
    "max_seq_len": 1024,
    "dropout": 0.1,
}
# 参数量: ~125M
# 显存需求: ~8GB per GPU
# 训练时间: ~24小时 (单卡)

# 配置2: 中等模型 (更有实用价值)
medium_config = {
    "vocab_size": 32000,
    "d_model": 2048,
    "num_heads": 32,
    "num_layers": 24,
    "d_ff": 5504,
    "max_seq_len": 2048,
    "dropout": 0.1,
}
# 参数量: ~1.3B
# 显存需求: ~40GB (2-4卡)
# 训练时间: ~72小时 (4卡)
```

### 3.2 数据准备

```python
# 数据源选择
datasets = {
    "中文": {
        "source": "WuDaoCorpora / MNBVC",
        "size": "10GB subset",
        "预处理": "去重、过滤、分词"
    },
    "英文": {
        "source": "OpenWebText / RedPajama",
        "size": "10GB subset", 
        "预处理": "去重、过滤"
    },
    "代码": {
        "source": "The Stack (Python subset)",
        "size": "5GB",
        "预处理": "去重、过滤低质量"
    }
}

# 数据处理流程
1. 下载原始数据
2. 去重 (MinHash/SimHash)
3. 质量过滤 (长度、语言检测)
4. 分词 (BPE)
5. 打包成固定长度序列
6. 保存为 Arrow/Parquet 格式
```

### 3.3 训练配置

```python
# DeepSpeed 配置 (8卡训练 1.3B 模型)
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # ZeRO-2 用于 1.3B
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
    },
    "gradient_accumulation_steps": 8,
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1}
    },
    "scheduler": {
        "type": "CosineDecay",
        "params": {"warmup_num_steps": 2000}
    }
}
```

### 3.4 训练监控

```python
# 关键指标
metrics = {
    "loss": "训练损失",
    "learning_rate": "学习率",
    "grad_norm": "梯度范数",
    "gpu_memory": "显存使用",
    "tokens_per_second": "训练吞吐量",
    "eval_loss": "验证损失",
    "perplexity": "困惑度"
}

# 监控工具
- Weights & Biases (wandb)
- TensorBoard
- 自定义脚本
```

### 预期产出
- [ ] 处理好的训练数据集
- [ ] 125M 参数小模型
- [ ] 1.3B 参数中等模型 (可选)
- [ ] 训练日志和分析报告
- [ ] 模型生成效果评估

---

## 4. Phase 3: SFT 监督微调

### 目标
让预训练模型学会遵循指令

### 4.1 SFT 数据准备

```python
# 数据来源选项
sft_datasets = {
    "通用指令": {
        "source": "Alpaca / BELLE / Firefly",
        "size": "100K-500K 条",
        "格式": {"instruction": "", "input": "", "output": ""}
    },
    "对话数据": {
        "source": "ShareGPT / UltraChat",
        "size": "100K 条",
        "格式": {"conversations": [{"from": "human", "value": ""}, ...]}
    },
    "Agent数据": {
        "source": "自行构造",
        "size": "10K 条",
        "格式": {"thought": "", "action": "", "observation": ""}
    }
}

# 数据质量要求
1. 指令多样性: 覆盖不同类型任务
2. 回复质量: 准确、详细、有帮助
3. 格式一致性: 统一的对话格式
4. 去重: 避免重复数据
```

### 4.2 SFT 训练配置

```python
# 方案1: 全参数微调 (8卡)
sft_full_config = {
    "model": "pretrained_1.3B",
    "epochs": 3,
    "batch_size": 64,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_seq_len": 2048,
    "bf16": True,
    "deepspeed_stage": 2
}
# 显存需求: ~40GB (2-4卡)
# 训练时间: ~12小时

# 方案2: QLoRA 微调 (单卡/2卡)
sft_qlora_config = {
    "model": "Qwen/Qwen2-7B",  # 使用开源7B模型
    "load_in_4bit": True,
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_len": 2048
}
# 显存需求: ~24GB (单卡)
# 训练时间: ~24小时
```

### 4.3 评估指标

```python
# 自动评估
auto_metrics = {
    "loss": "训练/验证损失",
    "token_accuracy": "token级别准确率",
    "rouge_score": "与参考答案的ROUGE分数"
}

# LLM评估 (用GPT-4打分)
llm_metrics = {
    "helpfulness": "回答有帮助程度",
    "relevance": "回答相关性",
    "accuracy": "回答准确性",
    "coherence": "回答连贯性"
}

# 人工评估
human_metrics = {
    "win_rate": "与baseline对比胜率",
    "preference": "人类偏好排序"
}
```

### 预期产出
- [ ] 处理好的SFT数据集
- [ ] SFT后的模型
- [ ] 评估报告 (自动 + 人工)
- [ ] 与base model的对比分析

---

## 5. Phase 4: RLHF/DPO 后训练

### 目标
通过人类偏好对齐，提升模型质量和安全性

### 5.1 方案选择: DPO vs RLHF

```
推荐: DPO (原因如下)

DPO 优势:
1. 实现简单: 不需要单独训练 Reward Model
2. 训练稳定: 不需要 PPO 的复杂调参
3. 显存友好: 比 RLHF 节省 ~50% 显存
4. 效果相当: 在很多场景下效果接近 RLHF

RLHF 优势:
1. 更灵活: 可以在线采样
2. 理论更强: 显式建模奖励函数
3. 可解释: 有独立的奖励模型

8卡3090的选择:
- DPO: 完全可行，4-8卡训练7B模型
- RLHF: 可行但紧张，需要优化
```

### 5.2 DPO 数据准备

```python
# 数据格式
dpo_data_format = {
    "prompt": "用户的问题或指令",
    "chosen": "人类偏好的回答",
    "rejected": "不被偏好的回答"
}

# 数据来源
dpo_datasets = {
    "开源数据": {
        "UltraFeedback": "大规模偏好数据",
        "HH-RLHF": "Anthropic人类反馈",
        "Nectar": "多维度偏好数据"
    },
    "自建数据": {
        "流程": [
            "1. 用SFT模型对同一prompt生成多个回答",
            "2. 人类/GPT-4标注偏好",
            "3. 构造chosen-rejected对"
        ],
        "规模": "10K-50K条"
    }
}
```

### 5.3 DPO 训练配置

```python
# DPO 训练配置
dpo_config = {
    "model": "sft_model",
    "beta": 0.1,  # KL惩罚系数
    "learning_rate": 5e-7,
    "batch_size": 32,
    "max_seq_len": 2048,
    "epochs": 1,
    "bf16": True,
    "deepspeed_stage": 3,  # 需要更多显存
    "gradient_checkpointing": True
}

# 显存优化技巧
optimization_tips = {
    "gradient_checkpointing": "用计算换显存",
    "bf16": "混合精度训练",
    "deepspeed_zero3": "参数分片",
    "packing": "多条数据打包成一个序列"
}
```

### 5.4 RLHF 配置 (备选方案)

```python
# 如果选择 RLHF，需要训练三个模型
rlhf_models = {
    "actor": "SFT后的模型 (策略模型)",
    "critic": "价值模型 (估计状态价值)",
    "reward": "奖励模型 (人类偏好)",
    "ref": "参考模型 (SFT模型，用于KL惩罚)"
}

# 显存估算 (7B模型)
# Actor: 14GB (bf16)
# Critic: 14GB
# Reward: 14GB  
# Ref: 14GB (可以offload到CPU)
# 总计: ~60GB + 优化器状态

# PPO 配置
ppo_config = {
    "ppo_epochs": 4,
    "mini_batch_size": 8,
    "learning_rate": 1.4e-5,
    "kl_penalty": 0.2,
    "gamma": 1.0,
    "lam": 0.95,
    "clip_range": 0.2
}
```

### 5.5 评估与对比

```python
# 评估维度
evaluation = {
    "安全性": {
        "指标": "有害回答比例",
        "方法": "红队测试"
    },
    "有帮助性": {
        "指标": "人类偏好胜率",
        "方法": "A/B测试"
    },
    "知识准确性": {
        "指标": "事实正确率",
        "方法": "QA测试集"
    },
    "指令遵循": {
        "指标": "格式正确率",
        "方法": "格式检查"
    }
}
```

### 预期产出
- [ ] DPO偏好数据集
- [ ] DPO对齐后的模型
- [ ] 与SFT模型的对比评估
- [ ] 安全性测试报告

---

## 6. Phase 5: Agent 系统构建

### 目标
构建可用的 Agent 系统，支持工具调用

### 6.1 Agent 架构升级

```python
# 当前项目的问题 (firstagent.py)
problems = {
    "解析脆弱": "正则表达式解析容易失败",
    "无重试": "解析失败直接退出",
    "无记忆": "没有长期记忆",
    "单Agent": "无法处理复杂任务"
}

# 改进方案
improvements = {
    "结构化输出": "使用 Function Calling 或 JSON Mode",
    "错误处理": "添加重试和降级机制",
    "记忆系统": "添加向量数据库存储历史",
    "多Agent": "支持Agent间协作"
}
```

### 6.2 本地模型推理服务

```python
# 使用 vLLM 部署本地模型
from vllm import LLM, SamplingParams

# 部署配置
vllm_config = {
    "model": "dpo_aligned_model",
    "tensor_parallel_size": 1,  # 单卡部署
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16"
}

# 或者使用 TGI (Text Generation Inference)
tgi_config = {
    "model_id": "dpo_aligned_model",
    "num_shard": 1,
    "max_input_tokens": 4096,
    "max_total_tokens": 8192
}
```

### 6.3 工具系统增强

```python
# 增强的工具系统
class EnhancedToolExecutor:
    def __init__(self):
        self.tools = {}
        self.tool_history = []  # 调用历史
        self.tool_cache = {}    # 结果缓存
    
    def register_tool(self, name, description, func, schema):
        """注册工具，包含参数schema"""
        self.tools[name] = {
            "description": description,
            "func": func,
            "schema": schema  # JSON Schema
        }
    
    def execute_with_retry(self, name, kwargs, max_retries=3):
        """带重试的工具执行"""
        for attempt in range(max_retries):
            try:
                result = self.tools[name]["func"](**kwargs)
                self.tool_history.append({
                    "tool": name,
                    "args": kwargs,
                    "result": result,
                    "success": True
                })
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: {e}"
                time.sleep(2 ** attempt)
```

### 6.4 Function Calling 支持

```python
# 使用 OpenAI 格式的 Function Calling
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 模型调用
response = client.chat.completions.create(
    model="local_model",
    messages=messages,
    tools=tools_schema,
    tool_choice="auto"
)

# 解析工具调用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
```

### 6.5 RAG 增强 (可选)

```python
# 简单的 RAG 系统
class SimpleRAG:
    def __init__(self, embedding_model="BAAI/bge-base-zh"):
        self.embedder = SentenceTransformer(embedding_model)
        self.vector_store = FAISS.from_documents(documents, self.embedder)
    
    def retrieve(self, query, k=3):
        """检索相关文档"""
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def generate_with_context(self, query):
        """带上下文的生成"""
        context = self.retrieve(query)
        prompt = f"基于以下信息回答问题:\n\n{context}\n\n问题: {query}"
        return llm.generate(prompt)
```

### 预期产出
- [ ] 增强的 Agent 框架
- [ ] 本地模型推理服务
- [ ] Function Calling 支持
- [ ] 工具调用测试报告
- [ ] RAG 系统 (可选)

---

## 7. Phase 6: 端到端集成与评估

### 目标
整合所有组件，进行全面评估

### 7.1 系统集成

```python
# 完整系统架构
class HappyAgentSystem:
    def __init__(self):
        # 1. 加载对齐后的模型
        self.llm = self.load_model()
        
        # 2. 初始化工具系统
        self.tools = self.init_tools()
        
        # 3. 初始化记忆系统 (可选)
        self.memory = self.init_memory()
        
        # 4. 初始化 RAG (可选)
        self.rag = self.init_rag()
    
    def process_query(self, user_input):
        """处理用户查询"""
        # 1. 检索相关记忆
        relevant_memory = self.memory.retrieve(user_input)
        
        # 2. 检索相关知识 (RAG)
        relevant_docs = self.rag.retrieve(user_input)
        
        # 3. 构建 prompt
        prompt = self.build_prompt(user_input, relevant_memory, relevant_docs)
        
        # 4. Agent 循环
        for step in range(max_steps):
            # 调用 LLM
            response = self.llm.generate(prompt)
            
            # 解析 action
            action = self.parse_action(response)
            
            # 检查是否完成
            if action.type == "finish":
                return action.answer
            
            # 执行工具
            observation = self.tools.execute(action)
            
            # 更新 prompt
            prompt += f"\n{response}\nObservation: {observation}"
        
        return "达到最大步数限制"
```

### 7.2 评估框架

```python
# 评估数据集
eval_datasets = {
    "单工具任务": {
        "数量": 100,
        "示例": "今天杭州天气怎么样？",
        "评估指标": ["完成率", "准确率", "步骤数"]
    },
    "多工具任务": {
        "数量": 50,
        "示例": "查一下杭州天气，然后推荐景点",
        "评估指标": ["完成率", "工具选择准确率", "最终答案质量"]
    },
    "复杂推理任务": {
        "数量": 30,
        "示例": "对比杭州和上海的天气，哪个更适合旅游",
        "评估指标": ["推理正确性", "答案完整性"]
    }
}

# 评估指标
metrics = {
    "功能指标": {
        "task_completion_rate": "任务完成率",
        "tool_accuracy": "工具选择准确率",
        "argument_accuracy": "参数提取准确率"
    },
    "性能指标": {
        "latency": "端到端延迟",
        "token_usage": "token消耗",
        "cost": "调用成本"
    },
    "质量指标": {
        "answer_relevance": "答案相关性",
        "answer_completeness": "答案完整性",
        "answer_accuracy": "答案准确性"
    }
}
```

### 7.3 对比实验

```python
# 对比方案
experiments = {
    "baseline": {
        "模型": "原始Qwen-7B",
        "Agent": "ReAct格式",
        "工具": "基础工具"
    },
    "sft_version": {
        "模型": "SFT后的模型",
        "Agent": "ReAct格式",
        "工具": "基础工具"
    },
    "dpo_version": {
        "模型": "DPO对齐后的模型",
        "Agent": "Function Calling",
        "工具": "增强工具"
    },
    "final_version": {
        "模型": "DPO对齐后的模型",
        "Agent": "Function Calling + RAG",
        "工具": "增强工具 + 记忆"
    }
}
```

### 7.4 Demo 构建

```python
# 使用 Gradio 构建 Demo
import gradio as gr

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Happy Agent Demo")
        
        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(label="输入你的问题")
                submit_btn = gr.Button("提交")
            
            with gr.Column():
                output_box = gr.Textbox(label="Agent回答")
                steps_box = gr.Textbox(label="推理步骤")
        
        submit_btn.click(
            fn=process_query,
            inputs=[input_box],
            outputs=[output_box, steps_box]
        )
    
    return demo

# 启动
demo = create_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 预期产出
- [ ] 集成的 Agent 系统
- [ ] 全面的评估报告
- [ ] 对比实验结果
- [ ] 可演示的 Demo

---

## 8. 时间规划与里程碑

### 8.1 总体时间线 (4-6周)

```
Week 1: Phase 1 - 基础组件
├── Day 1-2: BPE分词器完善
├── Day 3-4: Transformer优化
└── Day 5-7: 训练基础设施搭建

Week 2: Phase 2 - 预训练
├── Day 1-2: 数据准备
├── Day 3-5: 125M模型训练
└── Day 6-7: 模型评估

Week 3: Phase 3 - SFT
├── Day 1-2: SFT数据准备
├── Day 3-5: SFT训练
└── Day 6-7: 评估与优化

Week 4: Phase 4 - DPO
├── Day 1-2: 偏好数据准备
├── Day 3-5: DPO训练
└── Day 6-7: 评估与对比

Week 5-6: Phase 5&6 - Agent与集成
├── Day 1-3: Agent系统构建
├── Day 4-5: 系统集成
└── Day 6-7: 评估与Demo
```

### 8.2 里程碑检查点

```
Milestone 1 (Week 1结束):
□ BPE分词器可以训练和使用
□ Transformer可以正常训练
□ 训练工具链完整

Milestone 2 (Week 2结束):
□ 125M模型训练完成
□ 模型可以生成合理文本
□ 有完整的训练日志

Milestone 3 (Week 3结束):
□ SFT模型效果明显提升
□ 指令遵循能力验证通过
□ 有对比评估报告

Milestone 4 (Week 4结束):
□ DPO模型安全性提升
□ 人类偏好胜率 > 60%
□ 有完整的后训练分析

Milestone 5 (Week 5-6结束):
□ Agent系统可以正常工作
□ 端到端评估完成
□ Demo可演示
```

---

## 9. 风险与备选方案

### 9.1 潜在风险

```
风险1: 显存不足
├── 原因: 模型太大或batch太大
├── 解决: 使用QLoRA、gradient checkpointing、deepspeed
└── 备选: 减小模型规模

风险2: 训练不稳定
├── 原因: 学习率、数据质量
├── 解决: 调整学习率、warmup、gradient clipping
└── 备选: 使用更稳定的优化器

风险3: 数据质量差
├── 原因: 噪声数据、标注错误
├── 解决: 数据清洗、去重、人工审核
└── 备选: 使用更小但更干净的数据集

风险4: 时间不足
├── 原因: 任务复杂、遇到问题
├── 解决: 优先级排序、简化方案
└── 备选: 跳过预训练，直接使用开源模型
```

### 9.2 简化方案

```
如果时间/资源紧张，可以简化:

简化方案A (跳过预训练):
├── 直接使用 Qwen-7B 作为base model
├── 进行 SFT + DPO
└── 节省时间: ~2周

简化方案B (单卡方案):
├── 使用 QLoRA 进行微调
├── 单卡训练7B模型
└── 适合快速实验

简化方案C (API方案):
├── 使用 OpenAI/Anthropic API
├── 专注于 Agent 框架
└── 适合应用开发
```

---

## 附录: 关键代码模板

### A. 训练脚本模板

```python
#!/usr/bin/env python
# train_pretrain.py

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import deepspeed

def train():
    # 1. 初始化分布式
    deepspeed.init_distributed()
    
    # 2. 加载模型
    model = TransformerModel(config)
    
    # 3. 配置 DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    # 4. 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()
            
            # 日志
            if step % log_interval == 0:
                wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    train()
```

### B. 评估脚本模板

```python
# evaluate_agent.py

def evaluate_agent(agent, test_cases):
    results = []
    
    for case in test_cases:
        try:
            # 运行 Agent
            output = agent.run(case["input"])
            
            # 评估结果
            result = {
                "input": case["input"],
                "expected": case["expected"],
                "actual": output,
                "correct": check_correctness(output, case["expected"]),
                "steps": agent.step_count
            }
        except Exception as e:
            result = {"error": str(e)}
        
        results.append(result)
    
    # 统计
    success_rate = sum(r["correct"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results if "steps" in r) / len(results)
    
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "details": results
    }
```

---

*文档生成日期: 2026-06-23*
*硬件配置: 8x RTX 3090*
*预计完成时间: 4-6周*
