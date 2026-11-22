# LLM 学习笔记

从 **shape** 出发理解大模型：不先啃论文和数学，先把每一层的张量形状走通。

## 三条路线分别擅长什么

三者不是三种架构，是同一个 block 的三种用法。结构上的差别只有三处：

| | 结构 | mask | cross-attn | 输出头 |
| --- | --- | --- | --- | --- |
| **Transformer 2017** | Enc×N + Dec×N | encoder 无 / decoder 下三角 | 有 | Linear(dim_model, vocab_size) + softmax |
| **BERT** | Enc×N | 无（双向） | 无 | Linear(dim_model, num_labels) |
| **GPT** | Dec×N | 下三角 | 无 | Linear(dim_model, vocab_size) + softmax |

去掉 cross-attn 之后，GPT 的层和 BERT 的层是同一个东西，只差一个 mask。
所以 "encoder-only / decoder-only" 描述的是**用法**，不是两套积木。

"擅长什么"则由**输出长什么样**决定：

| | 输出形态 | 典型任务 | 工程特点 | 代表 |
| --- | --- | --- | --- | --- |
| **Transformer**<br/>enc-dec | 另一条序列，和输入逐位置对齐 | 机器翻译、语音识别、TTS、OCR 后处理 | 输入看全、输出因果，cross-attn 做对齐；结构最重 | 原始论文、T5、BART |
| **BERT**<br/>encoder-only | 一个标签 / 一段 span / 一个向量 | 文本分类、情感、意图、NER、抽取式 QA、句向量、检索召回与 rerank | 一次看全句（双向）；每个 token 出上下文向量，接个小 head 微调；**延迟低、吞吐高、输出稳定、便宜** | BERT、RoBERTa、DeBERTa |
| **GPT**<br/>decoder-only | 一段新的文本 | 对话、写作、代码生成、摘要、翻译、工具调用 / agent、多步推理 | 自回归逐 token 生成；一个模型覆盖多种任务，few-shot 强；**延迟高、贵、输出不可控** | GPT、Llama、Qwen |

怎么选：

1. **输出是一个标签 / 一个 span / 一个向量** → encoder
2. **输出是一段新文本，格式事先说不准** → decoder
3. **输入输出都是序列，且要求逐位置对齐** → enc-dec

还有第四条：**输出是标签，但标注数据很少** —— 用 decoder 做 few-shot 可能比训 encoder
更省事，代价是延迟和成本。边界一直在移动，不是硬规则。

## 目录

```
llm/
├── README.md                 # 本文件：路线 + 进度
├── transformer/
│   ├── architecture.md       # Transformer 架构图 + shape + 参数/FLOPs/访存
│   ├── model.py              # 自写 enc-dec，加载 opus-mt 权重，打印真实 shape
│   ├── data/pairs.tsv
│   └── runs/model_py_flow.log
├── gpt/
│   ├── README.md             # GPT-2 架构 + shape + 参数/FLOPs/KV cache
│   ├── model.py              # 自写 GPT-2，加载 HF 权重，prefill + decode
│   └── runs/gpt2_shapes.log
└── bert/
    ├── architecture.md       # BERT 架构图 + shape + 参数/FLOPs/访存
    ├── model.py              # 自写 encoder-only，加载 bert-base-uncased
    └── runs/bert_flow.log
```

## 三层目标

| 层 | 目标 | 产出 | 怎么算做完 |
| --- | --- | --- | --- |
| **L1 读图** | 看懂架构，**精确**记住 shape | 带形状标注的架构图 + 30 秒口述 | 遮住文档默画，形状零误差；能说出每个维度为什么是这个值 |
| **L2 真机** | 真实 prompt 喂进去，打印所有真实 shape | `model.py` 输出日志 | 每个 module 的 in/out shape 与 L1 预测逐项 diff，**差异为空**；含 prefill 和 decode 两步 |
| **L3a 理论** | 逐算子算 FLOPs / Bytes，算出"应该是多少时间" | 每个模型一张账表 | 两种算法互相验证相等；能对任意 config 报出参数 / 每层 FLOPs / 每 token FLOPs / decode 字节 |
| **L3b 实测** | 跑 profiler，填实测列 | 同一张表多两列：`实测 t` / `倍数` | 每个算子误差 < 3×，否则能归因（launch-bound / memory-bound / 算错了） |

两点说明：

- **L3a 和硬件无关。** 换机器只重跑 L3b，表结构、公式、预测值一个字不动。
  所以它是唯一能让你在拿到任何机器之前，就判断"这个模型能不能跑"的东西。
- **L2 必须打两遍：prefill 和 decode。** prefill 时 `T = prompt_len`，
  decode 时 `T = 1` 且带 KV cache。同一个模型、同一份代码、两套形状 ——
  "推理为什么分两阶段"只有在这里才看得见。

## 模型阶梯

| # | 载体 | 相比上一个新增什么 | L1 | L2 | L3a | L3b |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | **Transformer 2017**（enc-dec） | 地基：MHA + FFN + 残差 + LN + 位置编码 | ✅ | ✅ | ✅ | [ ] |
| 1 | **GPT-2 124M** | decoder-only；删掉 encoder 和 cross-attn；学习式位置编码；weight tying | ✅ | ✅ | ✅ | [ ] |
| 2 | **HY3** | 待查 config | [ ] | [ ] | [ ] | [ ] |
| 3 | **HY4 / HY4-Flash** | 待查 config | [ ] | [ ] | [ ] | [ ] |
| — | **BERT-base**（旁支） | encoder-only；双向、无 causal mask；可学习位置；三种 embedding；MLM 头 | ✅ | ✅ | ✅ | [ ] |

不做：Llama 权重、DeepSeek-V4.1-Flash。

为什么第 1 档是 GPT-2 124M：

- **能真跑**：124M 参数，FP32 约 498MB，prefill 和 decode 都能在这台 M4 上跑完；GPT-3 175B 只能纸面算。
- **只改四件事**：相对第 0 档，只删 encoder/cross-attn、换学习式位置编码、pre-LN、GELU。一次只加一个维度，shape 差异看得清。
- **有真实权重**：官方 `config.json` + state_dict + 标准 tokenizer，L2 能对真实张量，不靠背论文。
- **缺的正好是下一档**：RoPE / GQA / SwiGLU / RMSNorm / MoE 都不在这里加，留到 HY3；否则一次变太多，shape 对不上时分不清是哪一步的锅。

GPT-1 和 GPT-2 是同一套 decoder block，不单独设档；GPT-2 的 medium/large/xl 也只是改 config。GPT-3 没有开放权重，纸面看论文即可，不占一个动手档位。

## 三条原则

1. **L1 不许"大概"。** L1 是 L2 的对照物；记个大概，L2 就退化成看热闹。
2. **L3 分两半。** 理论现在做完，实测跟着机器走；没有 GPU 不是不做 L3 的理由。
3. **shape 的唯一真源是 `config.json`，不是记忆。** 每档开工第一步，把
   `hidden_size / num_attention_heads / num_key_value_heads / intermediate_size /
   num_experts / moe_top_k` 抄成一张表。第 2、3 档尤其如此 —— 名字对不上就先纠正。

从 GPT-2 直接跳到 HY3，中间的零件（RoPE / GQA / SwiGLU / KV cache 之类）不单独设一档，
等 HY3 的 config 到手，按真实字段逐个对。

## 已知数：GPT-2 124M

参数是真实 `state_dict` 数的；FLOPs 和字节按公式算。shape 见 `gpt/runs/gpt2_shapes.log`：

| 项 | 值 |
| --- | --- |
| 参数 | 124,439,808（约 124.4M） |
| prefill `T=1024` | 285M FLOPs / token |
| decode `T=1` | 247M FLOPs / token（context=1；context=1024 时 attention 再加 ~37.7M） |
| decode 权重 | 249 MB（FP16；FP32 约 498 MB） |
| KV cache `T=1024` | 38 MB（FP16） |

> 之前写的 `124,402,944` 少了 12 层 `c_fc` 的 bias（`12 × 3072 = 36,864`）。以真实 `state_dict` 的 **124,439,808** 为准。

decode 每生成一个 token 都要把 249 MB 权重读一遍 —— 这就是带宽瓶颈的来源，
也是为什么 decode 要看字节而不是 FLOPs。

## 这台机器

```
Apple M4 / 16 GB / arm64 / 无 CUDA / MPS 可用 / transformers 装在 ~/.venvs/llm-gpt / tiktoken 已装
```

L1 / L2 / L3a 都能做。L3b 等换机器：只需测本机的两个常数（可达算力、可达带宽），
再跑一遍 profiler 填实测列。

## 下一步

- [x] `transformers` 装到 `~/.venvs/llm-gpt`
- [x] GPT-2 `model.py` + `runs/gpt2_shapes.log`（prefill + decode 都有）
- [x] GPT-2 的 L1：单栈、causal mask、KV cache、参数量/FLOPs/访存
- [ ] HY3：拿到 config 后，按 `hidden_size / num_attention_heads / num_key_value_heads / intermediate_size / num_experts / moe_top_k` 逐项对
- [ ] L3b：换机器后跑 profiler，把 `实测 t / 倍数` 填进账表
