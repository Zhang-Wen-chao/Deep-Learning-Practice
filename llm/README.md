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

## 心算约定

不用计算器，看公式就能报出量级。三条：

1. **乘的过程中，凑出 1000 或 1024 就立刻换成 `1K` 带走**，剩下的继续乘。

   ```
   768 × 3072 = (3×256) × (3×1024)
              = 9 × 256 × 1K        ← 1024 变 1K，带走
              = 2,304K
   ```

   `4 × 256`、`2 × 512`、`8 × 128` 都能凑 1024，随手用。

2. **K 换 M 直接除 1000**，只为了读量级。

   ```
   2,304K ≈ 2.3M
   ```

   **不要在最后再除 1024**——那是画蛇添足。

3. **FLOPs 按矩阵乘算**：

   ```
   (m, k) @ (k, n)  →  m × k × n 次乘加
   × 2  →  FLOPs
   ```

   一层里的 Linear 全加起来，才等价于 `2 × 参数量 × token 数`。
   后者只是合并后的简写，**看不懂就用 `2mnk`**，别拿简写当定义。

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
│   ├── architecture.md       # GPT-2 架构 + shape + 参数/FLOPs/KV cache
│   ├── config.json           # HF 官方 config（shape 的唯一真源）
│   ├── model.py              # 自写 GPT-2，加载 HF 权重，prefill + decode
│   └── runs/gpt2_shapes.log
├── bert/
│   ├── architecture.md       # BERT 架构图 + shape + 参数/FLOPs/访存
│   ├── config.json           # HF 官方 config（shape 的唯一真源）
│   ├── model.py              # 自写 encoder-only，加载 bert-base-uncased
│   └── runs/bert_flow.log
├── hy3/
│   ├── architecture.md       # HY3 架构 + shape + 参数/FLOPs/访存
│   ├── config.json           # 官方 config（唯一真源）
│   ├── accounting.py         # 只读 config 的纸面账本
│   └── runs/accounting.log   # 原始输出
├── hy4/
│   ├── README.md             # HY4 入口：结论 + 文件导航
│   ├── architecture.md       # 结构、shape、参数、FLOPs、KV cache
│   ├── calc.py               # 只读 config 的账本
│   ├── config.json           # tencent/Hy4 config（唯一真源）
│   └── runs/calc.log
└── hy4_flash/
    ├── README.md             # HY4-Flash 结论 + config 速查 + 代码链接
    ├── architecture.md       # MLA / DSA / SWA / LatentMoE / Engram / iHC 形状
    ├── accounting.py         # 只读 config 的账本
    ├── config.json           # 官方 config（唯一真源）
    └── runs/                 # accounting.log + accounting_1m.log
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
| 2 | **HY3**（295B MoE） | MoE 192选8 + shared；GQA 64/8；QK-Norm；SwiGLU；RoPE；MTP | ✅ | — | ✅ | [ ] |
| 3 | **HY4**（774B，多模态） | Gated MLA + DSA + iHC + 256选8；ViT 1.23B + DFly2 3.3B；MLA 把 KV 压到 93KiB/token | ✅ | — | ✅ | [ ] |
| 4 | **HY4-Flash**（220B） | 12 层 DSA + 30 层 SWA；LatentMoE 512选16；Engram；ViT；DFly2 | ✅ | — | ✅ | [ ] |
| — | **BERT-base**（旁支） | encoder-only；双向、无 causal mask；可学习位置；三种 embedding；MLM 头 | ✅ | ✅ | ✅ | [ ] |

不做：Llama 权重、DeepSeek-V4.1-Flash。

HY3 的 L2 标 `—`：权重 BF16 约 597.6 GB，这台 16 GB 的 M4 装不下，也没有 8 卡机器。
能做的是 L1（架构 + shape）和 L3a（纸面账本）；MTP 的 3.8B 权重也包含在 checkpoint 里。

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

从 GPT-2 直接跳到 HY3 / HY4，中间的零件（RoPE / GQA / SwiGLU / KV cache / MLA /
DSA / MoE / iHC 之类）不单独设一档，直接在真实 config 上逐个对：

- `llm/hy3/`：295B MoE，GQA + QK-Norm + SwiGLU + RoPE + MTP；
- `llm/hy4/`：774B 多模态，Gated MLA + DSA + iHC + ViT + DFly2；
- `llm/hy4_flash/`：220B，混合 SWA/DSA + LatentMoE + Engram + ViT + DFly2。

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

## 已知数：HY3

纯纸面算，权重没下载。账本在 `hy3/runs/accounting.log`，所有数字由 `hy3/accounting.py`
只读 `hy3/config.json` 生成：

| 项 | 值 |
| --- | --- |
| 参数（主干 / 含 MTP） | 295,033,528,320（295.03B）/ 298,786,140,416（298.79B） |
| 激活参数 | 20,675,715,072（20.68B） |
| prefill `T=4096` | 209.8T FLOPs（51.2G / token） |
| decode `C=0` | 40.4G FLOPs / token |
| decode `C=128K` | 387.3G FLOPs / token |
| 权重（BF16） | 590.07 GB（含 MTP 597.57 GB） |
| decode 权重 / token | 40.36 GB（只读 top-8 专家） |
| KV cache | 320 KiB / token；256K 上下文 = 80 GiB |
| decode 算术强度 | 1.00 FLOP/B（C=0）→ 带宽瓶颈 |

含 MTP 的 298,786,140,416 和 HF safetensors 的 BF16 计数**逐位相等**，
说明 config 没抄错、MTP 没重复算 embedding/lm_head。

## 已知数：HY4 / HY4-Flash

纯纸面算，权重没下载。HY4 的账本在 `hy4/runs/calc.log`，Flash 在
`hy4_flash/runs/accounting.log`：

| 项 | HY4（正式版） | HY4-Flash |
| --- | --- | --- |
| checkpoint 参数 | 774.43B（769.91B text + 1.23B ViT + 3.30B DFly2） | 220.25B（含 Engram 63.46B） |
| 激活 / token | 49.06B text（有图 +1.23B） | 11.27B（text） |
| prefill | 107.45G FLOPs/token（S=2K） | 24.14G FLOPs/token（S=4K） |
| decode | 107.45G FLOPs/token（C=2K） | 25.98G FLOPs/token（C=128K） |
| decode 权重 / token | 89.99GiB（BF16） | 21.68GiB（含 fp32 lm_head） |
| KV / token | 93.0KiB（全历史） | 50.25KiB（上界）/ 16.50KiB（HF 默认） |
| 算术强度 | 1.11（C=2K） | 0.91–21.0（HF 默认；看上下文/是否含池化） |

HY4 的 checkpoint `774,434,002,142` 和 HF safetensors **逐位相等**，DFly2 3.297B
是残差；Flash 的 `220,255,011,261`（DFly2=0.85B 口径）和 `220,251,201,958`
差 0.0017%。

Flash 的 L1 细节（42 层排布、MLA/DSA/SWA 的每一个 shape、Engram 的 16 张
哈希表、iHC 的 4 条流）在 `hy4_flash/architecture.md`；L2 标 `—` 是因为
BF16 440.5 GB、4-bit 仍约 110 GB，16 GB 的 M4 装不下。

## 这台机器

```
Apple M4 / 16 GB / arm64 / 无 CUDA / MPS 可用 / transformers 装在 ~/.venvs/llm-gpt / tiktoken 已装
```

L1 / L2 / L3a 都能做。L3b 等换机器：只需测本机的两个常数（可达算力、可达带宽），
再跑一遍 profiler 填实测列。HY3 / HY4 / HY4-Flash 是例外：权重分别是 597.6 GB、
1.55 TB、440 GB，本机 L2 都做不了，只做 L1 + L3a。

## 下一步

- [x] `transformers` 装到 `~/.venvs/llm-gpt`
- [x] GPT-2 `model.py` + `runs/gpt2_shapes.log`（prefill + decode 都有）
- [x] GPT-2 的 L1：单栈、causal mask、KV cache、参数量/FLOPs/访存
- [x] HY3：config + 架构 + 参数/FLOPs/访存账本（`llm/hy3/`）
- [x] HY4：config + 架构 + 参数/FLOPs/访存账本（`llm/hy4/`）
- [x] HY4-Flash：config + 架构 + 参数/FLOPs/访存账本（`llm/hy4_flash/`）
- [ ] HY3 的 tiny config：`H=64, L=3, E=8, top_k=2` 跑同构代码，验证 shape
- [x] HY4-Flash 的 L1 架构文档（`llm/hy4_flash/architecture.md`）
- [ ] L3b：换机器后跑 profiler，把 `实测 t / 倍数` 填进账表
