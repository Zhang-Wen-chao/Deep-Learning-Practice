# BERT 架构

## 1. 任务

BERT 做的是**完形填空**（Masked Language Modeling）：盖住句子里的一些词，让模型猜。

```
输入   The capital of France is [MASK] .
输出   每个位置一个 30522 维的分布，[MASK] 那一位最高的是 "paris"
```

和 Transformer 的翻译任务比，最大的区别是：**输入一句话，输出还是这句话的每个位置**，
没有"生成另一条序列"这回事。所以它只需要 encoder。

论文在 GLUE 等 11 个理解任务上验证，另外还有两个预训练目标（MLM + NSP）。

## 2. 记号

| 名字 | bert-base-uncased | 是什么 |
| --- | --- | --- |
| `batch_size` | 1 | 几条句子 |
| `seq_len` | 9 | 一条句子的 token 数（含 `[CLS]`、`[SEP]`） |
| `dim_model` | 768 | 主干宽度 |
| `num_heads` | 12 | 头数 |
| `dim_k` | 64 | 每头维度 = 768 / 12 |
| `dim_ff` | 3072 | FFN 中间宽度 |
| `vocab_size` | 30522 | 词表大小 |
| `num_layers` | 12 | encoder 层数 |
| `max_pos` | 512 | 最长支持 512 个 token |

`config.json` 里叫 `hidden_size` / `num_attention_heads` / `intermediate_size` / `num_hidden_layers`。

## 3. 结构

### 3.1 和 Transformer encoder 比，差在哪

| 项 | Transformer encoder | BERT |
| --- | --- | --- |
| 注意力方向 | 双向 | **一样，双向** |
| causal mask | 没有 | **没有** |
| decoder | 有 | 删掉 |
| cross-attention | 有 | 删掉 |
| 位置编码 | sin/cos，固定 | **`nn.Embedding`，可学习** |
| 额外输入 | 无 | **`token_type_ids`（句子 A/B）** |
| 输出头 | 无 | **MLM + NSP** |
| 层数 | 6 | 12 |

**没有 causal mask 是重点**：BERT 不需要"预测下一个词"，它要的是每个位置都能看到左右两边，
才能从上下文同时推断被盖住的词。

### 3.2 图

```mermaid
flowchart TD
    ID(["ids<br/>(batch_size, seq_len)"]) --> W["word_emb<br/>(batch_size, seq_len, dim_model)"]
    TT(["token_type_ids"]) --> S["seg_emb<br/>(batch_size, seq_len, dim_model)"]
    P["pos_emb<br/>(seq_len, dim_model)"] --> ADD
    W --> ADD(("+"))
    S --> ADD
    ADD --> LN["LayerNorm"]
    LN --> BLK["Encoder Layer × 12<br/>self-attn（双向）→ +LN → FFN → +LN"]
    BLK --> H["每个位置<br/>(batch_size, seq_len, dim_model)"]
    H --> MLM["MLM 头<br/>(batch_size, seq_len, vocab_size)"]
    H --> C["取第 0 个位置<br/>[CLS] (batch_size, dim_model)"]
    C --> POOL["pooler + tanh<br/>(batch_size, dim_model)"]
```

### 3.3 shape 流水（真实跑出来的）

`"The capital of France is [MASK]."` → 9 个 token：

| 步骤 | 形状 |
| --- | --- |
| ids | `(1, 9)` |
| word_emb | `(1, 9, 768)` |
| pos_emb | `(1, 9, 768)`（广播） |
| seg_emb | `(1, 9, 768)` |
| 三者相加 + LN | `(1, 9, 768)` |
| q/k/v 拆头 | `(1, 12, 9, 64)` |
| scores | `(1, 12, 9, 9)` |
| softmax | `(1, 12, 9, 9)` |
| attn @ V + 拼头 | `(1, 9, 768)` |
| intermediate | `(1, 9, 3072)` |
| output.dense | `(1, 9, 768)` |
| [CLS] | `(1, 768)` |
| MLM logits | `(1, 9, 30522)` |

**全程没有一处 `seq_len` 和另一个长度相遇**——这是和 Transformer 最大的形状差别。
没有 decoder、没有 cross-attention，就没有 `tgt_len × src_len`。

### 3.4 三种 embedding 为什么要相加

```
word_emb      这个词是谁
pos_emb       它在第几位（可学习）
seg_emb       它属于第几句（A 还是 B）
```

三者都是 `(batch_size, seq_len, dim_model)`，逐元素相加，宽度不变。
`seg_emb` 只对"输入两句话"的任务（如 NSP、问答）有意义，单句时全是 0。

### 3.5 MLM 头

```
每个位置 (1, 9, 768)
  → Linear(768, 768) + GELU + LayerNorm   (1, 9, 768)
  → Linear(768, 30522)                    (1, 9, 30522)
```

最后一层的权重和 `word_emb` **共享**（weight tying），只额外加一个 30522 的 bias。

## 4. 参数

### 一层

| 部分 | 形状 | 参数量 |
| --- | --- | --- |
| `q/k/v/out` 四个投影 | 4 × (768×768 + 768) | 2,363,904 |
| `intermediate` | 3072×768 + 3072 | 2,362,368 |
| `output.dense` | 768×3072 + 768 | 2,360,064 |
| `LayerNorm` ×2 | γ、β 各 768 | 3,072 |
| **一层合计** | | **7,087,872** |

`768² = 589,824`，`768×3072 = 2,359,296`——和 Transformer 一样，FFN 是注意力的两倍。

### 整个模型

| 部分 | 参数量 |
| --- | --- |
| embedding（word 23.4M + pos 0.4M + seg + LN） | 23,837,184 |
| 12 层 | 85,054,464 |
| pooler | 590,592 |
| MLM 头（transform + LN + bias） | 622,650 |
| **合计** | **110,104,890** |

和 HuggingFace 的 `BertForMaskedLM`（109,514,298）差 590,592，**正好是 pooler**——
HF 的 MLM 版本不带 pooler，我们的保留着（分类任务要用）。

## 5. 计算量

一条规则：`FLOPs = 2 × 参数量 × token 数`。

```
12 层 encoder = 2 × 85,054,464 × 9 = 1,530,980,352 ≈ 1.43G
MLM 头        = 2 × 768 × 30522 × 9 =   421,936,128 ≈ 0.39G
                                    ─────────────
                                      1.82G
```

**FFN 仍占 2/3**：`4.72M / 7.09M`。

注意 `seq_len = 9` 时，attention 的 `9×9` 项只有 `2 × 12 × 9 × 9 × 64 = 124K`，可以忽略；
但 `seq_len = 512` 时它变成 `2 × 12 × 512 × 512 × 64 = 403M`，接近 FFN 的量级。

## 6. 访存

BERT 只有**一次前向**，不像 GPT 要逐 token 生成：

```
权重 110,104,890 × 4 字节（fp32）≈ 420M
```

跑一次就要把 420M 权重读一遍。所以 BERT 的算术强度是 `1.82G / 420M ≈ 4.3 flop/byte`，
比 GPT 的 decode（0.61）高得多——**BERT 更容易跑满算力，GPT 的 decode 卡带宽**。

这也是为什么批量推理时 BERT 吞吐高：batch 越大，权重摊得越薄。

## 7. 位置编码：可学习的

BERT 用 `nn.Embedding(512, 768)`，不是 sin/cos：

| | Transformer | BERT |
| --- | --- | --- |
| 怎么来 | 公式算，固定 | 训练学出来 |
| 参数 | 0 | 512 × 768 = 393,216 |
| 上限 | 由公式决定，理论可外推 | 硬上限 512 个位置 |

代价是丢了外推能力：句子超过 512 个 token 直接截断。
好处是位置表示可以跟着任务调，短句子上通常更好。

## 8. 一句话总结

> BERT = Transformer 的 encoder × 12 + 三种 embedding 相加 + MLM 头。
> 没有 decoder、没有 cross-attention、没有 causal mask，所以整个模型里
> **长度和长度从不相遇**，`scores` 永远是 `(seq_len, seq_len)` 的方阵。
