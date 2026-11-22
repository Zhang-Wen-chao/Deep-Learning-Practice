# GPT-2（124M）

## 1. 任务

GPT-2 做的是**下一 token 预测**：给一段文本，预测下一个 token；把预测接回输入，重复，就是生成。

它和 Transformer 2017 用的是同一套 block，只保留 decoder：

- 删掉整个 encoder
- 删掉 cross-attention
- 只剩一条序列，`q_len` 和 `kv_len` 不再来自两个长度
- 位置编码从 sin/cos 换成可学习的 `wpe`

训练时整句喂进去，每个位置都预测下一个 token；推理时一个 token 一个 token 往后接。
训练和推理的 shape 在 prefill 阶段相同，差别在 decode 阶段：`q_len = 1`，并且要带 KV cache。

## 2. 记号

和 Transformer 那篇相同的部分不再重复，只列 GPT-2 新出现的：

| 名字 | 是什么 |
| --- | --- |
| `seq_len` | 一条序列的 token 数；prefill 时是 prompt 长度，decode 时是 1 |
| `q_len` | 当前这次 forward 的 query 数：prefill = `seq_len`，decode = 1 |
| `kv_len` | K/V 的长度：prefill = `seq_len`，decode = 历史长度 + 1 |
| `max_pos` | 位置嵌入表的最大长度，GPT-2 是 1024 |
| `c_attn` | combined attention：一个 Linear 一次算出 Q/K/V：768 → 2304，再切成三份 |
| `c_proj` | Conv1D projection：attention 的 `c_proj` 就是 `out_proj`，MLP 的 `c_proj` 就是 `fc2` |
| `wte` / `wpe` | weight token embedding / weight position embedding |
| KV cache | decode 时缓存每层历史 K/V，避免重算 |

GPT-2 124M 的配置。本地 [`config.json`](config.json) 是 Hugging Face 官方 config 的副本，`model.py` 直接读它；下表是字段映射：

| 名字 | 值 | `config.json` 字段 |
| --- | --- | --- |
| `vocab_size` | 50257 | `vocab_size` |
| `dim_model` | 768 | `n_embd` |
| `num_layers` | 12 | `n_layer` |
| `num_heads` | 12 | `n_head` |
| `dim_k` | 64 | `n_embd / n_head` |
| `dim_ff` | 3072 | `n_inner`（null → `4 × n_embd`） |
| `max_pos` | 1024 | `n_positions` / `n_ctx` |
| 参数 | **124,439,808** | 由真实 `state_dict` 数出 |

`batch_size` 和 `seq_len` 不在 `config.json` 里：它们是运行时输入。`config.json` 只给 `seq_len` 的上限 `n_positions = 1024`，不给具体这次输入几个 token。

## 3. 结构

### 3.1 怎么读图

和 Transformer 那篇的图例一致：

| 记号 | 含义 |
| --- | --- |
| 箭头 | 数据流方向，和代码同向（从上往下） |
| 方框第二行 | 该步的关键张量形状 |
| 虚线 | 残差，从子层输入绕过子层 |
| `(+)` 圆圈 | 逐元素相加 |
| `× N` | 重复 N 次，每层权重独立，形状不变 |

### 3.2 GPT 单栈

同一套零件，删掉 encoder 和 cross-attention 之后的样子：

```mermaid
flowchart TB
    TK(["token ids<br/>(batch_size, seq_len)"]) --> E["Embedding  vocab_size×dim_model<br/>(batch_size, seq_len, dim_model)"]
    E --> P["+ PosEnc（可学习 wpe）<br/>(batch_size, seq_len, dim_model)"]

    subgraph BLK["Block × N"]
        direction TB
        IN(["x (batch_size, seq_len, dim_model)"]) --> L1["LayerNorm"]
        L1 --> A["Masked Self-Attention<br/>scores (batch_size, num_heads, seq_len, seq_len)"]
        A --> R1(("+"))
        IN -.->|residual| R1
        R1 --> L2["LayerNorm"]
        L2 --> F["Feed-Forward<br/>fc1 (batch_size, seq_len, dim_ff)<br/>fc2 (batch_size, seq_len, dim_model)"]
        F --> R2(("+"))
        R1 -.->|residual| R2
    end

    P --> IN
    R2 --> LN["Final LayerNorm<br/>(batch_size, seq_len, dim_model)"]
    LN --> HD["Linear dim_model → vocab_size<br/>（与 Embedding 共享权重）"]
    HD --> LG["logits<br/>(batch_size, seq_len, vocab_size)"]
    LG --> LAST["只取最后一个位置<br/>(batch_size, 1, vocab_size)"] --> SM["softmax<br/>→ 下一个 token"]
```

和 Transformer 的差异：

| | Transformer 2017 | GPT-2 |
| --- | --- | --- |
| 结构 | Enc×N + Dec×N | Dec×N |
| cross-attention | 有 | 无 |
| 序列 | `src_len` 和 `tgt_len` 两条 | 只有一条 `seq_len` |
| 位置编码 | sin/cos，固定 | `wpe`，可学习 |
| LayerNorm | post-LN | **pre-LN**：`x + 子层(LN(x))` |
| FFN 激活 | ReLU | GELU（tanh 近似） |
| Q/K/V | 三个独立 Linear | 一个 `c_attn`（768→2304），再切 |

### 3.3 c_attn / c_proj 和 wte / wpe

这三个名字来自 GPT-2 的实现，不是论文里的数学符号。先把全称写出来：

- `c_attn` = **combined attention**：把 Q/K/V 三个投影合在一个 Linear 里
- `c_proj` = **Conv1D projection**：GPT-2 对 Linear 的叫法；attention 里的 `c_proj` 就是 `out_proj`，MLP 里的 `c_proj` 就是 `fc2`
- `wte` = **weight token embedding**：token 的嵌入权重矩阵
- `wpe` = **weight position embedding**：位置的嵌入权重矩阵

图里出现过，这里拆开看。

#### c_attn：一次算出 Q/K/V

Transformer 写三个独立的 Linear：

```python
q = q_proj(x)   # 768 → 768
k = k_proj(x)
v = v_proj(x)
```

GPT-2 合成一个：

```python
qkv = c_attn(x)          # (batch_size, seq_len, 2304)
q, k, v = qkv.split(768, dim=-1)
```

`c_attn` 的权重是 `(2304, 768)`、bias 是 `(2304,)`。`2304 = 3 × 768`，按最后一维切成三段，分别就是 Q、K、V。下面用 `B = batch_size`、`T = seq_len`：

| 切片 | 形状 | 是什么 |
| --- | --- | --- |
| `[..., :768]` | `(B, T, 768)` | Q |
| `[..., 768:1536]` | `(B, T, 768)` | K |
| `[..., 1536:]` | `(B, T, 768)` | V |

然后再拆头：`(B, T, 768) → (B, 12, T, 64)`。

数学上和三套独立 Linear 完全等价；合成一个只是少两次矩阵乘、实现更紧凑。Hugging Face 的 checkpoint 把它存成 Conv1D，权重形状是 `(768, 2304)`（in, out）；本文件的 `nn.Linear` 要 `(2304, 768)`（out, in），所以加载时转置。

#### c_proj：两个 c_proj，别混

`c_proj` = **Conv1D projection**。GPT-2 把线性层实现成 1×1 卷积，所以名字里带 `c`；它和 `nn.Linear` 做的是同一件事。

一个 block 里有两个 `c_proj`：

| 位置 | 对应 Transformer 的名字 | 形状 |
| --- | --- | --- |
| attention 输出 | `out_proj` / `Wo` | `(B, T, 768) @ (768, 768)` |
| MLP 输出 | `fc2` / `W2` | `(B, T, 3072) @ (3072, 768)` |

log 里两行都叫 `c_proj`，看上下文：`拼头` 之后的是 attention 的，`gelu` 之后的是 MLP 的。现在 trace 已经分别写成 `c_proj (attn)` 和 `c_proj (mlp)`。

#### wte / wpe：两张查找表

GPT-2 的第一行不是矩阵乘，是两次查表：

```python
x = wte[token_ids] + wpe[positions]
```

`wte` = **w**eight **t**oken **e**mbedding，`wpe` = **w**eight **p**osition **e**mbedding。
`w` 表示这是一个权重矩阵，`te` / `pe` 分别是 token embedding / position embedding。

“查表”就是按 id 取矩阵的一行，不做乘法：`wte[token_ids]` 的输出形状是 `(batch_size, seq_len, 768)`。`50257` 和 `768` 来自 `config.json`；`seq_len` 是这次输入的 token 数，运行时才知道，`config.json` 只给上限 `n_positions = 1024`。

| 名字 | 全称 | 形状 | 输入 | 输出 | 是否训练 |
| --- | --- | --- | --- | --- | --- |
| `wte` | weight token embedding | 50257 × 768 | token id | 768 维向量 | 是 |
| `wpe` | weight position embedding | 1024 × 768 | 位置 `0…T-1` | 768 维向量 | 是 |

- `wte`：词表里每个 token 一行。`"A long time ago"` 切成 `[32, 890, 640, 2084]`，查出来就是 `(1, 4, 768)`。
- `wpe`：每个位置一行。位置 `0, 1, 2, 3` 查出来是 `(4, 768)`，广播到 batch 维后相加，得到 `(1, 4, 768)`。
- `wte` 和输出头 `lm_head` 共享同一个矩阵（weight tying）；`wpe` 不共享。
- `wpe` 是可学习的，这点和 Transformer 2017 的固定 sin/cos 不同；最大 1024 个位置。

所以 embedding 这一步的完整形状：

```
token_ids        (1, 4)
wte[token_ids]   (1, 4, 768)
wpe[0..3]        (4, 768)
相加              (1, 4, 768)
```

> 这套命名来自 OpenAI 的 GPT-2 代码；Hugging Face 里叫 `transformer.wte` / `transformer.wpe`。

### 3.4 causal mask

GPT-2 的 self-attention 是因果的：位置 `t` 只能看 `≤ t`。

prefill 时 `q_len = kv_len = seq_len`，mask 是下三角（行是 q、列是 k：左下含对角线可见，右上被屏蔽）。以 `seq_len = 4` 为例：

|  | k0 | k1 | k2 | k3 | softmax 后 |
| --- | --- | --- | --- | --- | --- |
| **q0** | ok | -inf | -inf | -inf | `[1, 0, 0, 0]` |
| **q1** | ok | ok | -inf | -inf | `[.5, .5, 0, 0]` |
| **q2** | ok | ok | ok | -inf | `[.33, .33, .33, 0]` |
| **q3** | ok | ok | ok | ok | `[.25, .25, .25, .25]` |

decode 时 `q_len = 1`，新的 query 可以看全部历史 K/V，所以 `scores` 是 `(batch_size, num_heads, 1, kv_len)`，不需要再遮未来——未来还没进 cache。

### 3.5 prefill 和 decode

`runs/gpt2_shapes.log` 是用真实 GPT-2 权重跑出来的，prompt 是 `"A long time ago"`（4 个 token）。

**prefill**：4 个 token 一次算完，`q_len = kv_len = 4`：

```
tok_emb        (1, 4, 768)
pos_emb        (4, 768)        ← 位置 0…3，表共 (1024, 768)
c_attn         (1, 4, 2304)
q/k/v          (1, 4, 768)      ← 切成三份
拆头 q          (1, 12, 4, 64)
scores         (1, 12, 4, 4)   ← 唯一的 T×T
softmax        (1, 12, 4, 4)
attn @ V       (1, 12, 4, 64)
拼头            (1, 4, 768)
c_proj (attn)  (1, 4, 768)
c_fc           (1, 4, 3072)
c_proj (mlp)   (1, 4, 768)
lm_head        (1, 4, 50257)   ← 只取最后一个位置
```

**decode**：每次 1 个 token，`q_len = 1`，`kv_len` 逐步增长：

```
step 1: 输入 (1,1)  → K/V cache (1,12,5,64) → scores (1,12,1,5)
step 2: 输入 (1,1)  → K/V cache (1,12,6,64) → scores (1,12,1,6)
```

每层缓存两份：K `(1, 12, kv_len, 64)` 和 V `(1, 12, kv_len, 64)`。12 层一共 24 个张量。
cache 只存历史，不存参数；`kv_len` 从 prompt 长度开始，每生成一个 token 加 1。

## 4. 参数

### 4.1 一个 block

一个 block 的参数量：

| 操作 | 完整式子 | 约算 | 参数量 |
| --- | --- | --- | --- |
| `c_attn` | 768×2304 + 2304 | 1,728K + 2.25K | ≈1.73M |
| `c_proj`（attention） | 768×768 + 768 | 576K + 0.75K | ≈0.58M |
| `c_fc` | 768×3072 + 3072 | 2,304K + 3K | ≈2.31M |
| `c_proj`（MLP） | 3072×768 + 768 | 2,304K + 0.75K | ≈2.30M |
| `ln1` + `ln2` | 2 × (768 + 768) | 3K | 3K |
| **合计** | | 6,912K | **≈6.9M** |

两个乘法怎么约的：

```
768 × 2304 = (3×256) × (9×256) = 27 × (256×256) = 27 × 64K  = 1,728K
768 × 3072 = (3×256) × (3×1024) = 9 × (256×1024) = 9 × 256K = 2,304K
```

### 4.2 整个模型

整个模型：

| 部分 | 约算 | 参数量 |
| --- | --- | --- |
| `wte` | 50257 × 768 | 37.7M |
| `wpe` | 1024 × 768 = 768K | 0.75M |
| 12 个 block | 6,912K × 12 = 82,944K | 82.9M |
| `ln_f` | 2 × 768 | 1.5K |
| `lm_head` | 和 `wte` 同一个矩阵 | 0 |
| **合计** | | **≈121.4M**（精确 124,439,808） |

> 上表的约算混用了 1024 和 1000 的进位，只用于心算；精确值以 `state_dict` 的 **124,439,808** 为准。

### 4.3 和论文的 117M 对得上吗

论文 Table 2 给了 GPT-2 四档参数：**117M / 345M / 762M / 1542M**（`×10⁶`，四舍五入），只给总数，没有逐模块拆解。
我们按 `config.json` + 真实 `state_dict` 数出来是 **124,439,808**（约 124.4M）。

和 117M 差约 6%。

我们这一路本来就在约算：K/M 的进位、bias 的取舍、词表/版本口径都可能差一点。
这些叠加起来，6% 的偏差很正常，说明不了谁对谁错。

所以 124.4M 只是**量级验证**——对上"一亿级"就说明架构和权重映射没跑偏。
引用时用论文的 117M。

## 5. 计算量（FLOPs）

### 5.1 规则

```
矩阵乘 (m, k) @ (k, n) → m × k × n 次乘加，× 2 = FLOPs
```

`m` 是 token 数：prefill 时 `m = seq_len`，decode 时 `m = 1`。

### 5.2 一个 Linear 展开

`c_attn`（`768 → 2304`），prefill `seq_len = 4`：

```
4 × 768 × 2304
= 4 × 1,728K          ← 768×2304 = 1,728K
= 6,912K 次乘加
= 6.9M

× 2 = 13.8M FLOPs
```

### 5.3 一个 block（prefill `seq_len = 4`）

| 操作 | m | k | n | 乘加 | FLOPs | 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| `c_attn` | 4 | 768 | 2304 | 4 × 1,728K = 6,912K | 13.8M | 25% |
| `scores`（12 头） | 4 | 64 | 4 | 12 × 4×64×4 = 12,288 | 24.6K | 0.04% |
| `attn @ V`（12 头） | 4 | 4 | 64 | 12 × 4×4×64 = 12,288 | 24.6K | 0.04% |
| `c_proj`（attn） | 4 | 768 | 768 | 4 × 576K = 2,304K | 4.6M | 8.3% |
| `c_fc` | 4 | 768 | 3072 | 4 × 2,304K = 9,216K | 18.4M | 33.3% |
| `c_proj`（MLP） | 4 | 3072 | 768 | 4 × 2,304K = 9,216K | 18.4M | 33.3% |
| **一层** | | | | **27,648K** | **≈55.3M** | 100% |

**FFN 占 2/3**（`c_fc` + `c_proj` = 36.8M），注意力投影占 1/3。

### 5.4 12 层 + 输出头

```
12 层   = 55.3M × 12              ≈ 0.66G
输出头  = 2 × 4 × 768 × 50257     ≈ 0.30G
                                  ─────────
                           合计   ≈ 0.97G（prefill 4 个 token）
```

### 5.5 decode：每生成 1 个 token

`m = 1`，其余不变：

| 操作 | 乘加 | FLOPs |
| --- | --- | --- |
| `c_attn` | 1,728K | 3.46M |
| `c_proj`（attn） | 576K | 1.15M |
| `c_fc` | 2,304K | 4.61M |
| `c_proj`（MLP） | 2,304K | 4.61M |
| **一层** | 6,912K | **≈13.8M** |

```
12 层   = 13.8M × 12           ≈ 166M
输出头  = 2 × 768 × 50257      ≈ 77M
                              ───────
                       合计   ≈ 243M / token
```

### 5.6 长上下文：注意力的平方项

`seq_len = T` 时，`scores` 和 `attn @ V` 两项：

```
4 × T × 768 × 12 层 = 36,864 × T
```

- `T = 4`：147K，可以忽略
- `T = 1024`：37.7M/ token，占 decode 的 13%
- `T = 8192`：302M，超过权重那 243M——这时注意力反超

权重那 243M 不随 `T` 变，只有这一项涨。

## 6. 访存量（Bytes）

### 6.1 规则

```
Bytes = (读入元素数 + 权重元素数 + 写出元素数) × 每元素字节
```

权重常驻内存，但每跑一次 forward 都要重新搬一遍。GPT-2 的 decode 每次只处理 1 个 token，所以读入/写出很小，**权重是大头**；prefill 时读入/写出乘 `T`，但相对权重仍是零头。

下面先用 FP32（4 字节）算，最后再换 FP16。

### 6.2 一个投影怎么算

以 `c_attn` 为例，decode 时 `q_len = 1`：

```
读入     1 × 768 = 768 ≈ 0.75K
权重     768 × 2304 = 1,728K
bias     2,304 ≈ 2.25K
写出     1 × 2304 = 2,304 ≈ 2.25K
         ─────────────────────
合计     1,728K + 0.75K + 2.25K + 2.25K ≈ 1,733K
× 4 字节 = 6,932K ≈ 6.9M
```

prefill `T = 4` 时，读入/写出乘 4，但 `1,728K` 的权重不变。

### 6.3 一层：逐算子

| 操作 | 权重 | bias | 读入 + 写出 | 字节 |
| --- | --- | --- | --- | --- |
| `c_attn` | 1,728K | 2.25K | 0.75K + 2.25K | 6.9M |
| `c_proj`（attn） | 576K | 0.75K | 0.75K + 0.75K | 2.3M |
| `c_fc` | 2,304K | 3K | 0.75K + 3K | 9.2M |
| `c_proj`（MLP） | 2,304K | 0.75K | 3K + 0.75K | 9.2M |
| `ln1` + `ln2` | 3K | — | 2 × (0.75K + 0.75K) | 0.02M |
| **一层合计** | 6,912K | 6.75K | ≈9K | **≈27.6M** |

注意 `c_proj` 有两个：`拼头` 之后的是 attention 的 `out_proj`，`gelu` 之后的是 MLP 的 `fc2`。

### 6.4 12 层 + 输出头

```
12 层   = 27.6M × 12   ≈ 331M
输出头  = (37.7M + 0.75K + 49K) × 4 ≈ 151M
                       ───────
                合计   ≈ 482M / token（FP32）
```

482M 是逐算子约算出来的；和 `124.4M × 4 ≈ 498M` 的差来自把 `wpe`（查表）和零头舍掉，以及 head / 每层的四舍五入。

### 6.5 prefill 和 decode

- **prefill（T=4）**：四个 token 一次算完，权重读一遍；读入/写出乘 4，总量约 `483M`（FP32）。FLOPs 约 `0.97G`，算术强度 `≈2.0`。
- **decode（T=1）**：每次 1 个 token，权重要重读；总量约 `482M`（FP32）。FLOPs 约 `243M`，算术强度 `≈0.50`。

decode 比 prefill 慢，不是因为 FLOPs 更多，而是因为每生成一个 token 都要把同一批权重再搬一遍。

### 6.6 KV cache 也算进 decode

KV cache 不是参数，是运行时缓存。每层 K、V 各 `(T, 768)`，FP16：

```
12 层 × 2 × T × 768 × 2 B = 36,864 × T 字节
T = 1024 → 37,748,736 B ≈ 38M
```

decode 到第 `T` 个 token 时，除了 249M 权重，还要读 38M 的 KV cache：

```
FP16 decode（T=1024）≈ 249M + 38M = 287M / token
```

`T` 小的时候可以忽略；`T=1024` 时 KV cache 约占 13%。

### 6.7 算术强度

| 阶段 | 访存 | FLOPs | 算术强度 |
| --- | --- | --- | --- |
| prefill T=4 | 483M（FP32） | 0.97G | ≈2.0 |
| decode T=1 | 482M（FP32） | 243M | ≈0.50 |
| decode T=1 | 249M（FP16） | 243M | ≈0.98 |
| decode T=1024 | 287M（FP16，含 KV） | 285M | ≈0.99 |

GPU 的 `算力/带宽` 比通常几十到几百，1.0 远低于它——**decode 是带宽瓶颈**。
每生成一个 token 都要搬 249M 权重，算力用不满。

## 7. GPT 这条线

### 7.1 论文这条线

| 年份 | 论文 | 架构上有新东西吗 | 真正的贡献 |
| --- | --- | --- | --- |
| 2018 | GPT-1《Improving Language Understanding by Generative Pre-Training》 | 没有，就是 Transformer decoder | "先预训练、再微调"这个范式 |
| 2019 | GPT-2《Language Models are Unsupervised Multitask Learners》 | 没有，放大 + 换 tokenizer | zero-shot，不微调也能做多任务 |
| 2020 | GPT-3《Language Models are Few-Shot Learners》 | 没有，放大到 175B | in-context learning |
| 2022 | InstructGPT《Training language models to follow instructions with human feedback》 | 没有 | RLHF，ChatGPT 的直接前身 |
| 2023 | GPT-4《GPT-4 Technical Report》 | **不公开** | 只有评测，参数 / 数据 / 架构全部保密 |
| 2024+ | GPT-4o / 4.1 / 5 | **不公开** | 只有 blog 和 system card |

两句话结论：

1. **对 shape 主线没有新增算子。** GPT-1/2 是同一套 decoder block，只差 vocab、context 和规模；
   GPT-3 的 alternating dense/sparse attention 改的是 mask pattern，主 shape 表仍是
   `(batch_size, num_heads, T, T)`。三篇的增量在数据、训练和规模，不在新算子。
2. **不是"后来不发论文了"，是不再公开技术细节。** GPT-4 起仍有 technical report，
   但里面没有参数量、没有数据规模、没有架构图，只剩评测和安全性说明。

所以不需要 GPT-1/2/3 各写一遍：会 GPT-2，换 config 就覆盖 GPT-1/2 的各尺寸；
GPT-3 没有开放权重，读论文即可。真正该动手的是下一档 HY3——RoPE、GQA、SwiGLU、
RMSNorm、MoE 才是新零件。

### 7.2 为什么不需要 GPT-1/2/3 各写一遍

把三者的配置摆在一起：

| | GPT-1 | GPT-2 small | GPT-3 |
| --- | --- | --- | --- |
| 参数 | 117M | 124M | 175B |
| 层数 | 12 | 12 | 96 |
| `dim_model` | 768 | 768 | 12288 |
| `num_heads` | 12 | 12 | 96 |
| `dim_ff` | 3072 | 3072 | 49152 |
| 上下文 | 512 | 1024 | 2048 |
| 词表 | 40478 | 50257 | 50257 |
| 权重 | 有 | 有 | 无 |

前三列的变化只是数字变大，算子没变：

- **GPT-1**：和 GPT-2 small 同构，换 `vocab_size` / `max_pos` 就能用同一份代码；没有新 shape。
- **GPT-2 medium/large/xl**：只改层数、宽度、头数；shape 表不变。
- **GPT-3**：没有开放权重，跑不了；它的 alternating dense/sparse attention 改的是 mask pattern，
  `scores` 的主 shape 仍是 `(batch_size, num_heads, T, T)`，稀疏层只是不把整张矩阵都算/存。
  读论文看训练、数据和规模即可，不需要写 175B 模型。

真正会改变 shape 和计算量的是下一档 HY3：RoPE、GQA、SwiGLU、RMSNorm、MoE。
如果以后想做 GPT-3 的 sparse attention，单独做一节 mask pattern 实验就行，不用建 175B。

## 8. 跑法

需要 `torch`、`transformers`、`tiktoken`。本机把 `transformers` 装在独立 venv 里：

```bash
python3 -m venv --system-site-packages ~/.venvs/llm-gpt
~/.venvs/llm-gpt/bin/pip install transformers
```

输出会打印每一步的真实 shape；本次运行用 `tee` 存到 `runs/gpt2_shapes.log`：

```bash
~/.venvs/llm-gpt/bin/python llm/gpt/model.py | tee llm/gpt/runs/gpt2_shapes.log
```

改 prompt 和生成长度：

```bash
~/.venvs/llm-gpt/bin/python llm/gpt/model.py --prompt "Hello, I am" --new 5
```

`model.py` 只做推理和 shape trace；训练不在这个文件里——GPT-2 的 prefill shape 和训练 forward 相同，训练多出来的是 loss 和反向。
