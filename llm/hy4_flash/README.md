# HY4-Flash

腾讯 `tencent/Hy4-flash` 的纸面账本。**权重没下载**，只读
[`config.json`](./config.json)，用
[`accounting.py`](./accounting.py) 复算参数量、FLOPs、KV cache 和访存。
官方代码已经找到：HuggingFace transformers 的
[PR #50903](https://github.com/huggingface/transformers/pull/50903)，模型卡见
[tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash)。

一句话：**220.25B 总参数，11.27B 激活，1M 上下文；42 层里 41 层是
LatentMoE，attention 是 12 层 DSA + 30 层 SWA，残差是 4 条 iHC 流。**

## 结论先看

| 项 | 值 |
| --- | ---: |
| 总参数 | 220,251,201,958（220.25B） |
| 文本主干（不含 Engram） | 155,483,166,797（155.48B） |
| Engram（表 + 投影） | 63,455,215,104（63.46B） |
| ViT（塔 + patch merger） | 466,629,360（0.467B） |
| DFly2 drafter | 846,190,697（0.846B） |
| 文本激活 / token | 11,270,639,693（11.27B） |
| checkpoint（BF16） | 440.51 GB / 410.26 GiB（运行时 fp32 lm_head 再 +0.74 GB） |
| decode 读权重 | 23.28 GB / token（含 fp32 lm_head） |
| KV / token（全历史上界） | 51,456 B = 50.25 KiB |
| KV / token（HF 默认，SWA 环形） | 16,896 B = 16.50 KiB（另加 16.9 MiB 固定窗口） |
| KV @ 1M（全历史上界） | 53.96 GB / 50.25 GiB |
| KV @ 1M（HF 默认） | 17.73 GB / 16.52 GiB |
| decode 1 token（C=1K） | 23.95 G FLOPs |
| decode 1 token（1M，理想） | 37.34 G FLOPs |
| prefill 1M | 39.15 P FLOPs |
| 对账 | 与 HF safetensors 总数 **差 0** |

> 官方卡片把总参数写成 220B、激活写成 12B；按 config 逐项算是
> **220.25B / 11.27B**。下文所有数字用精确值，四舍五入只在标题里出现。
>
> "全历史上界" 是 42 层都按全量 cache 存；HF `DynamicCache(config=...)`
> 会把 `sliding_attention` 层建成 `DynamicSlidingWindowLayer`，所以默认是
> 右列（12 个 DSA 层全历史 + 30 个 SWA 层 512 token 环形）。两者差 3 倍。

## 和 Transformer / GPT-2 的差别

| 零件 | Transformer / GPT-2 | HY4-Flash |
| --- | --- | --- |
| 层 | 6 / 12 | 42（1 dense + 41 MoE） |
| attention | MHA / GQA | **MLA**：K/V 压成 512+64 维 latent |
| 全局 attention | 全看 | **DSA**：indexer 先选 2048 token，再做 attention |
| 局部 attention | 无 | **SWA**：30 层 512 token 因果窗口 |
| FFN | 4× 宽 SwiGLU | **LatentMoE**：512 选 16，expert 在 1536 维里算 |
| 残差 | 1 条 | **iHC**：4 条流，子层输出按门加到 4 条上 |
| 位置 | 绝对 / RoPE | **MRoPE**：3D 时间/高/宽，两套 theta（1e7 / 1e4） |
| 额外记忆 | 无 | **Engram**：4 层 2/3-gram 哈希查表，63.36B 参数 |
| 多模态 | 无 | SigLIP2-So400m ViT + patch merger |
| 投机解码 | 无 | DFly2：5 层 drafter，复用目标 KV / indexer 选择 |

## config 速查

| 字段 | 值 | 说明 |
| --- | ---: | --- |
| `hidden_size` | 3072 | 主干宽度 |
| `num_hidden_layers` | 42 | 1 dense + 41 MoE |
| `vocab_size` | 120832 | 120832 × 3072 ≈ 371M / 份 |
| `num_attention_heads` | 32 | DSA 层 |
| `swa_num_attention_heads` | 64 | SWA 层，宽一倍 |
| `q_lora_rank` / `kv_lora_rank` | 1024 / 512 | MLA 压缩秩 |
| `qk_nope_head_dim` + `qk_rope_head_dim` | 192 + 64 = 256 | **不是** `head_dim=64` |
| `v_head_dim` | 256 | |
| `intermediate_size` | 13824 | 只在 layer 0 的 dense FFN |
| `n_routed_experts` / `num_experts_per_tok` | 512 / 16 | top-16 |
| `n_shared_experts` | 1 | |
| `routed_expert_hidden_size` | 1536 | LatentMoE 的 latent 宽度 |
| `moe_intermediate_size` | 1536 | 每个 expert 的中间宽度 |
| `sliding_window` | 512 | 511 前文 + 自己 |
| `index_topk` | 2048 | DSA 最多选 2048 token |
| `compressed_indexer_compress_ratio` | 4 | 4 个 token 压成 1 组 |
| `index_n_heads` / `index_head_dim` | 16 / 128 | indexer |
| `hc_mult` | 4 | iHC 的 4 条 residual stream |
| `engram_layer_ids` | [2,12,22,32] | **1-based**；0-based 是 1/11/21/31 |
| `engram_max_ngram_size` | 3 | 2-gram + 3-gram |
| `engram_n_head_per_ngram` / `engram_n_embed_per_ngram` | 8 / 768 | 8 头 × 96 维 |
| `max_position_embeddings` | 1048576 | 1M |
| vision `hidden_size` / layers / heads | 1152 / 27 / 16 | SigLIP2-So400m 形状 |
| vision `patch_size` / `out_hidden_size` | 16 / 3072 | |
| `dfly_num_hidden_layers` | 5 | drafter；参数 config 没给全 |

完整推导、逐层 shape、每个矩阵的形状见 [`architecture.md`](./architecture.md)。

## 三张账

### 参数

```
155.48B  文本主干（embedding + lm_head + 42 层 + iHC）
 63.46B  Engram（63.36B 表 + 0.095B 投影）
  0.467B ViT（0.431B 塔 + 0.035B merger）
  0.846B DFly2
────────
220.25B  总计
```

文本主干里最大的是 MoE：41 层 × 3.649B = **149.61B**，其中 routed experts
148.58B。每 token 只激活 top-16，所以动态部分是 11.27B，静态部分仍是
155.48B + 63.36B 表。

### 计算

| `C` | decode / token（理想） | decode / token（HF eager 池化） |
| ---: | ---: | ---: |
| 1K | 23.95 G | 23.95 G |
| 8K | 24.45 G | 30.90 G |
| 128K | 25.98 G | 129.06 G |
| 1M | 37.34 G | 861.97 G |

短上下文几乎就是 `2 × 11.27B = 22.54G` 的线性层；长上下文多出来的是
DSA indexer 的 `O(C)` 选择。HF eager 每个 decode step 重算所有压缩组的
池化，1M 时约 824.6G/token；生产 kernel 会缓存 pooled keys，所以主表用
"理想" 列，eager 列单独注明。

prefill 的 `T×T` 项按完整矩阵乘算：1M 时 39.15P FLOPs，其中 indexer
13.62P（causal 下界 6.81P）。4K 时 98.88T，平均 24.14G/token。

### 访存

```
checkpoint 常驻          440.51 GB
Engram 表                126.72 GB   ← 只查行，不整张读
decode 读权重             23.28 GB / token
prefill 读权重（4K）      312.86 GB
KV / token（全历史上界）  50.25 KiB
KV / token（HF 默认）     16.50 KiB
KV @ 1M（全历史上界）     50.25 GiB
KV @ 1M（HF 默认）        16.52 GiB
```

decode 的算术强度：短上下文约 1.03 FLOP/B，1M 约 0.91 FLOP/B（HF 默认
KV，理想）或 0.48（全历史上界）；含 eager 池化则是 21.0 / 11.16。都远低于
B200 的 281 FLOP/B —— **decode 始终是访存瓶颈**。prefill 1M 约 116,000
FLOP/B（HF 默认 KV），是 compute-bound。

## 跑法

```bash
# 默认：prefill 4096，decode 上下文 131072，8×H20
python llm/hy4_flash/accounting.py

# 1M 上下文，8×B200
python llm/hy4_flash/accounting.py \
  --prefill 1048576 --context 1048576 --hardware b200x8
```

原始输出：

- [`runs/accounting.log`](./runs/accounting.log)：4K prefill / 128K decode
- [`runs/accounting_1m.log`](./runs/accounting_1m.log)：1M prefill / 1M decode

脚本不依赖 torch，只读 `config.json`；`--config` 可以换别的 checkpoint。

## 官方代码在哪

| 内容 | 链接 |
| --- | --- |
| 模型卡 / config | https://huggingface.co/tencent/Hy4-flash |
| 官方 GitHub（权重说明 / DFly2 / finetune） | https://github.com/Tencent-Hunyuan/Hy4-flash |
| transformers PR（modeling 代码） | https://github.com/huggingface/transformers/pull/50903 |
| `modeling_hy_v4_flash.py` | [commit b865850](https://github.com/huggingface/transformers/blob/b865850d05326cf20e306b31f14b0027de802da3/src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py) |
| `configuration_hy_v4_flash.py` | [同 commit](https://github.com/huggingface/transformers/blob/b865850d05326cf20e306b31f14b0027de802da3/src/transformers/models/hy_v4_flash/configuration_hy_v4_flash.py) |
| HF 文档 | [hy_v4_flash.md](https://github.com/huggingface/transformers/blob/b865850d05326cf20e306b31f14b0027de802da3/docs/source/en/model_doc/hy_v4_flash.md) |
| vLLM recipe | https://recipes.vllm.ai/tencent/Hy4-flash |
| SGLang cookbook | https://docs.sglang.io/cookbook/autoregressive/Tencent/Hy4-flash |
| DFly2 / AngelSpec | https://github.com/Tencent/AngelSpec |

模型卡说 "Support for Hy4 flash will be merged into transformers main
soon"，我们拉取时代码在 PR #50903；以 `transformers` main 的最终文件为准。

## 本机能做 L1 / L3a，做不了 L2 / L3b

```
Apple M4 / 16 GB / arm64 / 无 CUDA / MPS 可用
```

- checkpoint BF16 440.51GB，FP8 约 220.25GB，4-bit 约 110.13GB —— 16GB
  统一内存**装不下**，连 4-bit 都不行。L2（真实权重打印 shape）和 L3b
  （profiler 实测）都需要 8×H200/B200 级别的机器。
- L1（架构 + shape）和 L3a（纸面 FLOPs / Bytes）已经做完；总参数与 HF
  safetensors 对账到 0，文本主干 / Engram / 激活参数都在 0.3% 以内。
- 如果只想验证 shape 逻辑，可以自己写一个 hidden=64、2 层的 tiny config
  随机初始化；那验证的是 `architecture.md` 的形状，不是这个 checkpoint。

## 对应 `llm/README.md` 的阶梯

| 层 | 状态 | 产出 |
| --- | --- | --- |
| L1 读图 | ✅ | `architecture.md`：42 层 + MLA + DSA + SWA + LatentMoE + Engram + iHC |
| L2 真机 | ❌ | 权重 440.51GB，本机装不下；需要 8×H200/B200 |
| L3a 理论 | ✅ | `accounting.py` + `runs/accounting.log` |
| L3b 实测 | ❌ | 等 GPU；vLLM `FLASHMLA_SPARSE` / SGLang 可直接跑 |
