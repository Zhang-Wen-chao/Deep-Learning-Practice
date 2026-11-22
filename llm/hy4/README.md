# HY4（tencent/Hy4）

HY4（腾讯混元 4 正式版）这一档，只做两件事：

1. **L1**：把 770B 文本主干 + ViT + DFly2 的 shape 和结构走通；
2. **L3a**：从 `config.json` 算参数 / FLOPs / 访存，**不下载权重**。

L2 / L3b 需要 1.55TB 权重和 8×B300 / 16×B200 级机器，本机（M4 / 16GB）做不了。
HY4-Flash 是另一档，账本在 [`../hy4_flash/`](../hy4_flash/)，这里不重复。

## 文件

```
llm/hy4/
├── README.md              # 本文件
├── architecture.md        # 结构、shape、参数、FLOPs、KV cache
├── calc.py                # 只读 config 的账本，纯标准库
├── config.json            # tencent/Hy4 的 config（唯一真源）
└── runs/
    └── calc.log           # calc.py 的输出
```

## 30 秒结论

| 项 | 值 |
| --- | --- |
| checkpoint 总参数 | **774.434B** |
| 文本主干 | **769.907B**（官方 770B） |
| ViT | **1.230B**（官方 1.23B） |
| DFly2 | **3.297B**（checkpoint 反推，官方 3.3B） |
| 文本激活 / token | **49.057B**（参与乘加 48.315B） |
| 有图时激活 / token | **50.287B**（text + Vision 1.23B） |
| 层数 / 宽度 | 78 / 6144 |
| 注意力 | 78 层 Gated MLA + DSA（top-2048），qk_norm |
| indexer | 32 头 × 128 维，21 个 full 层，57 个 shared 层复用 |
| MoE | 256 专家 top-8 + 1 共享，专家中间维 2048 |
| 残差 | iHC 4 条流 |
| 位置编码 | mRoPE（3D：时间/高/宽） |
| KV cache / token | **93.0KiB** |
| decode 权重 / token | **89.994GiB**（text） |
| decode FLOPs / token（KV=2K） | **107.454G** |
| decode 算术强度 | **1.11** |

一句话：**存 774B（770B text + 1.23B ViT + 3.3B DFly2），算 49B text；
decode 每 token 搬 90GiB 文本权重，算术强度约 1，天然带宽瓶颈。**

## 怎么跑

```bash
# 默认 config：llm/hy4/config.json（tencent/Hy4）
.venv/bin/python llm/hy4/calc.py

# 换 prefill 长度 / decode KV
.venv/bin/python llm/hy4/calc.py --seq-len 8192 --decode-kv 131072
```

重新生成 `runs/calc.log`：

```bash
.venv/bin/python llm/hy4/calc.py > llm/hy4/runs/calc.log
```

`calc.py` 不依赖 torch / transformers，只读 JSON。参数、FLOPs、Bytes 的公式在
`architecture.md` 第 5–7 节。

## 对账

| 检查 | 结果 |
| --- | --- |
| text backbone vs 官方 770B | 769.907B，差 0.012% |
| checkpoint vs HF safetensors | 774,434,002,142，逐位相等 |
| text active vs 官方 49B | 49.057B |
| ViT vs 官方 1.23B | 1.230B |
| DFly2 vs 官方 3.3B | 3.297B（残差） |
| KV/token vs 社区实测 ~95KB | 95,232B |

两种算法互相验证：

- **公式展开**：按 config 一层一层乘、加；
- **总数对账**：和 HF safetensors header 的总参数比。

## HY4 的代码在哪

| 位置 | 内容 |
| --- | --- |
| `huggingface/transformers` → `src/transformers/models/hy_v4/` | 文本主干参考实现（`HYV4ForCausalLM`） |
| `vllm-project/vllm` → `vllm/models/hy_v4/` | 推理实现：MLA / DSA / iHC / MoE / VL / DFly |
| `sgl-project/sglang` → `srt/models/hunyuan_v4.py` | 另一份推理实现 |
| `Tencent-Hunyuan/Hy4` | 官方仓库：微调、部署、DFly2 说明 |

链接：

- <https://huggingface.co/tencent/Hy4>
- <https://github.com/Tencent-Hunyuan/Hy4>
- <https://github.com/huggingface/transformers/tree/main/src/transformers/models/hy_v4>
- <https://github.com/vllm-project/vllm/tree/main/vllm/models/hy_v4>

## 不做什么

- **不下载权重**：checkpoint 1.55TB，本机装不下，也没必要。
- **不做 L2**：没有权重，没有对应 GPU；本机 `transformers 4.57` 也不带 `hy_v4`。
- **不重复 Flash**：Flash 的 config、参数预算、KV 压缩和代码状态见
  [`../hy4_flash/`](../hy4_flash/)。
- **不把 ViT / DFly2 的估算当逐模块实测**：ViT 用官方 1.23B，DFly2 用 checkpoint
  残差 3.297B；两者的逐层展开还没有公开的 `modeling_hy4_vl.py` 可对。

## 下一步

- [ ] 等 `modeling_hy4_vl.py` / vLLM VL 代码稳定后，把 ViT 和 DFly2 拆到逐层；
- [ ] GPT-2 的 L1 / L2 还欠着，先把 124M 的真机 shape 跑通；
- [ ] 有 8×B300 / 16×B200 再谈 L3b。
