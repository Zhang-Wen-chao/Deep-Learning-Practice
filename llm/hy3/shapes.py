#!/usr/bin/env python3
"""HY3 的 shape 流水：只按 config.json 推形状，不加载权重、不分配大张量。

这不是"跑模型"，是把每个算子的 in/out shape 走一遍。和
transformer/runs/model_py_flow.log、gpt/runs/gpt2_shapes.log、bert/runs/bert_flow.log
同样的读法：左边算子，中间 shape，右边说明。

用法：
    python llm/hy3/shapes.py
"""

import os
import unicodedata

import accounting

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")

cfg = accounting.load_config(CONFIG)
d = accounting.dims(cfg)
p = accounting.param_ledger(d)

H = d["H"]
HEADS = d["heads"]
KV_HEADS = d["kv_heads"]
HEAD_DIM = d["head_dim"]
HQ = d["q_dim"]
HKV = d["kv_dim"]
E = d["E"]
TOPK = d["top_k"]
IM = d["I_moe"]
I = d["I"]
V = d["V"]
L = d["L"]
BATCH_SIZE = 1
SEQ_LEN = 4096
KV_LEN = 4097  # prefill 4096 + decode 新 token 1

W = 26  # shape 列宽


def sep():
    print("=" * 78)


def dwidth(text):
    """终端显示宽度：中文/全角算 2。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def line(name, shape, note=""):
    pad = " " * max(1, 22 - dwidth(name))
    print(f"  {name}{pad}{str(shape):<{W}}{note}")


def h3(title):
    print(f"── {title} " + "─" * max(0, 48 - dwidth(title)))


def attention(b, q_len, kv_len, note=""):
    line("input RMSNorm", (b, q_len, H), "每个 token 独立，形状不变")
    line("q_proj", (b, q_len, HQ), f"{H} → {HEADS}×{HEAD_DIM}")
    line("k_proj", (b, q_len, HKV), f"{H} → {KV_HEADS}×{HEAD_DIM}")
    line("v_proj", (b, q_len, HKV), f"{H} → {KV_HEADS}×{HEAD_DIM}")
    line("拆头 q", (b, HEADS, q_len, HEAD_DIM),
         f"view(1,{q_len},{HEADS},{HEAD_DIM}).transpose(1,2)")
    line("拆头 k", (b, KV_HEADS, q_len, HEAD_DIM),
         f"view(1,{q_len},{KV_HEADS},{HEAD_DIM}).transpose(1,2)")
    line("拆头 v", (b, KV_HEADS, q_len, HEAD_DIM), "同 k")
    line("q_norm", (b, HEADS, q_len, HEAD_DIM), f"每头 {HEAD_DIM} 维 RMSNorm")
    line("k_norm", (b, KV_HEADS, q_len, HEAD_DIM), f"每头 {HEAD_DIM} 维 RMSNorm")
    line("RoPE q/k", f"{q_len} 个位置", f"theta = {cfg['rope_parameters']['rope_theta']:.0f}")
    line("repeat_kv", (b, HEADS, kv_len, HEAD_DIM), f"每个 KV 头复制 {d['group']} 份")
    line("scores", (b, HEADS, q_len, kv_len),
         f"q @ kᵀ / √{HEAD_DIM}" + ("，不 materialize" if q_len * kv_len > 1_000_000 else ""))
    line("+ causal mask", (b, HEADS, q_len, kv_len), "右上角遮住未来")
    line("softmax", (b, HEADS, q_len, kv_len), "dim=-1，每个 query 对全部 key 归一化")
    line("attn @ V", (b, HEADS, q_len, HEAD_DIM), "加权平均")
    line("拼头", (b, q_len, HQ), f"{HEADS} 个头拼回 {HQ}")
    line("o_proj", (b, q_len, H), f"{HQ} → {H}")
    line("+ 残差", (b, q_len, H), "attention 输出 + 子层输入")
    if note:
        print(f"  {'':<22}{'':<{W}}{note}")


def dense_ffn(b, q_len):
    h3("dense FFN（layer 0）")
    line("post RMSNorm", (b, q_len, H), "第二个 RMSNorm")
    line("gate_proj", (b, q_len, I), f"{H} → {I}")
    line("up_proj", (b, q_len, I), f"{H} → {I}")
    line("silu(gate) * up", (b, q_len, I), "逐元素")
    line("down_proj", (b, q_len, H), f"{I} → {H}")
    line("+ 残差", (b, q_len, H), "FFN 输出 + 子层输入")


def moe(b, q_len, n_e, expert_note=""):
    h3("MoE")
    line("post RMSNorm", (b, q_len, H), "第二个 RMSNorm")
    line("x flatten", (b * q_len, H), "按 token 路由")
    line("router", (b * q_len, E), f"x @ W_rᵀ，fp32；{E} 个专家")
    line("sigmoid + bias", (b * q_len, E), "bias 只影响 top-k 选择")
    line("top-k", (b * q_len, TOPK), f"每 token 选 {TOPK} 个")
    line("路由权重", (b * q_len, TOPK), "归一化后 × 2.826")
    line("每个专家 gate_up", (n_e, 2 * IM), expert_note)
    line("silu(gate) * up", (n_e, IM), "逐元素")
    line("每个专家 down", (n_e, H), f"{IM} → {H}")
    line("shared gate/up", (b * q_len, IM), "shared expert，所有 token 都算")
    line("shared down", (b * q_len, H), f"{IM} → {H}")
    line("合并", (b, q_len, H), "routed + shared，reshape 回 3D")
    line("+ 残差", (b, q_len, H), "MoE 输出 + 子层输入")


def mtp():
    print()
    h3("MTP（第 81 层，draft 1 token）")
    line("token emb", (1, 1, H), "共享主干的 embedding")
    line("上一 token hidden", (1, 1, H), "主模型最后一个 hidden")
    line("enorm", (1, 1, H), "RMSNorm")
    line("hnorm", (1, 1, H), "RMSNorm")
    line("cat", (1, 1, 2 * H), f"{H} + {H}")
    line("eh_proj", (1, 1, H), f"{2 * H} → {H}")
    line("decoder", (1, 1, H), "1 层 MoE decoder，内部同上")
    line("shared_head.norm", (1, 1, H), "RMSNorm")
    line("lm_head", (1, 1, V), f"{H} → {V}（和主干共享）")


def main():
    sep()
    print("HY3 shape 流水（shape-only：不加载权重，只按 config.json 推形状）")
    print(f"config: {os.path.relpath(CONFIG)}")
    print(f"  hidden_size={H}  num_hidden_layers={L}  "
          f"num_attention_heads={HEADS}  num_key_value_heads={KV_HEADS}  head_dim={HEAD_DIM}")
    print(f"  num_experts={E}  num_experts_per_tok={TOPK}  "
          f"num_shared_experts={d['shared']}  moe_intermediate_size={IM}")
    print(f"  vocab_size={V}  max_position_embeddings={d['max_pos']}")
    print(f"  参数: {p['main']:,}（主干）/ {p['total']:,}（含 MTP）")
    print(f"  KV cache: {2 * KV_HEADS * HEAD_DIM * d['dtype_bytes']:,} B / 层 / token"
          f"（80 层 = {2 * L * KV_HEADS * HEAD_DIM * d['dtype_bytes']:,} B / token）")
    sep()

    print()
    print("[1] 输入（没有 tokenizer，直接给长度）")
    line("ids", (BATCH_SIZE, SEQ_LEN), f"batch_size = {BATCH_SIZE}, seq_len = {SEQ_LEN}")
    line("tok_emb", (BATCH_SIZE, SEQ_LEN, H), "ids 查表 → (batch_size, seq_len, hidden_size)")

    print()
    print("[2] prefill：block 0（dense）")
    h3("attention")
    attention(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    dense_ffn(BATCH_SIZE, SEQ_LEN)

    print()
    print("[3] prefill：block 1（MoE）")
    h3("attention")
    attention(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    moe(BATCH_SIZE, SEQ_LEN, (BATCH_SIZE * SEQ_LEN * TOPK + E - 1) // E,
        f"平均 n_e ≈ {BATCH_SIZE * SEQ_LEN}×{TOPK}/{E} ≈ {(BATCH_SIZE * SEQ_LEN * TOPK + E - 1) // E}")
    print(f"  {'block 2–79':<22}{str((BATCH_SIZE, SEQ_LEN, H)):<{W}}同 block 1，形状不变")

    print()
    print("[4] prefill：收尾")
    line("final RMSNorm", (BATCH_SIZE, SEQ_LEN, H), "最后一个 RMSNorm")
    line("lm_head", (BATCH_SIZE, SEQ_LEN, V), f"{H} → {V}（不共享）")
    line("logits", (BATCH_SIZE, SEQ_LEN, V), f"{SEQ_LEN} 个位置各 {V} 个 logits")
    print(f"  只取最后一个位置: {(BATCH_SIZE, 1, V)}  →  下一个 token")

    mtp()

    print()
    print(f"[5] decode：1 token，kv_len = {KV_LEN}（4096 past + 1 new）")
    line("ids", (BATCH_SIZE, 1), "q_len = 1")
    line("tok_emb", (BATCH_SIZE, 1, H), "ids 查表")
    h3("block 0 · attention + dense FFN")
    # decode attention: q_len=1, kv_len=KV_LEN
    line("input RMSNorm", (BATCH_SIZE, 1, H), "")
    line("q_proj", (BATCH_SIZE, 1, HQ), f"{H} → {HQ}")
    line("k_proj / v_proj", (BATCH_SIZE, 1, HKV), f"{H} → {HKV}")
    line("拆头 q", (BATCH_SIZE, HEADS, 1, HEAD_DIM),
         f"view(1,1,{HEADS},{HEAD_DIM}).transpose(1,2)")
    line("拆头 k / v", (BATCH_SIZE, KV_HEADS, 1, HEAD_DIM),
         f"view(1,1,{KV_HEADS},{HEAD_DIM}).transpose(1,2)")
    line("q_norm", (BATCH_SIZE, HEADS, 1, HEAD_DIM), "")
    line("k_norm", (BATCH_SIZE, KV_HEADS, 1, HEAD_DIM), "")
    line("K cache", (BATCH_SIZE, KV_HEADS, KV_LEN, HEAD_DIM), f"past {SEQ_LEN} + new 1 → kv_len {KV_LEN}")
    line("V cache", (BATCH_SIZE, KV_HEADS, KV_LEN, HEAD_DIM), "")
    line("repeat_kv", (BATCH_SIZE, HEADS, KV_LEN, HEAD_DIM), f"每个 KV 头复制 {d['group']} 份")
    line("scores", (BATCH_SIZE, HEADS, 1, KV_LEN), "q(1) @ kᵀ(kv_len)")
    line("+ causal mask", (BATCH_SIZE, HEADS, 1, KV_LEN), "没有未来位置可遮")
    line("softmax", (BATCH_SIZE, HEADS, 1, KV_LEN), "dim=-1")
    line("attn @ V", (BATCH_SIZE, HEADS, 1, HEAD_DIM), "")
    line("拼头", (BATCH_SIZE, 1, HQ), "")
    line("o_proj", (BATCH_SIZE, 1, H), "")
    line("+ 残差", (BATCH_SIZE, 1, H), "")
    line("dense FFN gate/up", (BATCH_SIZE, 1, I), "silu(gate) * up → down")
    line("dense FFN down", (BATCH_SIZE, 1, H), "")
    print("  block 1 · MoE（decode）")
    moe(BATCH_SIZE, 1, 1, f"{TOPK} 个被选中的专家各 1 token")
    line("block 2–79", (BATCH_SIZE, 1, H), "同 block 1，形状不变")
    line("final RMSNorm", (BATCH_SIZE, 1, H), "")
    line("lm_head", (BATCH_SIZE, 1, V), f"{H} → {V}")
    print("  每生成 1 个 token：KV 长度 +1，q_len 始终 = 1")

    print()
    print("形状结论：")
    print(f"  prefill 的 scores 是 {(BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN)}，"
          f"{HEADS * SEQ_LEN * SEQ_LEN / 1e9:.2f}B 个数；不能 materialize，必须 FlashAttention。")
    print(f"  decode 的 scores 是 {(BATCH_SIZE, HEADS, 1, KV_LEN)}，q_len 永远是 1，"
          f"kv_len 从 {SEQ_LEN} 开始涨。")
    print(f"  权重形状不随 seq_len 变；变的是激活和 KV cache。")


if __name__ == "__main__":
    main()
