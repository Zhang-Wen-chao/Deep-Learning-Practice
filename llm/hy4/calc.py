#!/usr/bin/env python3
"""HY4（tencent/Hy4，官方正式版）的纸面账本：只读 config.json，不下载权重。

算三样东西：

    参数    有多少个可学习标量
    FLOPs   一次前向要做多少次乘加（×2）
    Bytes   一次前向要搬多少字节

用法：

    python llm/hy4/calc.py
    python llm/hy4/calc.py --seq-len 8192 --decode-kv 131072

HY4-Flash 的账本在 `llm/hy4_flash/accounting.py`，不在这里重复。

口径说明：

- 参数按 config 的字段逐项展开，和 HF safetensors 的总数对账；
- FLOPs 只算矩阵乘 + attention 核心 + DSA indexer 打分，norm / sigmoid /
  SiLU 这类逐元素算子另记，不进主表；
- prefill 的 attention / indexer 按“每个 query 看满 kv_len”算，不折半；
  真实 causal 前向平均少一半，属于同一量级；
- decode 的权重字节只算“这个 token 真正会读到的参数”：MoE 只读 top-k
  专家，但 KV cache 的容量仍按整条序列算，两者不要混。
"""

import argparse
import json
import os
from collections import OrderedDict

K = 1024
M = K * K
G = K * M
T = K * G

# HF API 的 safetensors 元数据（只读 header，不下载权重）：
#   https://huggingface.co/api/models/tencent/Hy4
CHECKPOINT_TOTAL = 774_434_002_142  # text backbone + ViT + DFly2
OFFICIAL_BACKBONE = 770e9
OFFICIAL_ACTIVE = 49e9
OFFICIAL_VIT = 1.23e9
OFFICIAL_DFLY = 3.3e9


# ─────────────────────────────────────────────────────────── 格式化
def nfmt(n):
    """1234567890 → 1.235B"""
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3f}{unit}"
    return f"{n:.0f}"


def bfmt(n):
    """字节按 1024 进制。"""
    n = float(n)
    for unit, div in (("TiB", T), ("GiB", G), ("MiB", M), ("KiB", K)):
        if abs(n) >= div:
            return f"{n / div:.3f}{unit}"
    return f"{n:.0f}B"


def row(name, value, note=""):
    print(f"  {name:<34}{str(value):>16}  {note}".rstrip())


def section(title):
    print()
    print(f"[{title}]")


# ─────────────────────────────────────────────────────────── 公共小工具
def n_layers(c):
    return c["num_hidden_layers"]


def dense_flags(c):
    types = c.get("mlp_layer_types")
    if not types:
        types = ["dense"] + ["sparse"] * (n_layers(c) - 1)
    return [t == "dense" for t in types]


def full_indexer_flags(c):
    types = c.get("indexer_types")
    if not types:
        types = ["full"] * n_layers(c)
    return [t == "full" for t in types]


def attn_params(c):
    """MLA + gate + sink 的参数量（不含 indexer）。"""
    h = c["hidden_size"]
    heads = c["num_attention_heads"]
    ql = c["q_lora_rank"]
    kvl = c["kv_lora_rank"]
    qn = c["qk_nope_head_dim"]
    qr = c["qk_rope_head_dim"]
    vh = c["v_head_dim"]
    return (
        h * ql + ql                       # q_a_proj + q_a_layernorm
        + ql * heads * (qn + qr)          # q_b_proj
        + h * (kvl + qr)                  # kv_a_proj_with_mqa
        + kvl                             # kv_a_layernorm
        + kvl * heads * (qn + vh)         # kv_b_proj
        + heads * vh * h                  # o_proj
        + h * heads * vh                  # gate_proj
        + heads                           # sinks
    )


def indexer_params(c):
    h = c["hidden_size"]
    ql = c["q_lora_rank"]
    ih = c["index_n_heads"]
    id_ = c["index_head_dim"]
    return ql * ih * id_ + h * id_ + 2 * id_ + h * ih


def ihc_params(c):
    h = c["hidden_size"]
    hc = c["hc_mult"]
    # 每个子层：fn(2*hc, hc*h) + base(2*hc) + scale(2)
    per = (2 * hc) * (hc * h) + 2 * hc + 2
    return 2 * per  # attn 一次 + ffn 一次


def layer_common_params(c):
    return (
        attn_params(c)
        + 2 * c["hidden_size"]  # input_layernorm + post_attention_layernorm
        + ihc_params(c)
    )


def dense_mlp_params(c):
    h = c["hidden_size"]
    ff = c["intermediate_size"]
    return 3 * h * ff


def moe_params(c):
    """一个 MoE 层的全部参数（不是每个 token 激活的部分）。"""
    h = c["hidden_size"]
    experts = c["n_routed_experts"]
    mi = c["moe_intermediate_size"]
    n_shared = c.get("n_shared_experts", 1)
    return (
        experts * 3 * h * mi       # 每个专家 gate_up(2*mi) + down(mi)
        + n_shared * 3 * h * mi    # 共享专家
        + experts * h              # router
    )


def active_moe_params(c):
    h = c["hidden_size"]
    topk = c["num_experts_per_tok"]
    mi = c["moe_intermediate_size"]
    n_shared = c.get("n_shared_experts", 1)
    return (
        topk * 3 * h * mi
        + n_shared * 3 * h * mi
        + c["n_routed_experts"] * h
    )


# ─────────────────────────────────────────────────────────── 参数
def model_params(c, expected):
    h = c["hidden_size"]
    v = c["vocab_size"]
    hc = c["hc_mult"]
    common = layer_common_params(c)
    dense = dense_flags(c)
    full = full_indexer_flags(c)
    idx = indexer_params(c)
    dense_mlp = dense_mlp_params(c)
    moe = moe_params(c)

    layers = []
    for i in range(n_layers(c)):
        p = common + (dense_mlp if dense[i] else moe) + (idx if full[i] else 0)
        layers.append({
            "i": i,
            "kind": ("dense" if dense[i] else "moe") + ("+full" if full[i] else "+shared"),
            "params": p,
        })

    groups = OrderedDict()
    for x in layers:
        groups.setdefault(x["kind"], []).append(x)

    embed = v * h
    lm_head = v * h
    hyper_head = hc * (hc * h) + hc + 1
    final_norm = h
    backbone = sum(x["params"] for x in layers) + embed + lm_head + hyper_head + final_norm
    checkpoint = expected
    non_text = checkpoint - backbone

    return {
        "common": common,
        "dense_mlp": dense_mlp,
        "moe": moe,
        "indexer": idx,
        "layers": layers,
        "groups": groups,
        "embed": embed,
        "lm_head": lm_head,
        "hyper_head": hyper_head,
        "final_norm": final_norm,
        "backbone": backbone,
        "checkpoint": checkpoint,
        "non_text": non_text,
    }


def active_params(c, p):
    h = c["hidden_size"]
    v = c["vocab_size"]
    dense = dense_flags(c)
    full = full_indexer_flags(c)
    common = p["common"]
    idx = p["indexer"]
    active_moe = active_moe_params(c)
    dense_mlp = p["dense_mlp"]

    total = 0
    for i in range(n_layers(c)):
        total += common + (dense_mlp if dense[i] else active_moe) + (idx if full[i] else 0)
    total += v * h  # embedding lookup（占“激活参数”，但不产生 FLOP）
    total += v * h  # lm_head
    return {
        "active": total,
        "active_compute": total - v * h,
        "active_moe": active_moe,
    }


def kv_cache(c):
    kvl = c["kv_lora_rank"]
    qr = c["qk_rope_head_dim"]
    full = full_indexer_flags(c)
    latent = n_layers(c) * (kvl + qr)
    index = sum(full) * c["index_head_dim"]
    return {
        "latent_elems": latent,
        "index_elems": index,
        "bytes": (latent + index) * 2,
        "latent_per_layer": kvl + qr,
        "n_full": sum(full),
    }


# ─────────────────────────────────────────────────────────── FLOPs / Bytes
def flops(c, active_compute, kv_len):
    heads = c["num_attention_heads"]
    qn = c["qk_nope_head_dim"]
    qr = c["qk_rope_head_dim"]
    vh = c["v_head_dim"]
    ih = c["index_n_heads"]
    id_ = c["index_head_dim"]
    topk = c["index_topk"]
    full = full_indexer_flags(c)

    # 所有矩阵乘（包含 MoE 的 top-k 专家、lm_head、iHC、indexer 投影）
    proj = 2 * active_compute
    # MLA 的 QK^T + P @ V；DSA 只看 top-k 个 key
    lk = min(kv_len, topk)
    attn_core = n_layers(c) * (2 * heads * lk * (qn + qr) + 2 * heads * lk * vh)
    # indexer 打分要扫满 kv_len，只有 full 层做
    idx_core = sum(full) * (2 * ih * id_ * kv_len + 2 * ih * kv_len)
    return {
        "proj": proj,
        "attn_core": attn_core,
        "idx_core": idx_core,
        "total": proj + attn_core + idx_core,
        "lk": lk,
    }


def bytes_decode(c, active_compute, kv_len, dtype_bytes=2):
    kvl = c["kv_lora_rank"]
    qr = c["qk_rope_head_dim"]
    id_ = c["index_head_dim"]
    topk = c["index_topk"]
    full = full_indexer_flags(c)

    weights = active_compute * dtype_bytes
    lk = min(kv_len, topk)
    selected_kv = n_layers(c) * lk * (kvl + qr) * dtype_bytes
    # indexer 的 key cache 是每个 full 层 128 维，decode 时全扫
    index_scan = sum(full) * kv_len * id_ * dtype_bytes
    return {
        "weights": weights,
        "selected_kv": selected_kv,
        "index_scan": index_scan,
        "total": weights + selected_kv + index_scan,
    }


# ─────────────────────────────────────────────────────────── 输出
def print_config(full, c):
    section("1. config")
    row("model_type", full.get("model_type", "?"))
    row("architectures", full.get("architectures", "?"))
    row("hidden_size", c["hidden_size"])
    row("num_hidden_layers", n_layers(c), "1 dense + %d MoE" % (n_layers(c) - 1))
    row("attention heads", c["num_attention_heads"])
    row("q_lora / kv_lora", f'{c["q_lora_rank"]} / {c["kv_lora_rank"]}')
    row("qk head dims",
        f'{c["qk_nope_head_dim"]}+{c["qk_rope_head_dim"]} = '
        f'{c["qk_nope_head_dim"] + c["qk_rope_head_dim"]}')
    row("v_head_dim", c["v_head_dim"])
    row("qk_norm", c.get("qk_norm", False))
    row("DSA top-k", c["index_topk"])
    row("indexer", f'{c["index_n_heads"]} heads × {c["index_head_dim"]}')
    row("full indexer layers", sum(full_indexer_flags(c)), "shared 层复用 top-k")
    row("routed experts", c["n_routed_experts"])
    row("top-k / shared", f'{c["num_experts_per_tok"]} / {c.get("n_shared_experts", 1)}')
    row("moe_intermediate", c["moe_intermediate_size"])
    row("vocab_size", c["vocab_size"])
    row("iHC streams", c["hc_mult"])
    row("context", c["max_position_embeddings"])
    if full.get("vision_config"):
        v = full["vision_config"]
        row("vision", f'{v["num_hidden_layers"]} 层 / {v["hidden_size"]} 宽',
            f'FFN {v["intermediate_size"]}，patch {v["patch_size"]}')
        row("vision out_hidden", v["out_hidden_size"])
    if full.get("dfly_num_hidden_layers"):
        row("DFly2", f'{full["dfly_num_hidden_layers"]} 层草稿',
            f'target layers {full.get("dfly_target_kv_layers")}')
    row("MTP layers", c.get("num_nextn_predict_layers", 0), "正式版没有 MTP")


def print_params(c, p):
    section("2. 参数量")
    print("  逐层展开（完整模型，不是每 token 激活）：")
    row("  一层公共部分（attn+norms+iHC）", nfmt(p["common"]),
        "attention 265.686M + norms 12K + iHC 393K")
    row("  第 0 层 dense MLP", nfmt(p["dense_mlp"]), "18432 中间维")
    row("  每个 MoE 层", nfmt(p["moe"]), "256 专家 + 1 共享")
    row("  每个 full indexer", nfmt(p["indexer"]), "21 层有，57 层共享")
    print()
    for kind, items in p["groups"].items():
        sub = sum(x["params"] for x in items)
        row(f"  {kind} × {len(items)}", nfmt(sub), f"每层 {nfmt(sub / len(items))}")
    row("  embedding", nfmt(p["embed"]), "120832 × 6144")
    row("  lm_head", nfmt(p["lm_head"]), "独立权重，不 tying")
    row("  final norm + HC head", nfmt(p["final_norm"] + p["hyper_head"]))
    row("text backbone", nfmt(p["backbone"]),
        f'官方 770B，差 {(p["backbone"] / OFFICIAL_BACKBONE - 1) * 100:+.3f}%')
    row("+ ViT", nfmt(OFFICIAL_VIT), "官方 1.23B（有图时激活）")
    row("+ DFly2", nfmt(p["non_text"] - OFFICIAL_VIT), "checkpoint 反推，官方 3.3B")
    row("checkpoint 合计", nfmt(p["checkpoint"]), "HF safetensors 精确值")
    row("非 text 合计", nfmt(p["non_text"]), "ViT + DFly2")


def print_active(c, a, full=None):
    section("3. 激活参数量（每 token）")
    row("text 激活之和", nfmt(a["active"]),
        f'官方 49B，差 {(a["active"] / OFFICIAL_ACTIVE - 1) * 100:+.2f}%')
    row("  其中 MoE 单层激活", nfmt(a["active_moe"]), "top-8 专家 + 共享 + router")
    row("  减去 embedding 查表", "-" + nfmt(c["vocab_size"] * c["hidden_size"]),
        "查表不产生 FLOP")
    row("参与乘加的 text 激活", nfmt(a["active_compute"]), "FLOPs / Bytes 用这个")
    if full and full.get("vision_config"):
        row("+ Vision（有图时）", nfmt(OFFICIAL_VIT), "官方 1.23B")
        row("text + vision", nfmt(a["active"] + OFFICIAL_VIT), "官方 50B 量级")


def print_kv(kv):
    section("4. KV cache")
    row("每层每 token", f'{kv["latent_per_layer"]} elems', "MLA 压缩：512 latent + 64 rope")
    row("全部 78 层每 token", f'{kv["latent_elems"]} elems', bfmt(kv["latent_elems"] * 2))
    row("indexer cache 每 token", f'{kv["index_elems"]} elems', f'{kv["n_full"]} 个 full 层 × 128')
    row("合计每 token", f'{kv["bytes"]} B', bfmt(kv["bytes"]))
    row("1K token", bfmt(kv["bytes"] * 1024))
    row("128K token", bfmt(kv["bytes"] * 131072))
    row("1M token", bfmt(kv["bytes"] * 1048576), "单条序列的常驻 KV")


def print_prefill(c, p, a, seq_len):
    print(f"\n  prefill：seq_len = {seq_len}，每个 query 看满 {seq_len} 个 key（不折半）")
    f = flops(c, a["active_compute"], seq_len)
    b = bytes_decode(c, a["active_compute"], seq_len)
    total_flops = f["total"] * seq_len
    # 2048 个 token 基本会碰到全部 256 个专家，权重按全量读一次；
    # decode 只读 top-8，所以两张表的口径不同。
    full_weights = p["backbone"] * 2
    kv_traffic = (b["selected_kv"] + b["index_scan"]) * seq_len / 2  # causal 平均折半
    total_bytes = full_weights + kv_traffic
    row("  矩阵乘", nfmt(f["proj"]), "2 × 激活参数 / token")
    row("  attention 核心", nfmt(f["attn_core"]), f'每层 top-k = {f["lk"]}')
    row("  indexer 打分", nfmt(f["idx_core"]), "21 个 full 层")
    row("  每 token 合计", nfmt(f["total"]), "GFLOPs")
    row("  整段合计", nfmt(total_flops), "FLOPs")
    row("  text 全量权重（全专家）", bfmt(full_weights), "1.54TB 量级，prefill 读一遍")
    row("  KV 读写（causal 折半）", bfmt(kv_traffic), "S × 每 token 选中 KV / 2")
    row("  整段总字节", bfmt(total_bytes))
    row("  算术强度", f'{total_flops / total_bytes:.1f}', "FLOPs / Byte，和硬件算力/带宽比")


def print_decode(c, a, decode_kv):
    print(f"\n  decode：每 token，KV 长度 = {decode_kv}")
    f = flops(c, a["active_compute"], decode_kv)
    b = bytes_decode(c, a["active_compute"], decode_kv)
    row("  矩阵乘", nfmt(f["proj"]), "2 × 激活参数")
    row("  attention 核心", nfmt(f["attn_core"]), f'每层 top-k = {f["lk"]}')
    row("  indexer 打分", nfmt(f["idx_core"]), "扫满 KV")
    row("  每 token 合计", nfmt(f["total"]), "GFLOPs")
    row("  text 权重字节", bfmt(b["weights"]), "MoE 只读 top-8")
    row("  选中 KV + indexer", bfmt(b["selected_kv"] + b["index_scan"]))
    row("  每 token 字节", bfmt(b["total"]))
    row("  算术强度", f'{f["total"] / b["total"]:.2f}',
        "FLOPs / Byte，< 硬件算力/带宽 → 带宽瓶颈")

    print("\n  不同 KV 长度下的 decode（每 token）：")
    print(f"  {'KV':>8}  {'FLOPs':>10}  {'权重':>10}  {'KV/索引':>10}  {'总字节':>10}  {'强度':>6}")
    for klen in (1, 2048, 131072, 1048576):
        ff = flops(c, a["active_compute"], klen)
        bb = bytes_decode(c, a["active_compute"], klen)
        print(f"  {klen:>8}  {nfmt(ff['total']):>10}  {bfmt(bb['weights']):>10}  "
              f"{bfmt(bb['selected_kv'] + bb['index_scan']):>10}  {bfmt(bb['total']):>10}  "
              f"{ff['total'] / bb['total']:>6.2f}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(here, "config.json"))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--decode-kv", type=int, default=2048)
    args = ap.parse_args()

    with open(args.config) as f:
        full = json.load(f)
    if full.get("model_type") == "hy_v4_vl":
        c = full["text_config"]
    elif full.get("model_type") == "hy_v4":
        c = full
    else:
        raise SystemExit(f"不是 HY4 config: {full.get('model_type')}；"
                         "HY4-Flash 用 llm/hy4_flash/accounting.py")

    print("=" * 78)
    print(f"HY4 config accounting — {os.path.basename(args.config)}")
    print("只读 config.json；没有下载任何权重")
    print("=" * 78)

    p = model_params(c, CHECKPOINT_TOTAL)
    a = active_params(c, p)
    kv = kv_cache(c)

    print_config(full, c)
    print_params(c, p)
    print_active(c, a, full)
    print_kv(kv)

    section("5. FLOPs / Bytes")
    print_prefill(c, p, a, args.seq_len)
    print_decode(c, a, args.decode_kv)


if __name__ == "__main__":
    main()
