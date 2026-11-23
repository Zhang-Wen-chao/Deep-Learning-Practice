#!/usr/bin/env python3
"""HY4-Flash 纸面账本：只读 config.json，不下载、不加载权重。

算五样东西：
    1. 参数量（总参数 / 激活参数 / 每层 / 每类）
    2. FLOPs（prefill 与 decode，含 LatentMoE top-k、DSA 选择器）
    3. 访存 Bytes（权重 / KV cache / Engram 表）
    4. 算术强度与 roofline 粗估
    5. 与官方数字对账

用法：
    python llm/hy4_flash/accounting.py
    python llm/hy4_flash/accounting.py --prefill 8192 --context 131072
    python llm/hy4_flash/accounting.py --prefill 1048576 --context 1048576 --hardware b200x8

约定：
    - 1B = 10^9，1T = 10^12；Bytes 同时给 GB（10^9）和 GiB（2^30）。
    - 权重按 config 的 bfloat16（2 B/元素）算；`enable_lm_head_fp32=true` 的
      fp32 lm_head 口径单独列。
    - prefill 的 T×T 注意力按完整矩阵乘算（HF eager 上界）；causal 实际约一半，
      需要时在文末注明。
    - DSA indexer 在上下文 C <= index_topk(2048) 时被整段跳过，不产生选择开销。
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

# 官方数字，用于对账（model card / HF safetensors index）
OFFICIAL_TEXT = 155.5e9          # 主干，不含 Engram / ViT / DFly2
OFFICIAL_ENGRAM = 63.5e9         # 表 + 投影
OFFICIAL_VIT_TOWER = 0.43e9      # 视觉塔，不含 patch merger
OFFICIAL_DFLY = 850_000_000       # 投机解码 drafter
OFFICIAL_ACTIVE = 11.3e9         # 文本每 token 激活
HF_SAFETENSORS_TOTAL = 220_251_201_958  # tencent/Hy4-flash model.safetensors.index.json

# ─────────────────────────────────────────────────────────── 单位


def num(n):
    """整数 + 十进制人类可读。"""
    if n >= 1e12:
        return f"{n:,} ({n / 1e12:.3f} T)"
    if n >= 1e9:
        return f"{n:,} ({n / 1e9:.3f} B)"
    if n >= 1e6:
        return f"{n:,} ({n / 1e6:.3f} M)"
    if n >= 1e3:
        return f"{n:,} ({n / 1e3:.3f} K)"
    return f"{n:,}"


def byt(n):
    """字节：GB + GiB。"""
    return f"{n:,} B ({n / 1e9:.2f} GB / {n / 2**30:.2f} GiB)"


def rate(n):
    """FLOPs 或 FLOP/s 的十进制。"""
    if n >= 1e15:
        return f"{n / 1e15:.3f} P"
    if n >= 1e12:
        return f"{n / 1e12:.3f} T"
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return f"{n:.0f}"


def pct(a, b):
    return (a - b) / b * 100


# ─────────────────────────────────────────────────────────── config


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dims(c):
    """把嵌套的 text_config / vision_config 整理成算账用的符号。"""
    t, v = c["text_config"], c["vision_config"]

    # 发布版 config 用的是旧字段名；HF 的 HYV4FlashTextConfig 会把它重映射成
    # index_kpool / indexed_attention。这里做同样的事，后面只认规范名。
    raw_layers = t["layer_types"]
    layer_types = [
        "indexed_attention"
        if x in ("deepseek_sparse_attention", "indexed_attention")
        else "sliding_attention"
        for x in raw_layers
    ]
    assert len(layer_types) == t["num_hidden_layers"]
    assert set(layer_types) <= {"indexed_attention", "sliding_attention"}

    qk_nope = t["qk_nope_head_dim"]
    qk_rope = t["qk_rope_head_dim"]
    d = {
        "model_type": c["model_type"],
        "arch": c["architectures"][0],
        "V": t["vocab_size"],
        "H": t["hidden_size"],
        "L": t["num_hidden_layers"],
        "heads": t["num_attention_heads"],
        "swa_heads": t["swa_num_attention_heads"],
        "q_lora": t["q_lora_rank"],
        "kv_lora": t["kv_lora_rank"],
        "qk_nope": qk_nope,
        "qk_rope": qk_rope,
        "qk_dim": qk_nope + qk_rope,
        "v_head": t["v_head_dim"],
        "I": t["intermediate_size"],
        "E": t["n_routed_experts"],
        "top_k": t["num_experts_per_tok"],
        "I_moe": t["moe_intermediate_size"],
        "shared": t["n_shared_experts"],
        "latent": t["routed_expert_hidden_size"],
        "first_dense": t["first_k_dense_replace"],
        "sliding": t["sliding_window"],
        "index_topk": t["index_topk"],
        "index_heads": t["index_n_heads"],
        "index_dim": t["index_head_dim"],
        "kpool": t.get("index_kpool", t.get("compressed_indexer_compress_ratio", 4)),
        "hc": t["hc_mult"],
        "engram_ids": t["engram_layer_ids"],
        "engram_orders": t["engram_max_ngram_size"] - 1,
        "engram_max": t["engram_max_ngram_size"],
        "engram_heads": t["engram_n_head_per_ngram"],
        "engram_dim": t["engram_n_embed_per_ngram"],
        "engram_vocab": t["engram_layer_head_vocab_sizes"],
        "max_pos": t["max_position_embeddings"],
        "dtype_bytes": 2 if c["dtype"] == "bfloat16" else 4,
        "lm_head_bytes": 4 if t.get("enable_lm_head_fp32", False) else (2 if c["dtype"] == "bfloat16" else 4),
        "layer_types": layer_types,
        "mlp_types": t["mlp_layer_types"],
        "vision": {
            "H": v["hidden_size"],
            "I": v["intermediate_size"],
            "L": v["num_hidden_layers"],
            "heads": v["num_attention_heads"],
            "patch": v["patch_size"],
            "channels": v["num_channels"],
            "max_image": v["max_image_size"],
            "out": v["out_hidden_size"],
            "merge": v["spatial_merge_size"],
        },
    }
    d["n_sparse"] = layer_types.count("indexed_attention")
    d["n_sliding"] = layer_types.count("sliding_attention")
    d["n_dense"] = d["mlp_types"].count("dense")
    d["n_moe"] = d["L"] - d["n_dense"]
    d["q_dim"] = d["heads"] * d["qk_dim"]
    d["swa_q_dim"] = d["swa_heads"] * d["qk_dim"]
    d["engram_head_dim"] = d["engram_dim"] // d["engram_heads"]
    d["engram_size"] = d["engram_orders"] * d["engram_dim"]
    assert d["n_sparse"] + d["n_sliding"] == d["L"]
    assert d["n_dense"] + d["n_moe"] == d["L"]
    return d


# ─────────────────────────────────────────────────────────── 参数量


def attn_params(d, indexed):
    """一层 MLA 的权重；indexer 单独算，不混进来。"""
    H = d["H"]
    heads = d["heads"] if indexed else d["swa_heads"]
    qk_dim, v_head = d["qk_dim"], d["v_head"]
    return {
        "q_a_proj": H * d["q_lora"],
        "q_a_layernorm": d["q_lora"],
        "q_b_proj": d["q_lora"] * heads * qk_dim,
        "kv_a_proj_with_mqa": H * (d["kv_lora"] + d["qk_rope"]),
        "kv_a_layernorm": d["kv_lora"],
        "kv_b_proj": d["kv_lora"] * heads * (d["qk_nope"] + v_head),
        "gate_proj": H * heads * v_head,
        "o_proj": heads * v_head * H,
        "post_proj_norm": H,
        "sinks": heads,
    }


def indexer_params(d):
    """一层压缩 indexer 的权重（只挂在 indexed_attention 层上）。"""
    H, ih, idim = d["H"], d["index_heads"], d["index_dim"]
    return {
        "query_projection": d["q_lora"] * ih * idim,
        "head_weight_projection": H * ih,
        "token_key_projection": H * idim,
        "token_key_norm": idim,
        "key_value_projection": idim * idim,
        "key_gate_projection": idim * idim,
        "key_norm": idim,
        "intra_group_ape": d["kpool"] * idim,
    }


def dense_mlp_params(d):
    H, I = d["H"], d["I"]
    return 3 * H * I + H  # SwiGLU 三个矩阵 + post_mlp_layernorm


def moe_params(d):
    H, E, top_k = d["H"], d["E"], d["top_k"]
    mi, latent = d["I_moe"], d["latent"]
    routed = E * (2 * mi * latent + latent * mi)
    shared = 3 * H * (mi * d["shared"])
    pre = H * latent
    post = latent * H
    router = E * H
    combine = H  # combine_output_norm
    return {
        "routed_experts": routed,
        "shared_expert": shared,
        "pre_expert_proj": pre,
        "post_expert_proj": post,
        "router_gate": router,
        "combine_output_norm": combine,
        "total": routed + shared + pre + post + router + combine,
        "active_total": (
            top_k * (2 * mi * latent + latent * mi) + shared + pre + post + router + combine
        ),
        "expert": 2 * mi * latent + latent * mi,
    }


def ihc_params(d):
    """一层两条独立 Hyper-Connection（attn 一条、ffn 一条）。"""
    H, n = d["H"], d["hc"]
    one = (2 * n) * (n * H) + (2 * n) + 2
    return 2 * one


def hyper_head_params(d):
    H, n = d["H"], d["hc"]
    return n * (n * H) + n + 1


def engram_layer_params(d, layer_id):
    """一层 Engram：哈希表 + 投影 / norm。表单独汇总。"""
    H, hc = d["H"], d["hc"]
    sizes = [x for order in d["engram_vocab"][str(layer_id)] for x in order]
    head_dim = d["engram_head_dim"]
    table = sum(sizes) * head_dim
    other = (
        len(sizes) * head_dim              # lookup_per_head_norms
        + d["engram_size"] * H + H         # value_proj (+bias)
        + hc * (d["engram_size"] * H + H)  # key_projs (+bias)
        + 2 * hc * H                       # norm_key + norm_query
        + H                                # unified_pre_gamma
        + hc * H                           # unified_per_hc_scales
    )
    return {"table": table, "other": other, "total": table + other}


def vision_params(d):
    """SigLIP2-So400m 形状的 ViT 塔 + patch merger。"""
    v = d["vision"]
    H, I, L = v["H"], v["I"], v["L"]
    patch = v["channels"] * v["patch"] ** 2 * H + H
    edge = v["max_image"] // v["patch"]
    pos = (edge * edge + 1) * H
    block = (
        2 * (2 * H)                                   # LayerNorm ×2
        + 4 * (H * H + H)                             # q/k/v/o
        + (H * I + I) + (I * H + H)                   # fc1 / fc2
    )
    tower = patch + pos + L * block
    merger = (
        H                                             # before_rms
        + (H * (H * v["merge"] ** 2) * v["merge"] ** 2 + H * v["merge"] ** 2)
        + ((H * v["merge"] ** 2) * v["out"] + v["out"])
        + v["out"]                                    # image_newline
    )
    return {
        "patch_embed": patch,
        "pos_embed": pos,
        "block": block,
        "tower": tower,
        "merger": merger,
        "total": tower + merger,
    }


def param_ledger(d):
    sparse_core = sum(attn_params(d, True).values())
    sparse_indexer = sum(indexer_params(d).values())
    sparse = sparse_core + sparse_indexer
    sliding = sum(attn_params(d, False).values())
    dense = dense_mlp_params(d)
    moe = moe_params(d)
    ihc = ihc_params(d)
    head = hyper_head_params(d)
    final_norm = d["H"]
    decoder_norm = 2 * d["H"]  # input_layernorm + post_attention_layernorm

    engram = {lid: engram_layer_params(d, lid) for lid in d["engram_ids"]}
    engram_tables = sum(e["table"] for e in engram.values())
    engram_other = sum(e["other"] for e in engram.values())

    # 每层拼起来，避免“层 0 是 dense + sparse attention”这类交叉组合漏算。
    # text 是官方 155.5B 的口径：不含 Engram 表 / 投影；Engram 单独成项。
    layer_core, layer_p = [], []
    for i in range(d["L"]):
        p = sparse if d["layer_types"][i] == "indexed_attention" else sliding
        p += ihc + decoder_norm
        if d["mlp_types"][i] == "dense":
            p += dense
        else:
            p += moe["total"]
        layer_core.append(p)
        layer_p.append(p + (engram[i + 1]["total"] if i + 1 in engram else 0))

    embed = d["V"] * d["H"]
    lm_head = d["V"] * d["H"]  # tie_word_embeddings=false，两份都存
    text = embed + lm_head + sum(layer_core) + head + final_norm
    text_with_engram = text + engram_tables + engram_other

    active_layers = []
    for i in range(d["L"]):
        p = sparse if d["layer_types"][i] == "indexed_attention" else sliding
        p += ihc + decoder_norm
        if d["mlp_types"][i] == "dense":
            p += dense
        else:
            p += moe["active_total"]
        if i + 1 in engram:
            p += engram[i + 1]["other"]  # 表只查行，不整张读
        active_layers.append(p)
    active_no_embed = sum(active_layers) + lm_head + head + final_norm
    active_with_embed = active_no_embed + embed

    vit = vision_params(d)
    without_dfly = text_with_engram + vit["total"]
    dfly_derived = HF_SAFETENSORS_TOTAL - without_dfly
    total = without_dfly + OFFICIAL_DFLY

    return {
        "attn_sparse_core": sparse_core,
        "attn_sparse_indexer": sparse_indexer,
        "attn_sparse": sparse,
        "attn_sliding": sliding,
        "attn_sparse_total": d["n_sparse"] * sparse,
        "attn_sliding_total": d["n_sliding"] * sliding,
        "attn_indexer_total": d["n_sparse"] * sparse_indexer,
        "attn_total": d["n_sparse"] * sparse + d["n_sliding"] * sliding,
        "dense_mlp": dense,
        "moe": moe,
        "moe_total": d["n_moe"] * moe["total"],
        "ihc_layer": ihc,
        "ihc_total": d["L"] * ihc,
        "decoder_norm_layer": decoder_norm,
        "decoder_norm_total": d["L"] * decoder_norm,
        "hyper_head": head,
        "final_norm": final_norm,
        "engram_layer": engram,
        "engram_tables": engram_tables,
        "engram_other": engram_other,
        "engram_total": engram_tables + engram_other,
        "embed": embed,
        "lm_head": lm_head,
        "layers": layer_p,
        "text": text,
        "text_with_engram": text_with_engram,
        "active_no_embed": active_no_embed,
        "active_with_embed": active_with_embed,
        "vit": vit,
        "without_dfly": without_dfly,
        "dfly_derived": dfly_derived,
        "total": total,
    }




# ─────────────────────────────────────────────────────────── FLOPs

# 一个 Linear: 2 × token 数 × in × out（乘加算两次）。
# 一个 routed expert: gate_up(2I×H) + down(H×I) = 3HI 次乘加。
# DSA indexer: 每个 query 对 C/kpool 个压缩组打分，16 头 × 128 维。


def sum_min(T, k):
    """sum_{i=1..T} min(i, k)：causal 下每个 query 最多看 k 个 key 的总对数。"""
    if T <= k:
        return T * (T + 1) // 2
    return k * (k + 1) // 2 + (T - k) * k


def flops_attn_core_decode(d, ctx):
    """decode 1 token 的 QK^T + P@V；C 是可见上下文长度（含自己）。"""
    sp = d["n_sparse"] * d["heads"] * min(ctx, d["index_topk"])
    sw = d["n_sliding"] * d["swa_heads"] * min(ctx, d["sliding"])
    qk = 2 * (sp + sw) * d["qk_dim"]
    av = 2 * (sp + sw) * d["v_head"]
    return qk + av, qk, av


def flops_attn_core_prefill(d, T):
    """prefill T 个 token 的 QK^T + P@V，按 causal 求和。"""
    sp_pairs = d["n_sparse"] * d["heads"] * sum_min(T, d["index_topk"])
    sw_pairs = d["n_sliding"] * d["swa_heads"] * sum_min(T, d["sliding"])
    qk = 2 * (sp_pairs + sw_pairs) * d["qk_dim"]
    av = 2 * (sp_pairs + sw_pairs) * d["v_head"]
    return qk + av, qk, av


def flops_indexer_score(d, queries, groups, causal=False):
    """压缩 indexer 的打分矩阵乘；groups = 可见压缩组数。"""
    if groups <= 0:
        return 0
    pairs = queries * groups
    if causal:
        pairs //= 2
    # 16 头 × 128 维的 QK，然后 16 个头加权求和
    return d["n_sparse"] * (2 * d["index_heads"] * d["index_dim"] * pairs + 2 * d["index_heads"] * pairs)


def flops_indexer_pool(d, ctx):
    """HF eager 每个 decode step 重算所有组的池化；优化 kernel 可缓存。"""
    if ctx <= d["index_topk"]:
        return 0
    groups = ctx // d["kpool"]
    per_group = d["kpool"] * 4 * d["index_dim"] ** 2  # 两个 128×128 Linear + softmax
    return d["n_sparse"] * groups * per_group


def flops_prefill(d, T, active):
    linear = 2 * active * T
    core, qk, av = flops_attn_core_prefill(d, T)
    if T > d["index_topk"]:
        groups = T // d["kpool"]
        # HF eager 对全部 query×group 打分，future 也先算再 mask；causal 下界约一半。
        indexer = flops_indexer_score(d, T, groups, causal=False)
        indexer_causal = flops_indexer_score(d, T, groups, causal=True)
        pool = d["n_sparse"] * groups * d["kpool"] * 4 * d["index_dim"] ** 2
    else:
        indexer = indexer_causal = pool = 0
    total = linear + core + indexer + pool
    return {
        "linear": linear,
        "attn_core": core,
        "qk": qk,
        "av": av,
        "indexer": indexer,
        "indexer_causal": indexer_causal,
        "indexer_pool": pool,
        "total": total,
        "per_token": total / T,
    }


def flops_decode(d, ctx, active):
    linear = 2 * active
    core, qk, av = flops_attn_core_decode(d, ctx)
    if ctx > d["index_topk"]:
        groups = ctx // d["kpool"]
        indexer = flops_indexer_score(d, 1, groups)
        pool = flops_indexer_pool(d, ctx)
    else:
        indexer = pool = 0
    total = linear + core + indexer + pool
    return {
        "linear": linear,
        "attn_core": core,
        "qk": qk,
        "av": av,
        "indexer": indexer,
        "indexer_pool": pool,
        "total": total,
        "total_no_pool": linear + core + indexer,
    }


# ─────────────────────────────────────────────────────────── 访存


def bytes_checkpoint(p, d):
    """常驻权重：全部参数，按 bf16。"""
    return p["total"] * d["dtype_bytes"]


def bytes_prefill_read(p, d, T):
    """prefill 读权重：假设全部 routed experts 都命中；Engram 只读命中的行。

    Engram 表本身不用整张读：每 token 16 个哈希头，每头 96 维。
    """
    text_weights = (p["text"] - p["lm_head"]) * d["dtype_bytes"] + p["lm_head"] * d["lm_head_bytes"]
    non_engram = text_weights + (p["engram_other"] + p["vit"]["total"]) * d["dtype_bytes"]
    engram_rows = T * len(d["engram_ids"]) * d["engram_heads"] * d["engram_head_dim"] * d["dtype_bytes"]
    return non_engram + engram_rows


def bytes_decode_read(p, d, ctx):
    """decode 一个 token：激活权重 + Engram 查表行；KV 单独算。"""
    engram_rows = len(d["engram_ids"]) * d["engram_heads"] * d["engram_head_dim"] * d["dtype_bytes"]
    weights = (p["active_no_embed"] - p["lm_head"]) * d["dtype_bytes"] + p["lm_head"] * d["lm_head_bytes"]
    return weights + engram_rows


def kv_token_full(d):
    """全历史 KV：每层 512 latent + 64 rope；indexed 层再加 128 维 indexer key。"""
    per_layer = (d["kv_lora"] + d["qk_rope"]) * d["dtype_bytes"]
    return d["L"] * per_layer + d["n_sparse"] * d["index_dim"] * d["dtype_bytes"]


def kv_token_marginal(d):
    """HF DynamicCache 默认：global 层全历史、SWA 层 512 环形窗口时的每 token 增量。"""
    per_layer = (d["kv_lora"] + d["qk_rope"]) * d["dtype_bytes"]
    return d["n_sparse"] * (per_layer + d["index_dim"] * d["dtype_bytes"])


def kv_at(d, ctx, mode="full"):
    if mode == "full":
        return kv_token_full(d) * ctx
    # DynamicSlidingWindowLayer 实际缓存 sliding_window - 1 个前文 token；
    # 当前 token 的 K/V 只参与本次 attention，不算进常驻缓存。
    ring = d["n_sliding"] * (d["kv_lora"] + d["qk_rope"]) * d["dtype_bytes"] * min(ctx, d["sliding"] - 1)
    return kv_token_marginal(d) * ctx + ring


def bytes_prefill_activations(d, T):
    """粗估：4 条 iHC 流 + 每层子层输出。只用于量级。"""
    streams = d["hc"] * d["H"] * d["dtype_bytes"]
    return (streams + d["H"] * d["dtype_bytes"] * 2) * T * d["L"]


def bytes_decode_activations(d):
    """粗估一层激活读写：iHC 流 + MLA 中间量 + MoE 激活专家。"""
    H = d["H"]
    total = 2 * d["hc"] * H  # 4 条流读 + 写
    for i in range(d["L"]):
        heads = d["heads"] if d["layer_types"][i] == "indexed_attention" else d["swa_heads"]
        attn = (
            H + d["q_lora"] + heads * d["qk_dim"]
            + d["kv_lora"] + d["qk_rope"]
            + heads * (d["qk_nope"] + d["v_head"])
            + heads * d["v_head"] + H
        )
        mlp = (
            d["latent"] + d["top_k"] * 2 * d["I_moe"] + d["top_k"] * d["I_moe"]
            + 3 * d["I_moe"] * d["shared"] + H + d["E"]
        )
        total += attn + (mlp if d["mlp_types"][i] == "sparse" else 3 * d["I"])
    return total * d["dtype_bytes"]


# ─────────────────────────────────────────────────────────── 主输出


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--prefill", type=int, default=4096, help="prefill 的 token 数")
    ap.add_argument("--context", type=int, default=131072, help="decode 时的上下文长度")
    ap.add_argument(
        "--hardware", default="h20x8", choices=["h20x8", "h100x8", "b200x8", "m4"]
    )
    args = ap.parse_args()

    c = load_config(args.config)
    d = dims(c)
    p = param_ledger(d)

    section("0. config：唯一真源")
    print(f"  文件              {args.config}")
    print(f"  model_type        {d['model_type']} / {d['arch']}")
    print(f"  主干层数          {d['L']}（dense {d['n_dense']} + MoE {d['n_moe']}）")
    print(f"  attention         12 × sparse(DSA) + 30 × sliding(SWA)，1:3 交替")
    print(f"  hidden_size       {d['H']}")
    print(f"  MLA               q_lora={d['q_lora']}, kv_lora={d['kv_lora']}, "
          f"QK={d['qk_nope']}+{d['qk_rope']}={d['qk_dim']}, V={d['v_head']}")
    print(f"  heads             sparse {d['heads']} / SWA {d['swa_heads']}"
          f"（config 里 head_dim={c['text_config']['head_dim']} 只是 RoPE 切片，别拿它当 QK 宽度）")
    print(f"  MoE               {d['E']} routed, top-{d['top_k']}, shared {d['shared']}, "
          f"latent {d['latent']}, expert intermediate {d['I_moe']}")
    print(f"  dense FFN         {d['I']}（只在 layer 0）")
    print(f"  DSA               index_topk={d['index_topk']}, group={d['kpool']}, "
          f"index {d['index_heads']}×{d['index_dim']}")
    print(f"  SWA               window={d['sliding']}")
    print(f"  iHC               {d['hc']} 条 residual stream")
    print(f"  Engram            layers={d['engram_ids']}（1-based）, "
          f"orders=2..{d['engram_max']}, "
          f"{d['engram_heads']} heads × {d['engram_head_dim']} dim")
    print(f"  vocab / max_pos   {d['V']} / {d['max_pos']}")
    print(f"  dtype             {c['dtype']}（{d['dtype_bytes']} B/元素）")
    print(f"  tie_word_embeddings {c['text_config']['tie_word_embeddings']} → embedding 与 lm_head 各存一份")

    section("1. 参数量：按模块数")
    print(f"  attention（sparse 一层）    {num(p['attn_sparse']):<34} MLA + indexer + sinks")
    print(f"    MLA core                  {num(p['attn_sparse_core'])}")
    print(f"    compressed indexer        {num(p['attn_sparse_indexer'])}")
    print(f"  attention（SWA 一层）       {num(p['attn_sliding'])}")
    print(f"  attention（12 sparse）      {num(p['attn_sparse_total'])}")
    print(f"  attention（30 SWA）         {num(p['attn_sliding_total'])}")
    print(f"  attention 合计              {num(p['attn_total'])}")
    print(f"  dense MLP（layer 0）        {num(p['dense_mlp'])}")
    print(f"  MoE（每个 MoE 层）          {num(p['moe']['total'])}")
    print(f"    routed experts            {num(p['moe']['routed_experts'])}  = {d['E']} × {num(p['moe']['expert'])}")
    print(f"    shared expert             {num(p['moe']['shared_expert'])}")
    print(f"    latent pre/post           {num(p['moe']['pre_expert_proj'] + p['moe']['post_expert_proj'])}")
    print(f"    router gate               {num(p['moe']['router_gate'])}")
    print(f"  MoE 合计（41 层）           {num(p['moe_total'])}")
    print(f"  iHC（42 层 + head）         {num(p['ihc_total'] + p['hyper_head'])}")
    print(f"  decoder 两条 RMSNorm（42 层）{num(p['decoder_norm_total'])}")
    print(f"  Engram 表                   {num(p['engram_tables'])}")
    print(f"  Engram 投影 / norm          {num(p['engram_other'])}")
    print(f"  embedding                   {num(p['embed'])}")
    print(f"  lm_head（不共享）           {num(p['lm_head'])}")
    print(f"  final norm                  {num(p['final_norm'])}")
    print(f"  ViT 塔（0.43B 口径）        {num(p['vit']['tower'])}")
    print(f"  ViT patch merger            {num(p['vit']['merger'])}")
    print()
    print(f"  文本主干（不含 Engram）     {num(p['text'])}")
    print(f"  文本主干 + Engram           {num(p['text_with_engram'])}")
    print(f"  Engram 合计                 {num(p['engram_total'])}")
    print(f"  ViT 合计                    {num(p['vit']['total'])}")
    print(f"  DFly2（config 未给）        {num(OFFICIAL_DFLY)}")
    print(f"  合计（官方 DFly2 口径）     {num(p['total'])}")

    section("2. 参数量：两种算法对账")
    print(f"  本文文本主干                {p['text']:,}  ({p['text'] / 1e9:.3f} B)")
    print(f"  官方口径「155.5B」          {OFFICIAL_TEXT:,.0f}")
    print(f"  差                          {p['text'] - OFFICIAL_TEXT:+,.0f} "
          f"({pct(p['text'], OFFICIAL_TEXT):+.3f}%)")
    print()
    print(f"  本文 Engram                 {p['engram_total']:,}  ({p['engram_total'] / 1e9:.3f} B)")
    print(f"  官方口径「63.5B」           {OFFICIAL_ENGRAM:,.0f}")
    print(f"  差                          {p['engram_total'] - OFFICIAL_ENGRAM:+,.0f} "
          f"({pct(p['engram_total'], OFFICIAL_ENGRAM):+.3f}%)")
    print()
    print(f"  本文 ViT 塔                 {p['vit']['tower']:,}  ({p['vit']['tower'] / 1e9:.3f} B)")
    print(f"  官方口径「0.43B」           {OFFICIAL_VIT_TOWER:,.0f}")
    print(f"  差                          {p['vit']['tower'] - OFFICIAL_VIT_TOWER:+,.0f} "
          f"({pct(p['vit']['tower'], OFFICIAL_VIT_TOWER):+.3f}%)")
    print()
    print(f"  本文不含 DFly2              {p['without_dfly']:,}  ({p['without_dfly'] / 1e9:.3f} B)")
    print(f"  HF safetensors 总数         {HF_SAFETENSORS_TOTAL:,}  ({HF_SAFETENSORS_TOTAL / 1e9:.3f} B)")
    print(f"  反推 DFly2                  {p['dfly_derived']:,}  ({p['dfly_derived'] / 1e9:.3f} B)")
    print(f"  官方 DFly2 口径「0.85B」    {OFFICIAL_DFLY:,.0f}")
    print(f"  差                          {p['dfly_derived'] - OFFICIAL_DFLY:+,.0f} "
          f"({pct(p['dfly_derived'], OFFICIAL_DFLY):+.3f}%)")
    print()
    print(f"  本文总参数（DFly2=0.85B）   {p['total']:,.0f}  ({p['total'] / 1e9:.3f} B)")
    print(f"  HF safetensors 总数         {HF_SAFETENSORS_TOTAL:,}  ({HF_SAFETENSORS_TOTAL / 1e9:.3f} B)")
    print(f"  差                          {p['total'] - HF_SAFETENSORS_TOTAL:+,.0f} "
          f"({pct(p['total'], HF_SAFETENSORS_TOTAL):+.4f}%)")

    section("3. 激活参数（每个文本 token 真正参与计算的权重）")
    print(f"  文本激活（不含 embedding）  {num(p['active_no_embed'])}")
    print(f"    其中 lm_head               {num(p['lm_head'])}")
    print(f"    其中 Engram 投影 / norm    {num(p['engram_other'])}")
    print(f"  官方口径「11.3B」           {OFFICIAL_ACTIVE:,.0f}  → 差 "
          f"{pct(p['active_no_embed'], OFFICIAL_ACTIVE):+.2f}%")
    print(f"  若把 embedding 整张也算     {num(p['active_with_embed'])}（查表只读一行，通常不计）")
    print(f"  视觉塔激活（有图时）        {num(p['vit']['tower'])}")
    print(f"  总参数 : 文本激活           {p['total'] / p['active_no_embed']:.2f} : 1")

    section(f"4. FLOPs：prefill T={args.prefill:,}")
    pre = flops_prefill(d, args.prefill, p["active_no_embed"])
    print(f"  线性层（2 × 激活参数 × T）  {rate(pre['linear'])}")
    print(f"  attention QK^T + P@V     {rate(pre['attn_core'])}")
    print(f"    QK^T                      {rate(pre['qk'])}")
    print(f"    P@V                    {rate(pre['av'])}")
    if pre["indexer"]:
        print(f"  DSA indexer 打分（eager）   {rate(pre['indexer'])}")
        print(f"    causal 下界（约一半）     {rate(pre['indexer_causal'])}")
        print(f"  DSA indexer 池化（eager）   {rate(pre['indexer_pool'])}")
    else:
        print(f"  DSA indexer                 0（T <= {d['index_topk']}，整段跳过）")
    print(f"  合计（eager 上界）          {rate(pre['total'])} FLOPs")
    print(f"  平均每 token                {rate(pre['per_token'])} FLOPs/token")
    print()
    print(f"  校验：2 × 激活参数 × T = 2 × {p['active_no_embed']:,} × {args.prefill:,}")
    print(f"        = {rate(pre['linear'])}  ← 不含 attention core / indexer / norm")

    section(f"5. FLOPs：decode 1 token（上下文 {args.context:,}）")
    dec = flops_decode(d, args.context, p["active_no_embed"])
    print(f"  线性层（2 × 激活参数）      {rate(dec['linear'])}")
    print(f"  attention QK^T + P@V     {rate(dec['attn_core'])}")
    print(f"    QK^T                      {rate(dec['qk'])}")
    print(f"    P@V                    {rate(dec['av'])}")
    if dec["indexer"]:
        print(f"  DSA indexer 打分            {rate(dec['indexer'])}")
        print(f"  DSA indexer 池化（eager）   {rate(dec['indexer_pool'])}  ← 优化 kernel 可缓存")
    else:
        print(f"  DSA indexer                 0（C <= {d['index_topk']}，整段跳过）")
    print(f"  合计（不含池化）            {rate(dec['total_no_pool'])} FLOPs/token")
    print(f"  合计（含 eager 池化）       {rate(dec['total'])} FLOPs/token")
    print()
    print(f"  校验：2 × 激活参数 = {rate(dec['linear'])}")
    print(f"  差值 = attention core + indexer 的零头")

    section("6. 访存：权重 + KV cache + 激活")
    ckpt = bytes_checkpoint(p, d)
    pre_w = bytes_prefill_read(p, d, args.prefill)
    dec_w = bytes_decode_read(p, d, args.context)
    kv_full = kv_at(d, args.context, "full")
    kv_min = kv_at(d, args.context, "min")
    kv_tok = kv_token_full(d)
    kv_tok_min = kv_token_marginal(d)
    act_pre = bytes_prefill_activations(d, args.prefill)
    act_dec = bytes_decode_activations(d)

    print(f"  checkpoint 常驻（BF16）     {byt(ckpt)}")
    print(f"    其中 Engram 表             {byt(p['engram_tables'] * d['dtype_bytes'])}")
    print(f"    其中 ViT                   {byt(p['vit']['total'] * d['dtype_bytes'])}")
    print()
    print(f"  prefill 读权重（全专家命中）{byt(pre_w)}")
    print(f"  prefill 激活读写（粗估）    {byt(act_pre)}  ← 多数在片上，不计入 roofline")
    print(f"  prefill 写 KV（T={args.prefill:,}）     {byt(kv_at(d, args.prefill, 'min'))}  ← HF 默认")
    print(f"    若 42 层全历史（上界）       {byt(kv_at(d, args.prefill, 'full'))}")
    print()
    print(f"  decode 读权重（top-{d['top_k']}，fp32 lm_head） {byt(dec_w)}")
    print(f"  decode 激活读写（粗估）     {byt(act_dec)}  ← 可忽略")
    print(f"  decode 读 KV（C={args.context:,}）   {byt(kv_min)}  ← HF 默认")
    print(f"    若 42 层全历史（上界）       {byt(kv_full)}")
    print(f"    KV / token / 全历史上界    {kv_tok:,} B ({kv_tok / 1024:.2f} KiB)")
    print(f"    KV / token / HF 默认       {kv_tok_min:,} B ({kv_tok_min / 1024:.2f} KiB)")
    print(f"    KV 占满 {d['max_pos']:,} 上下文  {byt(kv_at(d, d['max_pos'], 'full'))}（上界）")
    print(f"    SWA 环形窗口（30 层）      {byt(d['n_sliding'] * (d['kv_lora'] + d['qk_rope']) * d['dtype_bytes'] * (d['sliding'] - 1))}")
    print(f"    HF 默认（C={args.context:,}）      {byt(kv_min)}")
    print()
    print(f"  说明：HF `DynamicCache(config=...)` 会把 `sliding_attention` 层建成")
    print(f"        `DynamicSlidingWindowLayer`，所以默认就是 12 个 global 层全历史 + 30 个")
    print(f"        SWA 层只留 {d['sliding']} token 环形窗口。全历史列是普通全量 cache 的上界。")

    section("7. 算术强度与 roofline")
    hw = {
        "h20x8": ("8×H20-3e", 8 * 148e12, 8 * 4.0e12, 8 * 96 * 2**30),
        "h100x8": ("8×H100 SXM", 8 * 989e12, 8 * 3.35e12, 8 * 80 * 2**30),
        "b200x8": ("8×B200", 8 * 2.25e15, 8 * 8.0e12, 8 * 192 * 2**30),
        "m4": ("Apple M4 10-core GPU", 4.6e12, 120e9, 16 * 2**30),
    }[args.hardware]
    name, peak_flops, peak_bw, mem = hw
    print(f"  机器                        {name}")
    print(f"  峰值 BF16 算力              {rate(peak_flops)} FLOP/s")
    print(f"  峰值带宽                    {rate(peak_bw)} B/s")
    print(f"  算力/带宽                   {peak_flops / peak_bw:.1f} FLOP/B")
    print(f"  显存                        {byt(mem)}")
    print()
    for label, fl, by in [
        (f"prefill T={args.prefill:,}（HF 默认 KV）", pre["total"], pre_w + kv_at(d, args.prefill, "min")),
        (f"decode C={args.context:,}（HF 默认 KV）", dec["total"], dec_w + kv_min),
        (f"decode C={args.context:,}（无 eager 池化）", dec["total_no_pool"], dec_w + kv_min),
    ]:
        ai = fl / by
        t_mem = by / peak_bw
        t_compute = fl / (peak_flops * 0.4)  # MFU 40%
        print(f"  {label:<34} FLOPs={rate(fl):>10}  Bytes={rate(by):>10}  "
              f"AI={ai:7.2f}  t_mem={t_mem * 1e3:8.3f} ms  t_compute@40%={t_compute * 1e3:8.3f} ms")
    print()
    print(f"  若按全历史 KV 上界：decode C={args.context:,} 用 {byt(dec_w + kv_full)}，"
          f"AI={dec['total_no_pool'] / (dec_w + kv_full):.2f}（理想）"
          f" / {dec['total'] / (dec_w + kv_full):.2f}（eager）")
    print(f"  权重装得下吗：{byt(ckpt)} vs 显存 {byt(mem)}"
          f"  → {'够' if ckpt < mem else '不够'}")
    print(f"  4-bit 量化后约 {byt(ckpt / 4)}；FP8 后约 {byt(ckpt / 2)}。")


if __name__ == "__main__":
    main()
