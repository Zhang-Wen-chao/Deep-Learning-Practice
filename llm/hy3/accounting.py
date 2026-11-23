#!/usr/bin/env python3
"""HY3 纸面账本：只读 config.json，不下载、不加载权重。

算四样东西：
    1. 参数量（总参数 / 激活参数 / 每层 / 每类）
    2. FLOPs（prefill 与 decode，含 MoE 只算 top-k）
    3. 访存 Bytes（权重 / 激活 / KV cache）
    4. 算术强度与 roofline 粗估

用法：
    python llm/hy3/accounting.py
    python llm/hy3/accounting.py --prefill 1024 --context 32768
    python llm/hy3/accounting.py --prefill 4096 --context 131072 --hardware h20x8

约定：
    - 参数量、FLOPs 用十进制（1B = 10^9，1T = 10^12）。
    - Bytes 同时给 GB（10^9）和 GiB（2^30）。
    - 权重按 config 的 bfloat16（2 B/元素）算；lm_head 的 fp32 口径单独列。
    - attention 的 T×T 项按完整矩阵乘算（上界）；causal 实际约一半，单独注明。
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

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


# ─────────────────────────────────────────────────────────── config


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dims(c):
    """把 config 里的字段整理成算账用的符号。"""
    d = {
        "V": c["vocab_size"],
        "H": c["hidden_size"],
        "I": c["intermediate_size"],
        "L": c["num_hidden_layers"],
        "heads": c["num_attention_heads"],
        "kv_heads": c["num_key_value_heads"],
        "head_dim": c["head_dim"],
        "E": c["num_experts"],
        "top_k": c["num_experts_per_tok"],
        "I_moe": c["moe_intermediate_size"],
        "shared": c["num_shared_experts"],
        "first_dense": c["first_k_dense_replace"],
        "mtp": c.get("num_nextn_predict_layers", 0),
        "max_pos": c["max_position_embeddings"],
        "dtype_bytes": 2,  # config.dtype = bfloat16
    }
    d["q_dim"] = d["heads"] * d["head_dim"]
    d["kv_dim"] = d["kv_heads"] * d["head_dim"]
    d["group"] = d["heads"] // d["kv_heads"]
    return d


# ─────────────────────────────────────────────────────────── 参数量


def param_ledger(d):
    H, I, V = d["H"], d["I"], d["V"]
    Hq, Hkv = d["q_dim"], d["kv_dim"]

    attn = H * Hq + 2 * H * Hkv + Hq * H + 2 * d["head_dim"]
    dense_mlp = 3 * H * I
    expert = 3 * H * d["I_moe"]
    experts = d["E"] * expert
    shared = d["shared"] * 3 * H * d["I_moe"]
    router = d["E"] * H
    norms = 2 * H

    layer_dense = attn + dense_mlp + norms
    layer_moe = attn + experts + shared + router + norms
    embed = V * H
    lm_head = V * H
    final_norm = H

    main = d["first_dense"] * layer_dense + (d["L"] - d["first_dense"]) * layer_moe + embed + lm_head + final_norm

    eh_proj = 2 * H * H
    mtp = layer_moe + eh_proj + 2 * H + H  # enorm + hnorm + shared_head.norm
    total = main + d["mtp"] * mtp

    active_layer_moe = attn + router + d["top_k"] * expert + shared + norms
    active_main = d["first_dense"] * layer_dense + (d["L"] - d["first_dense"]) * active_layer_moe + embed + lm_head + final_norm
    active_mtp = active_layer_moe + eh_proj + 2 * H + H

    return {
        "attn": attn,
        "dense_mlp": dense_mlp,
        "expert": expert,
        "experts": experts,
        "shared": shared,
        "router": router,
        "norms": norms,
        "layer_dense": layer_dense,
        "layer_moe": layer_moe,
        "embed": embed,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "main": main,
        "mtp_each": mtp,
        "mtp_total": d["mtp"] * mtp,
        "total": total,
        "active_layer_moe": active_layer_moe,
        "active_main": active_main,
        "active_mtp": active_mtp,
        "active_total_with_mtp": active_main + d["mtp"] * active_mtp,
    }


# ─────────────────────────────────────────────────────────── FLOPs

# 一个 Linear: 2 × token 数 × in × out（乘加算两次）
# 一个 expert 的 SwiGLU: gate_up(2I×H) + down(H×I) = 3HI 次乘加
# RMSNorm: 每元素约 5 次；softmax: 每个 score 约 5 次；RoPE: 每元素约 6 次。


def flops_attention(d, T, kv_len, causal=False):
    H, Hq, Hkv = d["H"], d["q_dim"], d["kv_dim"]
    q = 2 * T * H * Hq
    k = 2 * T * H * Hkv
    v = 2 * T * H * Hkv
    o = 2 * T * Hq * H
    pairs = T * (T + 1) // 2 if causal else T * kv_len
    scores = 2 * d["heads"] * pairs * d["head_dim"]
    av = scores
    qk_norm = 2 * 5 * T * Hq + 2 * 5 * T * Hkv
    rope = 6 * T * (Hq + Hkv)
    softmax = 5 * d["heads"] * pairs
    return {
        "q_proj": q,
        "k_proj": k,
        "v_proj": v,
        "o_proj": o,
        "scores": scores,
        "attn_v": av,
        "qk_norm": qk_norm,
        "rope": rope,
        "softmax": softmax,
        "linear": q + k + v + o,
        "total": q + k + v + o + scores + av + qk_norm + rope + softmax,
    }


def flops_moe(d, T):
    H, I = d["H"], d["I_moe"]
    router = 2 * T * H * d["E"]
    routed = d["top_k"] * 6 * T * H * I
    shared = d["shared"] * 6 * T * H * I
    return {
        "router": router,
        "routed": routed,
        "shared": shared,
        "total": router + routed + shared,
    }


def flops_dense_mlp(d, T):
    return 6 * T * d["H"] * d["I"]


def flops_layer(d, T, kv_len, dense=False, causal=False):
    a = flops_attention(d, T, kv_len, causal)
    if dense:
        m = flops_dense_mlp(d, T)
        mlp_detail = {"dense": m}
        mlp = m
    else:
        m = flops_moe(d, T)
        mlp_detail = m
        mlp = m["total"]
    norms = 2 * 5 * T * d["H"]
    return {
        "attn": a,
        "mlp": mlp_detail,
        "norms": norms,
        "total": a["total"] + mlp + norms,
    }


def flops_prefill(d, T, causal=False):
    dense = flops_layer(d, T, T, dense=True, causal=causal)
    moe = flops_layer(d, T, T, dense=False, causal=causal)
    lm_head = 2 * T * d["H"] * d["V"]
    final_norm = 5 * T * d["H"]
    total = d["first_dense"] * dense["total"] + (d["L"] - d["first_dense"]) * moe["total"] + lm_head + final_norm
    return {
        "dense": dense,
        "moe": moe,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "total": total,
        "per_token": total / T,
        "tokens": T,
    }


def flops_decode(d, ctx):
    """解码一个 token。ctx = 生成这个 token 时可见的上下文长度（含自己）。"""
    dense = flops_layer(d, 1, ctx, dense=True)
    moe = flops_layer(d, 1, ctx, dense=False)
    lm_head = 2 * d["H"] * d["V"]
    final_norm = 5 * d["H"]
    total = d["first_dense"] * dense["total"] + (d["L"] - d["first_dense"]) * moe["total"] + lm_head + final_norm
    return {
        "dense": dense,
        "moe": moe,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "total": total,
    }


# ─────────────────────────────────────────────────────────── 访存


def bytes_prefill_weights(d, p, fp32_lm_head=False):
    """prefill 要把所有专家都读一遍（T 大时 192 个专家基本都会被命中）。"""
    lm = p["lm_head"] * (4 if fp32_lm_head else d["dtype_bytes"])
    return (p["main"] - p["lm_head"]) * d["dtype_bytes"] + lm


def bytes_decode_weights(d, p, fp32_lm_head=False):
    """decode 一个 token 只读被激活的专家。"""
    touched = (
        d["first_dense"] * p["layer_dense"]
        + (d["L"] - d["first_dense"]) * p["active_layer_moe"]
        + d["H"]  # embedding 只读一行
        + p["final_norm"]
        + p["lm_head"]
    )
    lm = p["lm_head"] * (4 if fp32_lm_head else d["dtype_bytes"])
    return (touched - p["lm_head"]) * d["dtype_bytes"] + lm


def bytes_kv_per_token(d):
    return 2 * d["L"] * d["kv_dim"] * d["dtype_bytes"]


def bytes_kv_at(d, ctx):
    return bytes_kv_per_token(d) * ctx


def bytes_decode_activations(d, ctx):
    """decode 一个 token 的粗略激活读写（不含 KV）。"""
    H, Hq, Hkv, V, I = d["H"], d["q_dim"], d["kv_dim"], d["V"], d["I_moe"]
    # 每个 Linear 读输入 + 写输出；MoE 只算 top-k + shared；router 读写 logits
    attn = (H + Hq) + (H + Hkv) * 2 + (Hq + H) + 2 * Hq + H  # qkv/o + qknorm/rope
    dense_mlp = (H + 2 * d["I"]) + (d["I"] + H)
    moe = (
        (H + d["E"])
        + d["top_k"] * ((H + 2 * d["I_moe"]) + (d["I_moe"] + H))
        + ((H + 2 * d["I_moe"]) + (d["I_moe"] + H))
    )
    lm = H + V
    elems = attn + dense_mlp + (d["L"] - d["first_dense"]) * moe + lm
    return elems * d["dtype_bytes"]


def bytes_prefill_activations(d, T):
    """prefill 的粗略激活读写：decode 的单 token 结构 × T。"""
    return bytes_decode_activations(d, 0) * T


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
    ap.add_argument("--hardware", default="h20x8", choices=["h20x8", "h100x8", "m4"])
    args = ap.parse_args()

    c = load_config(args.config)
    d = dims(c)
    p = param_ledger(d)

    section("0. config：唯一真源")
    print(f"  文件              {os.path.relpath(args.config)}")
    print(f"  model_type        {c['model_type']} / {c['architectures'][0]}")
    print(f"  hidden_size       {d['H']}")
    print(f"  num_hidden_layers {d['L']}（不含 MTP；MTP 层数 {d['mtp']}）")
    print(f"  attention         {d['heads']} Q heads / {d['kv_heads']} KV heads, head_dim {d['head_dim']}")
    print(f"  q_dim / kv_dim    {d['q_dim']} / {d['kv_dim']}（GQA group = {d['group']}）")
    print(f"  intermediate_size {d['I']}（dense 层）")
    print(f"  MoE               {d['E']} experts, top-{d['top_k']}, shared {d['shared']}, "
          f"expert intermediate {d['I_moe']}")
    print(f"  dense / MoE 层    {d['first_dense']} / {d['L'] - d['first_dense']}")
    print(f"  vocab / max_pos   {d['V']} / {d['max_pos']}")
    print(f"  dtype              {c['dtype']}（{d['dtype_bytes']} B/元素）")
    print(f"  rope              theta={c['rope_parameters']['rope_theta']:.0f}, "
          f"type={c['rope_parameters']['rope_type']}")
    print(f"  tie_word_embeddings {c['tie_word_embeddings']}")

    section("1. 参数量：按模块数")
    print(f"  attention（一层）           {num(p['attn']):<32} q/k/v/o + q_norm/k_norm")
    print(f"    q_proj                    {num(d['H'] * d['q_dim'])}")
    print(f"    k_proj                    {num(d['H'] * d['kv_dim'])}")
    print(f"    v_proj                    {num(d['H'] * d['kv_dim'])}")
    print(f"    o_proj                    {num(d['q_dim'] * d['H'])}")
    print(f"    q_norm + k_norm           {num(2 * d['head_dim'])}")
    print(f"  dense MLP（layer 0）        {num(p['dense_mlp'])}")
    print(f"  MoE（每个 MoE 层）          {num(p['experts'] + p['shared'] + p['router'])}")
    print(f"    routed experts            {num(p['experts'])}  = {d['E']} × {num(p['expert'])}")
    print(f"    shared expert             {num(p['shared'])}")
    print(f"    router gate               {num(p['router'])}  ({d['E']}×{d['H']}, fp32)")
    print(f"    e_score_correction_bias   {d['E']}（buffer，不算参数）")
    print(f"  layer 0（dense）            {num(p['layer_dense'])}")
    print(f"  layer 1–{d['L'] - 1}（MoE，每层）     {num(p['layer_moe'])}")
    print(f"  embedding                   {num(p['embed'])}")
    print(f"  lm_head（不共享）           {num(p['lm_head'])}")
    print(f"  final norm                  {num(p['final_norm'])}")
    print()
    print(f"  主干合计（不含 MTP）        {num(p['main'])}")
    print(f"  MTP 一层 + eh_proj          {num(p['mtp_each'])}")
    print(f"  MTP 合计                    {num(p['mtp_total'])}")
    print(f"  含 MTP 的 checkpoint        {num(p['total'])}")

    section("2. 参数量：两种算法对账")
    official_main = 295e9
    official_ckpt = 298_786_140_416  # HF safetensors BF16 参数
    print(f"  本文主干                    {p['main']:,}  ({p['main'] / 1e9:.3f} B)")
    print(f"  官方口径「295B」            {official_main:,.0f}")
    print(f"  差                          {p['main'] - official_main:,.0f} "
          f"({(p['main'] - official_main) / official_main * 100:+.3f}%)")
    print()
    print(f"  本文含 MTP                  {p['total']:,}  ({p['total'] / 1e9:.3f} B)")
    print(f"  HF safetensors BF16         {official_ckpt:,}  ({official_ckpt / 1e9:.3f} B)")
    print(f"  差                          {p['total'] - official_ckpt:+,.0f}  ← 应为 0")
    print()
    print(f"  F32 张量                    {c['num_experts'] * d['L']:,} 个 "
          f"= e_score_correction_bias（每层 {c['num_experts']} 个 buffer）")

    section("3. 激活参数（每个 token 真正参与计算的权重）")
    print(f"  MoE 层激活（每层）          {num(p['active_layer_moe'])}")
    print(f"    attention                 {num(p['attn'])}")
    print(f"    router                    {num(p['router'])}")
    print(f"    top-{d['top_k']} experts           {num(d['top_k'] * p['expert'])}")
    print(f"    shared expert             {num(p['shared'])}")
    print(f"    norms                     {num(p['norms'])}")
    print(f"  layer 0（dense）            {num(p['layer_dense'])}")
    print(f"  embedding + lm_head         {num(p['embed'] + p['lm_head'])}")
    print(f"  主干激活合计                {num(p['active_main'])}")
    print(f"  官方口径「21B」             {21e9:,.0f}  → 四舍五入对得上")
    print(f"  开 MTP 时再加               {num(p['active_mtp'])} / draft token")
    print(f"  总参数 : 激活参数           {p['main'] / p['active_main']:.2f} : 1")

    section(f"4. FLOPs：prefill seq_len={args.prefill}（完整 seq_len × seq_len，未按 causal 减半）")
    pre = flops_prefill(d, args.prefill)
    print(f"  layer 0（dense）            {rate(pre['dense']['total'])} FLOPs")
    print(f"  MoE 层（每层）              {rate(pre['moe']['total'])} FLOPs")
    print(f"    attention                 {rate(pre['moe']['attn']['total'])}")
    print(f"      q/k/v/o                 {rate(pre['moe']['attn']['linear'])}")
    print(f"      scores + P@V         {rate(pre['moe']['attn']['scores'] * 2)}")
    print(f"    MoE                       {rate(pre['moe']['mlp']['total'])}")
    print(f"      router                  {rate(pre['moe']['mlp']['router'])}")
    print(f"      top-{d['top_k']} experts           {rate(pre['moe']['mlp']['routed'])}")
    print(f"      shared                  {rate(pre['moe']['mlp']['shared'])}")
    print(f"    norms                     {rate(pre['moe']['norms'])}")
    print(f"  {d['L'] - d['first_dense']} 个 MoE 层合计          {rate((d['L'] - d['first_dense']) * pre['moe']['total'])}")
    print(f"  lm_head                     {rate(pre['lm_head'])}")
    print(f"  合计                        {rate(pre['total'])} FLOPs")
    print(f"  平均每 token                {rate(pre['per_token'])} FLOPs/token")
    print()
    linear_check = (p["active_main"] - p["embed"]) * 2 * args.prefill
    print(f"  校验：2 × (激活参数 − embedding) × seq_len")
    print(f"        = 2 × {p['active_main'] - p['embed']:,} × {args.prefill}")
    print(f"        = {rate(linear_check)}  ← 不含 T×T 注意力与 norm/softmax")

    section(f"5. FLOPs：decode 1 token（kv_len = {args.context}）")
    dec = flops_decode(d, args.context)
    print(f"  layer 0（dense）            {rate(dec['dense']['total'])} FLOPs")
    print(f"  MoE 层（每层）              {rate(dec['moe']['total'])} FLOPs")
    print(f"    attention                 {rate(dec['moe']['attn']['total'])}")
    print(f"      q/k/v/o（只算新 token）  {rate(dec['moe']['attn']['linear'])}")
    print(f"      scores + P@V（kv_len 项）  {rate(dec['moe']['attn']['scores'] * 2)}")
    print(f"    MoE                       {rate(dec['moe']['mlp']['total'])}")
    print(f"  {d['L'] - d['first_dense']} 个 MoE 层合计          {rate((d['L'] - d['first_dense']) * dec['moe']['total'])}")
    print(f"  lm_head                     {rate(dec['lm_head'])}")
    print(f"  合计                        {rate(dec['total'])} FLOPs/token")
    print()
    print(f"  校验：2 × (激活参数 − embedding) = {rate((p['active_main'] - p['embed']) * 2)}")
    print(f"  差值 = T×T 注意力 + norm/softmax 的零头")

    section("6. 访存：权重 + KV cache + 激活")
    pre_w = bytes_prefill_weights(d, p)
    pre_w32 = bytes_prefill_weights(d, p, fp32_lm_head=True)
    dec_w = bytes_decode_weights(d, p)
    dec_w32 = bytes_decode_weights(d, p, fp32_lm_head=True)
    kv_tok = bytes_kv_per_token(d)
    kv_ctx = bytes_kv_at(d, args.context)
    act_dec = bytes_decode_activations(d, args.context)
    act_pre = bytes_prefill_activations(d, args.prefill)

    print(f"  权重总量（主干，BF16）      {byt(p['main'] * d['dtype_bytes'])}")
    print(f"  权重总量（含 MTP，BF16）    {byt(p['total'] * d['dtype_bytes'])}")
    print()
    print(f"  prefill 读权重（全部专家）  {byt(pre_w)}")
    print(f"    若 lm_head 按 fp32 读       {byt(pre_w32)}")
    print(f"  prefill 激活读写（粗估）    {byt(act_pre)}")
    print(f"  prefill 写 KV（seq_len={args.prefill}）      {byt(kv_tok * args.prefill)}")
    print()
    print(f"  decode 读权重（只读 top-{d['top_k']}）  {byt(dec_w)}")
    print(f"    若 lm_head 按 fp32 读       {byt(dec_w32)}")
    print(f"  decode 激活读写（粗估）     {byt(act_dec)}")
    print(f"  decode 读 KV（kv_len={args.context}）    {byt(kv_ctx)}")
    print(f"    KV / token / layer        {byt(kv_tok // d['L'])}")
    print(f"    KV / token（{d['L']} 层）        {byt(kv_tok)}")
    print(f"    KV 占满 {d['max_pos']} 上下文     {byt(bytes_kv_at(d, d['max_pos']))}")

    section("7. 算术强度与 roofline")
    hw = {
        "h20x8": ("8×H20-3e", 8 * 148e12, 8 * 4.0e12, 8 * 96e9),
        "h100x8": ("8×H100 SXM", 8 * 989e12, 8 * 3.35e12, 8 * 80e9),
        "m4": ("Apple M4 10-core GPU", 4.6e12, 120e9, 16e9),
    }[args.hardware]
    name, peak_flops, peak_bw, mem = hw
    print(f"  机器                        {name}")
    print(f"  峰值 BF16 算力              {rate(peak_flops)} FLOP/s")
    print(f"  峰值带宽                    {rate(peak_bw)} B/s")
    print(f"  算力/带宽                   {peak_flops / peak_bw:.1f} FLOP/B")
    print(f"  显存                        {byt(mem)}")
    print()
    for label, fl, by in [
        (f"prefill seq_len={args.prefill}", pre["total"], pre_w + act_pre + kv_tok * args.prefill),
        (f"decode kv_len={args.context}", dec["total"], dec_w + act_dec + kv_ctx),
    ]:
        ai = fl / by
        t_mem = by / peak_bw
        t_compute = fl / (peak_flops * 0.4)  # MFU 40%
        print(f"  {label:<24} FLOPs={rate(fl):>10}  Bytes={rate(by):>10}  "
              f"AI={ai:6.2f}  t_mem={t_mem * 1e3:8.3f} ms  t_compute@40%={t_compute * 1e3:8.3f} ms")
    print()
    print(f"  权重装得下吗：{byt(p['total'] * d['dtype_bytes'])} vs 显存 {byt(mem)}"
          f"  → {'够' if p['total'] * d['dtype_bytes'] < mem else '不够'}")

    section("8. 扫描：prefill 与 decode")
    print("  prefill（全部专家；激活按未融合的粗略上界）")
    print(f"  {'seq_len':>8} {'FLOPs':>12} {'每 token':>10} {'Bytes':>12} {'AI':>9} "
          f"{'t_mem':>9} {'t_compute@40%':>14}")
    for seq_len in (256, 1024, 4096, 16384, 65536, 131072, 262144):
        f = flops_prefill(d, seq_len)
        by = pre_w + bytes_prefill_activations(d, seq_len) + kv_tok * seq_len
        print(f"  {seq_len:>8} {rate(f['total']):>12} {rate(f['per_token']):>10} {rate(by):>12} "
              f"{f['total'] / by:>9.2f} {by / peak_bw * 1e3:>7.2f}ms "
              f"{f['total'] / (peak_flops * 0.4) * 1e3:>12.1f}ms")
    print()
    print("  decode 1 token（只读 top-k 专家）")
    print(f"  {'kv_len':>8} {'FLOPs':>12} {'Bytes':>12} {'AI':>9} {'t_mem':>9} "
          f"{'t_compute@40%':>14} {'tok/s 上限':>12}")
    for kv_len in (0, 1024, 8192, 32768, 65536, 131072, 262144):
        f = flops_decode(d, kv_len)
        by = dec_w + act_dec + bytes_kv_at(d, kv_len)
        print(f"  {kv_len:>8} {rate(f['total']):>12} {rate(by):>12} {f['total'] / by:>9.2f} "
              f"{by / peak_bw * 1e3:>7.2f}ms {f['total'] / (peak_flops * 0.4) * 1e3:>12.3f}ms "
              f"{peak_bw / by * 1:>10.1f}")


if __name__ == "__main__":
    main()
