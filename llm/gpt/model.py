"""GPT-2 124M（decoder-only），全流程在一个文件里。

和 `transformer/model.py` 的对应关系：

    Transformer 2017          GPT-2
    ─────────────────         ─────────────────
    encoder + decoder         只有 decoder
    cross-attention           没有
    sin/cos 位置编码          可学习位置嵌入 wpe
    post-LN                   pre-LN
    ReLU FFN                  GELU FFN
    分开的 q/k/v 投影         一个 c_attn（768 → 2304），再切成 q/k/v
    lm_head 和 embedding 分开  lm_head 和 wte 共享权重

跑法（需要 transformers 和 tiktoken）：

    python llm/gpt/model.py
    python llm/gpt/model.py --prompt "Hello, I am" --new 5
"""

import argparse
import json
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────── 形状打印
DEPTH = 0
TRACE = True


def shape(t):
    return str(tuple(t.shape)) if hasattr(t, "shape") else str(type(t).__name__)


def sp(t):
    return str(tuple(t.shape)).replace(" ", "")


def trace(name, t, note=""):
    if TRACE:
        print(f"{'  ' * DEPTH}{name:<24}{shape(t):<24}{note}")


class stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global DEPTH
        if TRACE:
            print(f"{'  ' * DEPTH}── {self.name} " + "─" * max(0, 42 - len(self.name)))
        DEPTH += 1

    def __exit__(self, *a):
        global DEPTH
        DEPTH -= 1


class silent:
    def __enter__(self):
        global TRACE
        self.prev = TRACE
        TRACE = False

    def __exit__(self, *a):
        global TRACE
        TRACE = self.prev


# ─────────────────────────────────────────────────────────── 1. tokenizer
class Tokenizer:
    """GPT-2 的 BPE，直接复用 tiktoken 的 gpt2 编码。"""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos = 50256

    def encode(self, text):
        return torch.tensor([self.enc.encode(text)], dtype=torch.long)   # (1, seq_len)

    def decode(self, ids):
        return self.enc.decode(ids)


# ─────────────────────────────────────────────────────────── 2. 模型
class CausalSelfAttention(nn.Module):
    """GPT-2 的 causal self-attention。

    Q/K/V 由一个 `c_attn` 一次算出：768 → 2304，再切成三份 768。
    好处是一次矩阵乘，坏处是读代码时要自己拆。
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_k = dim // heads
        self.c_attn = nn.Linear(dim, 3 * dim)     # (dim_model) → (3 * dim_model)
        self.c_proj = nn.Linear(dim, dim)         # (dim_model) → (dim_model)

    def forward(self, x, cache=None, pos_offset=0):
        b, q_len, d = x.shape
        qkv = self.c_attn(x)                      # (batch_size, q_len, 3 * dim_model)
        trace("c_attn", qkv, f"x{sp(x)} @ Wᵀ{sp(self.c_attn.weight)} + b{sp(self.c_attn.bias)}")

        q, k, v = qkv.split(d, dim=-1)             # 三个 (batch_size, q_len, dim_model)
        trace("q/k/v", q, "c_attn 输出按最后一维切成三份，每份 768")

        q = q.view(b, q_len, self.heads, self.dim_k).transpose(1, 2)   # (b, heads, q_len, dim_k)
        k = k.view(b, q_len, self.heads, self.dim_k).transpose(1, 2)   # (b, heads, q_len, dim_k)
        v = v.view(b, q_len, self.heads, self.dim_k).transpose(1, 2)   # (b, heads, q_len, dim_k)
        trace("拆头 q", q, f"{sp(x)} view({b},{q_len},{self.heads},{self.dim_k}).transpose(1,2) → (batch_size, num_heads, q_len, dim_k)")
        trace("拆头 k", k, "同上")
        trace("拆头 v", v, "同上")

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)      # (b, heads, past_len + q_len, dim_k)
            v = torch.cat([past_v, v], dim=2)
            trace("K cache", k, f"拼接 past {past_k.shape[2]} + new {q_len} → kv_len {k.shape[2]}")
            trace("V cache", v, f"拼接 past {past_v.shape[2]} + new {q_len} → kv_len {v.shape[2]}")
        else:
            trace("K cache", k, f"新建 cache，kv_len {k.shape[2]}")
            trace("V cache", v, f"新建 cache，kv_len {v.shape[2]}")

        kv_len = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim_k)   # (b, heads, q_len, kv_len)
        trace("scores", scores, f"q{sp(q)} @ kᵀ{sp(k.transpose(-2, -1))} / √{self.dim_k}")

        # 因果 mask：prefill 时 q_len = kv_len，遮右上角；
        # decode 时 q_len = 1，新的 query 可以看全部 cache，mask 自然为空。
        mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool), diagonal=1 + pos_offset)
        n_ban = int(mask.sum())
        if n_ban:
            scores = scores.masked_fill(mask, float("-inf"))
            trace("+ causal mask", scores, f"{q_len}×{kv_len}，遮住右上角 {n_ban} 格")
        else:
            trace("+ causal mask", scores, f"{q_len}×{kv_len}，没有未来位置可遮")

        P = scores.softmax(-1)                      # (b, heads, q_len, kv_len)
        trace("softmax → P", P, "dim=-1，每个 query 对全部 key 归一化")

        out = P @ v                                 # (b, heads, q_len, dim_k)
        trace("P @ V", out, f"Σ_k P{sp(P)}·v{sp(v)}，加权平均")
        out = out.transpose(1, 2).contiguous().view(b, q_len, d)   # (b, q_len, dim_model)
        trace("拼头", out, f"{self.heads} 个头 view({b},{q_len},{d})，无计算")
        out = self.c_proj(out)                      # (b, q_len, dim_model)
        trace("c_proj (attn)", out, f"拼接结果{sp(out)} @ Wᵀ{sp(self.c_proj.weight)} + b{sp(self.c_proj.bias)}")
        return out, (k, v)


class MLP(nn.Module):
    """GPT-2 的 FFN：768 → 3072 → 768，激活是 GELU（tanh 近似）。"""

    def __init__(self, dim, dim_ff):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim_ff)          # (dim_model) → (dim_ff)
        self.c_proj = nn.Linear(dim_ff, dim)        # (dim_ff) → (dim_model)

    def forward(self, x):
        h = self.c_fc(x)                            # (b, q_len, dim_ff)
        trace("c_fc", h, f"x{sp(x)} @ Wᵀ{sp(self.c_fc.weight)} + b{sp(self.c_fc.bias)}")
        h = F.gelu(h, approximate="tanh")           # (b, q_len, dim_ff)
        trace("gelu", h, "逐元素，形状不变（GPT-2 用 tanh 近似）")
        h = self.c_proj(h)                          # (b, q_len, dim_model)
        trace("c_proj (mlp)", h, f"h{sp(h)} @ Wᵀ{sp(self.c_proj.weight)} + b{sp(self.c_proj.bias)}")
        return h


class Block(nn.Module):
    """GPT-2 是 pre-LN：x = x + Attention(LN(x))，再 x = x + MLP(LN(x))。"""

    def __init__(self, dim, heads, dim_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim_ff)

    def forward(self, x, cache=None, pos_offset=0):
        residual = x
        h = self.ln1(x)                             # (b, q_len, dim_model)
        trace("ln1", h, "LayerNorm，形状不变")
        h, new_cache = self.attn(h, cache=cache, pos_offset=pos_offset)
        x = residual + h                            # (b, q_len, dim_model)
        trace("+ 残差", x, f"残差{sp(residual)} + 注意力输出{sp(h)}")

        residual = x
        h = self.ln2(x)
        trace("ln2", h, "LayerNorm，形状不变")
        h = self.mlp(h)
        x = residual + h
        trace("+ 残差", x, f"残差{sp(residual)} + MLP输出{sp(h)}")
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, vocab=50257, dim=768, heads=12, dim_ff=3072,
                 layers=12, max_pos=1024):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(vocab, dim)         # (vocab_size, dim_model)
        self.pos_emb = nn.Embedding(max_pos, dim)       # (max_pos, dim_model)
        self.blocks = nn.ModuleList(Block(dim, heads, dim_ff) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight       # weight tying

    def forward(self, ids, caches=None, pos_offset=0):
        b, q_len = ids.shape
        x = self.tok_emb(ids)                           # (b, q_len, dim_model)
        trace("tok_emb", x, f"ids{sp(ids)} 查表 → (batch_size, q_len, dim_model)")

        pos = torch.arange(pos_offset, pos_offset + q_len)
        p = self.pos_emb(pos)                           # (q_len, dim_model)
        trace("pos_emb", p, f"位置 {pos_offset}…{pos_offset + q_len - 1}，表共 {tuple(self.pos_emb.weight.shape)}")
        x = x + p                                       # (b, q_len, dim_model)
        trace("emb + pos", x, "逐元素相加，宽度不变")

        new_caches = []
        for i, block in enumerate(self.blocks):
            if i == 0:
                with stage("block 0（细节）"):
                    x, c = block(x, cache=caches[i] if caches else None, pos_offset=pos_offset)
            else:
                with silent():
                    x, c = block(x, cache=caches[i] if caches else None, pos_offset=pos_offset)
            new_caches.append(c)
        trace(f"block 1–{len(self.blocks) - 1}", x, "内部同 block 0，形状不变")

        x = self.ln_f(x)                                # (b, q_len, dim_model)
        trace("ln_f", x, "最后一个 LayerNorm，形状不变")
        logits = self.lm_head(x)                        # (b, q_len, vocab_size)
        trace("lm_head", logits, f"x{sp(x)} @ Wᵀ{sp(self.lm_head.weight)}（和 tok_emb 共享）")
        return logits, new_caches

    @torch.no_grad()
    def generate(self, tok, prompt, max_new=8):
        ids = tok.encode(prompt)                        # (1, prompt_len)
        trace("ids", ids, f"batch_size = {ids.shape[0]}, q_len = {ids.shape[1]}")

        print("\n[3] prefill（整段 prompt 一次算完，建 KV cache）")
        logits, caches = self.forward(ids)              # (1, prompt_len, vocab_size)
        nxt = logits[:, -1].argmax(-1, keepdim=True)    # (1, 1)
        ids = torch.cat([ids, nxt], dim=1)              # (1, prompt_len + 1)
        print(f"  prefill 后第一个新 token: {nxt.item()} {tok.decode([nxt.item()])!r}")

        print(f"\n[4] decode（每次 1 个 token，读 cache）")
        for step in range(max_new - 1):
            pos_offset = ids.shape[1] - 1                # 新 token 在序列里的位置
            trace("ids", nxt, f"decode step {step + 1}，pos_offset = {pos_offset}")
            logits, caches = self.forward(nxt, caches=caches, pos_offset=pos_offset)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            print(f"  step {step + 1}: 新 token {nxt.item()} {tok.decode([nxt.item()])!r}，"
                  f"cache 长度 {caches[0][0].shape[2]}")
        print(f"\n  生成结果：{tok.decode(ids[0].tolist())!r}")
        return ids


# ─────────────────────────────────────────────────────────── 3. 加载 config 和权重
def load_config(path):
    """读取 HF 的 config.json，映射成本文件的参数名。

    GPT-2 的 config 字段和论文符号不是一一同名：
        n_embd      → dim_model
        n_head      → num_heads
        n_layer     → num_layers
        n_positions → max_pos
        n_inner     → dim_ff（null 时按 4 × n_embd）
    """
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    dim = c["n_embd"]
    return dict(
        vocab=c["vocab_size"],
        dim=dim,
        heads=c["n_head"],
        dim_ff=c.get("n_inner") or 4 * dim,
        layers=c["n_layer"],
        max_pos=c["n_positions"],
    )


def load_pretrained(model, name="gpt2"):
    """把 HF 的 GPT-2 state_dict 映射到本文件。

    容易踩的坑：HF 的 c_attn / c_proj / c_fc 是 Conv1D，权重形状是 (in, out)；
    本文件的 nn.Linear 是 (out, in)，所以加载时要转置。
    """
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name)
    sd = hf.state_dict()
    new = {}

    new["tok_emb.weight"] = sd["transformer.wte.weight"]
    new["pos_emb.weight"] = sd["transformer.wpe.weight"]

    for i in range(len(model.blocks)):
        p = f"transformer.h.{i}."
        q = f"blocks.{i}."

        new[q + "ln1.weight"] = sd[p + "ln_1.weight"]
        new[q + "ln1.bias"] = sd[p + "ln_1.bias"]
        new[q + "ln2.weight"] = sd[p + "ln_2.weight"]
        new[q + "ln2.bias"] = sd[p + "ln_2.bias"]

        # c_attn: (768, 2304) → 转置 (2304, 768)；运行时再按最后一维切成 q/k/v
        w = sd[p + "attn.c_attn.weight"].t()
        b = sd[p + "attn.c_attn.bias"]
        new[q + "attn.c_attn.weight"] = w
        new[q + "attn.c_attn.bias"] = b
        new[q + "attn.c_proj.weight"] = sd[p + "attn.c_proj.weight"].t()
        new[q + "attn.c_proj.bias"] = sd[p + "attn.c_proj.bias"]

        new[q + "mlp.c_fc.weight"] = sd[p + "mlp.c_fc.weight"].t()
        new[q + "mlp.c_fc.bias"] = sd[p + "mlp.c_fc.bias"]
        new[q + "mlp.c_proj.weight"] = sd[p + "mlp.c_proj.weight"].t()
        new[q + "mlp.c_proj.bias"] = sd[p + "mlp.c_proj.bias"]

    new["ln_f.weight"] = sd["transformer.ln_f.weight"]
    new["ln_f.bias"] = sd["transformer.ln_f.bias"]
    new["lm_head.weight"] = sd["transformer.wte.weight"]   # tied

    model.load_state_dict(new, strict=True)
    print(f"  加载 {len(new)} 个张量；c_attn/c_proj/c_fc 已从 Conv1D 转置成 Linear")
    return model


# ─────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A long time ago")
    ap.add_argument("--new", type=int, default=4)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    args = ap.parse_args()

    print("=" * 70)
    print("[1] tokenizer")
    tok = Tokenizer()
    print(f"  GPT-2 BPE，eos = {tok.eos}")

    print("\n[2] config + 模型 + 加载权重")
    cfg = load_config(args.config)
    print(f"  config: {args.config}")
    print(f"  n_embd={cfg['dim']}  n_head={cfg['heads']}  n_layer={cfg['layers']}  "
          f"n_positions={cfg['max_pos']}  vocab={cfg['vocab']}  n_inner={cfg['dim_ff']}")
    model = GPT(**cfg)
    load_pretrained(model)
    model.eval()
    print(f"  参数 {sum(p.numel() for p in model.parameters()):,}")
    print(f"  vocab={model.tok_emb.num_embeddings}, dim={model.dim}, "
          f"blocks={len(model.blocks)}, max_pos={model.pos_emb.num_embeddings}")

    print("\n" + "=" * 70)
    print(f"prompt: {args.prompt!r}")
    model.generate(tok, args.prompt, max_new=args.new)


if __name__ == "__main__":
    main()
