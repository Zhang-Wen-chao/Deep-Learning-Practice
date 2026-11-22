"""一个自己写的 Transformer（encoder-decoder），全流程都在这个文件里。

tokenizer → 加载权重 → 加载数据 → encoder → decoder → 训练 → 推理

每一行产生张量的代码后面都注了形状，注释里的符号含义：

    batch_size   一次几条句子
    src_len      源句 token 数
    tgt_len      目标句 token 数
    q_len        Q 的长度（query 有几个位置）
    kv_len       K 和 V 的长度（两者必然相等：由同一个输入投影出来，
                 并且 attn @ V 要求它对得上 scores 的最后一维）
    dim_model    主干宽度（512）
    num_heads    头数（8）
    dim_k        每头维度 = dim_model / num_heads（64）
    dim_ff       FFN 中间宽度（2048）
    vocab_size   词表大小（65001）

跑法：
    python llm/transformer/model.py                 # 完整流程
    python llm/transformer/model.py --no-train      # 跳过训练
"""

import argparse
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────── 形状打印
DEPTH = 0
TRACE = True


def shape(t):
    return str(tuple(t.shape)) if hasattr(t, "shape") else str(type(t).__name__)


def trace(name, t, note=""):
    if TRACE:
        print(f"{'  ' * DEPTH}{name:<24}{shape(t):<24}{note}")


def sp(t):
    """张量 → (1,5,512)，紧凑写法。"""
    return str(tuple(t.shape)).replace(" ", "")


def lin_note(lhs, lhs_name, wname, lin):
    """生成 `x(1,5,512) @ Wqᵀ(512,512) + b(512)` 这种注释。"""
    wt = str(tuple(reversed(lin.weight.shape))).replace(" ", "")
    bt = str(tuple(lin.bias.shape)).replace(" ", "")
    return f"{lhs_name}{sp(lhs)} @ {wname}ᵀ{wt} + b{bt}"


class silent:
    """with silent(): 里面的 trace 不打印（用来跳过重复的层）。"""

    def __enter__(self):
        global TRACE
        self.prev = TRACE
        TRACE = False

    def __exit__(self, *a):
        global TRACE
        TRACE = self.prev


class stage:
    """with stage('layer 0'): 缩进一层，并打一个分隔。"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global DEPTH
        if TRACE:
            print(f"{'  ' * DEPTH}── {self.name} " + "─" * max(0, 46 - len(self.name)))
        DEPTH += 1

    def __exit__(self, *a):
        global DEPTH
        DEPTH -= 1


# ─────────────────────────────────────────────────────────── 1. tokenizer
class Tokenizer:
    """用现成的 sentencepiece 词表，不自己训 BPE。"""

    def __init__(self, name="Helsinki-NLP/opus-mt-en-zh"):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.pad = self.tok.pad_token_id      # 65000
        self.eos = self.tok.eos_token_id      # 0
        self.start = self.pad                 # 解码从 pad 开始，Marian 的约定

    def encode(self, text):
        return self.tok(text, return_tensors="pt").input_ids   # (batch_size, src_len)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────── 2. 模型
class Attention(nn.Module):
    """四个 Linear + 五步张量运算，就是注意力的全部。

    自注意力：kv_input 不传，K/V 和 Q 同源，q_len = kv_len
    交叉注意力：kv_input 传 encoder 输出，q_len = tgt_len，kv_len = src_len
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_k = dim // heads
        self.q_proj = nn.Linear(dim, dim)     # (dim_model) → (dim_model)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, kv_input=None, mask=None):
        b, q_len, d = x.shape                 # x: (batch_size, q_len, dim_model)
        is_self = kv_input is None
        kv_input = x if is_self else kv_input             # (batch_size, kv_len, dim_model)
        kv_len = kv_input.shape[1]          # 取输入的第 1 维；Linear 不改它，只改最后一维
        kv_name = "x" if is_self else "enc"               # 自注意力: x 自己；交叉注意力: encoder 输出

        # Q 永远来自 x（当前正在算的这一摞）；K/V 来自 kv_input：
        #     自注意力    kv_input = x      → Q/K/V 同源，kv_len = q_len
        #     交叉注意力  kv_input = enc    → Q 来自 decoder，K/V 来自 encoder，kv_len = src_len
        #
        # 三个投影都是方阵，只换一组基，不改宽度：
        #     Q = x @ Wqᵀ + bq    (batch_size, q_len,  dim_model) @ (dim_model, dim_model)
        #     K = 源 @ Wkᵀ + bk   (batch_size, kv_len, dim_model) @ (dim_model, dim_model)
        #     V = 源 @ Wvᵀ + bv   (batch_size, kv_len, dim_model) @ (dim_model, dim_model)
        q = self.q_proj(x)                    # (batch_size, q_len,  dim_model)   Q ← x
        k = self.k_proj(kv_input)             # (batch_size, kv_len, dim_model)   K ← x 或 enc
        v = self.v_proj(kv_input)             # (batch_size, kv_len, dim_model)   V ← x 或 enc
        trace("q_proj", q, "Q← " + lin_note(x, "x", "Wq", self.q_proj))
        trace("k_proj", k, "K← " + lin_note(kv_input, kv_name, "Wk", self.k_proj))
        trace("v_proj", v, "V← " + lin_note(kv_input, kv_name, "Wv", self.v_proj))

        # 拆头：把 dim_model 平均切成 num_heads 份，每份 dim_k
        q = q.view(b, q_len, self.heads, self.dim_k).transpose(1, 2)      # (batch_size, num_heads, q_len, dim_k)
        k = k.view(b, kv_len, self.heads, self.dim_k).transpose(1, 2)     # (batch_size, num_heads, kv_len, dim_k)
        v = v.view(b, kv_len, self.heads, self.dim_k).transpose(1, 2)     # (batch_size, num_heads, kv_len, dim_k)
        trace("拆头 q", q, f"{sp(x)} view({b},{q_len},{self.heads},{self.dim_k}).transpose(1,2) → (batch_size, num_heads, q_len, dim_k)")
        trace("拆头 k", k, f"{sp(kv_input)} view({b},{kv_len},{self.heads},{self.dim_k}).transpose(1,2) → (batch_size, num_heads, kv_len, dim_k)")

        # 打分：唯一的 Tq × Tk
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim_k)    # (batch_size, num_heads, q_len, kv_len)
        trace("scores", scores, f"q{sp(q)} @ kᵀ{sp(k.transpose(-2, -1))} / √{self.dim_k}")
        if mask is not None:
            scores = scores.masked_fill(mask[:q_len, :kv_len], float("-inf"))   # (batch_size, num_heads, q_len, kv_len)
            trace("+ mask", scores, "右上角填 -inf（遮住）")
        attn = scores.softmax(-1)                                   # 在最后一维(kv_len)归一化，每个 query 的权重和=1
        #                                                            (batch_size, num_heads, q_len, kv_len)
        trace("softmax", attn, "= attn 权重；dim=-1，每个 query 对全部 key 归一化")

        # 加权求和：out[q] = Σ_k attn[q,k] · v[k]，中间那维 kv_len 被求和掉
        out = attn @ v                                              # (batch_size, num_heads, q_len, dim_k)
        trace("attn @ V", out, f"Σ_k attn{sp(attn)}·v{sp(v)}，加权平均")
        out = out.transpose(1, 2).contiguous().view(b, q_len, d)     # (batch_size, q_len, dim_model)
        trace("拼头", out, f"{self.heads} 个头的输出 view({b},{q_len},{d})，无计算 → 该位置的上下文向量")
        out = self.out_proj(out)                                    # (batch_size, q_len, dim_model)
        trace("out_proj", out, lin_note(out, "拼接结果", "Wo", self.out_proj))
        # 注意：残差和 LayerNorm 不在这里做。Attention 只负责算注意力，
        # 残差 + LN 由外层的 EncoderLayer / DecoderLayer 负责（见下面的代码）。
        return out


class EncoderLayer(nn.Module):
    """post-LN，和 2017 论文一致：x = LN(x + 子层(x))。

    每个子层的输入输出都是 (batch_size, src_len, dim_model)。
    """

    def __init__(self, dim, heads, dim_ff):
        super().__init__()
        self.self_attn = Attention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)   # 512 维，可学 γ/β
        self.fc1 = nn.Linear(dim, dim_ff)     # (dim_model) → (dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)     # (dim_ff) → (dim_model)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        residual = x                                                # (batch_size, src_len, dim_model)

        with stage("① 自注意力（双向，不遮）"):
            x = self.self_attn(x, mask=mask)                            # (batch_size, src_len, dim_model)
        x = self.self_attn_layer_norm(residual + x)                 # (batch_size, src_len, dim_model)
        trace("+ 残差 + LayerNorm", x, f"残差{sp(residual)} + 子层输出{sp(x)} → LayerNorm")

        residual = x                                                # (batch_size, src_len, dim_model)
        with stage("② FFN"):
            h = self.fc1(x)                                         # (batch_size, src_len, dim_ff)
            trace("fc1", h, lin_note(x, "x", "W1", self.fc1))
            h = F.silu(h)                                           # (batch_size, src_len, dim_ff)
            trace("silu", h, "逐元素，形状不变")
            x = self.fc2(h)                                         # (batch_size, src_len, dim_model)
            trace("fc2", x, lin_note(h, "h", "W2", self.fc2))
        x = self.final_layer_norm(residual + x)                     # (batch_size, src_len, dim_model)
        trace("+ 残差 + LayerNorm", x, f"残差{sp(residual)} + FFN输出{sp(x)} → LayerNorm")
        return x


class DecoderLayer(nn.Module):
    """比 encoder 多一组 encoder_attn：Q 来自 decoder，K/V 来自 encoder。"""

    def __init__(self, dim, heads, dim_ff):
        super().__init__()
        self.self_attn = Attention(dim, heads)             # q_len = kv_len = tgt_len
        self.self_attn_layer_norm = nn.LayerNorm(dim)   # 512 维，可学 γ/β
        self.encoder_attn = Attention(dim, heads)          # q_len = tgt_len, kv_len = src_len  ★
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, y, enc, causal_mask=None, enc_mask=None):
        # y:   (batch_size, tgt_len, dim_model)
        # enc: (batch_size, src_len, dim_model)
        residual = y                                                # (batch_size, tgt_len, dim_model)

        with stage("① 自注意力（遮住右上角）"):
            y = self.self_attn(y, mask=causal_mask)                     # (batch_size, tgt_len, dim_model)
        y = self.self_attn_layer_norm(residual + y)                 # (batch_size, tgt_len, dim_model)
        trace("+ 残差 + LayerNorm", y, f"残差{sp(residual)} + 自注意力输出{sp(y)} → LayerNorm")

        residual = y                                                # (batch_size, tgt_len, dim_model)
        with stage("② 交叉注意力（K/V ← encoder）"):
            y = self.encoder_attn(y, kv_input=enc, mask=enc_mask)      # (batch_size, tgt_len, dim_model)
        y = self.encoder_attn_layer_norm(residual + y)              # (batch_size, tgt_len, dim_model)
        trace("+ 残差 + LayerNorm", y, f"残差{sp(residual)} + 交叉注意力输出{sp(y)} → LayerNorm")

        residual = y                                                # (batch_size, tgt_len, dim_model)
        with stage("③ FFN"):
            h = self.fc1(y)                                         # (batch_size, tgt_len, dim_ff)
            trace("fc1", h, lin_note(y, "y", "W1", self.fc1))
            h = F.silu(h)                                           # (batch_size, tgt_len, dim_ff)
            trace("silu", h, "逐元素，形状不变")
            y = self.fc2(h)                                         # (batch_size, tgt_len, dim_model)
            trace("fc2", y, lin_note(h, "h", "W2", self.fc2))
        y = self.final_layer_norm(residual + y)                     # (batch_size, tgt_len, dim_model)
        trace("+ 残差 + LayerNorm", y, f"残差{sp(residual)} + FFN输出{sp(y)} → LayerNorm")
        return y


class Transformer(nn.Module):
    def __init__(self, vocab, dim=512, heads=8, dim_ff=2048,
                 enc_layers=6, dec_layers=6, max_pos=512, pad=65000):
        super().__init__()
        self.dim = dim
        self.pad = pad
        self.tok_emb = nn.Embedding(vocab, dim, padding_idx=pad)    # (vocab_size, dim_model)
        self.register_buffer("pos_emb", torch.zeros(max_pos, dim))  # (max_pos, dim_model)
        self.enc_layers = nn.ModuleList(EncoderLayer(dim, heads, dim_ff) for _ in range(enc_layers))
        self.enc_norm = nn.Identity()      # 这个 checkpoint 没有末尾 LN
        self.dec_layers = nn.ModuleList(DecoderLayer(dim, heads, dim_ff) for _ in range(dec_layers))
        self.dec_norm = nn.Identity()
        self.lm_head = nn.Linear(dim, vocab, bias=False)            # (dim_model) → (vocab_size)
        self.lm_head.weight = self.tok_emb.weight                   # weight tying，同一个矩阵
        self.final_logits_bias = nn.Parameter(torch.zeros(1, vocab))  # (1, vocab_size)

    # ---- encoder ----
    def encode(self, ids):
        # ids: (batch_size, src_len)
        with stage("encoder"):
            # ×√d：位置编码的范数固定在 √(d/2)≈16，而训练后的 embedding 范数只有 ~1，
            # 不缩放会被位置信息淹没。乘 √d 后两边同量级（实测 21 : 16）。
            x = self.tok_emb(ids) * math.sqrt(self.dim)             # (batch_size, src_len, dim_model)
            trace("tok_emb ×√d", x, "id → 向量，再乘 √dim")
            # 全表 (max_pos, dim_model) = (512, 512)；这句只有 src_len 个位置，切前 src_len 行
            pos = self.pos_emb[: ids.shape[1]]                      # (src_len, dim_model)
            trace("pos_emb 切片", pos, f"查表[:{ids.shape[1]}]，表共 {tuple(self.pos_emb.shape)}")
            x = x + pos                                             # (batch_size, src_len, dim_model)
            trace("emb + pos", x, "逐元素相加，宽度不变")
            for i, layer in enumerate(self.enc_layers):
                if i == 0:
                    with stage("layer 0（细节）"):
                        x = layer(x)                                # (batch_size, src_len, dim_model)
                else:
                    with silent():                                  # 1~N-1 形状完全一样，不重复打印
                        x = layer(x)
            trace(f"layer 1–{len(self.enc_layers) - 1}", x, "内部同第 0 层，形状不变")
            x = self.enc_norm(x)                                    # (batch_size, src_len, dim_model)
            trace("enc_norm", x, "Identity，不做事：本 checkpoint 没有末尾 LayerNorm")
            return x

    # ---- decoder ----
    def decode(self, tgt_ids, enc):
        # tgt_ids: (batch_size, tgt_len)   enc: (batch_size, src_len, dim_model)
        with stage("decoder"):
            y = self.tok_emb(tgt_ids) * math.sqrt(self.dim)         # (batch_size, tgt_len, dim_model)
            trace("tok_emb ×√d", y)
            y = y + self.pos_emb[: tgt_ids.shape[1]]                # (batch_size, tgt_len, dim_model)
            trace("emb + pos", y, "逐元素相加，宽度不变")

            t = tgt_ids.shape[1]
            causal = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)   # (tgt_len, tgt_len)
            n_ban = int(causal.sum())     # True = 遮住
            trace("causal mask", causal,
                  f"{t}×{t}，遮住右上角，共 {n_ban} 格"
                  + ("（只有 1 个 token，没有未来可禁）" if t == 1 else ""))
            for i, layer in enumerate(self.dec_layers):
                if i == 0:
                    with stage("layer 0（细节）"):
                        y = layer(y, enc, causal_mask=causal)       # (batch_size, tgt_len, dim_model)
                else:
                    with silent():
                        y = layer(y, enc, causal_mask=causal)
            trace(f"layer 1–{len(self.dec_layers) - 1}", y, "内部同第 0 层，形状不变")
            y = self.dec_norm(y)                                    # (batch_size, tgt_len, dim_model)
            trace("dec_norm", y, "Identity，不做事：本 checkpoint 没有末尾 LayerNorm")
            return y

    def logits(self, y):
        # y: (batch_size, tgt_len, dim_model)
        with stage("输出头"):
            z = self.lm_head(y)                                     # (batch_size, tgt_len, vocab_size)
            wt = str(tuple(reversed(self.lm_head.weight.shape))).replace(" ", "")
            trace("lm_head", z, f"y{sp(y)} @ Wᵀ{wt}")
            z = z + self.final_logits_bias                          # (batch_size, tgt_len, vocab_size)
            trace("+ final_logits_bias", z, "逐元素相加")
            return z

    def forward(self, src, tgt_in):
        # src: (batch_size, src_len)   tgt_in: (batch_size, tgt_len)
        enc = self.encode(src)                                      # (batch_size, src_len, dim_model)
        y = self.decode(tgt_in, enc)                                # (batch_size, tgt_len, dim_model)
        return self.logits(y)                                       # (batch_size, tgt_len, vocab_size)


# ─────────────────────────────────────────────────────────── 3. 加载权重
def load_pretrained(model, name="Helsinki-NLP/opus-mt-en-zh"):
    """把 HF 的 state_dict 按名字映射到本文件的模块。"""
    from transformers import AutoModelForSeq2SeqLM

    hf = AutoModelForSeq2SeqLM.from_pretrained(name)
    sd, new, skipped = hf.state_dict(), {}, []
    for k, v in sd.items():
        if k == "model.shared.weight":                       # (vocab_size, dim_model)
            new["tok_emb.weight"] = v
        elif k == "model.encoder.embed_tokens.weight":       # 和 shared 同一份，跳过
            skipped.append(k)
        elif k == "model.decoder.embed_tokens.weight":
            skipped.append(k)
        elif k == "model.encoder.embed_positions.weight":    # (max_pos, dim_model)
            new["pos_emb"] = v
        elif k == "model.decoder.embed_positions.weight":
            skipped.append(k)
        elif k.startswith("model.encoder.layers."):
            new["enc_layers." + k[len("model.encoder.layers."):]] = v
        elif k.startswith("model.decoder.layers."):
            new["dec_layers." + k[len("model.decoder.layers."):]] = v
        elif k == "final_logits_bias":                       # (1, vocab_size)
            new["final_logits_bias"] = v
        elif k.startswith("lm_head"):                        # (vocab_size, dim_model)
            new["lm_head.weight"] = v                        # 和 tok_emb 是同一个矩阵
        else:
            skipped.append(k)
    model.load_state_dict(new, strict=True)
    print(f"  加载 {len(new)} 个张量，跳过 {len(set(skipped))} 个重复/绑定项")
    return model


# ─────────────────────────────────────────────────────────── 4. 数据集
def load_pairs(path="llm/transformer/data/pairs.tsv"):
    """每行 `英文<TAB>中文`。文件不存在就用内置的小样本。"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip() and not ln.startswith("#")]
        return [(a, b) for a, b in rows if a and b]
    return [
        ("I love you", "我爱你"),
        ("Good morning", "早上好"),
        ("Thank you very much", "非常感谢你"),
        ("What is your name", "你叫什么名字"),
        ("I am a student", "我是一名学生"),
        ("This is a book", "这是一本书"),
        ("He is my friend", "他是我的朋友"),
        ("Where are you from", "你来自哪里"),
        ("See you tomorrow", "明天见"),
        ("How much is it", "多少钱"),
    ]


# ─────────────────────────────────────────────────────────── 5. 训练
def train(model, tok, pairs, steps=3, lr=1e-5, batch=4):
    """演示用：几个 step。形状和推理走同一段代码，所以训练时关掉 trace。"""
    global TRACE
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    TRACE = False
    print(f"\n训练 {steps} 步，batch={batch}，lr={lr}（训练时 shape 和推理相同，略）")
    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for en, zh in pairs[:batch]:
            src = tok.tok(en, return_tensors="pt").input_ids      # (1, src_len)
            tgt = tok.tok(zh, return_tensors="pt").input_ids      # (1, tgt_len + 1)
            logits = model(src, tgt[:, :-1])                       # (1, tgt_len, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),             # (tgt_len, vocab_size)
                tgt[:, 1:].reshape(-1),                           # (tgt_len,)
            ) / batch
            loss.backward()
            total += loss.item()
        opt.step()
        print(f"  step {step}: loss = {total:.4f}")
    TRACE = True
    model.eval()


# ─────────────────────────────────────────────────────────── 6. 推理
@torch.no_grad()
def greedy(model, tok, text, max_new=16):
    src = tok.tok(text, return_tensors="pt").input_ids             # (1, src_len)
    trace("ids", src, f"batch_size = {src.shape[0]}, src_len = {src.shape[1]}")
    enc = model.encode(src)                                        # (1, src_len, dim_model)
    trace("enc_out", enc, "只算一次")
    ids = torch.tensor([[tok.tok.pad_token_id]])                   # (1, 1)
    trace("decoder 起点", ids, f"id={tok.tok.pad_token_id}（Marian 用 pad 当起始符）")
    for step in range(max_new):
        logits = model.logits(model.decode(ids, enc))              # (1, tgt_len, vocab_size)
        nxt = logits[:, -1].argmax(-1, keepdim=True)               # (1, 1)
        ids = torch.cat([ids, nxt], dim=1)                         # (1, tgt_len + 1)
        print(f"  step {step}: 输入 tgt_len={ids.shape[1] - 1} → 新 token {nxt.item()} "
              f"{tok.decode([nxt.item()])!r}")
        if nxt.item() == tok.eos:
            break
    return tok.decode(ids[0, 1:])


# ─────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="A long time ago")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--train-steps", type=int, default=3)
    args = ap.parse_args()

    print("=" * 70)
    print("[1] tokenizer")
    tok = Tokenizer()

    print("\n[2] 模型 + 加载权重")
    model = Transformer(vocab=65001)
    load_pretrained(model)
    model.eval()
    print(f"  位置编码表 pos_emb（全表）: {tuple(model.pos_emb.shape)}"
          f" = max_pos {model.pos_emb.shape[0]} 行，每行 {model.pos_emb.shape[1]} 维")
    print(f"  参数 {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3] 数据集")
    pairs = load_pairs()
    print(f"  {len(pairs)} 句对，例：{pairs[0]}")

    if not args.no_train:
        print("\n[4] 训练")
        train(model, tok, pairs, steps=args.train_steps)

    print("\n[5] 推理")
    out = greedy(model, tok, args.text)
    print(f"\n  译文：{out!r}")


if __name__ == "__main__":
    main()
