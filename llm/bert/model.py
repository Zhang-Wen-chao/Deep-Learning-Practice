"""自己写的 BERT（encoder-only），全流程在这个文件里。

tokenizer → 加载权重 → embedding → encoder × 12 → [CLS] / MLM 头 → 预测

和 Transformer 的 encoder 比，少了三样、多了两样：

  少了：decoder、cross-attention、causal mask（BERT 是双向的，不需要遮）
  多了：token_type embedding（区分句子 A/B）、MLM 头（预测被遮住的词）

形状记号：
    batch_size  几条句子        seq_len   一条句子几个 token
    dim_model   主干宽度 768    num_heads 12      dim_k 64
    dim_ff      3072           vocab_size 30522   max_pos 512

跑法：
    python llm/bert/model.py
"""

import argparse
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

DEPTH = 0
TRACE = True


def sp(t):
    return str(tuple(t.shape)).replace(" ", "")


def lin_note(lhs, lhs_name, wname, lin):
    wt = str(tuple(reversed(lin.weight.shape))).replace(" ", "")
    bt = str(tuple(lin.bias.shape)).replace(" ", "")
    return f"{lhs_name}{sp(lhs)} @ {wname}ᵀ{wt} + b{bt}"


def trace(name, t, note=""):
    if TRACE:
        print(f"{'  ' * DEPTH}{name:<24}{sp(t):<26}{note}")


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


# ─────────────────────────────────────────────────────────── Embedding
class BertEmbeddings(nn.Module):
    """BERT 有三种 embedding，相加：词、位置（可学习）、句子编号。"""

    def __init__(self, vocab, dim, max_pos=512, type_vocab=2):
        super().__init__()
        self.word = nn.Embedding(vocab, dim)              # (vocab_size, dim_model)
        self.pos = nn.Embedding(max_pos, dim)             # (max_pos, dim_model)
        self.seg = nn.Embedding(type_vocab, dim)          # (2, dim_model)  句子 A/B
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, ids, token_type_ids=None):
        b, s = ids.shape
        w = self.word(ids)                                 # (batch_size, seq_len, dim_model)
        trace("word_emb", w, "查表")
        p = self.pos(torch.arange(s).unsqueeze(0))         # (1, seq_len, dim_model)
        trace("pos_emb", p, "可学习，不是 sin/cos")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(ids)
        t = self.seg(token_type_ids)                       # (batch_size, seq_len, dim_model)
        trace("token_type_emb", t, "句子 A=0 / B=1")
        x = w + p + t                                      # (batch_size, seq_len, dim_model)
        trace("三者相加", x, "宽度不变")
        x = self.norm(x)                                   # (batch_size, seq_len, dim_model)
        trace("LayerNorm", x, "embedding 后还有一次 LN")
        return x


# ─────────────────────────────────────────────────────────── 注意力
class BertSelfAttention(nn.Module):
    """和 Transformer 一样，只是没有 mask（双向可见）。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.dim_k = heads, dim // heads
        self.q = nn.Linear(dim, dim)                       # (dim_model, dim_model)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, d = x.shape
        q = self.q(x)                                       # (b, s, hidden_size)
        k = self.k(x)
        v = self.v(x)
        trace("q_proj", q, lin_note(x, "x", "Wq", self.q))
        trace("k_proj", k, lin_note(x, "x", "Wk", self.k))
        trace("v_proj", v, lin_note(x, "x", "Wv", self.v))
        q = q.view(b, s, self.heads, self.dim_k).transpose(1, 2)   # (b, heads, s, dim_k)
        k = k.view(b, s, self.heads, self.dim_k).transpose(1, 2)
        v = v.view(b, s, self.heads, self.dim_k).transpose(1, 2)
        trace("拆头 q", q, f"{sp(x)} view({b},{s},{self.heads},{self.dim_k}).transpose(1,2)")
        trace("拆头 k", k, f"{sp(x)} view({b},{s},{self.heads},{self.dim_k}).transpose(1,2)")
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim_k)           # (b, heads, s, s)
        trace("scores", scores, f"q{sp(q)} @ kᵀ{sp(k.transpose(-2,-1))} / √{self.dim_k}")
        P = scores.softmax(-1)                                             # (b, heads, s, s)
        trace("softmax → P", P, "P 是概率矩阵；dim=-1，每行和=1")
        h = P @ v                                                          # (b, heads, s, dim_k)
        trace("P @ V", h, f"P{sp(P)} @ v{sp(v)}")
        out = h.transpose(1, 2).contiguous().view(b, s, d)                 # (b, s, hidden_size)
        trace("拼头", out, f"({b},{s},{self.heads},{self.dim_k}) view({b},{s},{d})")
        return out


class BertAttention(nn.Module):
    """self-attention + 输出投影 + 残差 + LN（post-LN，和论文一致）。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.attn = BertSelfAttention(dim, heads)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x):
        residual = x                                       # (b, s, dim_model)
        h = self.attn(x)
        h = self.out(h)                                    # (b, s, hidden_size)
        trace("out_dense", h, lin_note(h, "x", "Wo", self.out))
        x = self.norm(residual + h)                        # (b, s, dim_model)
        trace("+ 残差 + LN", x, f"残差{sp(residual)} + 注意力输出{sp(h)} → LN")
        return x


class BertLayer(nn.Module):
    def __init__(self, dim, heads, dim_ff):
        super().__init__()
        self.attention = BertAttention(dim, heads)
        self.inter = nn.Linear(dim, dim_ff)                # (dim_ff, dim_model)
        self.out = nn.Linear(dim_ff, dim)                  # (dim_model, dim_ff)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x):
        x = self.attention(x)                              # (b, s, dim_model)
        residual = x
        h = self.inter(x)                                  # (b, s, intermediate_size)
        trace("intermediate", h, lin_note(x, "x", "W1", self.inter))
        h = F.gelu(h)                                      # (b, s, intermediate_size)
        trace("gelu", h, "逐元素，形状不变")
        h = self.out(h)                                    # (b, s, hidden_size)
        trace("output.dense", h, lin_note(h, "h", "W2", self.out))
        x = self.norm(residual + h)                        # (b, s, hidden_size)
        trace("+ 残差 + LN", x, f"残差{sp(residual)} + FFN输出{sp(h)} → LN")
        return x


# ─────────────────────────────────────────────────────────── 模型
class BertModel(nn.Module):
    def __init__(self, vocab, dim=768, heads=12, dim_ff=3072, layers=12):
        super().__init__()
        self.emb = BertEmbeddings(vocab, dim)
        self.layers = nn.ModuleList(BertLayer(dim, heads, dim_ff) for _ in range(layers))
        self.pooler = nn.Linear(dim, dim)                  # (dim_model, dim_model)

    def forward(self, ids, token_type_ids=None):
        with stage("embedding"):
            x = self.emb(ids, token_type_ids)              # (b, s, dim_model)
        for i, layer in enumerate(self.layers):
            if i == 0:
                with stage("layer 0（细节）"):
                    x = layer(x)
            else:
                global_trace = TRACE
                with silent():
                    x = layer(x)
        trace(f"layer 1–{len(self.layers) - 1}", x, "内部同第 0 层")
        # pooler 只取 [CLS]（第一个位置）
        cls = x[:, 0]                                      # (b, dim_model)
        trace("[CLS]", cls, "取第 0 个位置")
        pooled = torch.tanh(self.pooler(cls))              # (b, dim_model)
        trace("pooler", pooled, "分类/句向量用这个")
        return x, pooled


class BertForMaskedLM(nn.Module):
    """MLM 头：768 → 768 → vocab_size，预测被 [MASK] 盖住的词。"""

    def __init__(self, vocab, dim=768, heads=12, dim_ff=3072, layers=12):
        super().__init__()
        self.bert = BertModel(vocab, dim, heads, dim_ff, layers)
        self.transform = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.to_vocab = nn.Linear(dim, vocab, bias=True)   # (vocab_size, hidden_size)
        self.to_vocab.weight = self.bert.emb.word.weight   # weight tying

    def forward(self, ids, token_type_ids=None):
        seq, pooled = self.bert(ids, token_type_ids)
        with stage("MLM 头"):
            h = self.transform(seq)                        # (b, s, hidden_size)
            trace("transform", h, lin_note(seq, "x", "W", self.transform))
            h = F.gelu(h)                                  # (b, s, hidden_size)
            trace("gelu", h, "逐元素")
            h = self.norm(h)                               # (b, s, hidden_size)
            trace("LayerNorm", h)
            logits = self.to_vocab(h)                      # (b, s, vocab_size)
            trace("to_vocab", logits, lin_note(h, "h", "W", self.to_vocab) + "  权重和 word_emb 共享")
        return logits, pooled


class silent:
    def __enter__(self):
        global TRACE
        self.prev = TRACE
        TRACE = False

    def __exit__(self, *a):
        global TRACE
        TRACE = self.prev


# ─────────────────────────────────────────────────────────── 加载权重
def load_pretrained(model, name="bert-base-uncased"):
    from transformers import BertForPreTraining as HF

    hf = HF.from_pretrained(name)
    sd = hf.state_dict()
    new = {}
    for k, v in sd.items():
        k2 = k
        k2 = k2.replace("bert.embeddings.word_embeddings", "bert.emb.word")
        k2 = k2.replace("bert.embeddings.position_embeddings", "bert.emb.pos")
        k2 = k2.replace("bert.embeddings.token_type_embeddings", "bert.emb.seg")
        k2 = k2.replace("bert.embeddings.LayerNorm", "bert.emb.norm")
        k2 = k2.replace("bert.encoder.layer.", "bert.layers.")
        k2 = k2.replace(".attention.self.query", ".attention.attn.q")
        k2 = k2.replace(".attention.self.key", ".attention.attn.k")
        k2 = k2.replace(".attention.self.value", ".attention.attn.v")
        k2 = k2.replace(".attention.output.dense", ".attention.out")
        k2 = k2.replace(".attention.output.LayerNorm", ".attention.norm")
        k2 = k2.replace(".intermediate.dense", ".inter")
        k2 = k2.replace(".output.dense", ".out")
        k2 = k2.replace(".output.LayerNorm", ".norm")
        k2 = k2.replace("bert.pooler.dense", "bert.pooler")
        k2 = k2.replace("cls.predictions.transform.dense", "transform")
        k2 = k2.replace("cls.predictions.transform.LayerNorm", "norm")
        k2 = k2.replace("cls.predictions.decoder", "to_vocab")
        if k2.startswith("cls.predictions.bias"):
            k2 = "to_vocab.bias"
        new[k2] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    print(f"  加载 {len(new)} 个张量")
    if missing:
        print(f"  缺失: {missing[:5]}")
    if unexpected:
        print(f"  多余: {unexpected[:5]}")
    return model


# ─────────────────────────────────────────────────────────── demo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The capital of France is [MASK].")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"model: bert-base-uncased")
    model = BertForMaskedLM(vocab=30522)
    load_pretrained(model)
    model.eval()
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n[1] tokenizer")
    enc = tok(args.text, return_tensors="pt")
    ids = enc["input_ids"]
    print(f"  text      {args.text!r}")
    print(f"  ids       {ids.tolist()[0][:10]}...")
    print(f"  形状      {sp(ids)}  seq_len = {ids.shape[1]}")

    print(f"\n[2] 前向")
    with torch.no_grad():
        logits, pooled = model(ids, enc.get("token_type_ids"))

    print(f"\n[3] 预测 [MASK]")
    pos = (ids[0] == tok.mask_token_id).nonzero()[0].item()
    probs = logits[0, pos].softmax(-1)
    top = probs.topk(5)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {p:6.2%}  {tok.decode([i])!r}")

    print(f"\n[4] [CLS] 向量")
    print(f"  pooled 形状 {sp(pooled)}  ← 分类头就用这个")


if __name__ == "__main__":
    main()
