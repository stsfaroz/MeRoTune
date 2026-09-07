"""
Shared helpers for merging two Qwen2.5-1.5B-Instruct fine-tunes with a
dual-sided ConstrainedM. Everything here is specific to this
architecture (28 layers, 12 query heads, 2 kv heads, head_dim=128,
q/k_proj carry a bias that rides along with the rotation since it's
added before RoPE) -- this isn't meant to generalize past Qwen2.5-1.5B.
"""
import torch
from torch.func import functional_call

N_LAYERS = 28
N_Q_HEADS = 12
N_KV_HEADS = 2
HEAD_DIM = 128
GROUP = N_Q_HEADS // N_KV_HEADS
HALF = HEAD_DIM // 2

KQ_SUFFIXES = (
    "self_attn.k_proj.weight", "self_attn.q_proj.weight",
    "self_attn.k_proj.bias", "self_attn.q_proj.bias",
)


def build_static_state_dict(sd_a, sd_b, alpha):
    """Everything except K/Q, plain-interpolated. K/Q gets the rotation
    treatment in build_kq_for_layer below."""
    out = {}
    for k in sd_a:
        if k.endswith(KQ_SUFFIXES):
            continue
        out[k] = ((1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()).to(torch.bfloat16)
    return out


def build_kq_for_layer(sd_a, sd_b, m_a, m_b, layer, alpha, device):
    """M_a rotates A's K/Q into the shared frame, M_b rotates B's --
    dual-sided, neither model is a fixed anchor. Bias rides along with
    the weight it sits next to since it's added before RoPE too."""
    p = f"model.layers.{layer}.self_attn."
    Wk_a = sd_a[p + "k_proj.weight"].to(device).float()
    Wk_b = sd_b[p + "k_proj.weight"].to(device).float()
    Wq_a = sd_a[p + "q_proj.weight"].to(device).float()
    Wq_b = sd_b[p + "q_proj.weight"].to(device).float()
    bk_a = sd_a[p + "k_proj.bias"].to(device).float()
    bk_b = sd_b[p + "k_proj.bias"].to(device).float()
    bq_a = sd_a[p + "q_proj.bias"].to(device).float()
    bq_b = sd_b[p + "q_proj.bias"].to(device).float()

    Wk_parts, Wq_parts, bk_parts, bq_parts = [], [], [], []
    for kv in range(N_KV_HEADS):
        M_a, M_b = m_a.get_M(layer, kv), m_b.get_M(layer, kv)
        Ma_inv_T = torch.linalg.inv(M_a).transpose(-1, -2)
        Mb_inv_T = torch.linalg.inv(M_b).transpose(-1, -2)

        sl = slice(kv * HEAD_DIM, (kv + 1) * HEAD_DIM)
        Wk_parts.append((1 - alpha) * (Ma_inv_T @ Wk_a[sl]) + alpha * (Mb_inv_T @ Wk_b[sl]))
        bk_parts.append((1 - alpha) * (Ma_inv_T @ bk_a[sl]) + alpha * (Mb_inv_T @ bk_b[sl]))

        for g in range(GROUP):
            qsl = slice((kv * GROUP + g) * HEAD_DIM, (kv * GROUP + g + 1) * HEAD_DIM)
            Wq_parts.append((1 - alpha) * (M_a @ Wq_a[qsl]) + alpha * (M_b @ Wq_b[qsl]))
            bq_parts.append((1 - alpha) * (M_a @ bq_a[qsl]) + alpha * (M_b @ bq_b[qsl]))

    return {
        p + "k_proj.weight": torch.cat(Wk_parts, 0).to(torch.bfloat16),
        p + "q_proj.weight": torch.cat(Wq_parts, 0).to(torch.bfloat16),
        p + "k_proj.bias": torch.cat(bk_parts, 0).to(torch.bfloat16),
        p + "q_proj.bias": torch.cat(bq_parts, 0).to(torch.bfloat16),
    }


def assemble_params(static_sd, sd_a, sd_b, m_a, m_b, alpha, device):
    params = dict(static_sd)
    for layer in range(N_LAYERS):
        params.update(build_kq_for_layer(sd_a, sd_b, m_a, m_b, layer, alpha, device))
    return params


def make_chat_batch(tok, examples, idxs, max_len=256):
    """Loss masked to the answer tokens only."""
    out = []
    for i in idxs:
        ex = examples[i % len(examples)]
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        plen = len(tok(prefix, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"])
        full = prefix + ex["answer"] + tok.eos_token
        enc = tok(full, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        out.append((enc["input_ids"], enc["attention_mask"], labels))
    return out


def forward_loss(model, params, input_ids, attn_mask, labels, device):
    out = functional_call(
        model, params, args=(), tie_weights=False,
        kwargs={"input_ids": input_ids.to(device), "attention_mask": attn_mask.to(device),
                "labels": labels.to(device)},
    )
    return out.loss


def split_train_val(examples, train_frac=0.8, seed=0):
    import random
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    cut = int(len(examples) * train_frac)
    return [examples[i] for i in idx[:cut]], [examples[i] for i in idx[cut:]]
