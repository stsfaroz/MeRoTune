#!/usr/bin/env python3
"""
Trains a dual-sided ConstrainedM for two Qwen2.5-1.5B-Instruct
fine-tunes. Both models get their own learned rotation (M_a, M_b) --
neither is a fixed anchor. By default the blend ratio (alpha) is fixed
at 0.5 for the whole run, the same way LoRA fixes its own scaling
hyperparameter before training rather than varying it -- pass
--alpha-min 0.05 --alpha-max 0.95 to resample alpha every step instead
(the randomized variant); both were tested and both work as a dial
after training, see the paper for the comparison.

Loss is unweighted retention on both tasks at whatever alpha is in
play that step: loss = loss_a + loss_b. Weighting the loss by alpha
would let the optimizer stop caring about model B whenever alpha
happened to favor A that step, which defeats the point of a dial you
can move later.

Example:
    python scripts/train.py \\
        --model-a arissuga/aurum-brain-ai --data-a data/task_a.json \\
        --model-b SakanaAI/TinySwallow-1.5B-Instruct --data-b data/task_b.json \\
        --out checkpoints/
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merotune import (
    ConstrainedM,
    HALF, N_KV_HEADS, N_LAYERS,
    assemble_params, build_static_state_dict, forward_loss,
    make_chat_batch, split_train_val,
)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    for tok in (tok_a, tok_b):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    task_a = json.load(open(args.data_a))
    task_b = json.load(open(args.data_b))
    train_a, _ = split_train_val(task_a)
    train_b, _ = split_train_val(task_b)
    print(f"train_a={len(train_a)}  train_b={len(train_b)}")

    # model A doubles as the live source of sdA
    model = AutoModelForCausalLM.from_pretrained(args.model_a, dtype=torch.bfloat16).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    sd_a = dict(model.state_dict(keep_vars=True))

    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, dtype=torch.bfloat16).to(device)
    sd_b = dict(model_b.state_dict())

    static_sd = build_static_state_dict(sd_a, sd_b, alpha=0.5)
    torch.cuda.empty_cache()

    m_a = ConstrainedM(N_LAYERS, N_KV_HEADS, HALF).to(device)
    m_b = ConstrainedM(N_LAYERS, N_KV_HEADS, HALF).to(device)
    opt = torch.optim.AdamW(list(m_a.parameters()) + list(m_b.parameters()), lr=args.lr)

    rng = torch.Generator().manual_seed(args.seed)
    py_rng = random.Random(args.seed + 1)
    log = []
    t0 = time.time()

    for step in range(args.steps):
        alpha = py_rng.uniform(args.alpha_min, args.alpha_max)
        opt.zero_grad()

        idx_a = torch.randint(0, len(train_a), (args.batch,), generator=rng).tolist()
        idx_b = torch.randint(0, len(train_b), (args.batch,), generator=rng).tolist()
        batch_a = make_chat_batch(tok_a, train_a, idx_a)
        batch_b = make_chat_batch(tok_b, train_b, idx_b)
        n_total = len(batch_a) + len(batch_b)

        # one example at a time, backward immediately, no retained graph
        # -- keeps peak memory at one example's footprint
        loss_a_sum = loss_b_sum = 0.0
        for input_ids, attn_mask, labels in batch_a:
            params = assemble_params(static_sd, sd_a, sd_b, m_a, m_b, alpha, device)
            loss = forward_loss(model, params, input_ids, attn_mask, labels, device) / n_total
            loss.backward()
            loss_a_sum += loss.item() * n_total
            del params, loss
            torch.cuda.empty_cache()
        for input_ids, attn_mask, labels in batch_b:
            params = assemble_params(static_sd, sd_a, sd_b, m_a, m_b, alpha, device)
            loss = forward_loss(model, params, input_ids, attn_mask, labels, device) / n_total
            loss.backward()
            loss_b_sum += loss.item() * n_total
            del params, loss
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(list(m_a.parameters()) + list(m_b.parameters()), max_norm=1.0)
        opt.step()

        loss_a, loss_b = loss_a_sum / len(batch_a), loss_b_sum / len(batch_b)
        log.append({"step": step, "alpha": alpha, "loss_a": loss_a, "loss_b": loss_b})

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step:4d}/{args.steps}  alpha={alpha:.3f}  "
                  f"loss_a={loss_a:.4f}  loss_b={loss_b:.4f}  ({time.time() - t0:.0f}s)")

        if step % args.ckpt_every == 0 or step == args.steps - 1:
            torch.save(m_a.state_dict(), os.path.join(args.out, "M_a.pt"))
            torch.save(m_b.state_dict(), os.path.join(args.out, "M_b.pt"))
            json.dump(log, open(os.path.join(args.out, "training_log.json"), "w"))

    print(f"done -- checkpoints in {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", required=True, help="HF repo id or local path, fine-tune A")
    p.add_argument("--model-b", required=True, help="HF repo id or local path, fine-tune B")
    p.add_argument("--data-a", required=True, help="JSON list of {prompt, answer}, A's specialty")
    p.add_argument("--data-b", required=True, help="JSON list of {prompt, answer}, B's specialty")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=2, help="examples per task per step")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--alpha-min", type=float, default=0.5,
                    help="default fixes alpha at one point (0.5), the same way LoRA "
                         "fixes its own scaling hyperparameter before training; "
                         "set --alpha-min 0.05 --alpha-max 0.95 for the randomized variant")
    p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
