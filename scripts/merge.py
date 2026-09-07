#!/usr/bin/env python3
"""
Builds a merged model at a chosen alpha (the fraction of model B in the
blend) from a trained ConstrainedM pair, and saves it as a normal HF
model directory -- load it afterward with AutoModelForCausalLM like
anything else.

alpha=0.0 is pure A, alpha=1.0 is pure B, alpha=0.5 is a balanced blend.
Note this is NOT the same as plain averaging even at alpha=0.5: K/Q
still gets rotated through M_a/M_b first, only everything outside
attention is a plain interpolation.

Example:
    python scripts/merge.py --model-a arissuga/aurum-brain-ai \\
        --model-b SakanaAI/TinySwallow-1.5B-Instruct \\
        --checkpoints checkpoints/ --alpha 0.1 --out merged-alpha0.1/
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merotune import ConstrainedM, HALF, N_KV_HEADS, N_LAYERS, build_kq_for_layer, build_static_state_dict


def merge(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, dtype=torch.bfloat16, device_map="cpu")
    sd_a = dict(model_a.state_dict())
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, dtype=torch.bfloat16, device_map="cpu")
    sd_b = dict(model_b.state_dict())

    m_a = ConstrainedM(N_LAYERS, N_KV_HEADS, HALF)
    m_a.load_state_dict(torch.load(os.path.join(args.checkpoints, "M_a.pt"), map_location="cpu", weights_only=True))
    m_b = ConstrainedM(N_LAYERS, N_KV_HEADS, HALF)
    m_b.load_state_dict(torch.load(os.path.join(args.checkpoints, "M_b.pt"), map_location="cpu", weights_only=True))
    m_a.to(device)
    m_b.to(device)

    merged = build_static_state_dict(sd_a, sd_b, args.alpha)
    for layer in range(N_LAYERS):
        merged.update(build_kq_for_layer(sd_a, sd_b, m_a, m_b, layer, args.alpha, device))

    out_model = AutoModelForCausalLM.from_pretrained(args.model_a, dtype=torch.bfloat16)
    out_model.load_state_dict({k: v.cpu() for k, v in merged.items()}, strict=True)
    out_model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model_a).save_pretrained(args.out)
    print(f"merged model (alpha={args.alpha}) saved to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", required=True)
    p.add_argument("--model-b", required=True)
    p.add_argument("--checkpoints", required=True, help="directory with M_a.pt / M_b.pt from train.py")
    p.add_argument("--alpha", type=float, required=True, help="0.0 = pure A, 1.0 = pure B")
    p.add_argument("--out", required=True)
    return p.parse_args()


if __name__ == "__main__":
    merge(parse_args())
