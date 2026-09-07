#!/usr/bin/env python3
"""
Benchmarks a set of saved model directories (base model, the two
fine-tunes, and whatever merge.py produced) on lm-evaluation-harness
tasks and prints a comparison table.

Example:
    python scripts/evaluate.py \\
        --models Qwen/Qwen2.5-1.5B-Instruct arissuga/aurum-brain-ai \\
                 SakanaAI/TinySwallow-1.5B-Instruct merged-alpha0.1/ \\
        --tasks include_base_44_indonesian include_base_44_japanese \\
        --limit 100
"""
import argparse
import json

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_tasks(model_path, tasks, limit):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)

    hflm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
    results = simple_evaluate(model=hflm, tasks=tasks, limit=limit, bootstrap_iters=0, verbosity="ERROR")

    del model
    torch.cuda.empty_cache()
    return {task: res.get("acc,none", res.get("exact_match,strict-match")) for task, res in results["results"].items()}


def main(args):
    all_results = {}
    for model_path in args.models:
        print(f"\n{model_path}")
        scores = run_tasks(model_path, args.tasks, args.limit)
        all_results[model_path] = scores
        for task, score in scores.items():
            print(f"  {task}: {score:.4f}" if score is not None else f"  {task}: n/a")

    print(f"\n{'model':40s}" + "".join(f"{t:>20s}" for t in args.tasks))
    for model_path, scores in all_results.items():
        print(f"{model_path:40s}" + "".join(f"{scores.get(t, 0):20.4f}" for t in args.tasks))

    if args.out:
        json.dump(all_results, open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="HF repo ids or local model directories")
    p.add_argument("--tasks", nargs="+", required=True, help="lm-eval-harness task names")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
