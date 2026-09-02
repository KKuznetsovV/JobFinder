"""Evaluate the quantized tier-1 model against the held-out eval split
(scripts/tier1/data/eval.jsonl, produced by split_dataset.py and never seen
during training). Reuses the ground-truth Claude labels already stored in
each eval record's "output" field - no live Claude calls needed.

Reports relevance accuracy, resume-variant accuracy (among relevant examples),
and combined accuracy (both correct). Record the combined accuracy in the PR
description before merging, per the feature's definition-of-done.

Run:
    python scripts/tier1/evaluate.py --model models/tier1_classifier.q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder.ai.tier1_schema import PROMPT_TEMPLATE, parse_model_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/tier1_classifier.q4_k_m.gguf"))
    parser.add_argument("--eval-set", type=Path, default=Path("scripts/tier1/data/eval.jsonl"))
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    from llama_cpp import Llama

    llm = Llama(model_path=str(args.model), n_ctx=1024, n_threads=args.n_threads, verbose=False)

    records = [json.loads(line) for line in args.eval_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("Eval set is empty.", file=sys.stderr)
        sys.exit(1)

    relevance_correct = 0
    resume_correct = 0
    resume_total = 0
    both_correct = 0

    for record in records:
        expected = json.loads(record["output"])
        prompt = PROMPT_TEMPLATE.format(
            instruction=record["instruction"], input=record["input"], output=""
        )
        output = llm(prompt, max_tokens=300, stop=["###"], temperature=0.0)
        text = output["choices"][0]["text"].strip()
        predicted = parse_model_output(text)

        if predicted is None:
            continue

        relevance_ok = bool(predicted.get("relevant")) == bool(expected["relevant"])
        relevance_correct += int(relevance_ok)

        resume_ok = True
        if expected["relevant"]:
            resume_total += 1
            resume_ok = predicted.get("resume_variant") == expected["resume_variant"]
            resume_correct += int(resume_ok)

        both_correct += int(relevance_ok and resume_ok)

    total = len(records)
    print(f"total examples: {total}")
    print(f"relevance accuracy: {relevance_correct / total:.4f}")
    if resume_total:
        print(f"resume-variant accuracy (relevant only): {resume_correct / resume_total:.4f}")
    print(f"combined accuracy: {both_correct / total:.4f}")


if __name__ == "__main__":
    main()
