"""Export tier1_training_examples from the DB and split into train/val/eval
JSONL files formatted for train.py (prompt/completion pairs using the shared
jobfinder.ai.tier1_schema template).

The eval split is held out and NEVER touched during training - evaluate.py
compares the fine-tuned model's predictions against these examples' existing
Claude-derived labels (no live Claude calls needed at eval time).

Run:
    python scripts/tier1/split_dataset.py --out-dir scripts/tier1/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder import config  # noqa: E402
from jobfinder.ai import resume_selector  # noqa: E402
from jobfinder.ai.tier1_schema import INSTRUCTION, build_input_text  # noqa: E402
from jobfinder.db import store  # noqa: E402
from jobfinder.db.models import ApplyMethod, JobPosting  # noqa: E402


def _best_guess_resume_variant(example) -> tuple[str, str]:
    """For not-relevant examples (no resume_variant was computed in
    production, since resume selection is skipped to save Claude calls), use
    the free local keyword heuristic to produce a genuine best-guess label
    instead of a fixed placeholder - matching the "give your best guess even
    if not relevant" instruction in tier1_schema.INSTRUCTION."""
    job = JobPosting(
        source="synthetic", external_id="", title=example.title, company=example.company,
        description=example.description, url="", apply_method=ApplyMethod.COMPANY_SITE,
    )
    matched = resume_selector.heuristic_select(job)
    variant = matched or resume_selector.ResumeVariant.FULLSTACK
    reason = (
        "Keyword heuristic best guess (posting was not relevant, so no resume was actually chosen)."
        if matched
        else "No keyword match; defaulting to fullstack as a best guess."
    )
    return variant.value, reason


def _example_to_record(example) -> dict:
    if example.resume_variant:
        resume_variant, resume_reason = example.resume_variant.value, example.resume_reason
    else:
        resume_variant, resume_reason = _best_guess_resume_variant(example)

    # No confidence fields here: round 1 trained the model to emit a fixed
    # placeholder confidence (0.9/0.95/0.5) regardless of the actual example,
    # which it dutifully memorized - useless for TIER1_CONFIDENCE_MIN
    # routing. Confidence is now derived at inference time from the model's
    # real output-token probabilities (see jobfinder.ai.tier1_classifier).
    output = json.dumps(
        {
            "relevant": example.relevant,
            "relevance_reason": example.relevance_reason,
            "resume_variant": resume_variant,
            "resume_reason": resume_reason,
        },
        ensure_ascii=False,
    )
    return {
        "instruction": INSTRUCTION,
        "input": build_input_text(example.title, example.company, example.description),
        "output": output,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_fixed_eval_inputs(path: Path) -> set[str]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["input"] for r in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/tier1/data"))
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.10, help="Of the remaining (non-eval) examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-eval-jsonl", type=Path, default=None,
        help="Reuse an existing eval.jsonl's exact examples (matched by input text) as the eval "
        "split, instead of re-shuffling a fresh one - keeps accuracy numbers comparable across "
        "training rounds when new examples are added to the DB. Examples not found among the "
        "current export are dropped with a warning.",
    )
    args = parser.parse_args()

    examples = store.export_tier1_training_examples(db_path=args.db)
    if not examples:
        print("No tier1_training_examples found - run label_examples.py first.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    if args.fixed_eval_jsonl and args.fixed_eval_jsonl.exists():
        fixed_inputs = _load_fixed_eval_inputs(args.fixed_eval_jsonl)
        eval_examples = [
            e for e in examples if build_input_text(e.title, e.company, e.description) in fixed_inputs
        ]
        remaining = [
            e for e in examples if build_input_text(e.title, e.company, e.description) not in fixed_inputs
        ]
        if len(eval_examples) != len(fixed_inputs):
            print(
                f"WARNING: matched {len(eval_examples)}/{len(fixed_inputs)} fixed eval examples by "
                "input text - some round-1 eval examples may no longer exist in the DB.",
                file=sys.stderr,
            )
        rng.shuffle(remaining)
        val_count = max(1, int(len(remaining) * args.val_fraction))
        val_examples = remaining[:val_count]
        train_examples = remaining[val_count:]
    else:
        rng.shuffle(examples)
        eval_count = max(1, int(len(examples) * args.eval_fraction))
        eval_examples = examples[:eval_count]
        remaining = examples[eval_count:]
        val_count = max(1, int(len(remaining) * args.val_fraction))
        val_examples = remaining[:val_count]
        train_examples = remaining[val_count:]

    _write_jsonl(args.out_dir / "train.jsonl", [_example_to_record(e) for e in train_examples])
    _write_jsonl(args.out_dir / "val.jsonl", [_example_to_record(e) for e in val_examples])
    _write_jsonl(args.out_dir / "eval.jsonl", [_example_to_record(e) for e in eval_examples])

    print(
        f"train={len(train_examples)} val={len(val_examples)} eval={len(eval_examples)} "
        f"(total={len(examples)}) written to {args.out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
