"""Label a verified-fresh holdout batch (see verify_fresh_holdout.py) via the
real Claude relevance/resume-selector calls - the same teacher-labeling
logic label_examples.py uses for actual training data - but writes DIRECTLY
to a standalone train/val/eval-format JSONL file instead of inserting into
the tier1_training_examples DB table.

This is deliberate: anything inserted into tier1_training_examples is fair
game for a future split_dataset.py export into train/val/eval splits. This
batch exists ONLY to validate round 3's already-locked model and thresholds
(see README's fresh-holdout validation section) and must never become
eligible for training or threshold re-tuning - so it never touches that
table at all.

Requires: ANTHROPIC_API_KEY in the environment. Makes real, paid Claude API
calls.

Run:
    python scripts/tier1/label_fresh_holdout.py \\
        --postings scripts/tier1/data/fresh_holdout_postings.jsonl \\
        --out scripts/tier1/data/fresh_holdout_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder.ai import relevance, resume_selector  # noqa: E402
from jobfinder.ai.client import get_client  # noqa: E402
from jobfinder.ai.tier1_schema import INSTRUCTION, build_input_text  # noqa: E402
from jobfinder.db.models import ApplyMethod, JobPosting  # noqa: E402


def _posting_from_record(record: dict) -> JobPosting:
    return JobPosting(
        source="synthetic",
        external_id=record["synthetic_id"],
        title=record["title"],
        company=record["company"],
        description=record["description"],
        url="https://example.invalid/synthetic",
        apply_method=ApplyMethod.COMPANY_SITE,  # placeholder, unused by relevance/resume logic
    )


def _best_guess_resume_variant(job: JobPosting) -> tuple[str, str]:
    """Same fallback split_dataset.py uses for not-relevant examples (no
    resume_variant was actually computed, since production/label_examples.py
    skips resume selection to save Claude calls when a posting is rejected)."""
    matched = resume_selector.heuristic_select(job)
    variant = matched or resume_selector.ResumeVariant.FULLSTACK
    reason = (
        "Keyword heuristic best guess (posting was not relevant, so no resume was actually chosen)."
        if matched
        else "No keyword match; defaulting to fullstack as a best guess."
    )
    return variant.value, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    client = get_client()
    records = [json.loads(line) for line in args.postings.read_text(encoding="utf-8").splitlines() if line.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labeled = 0
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            job = _posting_from_record(record)
            passes, score, reason = relevance.is_relevant(job, client=client)
            if passes:
                variant, resume_reason = resume_selector.select_resume(job, client=client)
                resume_variant, resume_variant_reason = variant.value, resume_reason
            else:
                resume_variant, resume_variant_reason = _best_guess_resume_variant(job)

            output = json.dumps(
                {
                    "relevant": passes,
                    "relevance_reason": reason,
                    "resume_variant": resume_variant,
                    "resume_reason": resume_variant_reason,
                },
                ensure_ascii=False,
            )
            out_record = {
                "instruction": INSTRUCTION,
                "input": build_input_text(job.title, job.company, job.description),
                "output": output,
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            labeled += 1
            if labeled % 25 == 0:
                print(f"  labeled {labeled}/{len(records)}...", file=sys.stderr)

    print(f"Labeled {labeled} fresh-holdout examples -> {args.out} (never inserted into the DB)", file=sys.stderr)


if __name__ == "__main__":
    main()
