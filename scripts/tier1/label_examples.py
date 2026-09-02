"""Label synthetic job postings (generate_synthetic_postings.py's output)
using the real, existing relevance/resume-selection Claude logic, then store
them as tier1_training_examples rows with source='synthetic'.

Reuses jobfinder.ai.relevance.is_relevant and jobfinder.ai.resume_selector.select_resume
directly - the exact same code path production traffic uses - so synthetic
and production-logged examples are labeled identically, and there's only one
place (jobfinder/ai/relevance.py, jobfinder/ai/resume_selector.py) where the
labeling logic itself lives.

Makes real Claude API calls - costs money. Start with a small --limit to
sanity check output before labeling the full synthetic set.

Requires: ANTHROPIC_API_KEY in the environment.

Run:
    python scripts/tier1/label_examples.py \\
        --postings scripts/tier1/data/synthetic_postings.jsonl \\
        --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder import config  # noqa: E402
from jobfinder.ai import relevance, resume_selector  # noqa: E402
from jobfinder.ai.client import get_client  # noqa: E402
from jobfinder.db import store  # noqa: E402
from jobfinder.db.models import ApplyMethod, JobPosting, Tier1TrainingExample  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postings", type=Path, default=Path("scripts/tier1/data/synthetic_postings.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Only label the first N postings.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    store.init_db(args.db)
    client = get_client()

    records = [json.loads(line) for line in args.postings.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    labeled = 0
    for record in records:
        job = _posting_from_record(record)
        passes, score, reason = relevance.is_relevant(job, client=client)
        variant = resume_reason = None
        if passes:
            variant, resume_reason = resume_selector.select_resume(job, client=client)

        example = Tier1TrainingExample(
            job_posting_id=None,
            title=job.title,
            company=job.company,
            description=job.description,
            relevant=passes,
            relevance_score=score,
            relevance_reason=reason,
            resume_variant=variant,
            resume_reason=resume_reason,
            source="synthetic",
        )
        store.insert_tier1_training_example(example, db_path=args.db)
        labeled += 1
        if labeled % 25 == 0:
            print(f"  labeled {labeled}/{len(records)}...", file=sys.stderr)

    print(f"Labeled {labeled} synthetic examples into {args.db}", file=sys.stderr)


if __name__ == "__main__":
    main()
