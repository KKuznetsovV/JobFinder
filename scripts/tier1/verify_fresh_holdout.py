"""Verify that a batch of newly-generated synthetic postings has zero text
overlap with anything already used in rounds 1-3 (training, val, eval, or
raw synthetic-postings files under scripts/tier1/data/).

Why this exists: generate_synthetic_postings.py draws from a small, finite
pool of titles/companies/snippets, so a "new" --seed alone does not
guarantee genuinely fresh text - it's entirely possible (especially in the
low-cardinality "unrelated" bucket) for a new run to regenerate a posting
byte-identical to one already labeled and used in an earlier round. "Fresh"
must be verified by hashing, not assumed from the seed.

Hashing is done on build_input_text(title, company, description) - the
exact text the model actually trains/evaluates on - via sha256, so a
posting-format record (title/company/description) and a dataset-format
record (instruction/input/output, where "input" already IS that text) can
be compared on equal footing.

Run:
    python scripts/tier1/verify_fresh_holdout.py \\
        --new scripts/tier1/data/fresh_holdout_postings.jsonl \\
        --data-dir scripts/tier1/data
Exits 1 (and prints every colliding record) if any overlap is found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder.ai.tier1_schema import build_input_text  # noqa: E402


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def posting_input_hash(title: str, company: str, description: str) -> str:
    """Hash of the exact text a posting-format record contributes to
    build_input_text() - what a raw generator record turns into once
    fed through the pipeline."""
    return _hash_text(build_input_text(title, company, description))


def record_input_hash(record: dict) -> str | None:
    """Hash a single JSONL record, auto-detecting its format:
    - dataset-format (train/val/eval.jsonl): has an "input" field already
      equal to build_input_text(...) output - hash it directly.
    - posting-format (synthetic_postings*.jsonl / a fresh batch pre-labeling):
      has "title"/"company"/"description" fields - compute build_input_text
      first.
    Returns None if the record matches neither shape.
    """
    if "input" in record:
        return _hash_text(record["input"])
    if "title" in record and "company" in record and "description" in record:
        return posting_input_hash(record["title"], record["company"], record["description"])
    return None


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_existing_hashes(data_dir: Path, exclude: Path | None = None) -> set[str]:
    """Hash every record in every *.jsonl file under data_dir (except
    `exclude`, typically the new batch's own output path if it happens to
    already live there) - covers raw synthetic-postings generator output
    from every round plus the accumulated train/val/eval splits and the
    round-1/round-2 eval snapshots."""
    hashes: set[str] = set()
    for path in sorted(data_dir.glob("*.jsonl")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        for record in _load_jsonl(path):
            h = record_input_hash(record)
            if h is not None:
                hashes.add(h)
    return hashes


def find_overlaps(new_records: list[dict], existing_hashes: set[str]) -> list[dict]:
    """Return the subset of new_records (posting-format) whose input hash
    already appears among existing_hashes."""
    overlaps = []
    for record in new_records:
        h = record_input_hash(record)
        if h is not None and h in existing_hashes:
            overlaps.append(record)
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True, help="Posting-format JSONL to verify.")
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/tier1/data"))
    args = parser.parse_args()

    new_records = _load_jsonl(args.new)
    existing_hashes = load_existing_hashes(args.data_dir, exclude=args.new)
    overlaps = find_overlaps(new_records, existing_hashes)

    if overlaps:
        print(
            f"FOUND {len(overlaps)}/{len(new_records)} overlapping posting(s) - NOT fresh:",
            file=sys.stderr,
        )
        for record in overlaps:
            print(f"  {record.get('synthetic_id', '?')}: {record.get('title', '?')!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Verified fresh: 0/{len(new_records)} overlap with existing data in {args.data_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
