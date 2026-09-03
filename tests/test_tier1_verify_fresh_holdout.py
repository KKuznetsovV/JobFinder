import json

from scripts.tier1 import verify_fresh_holdout as vfh


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_posting_input_hash_matches_dataset_record_hash():
    """A posting-format record and its equivalent dataset-format record
    (whose "input" field is exactly build_input_text(...)) must hash the
    same way - that's what lets the two file formats be compared."""
    title, company, description = "Backend Engineer", "Acme", "Build things with Python."
    posting_record = {"synthetic_id": "s-1", "title": title, "company": company, "description": description}
    dataset_record = {
        "instruction": "irrelevant for hashing",
        "input": vfh.build_input_text(title, company, description),
        "output": "{}",
    }
    assert vfh.record_input_hash(posting_record) == vfh.record_input_hash(dataset_record)


def test_record_input_hash_differs_for_different_text():
    a = {"title": "Backend Engineer", "company": "Acme", "description": "Python."}
    b = {"title": "Backend Engineer", "company": "Acme", "description": "Go."}
    assert vfh.record_input_hash(a) != vfh.record_input_hash(b)


def test_record_input_hash_returns_none_for_unrecognized_shape():
    assert vfh.record_input_hash({"foo": "bar"}) is None


def test_load_existing_hashes_reads_both_formats_and_excludes_given_path(tmp_path):
    postings_path = tmp_path / "synthetic_postings.jsonl"
    _write_jsonl(
        postings_path,
        [{"synthetic_id": "s-1", "title": "A", "company": "B", "description": "C"}],
    )
    dataset_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        dataset_path,
        [{"instruction": "x", "input": vfh.build_input_text("D", "E", "F"), "output": "{}"}],
    )
    excluded_path = tmp_path / "fresh_holdout_postings.jsonl"
    _write_jsonl(
        excluded_path,
        [{"synthetic_id": "s-2", "title": "should not count", "company": "X", "description": "Y"}],
    )

    hashes = vfh.load_existing_hashes(tmp_path, exclude=excluded_path)

    assert vfh.posting_input_hash("A", "B", "C") in hashes
    assert vfh.posting_input_hash("D", "E", "F") in hashes
    assert vfh.posting_input_hash("should not count", "X", "Y") not in hashes


def test_find_overlaps_detects_exact_duplicate_and_ignores_novel_records():
    existing_hashes = {vfh.posting_input_hash("Backend Engineer", "Acme", "Build things with Python.")}
    new_records = [
        {"synthetic_id": "s-1", "title": "Backend Engineer", "company": "Acme", "description": "Build things with Python."},
        {"synthetic_id": "s-2", "title": "Frontend Engineer", "company": "Acme", "description": "Build things with React."},
    ]

    overlaps = vfh.find_overlaps(new_records, existing_hashes)

    assert [r["synthetic_id"] for r in overlaps] == ["s-1"]
