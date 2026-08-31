"""Cache parsed resume JSON next to the source .docx so repeated calls (job
polling loops, cover-letter generation, resume selection) don't reparse the
file every time. Cache is invalidated automatically when the source file's
mtime changes, so edits to the .docx are always picked up.
"""
from __future__ import annotations

import json
from pathlib import Path

from jobfinder import config
from jobfinder.db.models import ResumeVariant
from jobfinder.resume.parser import parse_resume

_FILENAMES = {
    ResumeVariant.FULLSTACK: "fullstack.docx",
    ResumeVariant.PROJECT_MANAGER: "project_manager.docx",
}


def resume_path(variant: ResumeVariant) -> Path:
    return config.RESUME_DIR / _FILENAMES[variant]


def _cache_path(variant: ResumeVariant) -> Path:
    return config.RESUME_CACHE_DIR / f"{variant.value}.json"


def get_or_parse_resume(variant: ResumeVariant, force: bool = False) -> dict:
    """Return the parsed resume as a dict, reparsing only when needed.

    Raises FileNotFoundError with a friendly message pointing at the expected
    path if the user hasn't placed their real resume there yet.
    """
    source = resume_path(variant)
    if not source.exists():
        raise FileNotFoundError(
            f"Expected resume at {source} - add your real .docx there "
            "(see README for the resume variants JobFinder expects)."
        )

    cache_file = _cache_path(variant)
    source_mtime = source.stat().st_mtime

    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("_source_mtime") == source_mtime:
                return cached["data"]
        except (json.JSONDecodeError, KeyError):
            pass  # cache file is corrupt/outdated shape; fall through and reparse

    data = parse_resume(source).as_dict()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"_source_mtime": source_mtime, "data": data}, indent=2),
        encoding="utf-8",
    )
    return data
