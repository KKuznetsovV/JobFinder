"""Tier-0 embedding pre-filter.

Runs entirely locally (no Anthropic API call) using a small sentence-transformer
model to score how similar a job posting is to the candidate's resumes via
cosine similarity. Postings that score below `config.TIER0_MIN_SIMILARITY`
never reach the (paid) Claude relevance call in `jobfinder.ai.relevance`.

The sentence-transformer model and the resume embeddings are both loaded/computed
once per process (module-level cache) since loading the model is comparatively
expensive and the resumes rarely change.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from jobfinder import config
from jobfinder.db.models import JobPosting, ResumeVariant
from jobfinder.resume.cache import get_or_parse_resume

logger = logging.getLogger(__name__)

_model = None
_resume_embeddings: dict[ResumeVariant, object] | None = None

# How much of the job description to embed alongside the title. Keeps the
# encode call cheap and avoids diluting the signal with long boilerplate.
_DESCRIPTION_CHARS = 500


@dataclass
class Tier0Result:
    score: float
    resume_hint: ResumeVariant | None


def _get_model():
    """Lazily load (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("tier0: loading embedding model %s", config.TIER0_MODEL_NAME)
        _model = SentenceTransformer(config.TIER0_MODEL_NAME)
    return _model


def _cosine_similarity(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_resume_embeddings(model=None, force: bool = False) -> dict[ResumeVariant, object]:
    """Return (and in-process cache) an embedding vector per resume variant.

    Variants whose resume file isn't present yet are skipped rather than
    raising, so a partially-configured install (only one resume placed) still
    works instead of crashing every posting.
    """
    global _resume_embeddings
    if _resume_embeddings is not None and not force:
        return _resume_embeddings

    model = model or _get_model()
    embeddings: dict[ResumeVariant, object] = {}
    for variant in (ResumeVariant.FULLSTACK, ResumeVariant.PROJECT_MANAGER):
        try:
            resume = get_or_parse_resume(variant)
        except FileNotFoundError:
            logger.warning("tier0: no resume on disk for variant %s, skipping", variant.value)
            continue
        embeddings[variant] = model.encode(resume["raw_text"])

    _resume_embeddings = embeddings
    return embeddings


def _posting_text(job: JobPosting) -> str:
    description = (job.description or "")[:_DESCRIPTION_CHARS]
    return f"{job.title}\n{description}"


def score_posting(job: JobPosting, model=None, resume_embeddings=None) -> Tier0Result:
    """Score a posting against both resume variants, returning the best match.

    If neither resume is available to compare against, returns a score of 1.0
    (i.e. never blocks the pipeline) rather than failing every posting.
    """
    model = model or _get_model()
    if resume_embeddings is None:
        resume_embeddings = get_resume_embeddings(model=model)

    if not resume_embeddings:
        return Tier0Result(score=1.0, resume_hint=None)

    posting_embedding = model.encode(_posting_text(job))

    best_variant: ResumeVariant | None = None
    best_score = -1.0
    for variant, resume_vector in resume_embeddings.items():
        similarity = _cosine_similarity(posting_embedding, resume_vector)
        if similarity > best_score:
            best_score = similarity
            best_variant = variant

    return Tier0Result(score=best_score, resume_hint=best_variant)


def summarize_scores(scores: list[float], min_similarity: float | None = None) -> dict:
    """Aggregate stats for a batch of tier0 scores, for cheap per-cycle logging."""
    threshold = config.TIER0_MIN_SIMILARITY if min_similarity is None else min_similarity
    if not scores:
        return {"count": 0, "passed": 0, "rejected": 0, "min": None, "max": None, "median": None}
    passed = sum(1 for s in scores if s >= threshold)
    return {
        "count": len(scores),
        "passed": passed,
        "rejected": len(scores) - passed,
        "min": min(scores),
        "max": max(scores),
        "median": statistics.median(scores),
    }
