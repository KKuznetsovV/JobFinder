import numpy as np

from jobfinder import config
from jobfinder.ai import tier0_filter
from jobfinder.db.models import ApplyMethod, JobPosting, ResumeVariant


def _job(title="Backend Engineer", description="Build things with Python and APIs."):
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title=title,
        company="Acme",
        description=description,
        url="https://example.com/jobs/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer: maps known strings to
    fixed vectors so cosine similarity is predictable in tests."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, text):
        return np.asarray(self.vectors[text], dtype=float)


def test_score_posting_prefers_closer_resume(mocker):
    fullstack_vec = [1.0, 0.0]
    pm_vec = [0.0, 1.0]
    posting_text = tier0_filter._posting_text(_job())
    model = _FakeModel({posting_text: [0.9, 0.1]})

    resume_embeddings = {
        ResumeVariant.FULLSTACK: np.asarray(fullstack_vec),
        ResumeVariant.PROJECT_MANAGER: np.asarray(pm_vec),
    }

    result = tier0_filter.score_posting(_job(), model=model, resume_embeddings=resume_embeddings)

    assert result.resume_hint == ResumeVariant.FULLSTACK
    assert result.score > 0.5


def test_score_posting_no_resumes_never_blocks():
    result = tier0_filter.score_posting(_job(), model=_FakeModel({}), resume_embeddings={})

    assert result.score == 1.0
    assert result.resume_hint is None


def test_cosine_similarity_identical_vectors_is_one():
    assert tier0_filter._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert tier0_filter._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero():
    assert tier0_filter._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_summarize_scores_empty():
    stats = tier0_filter.summarize_scores([])
    assert stats["count"] == 0
    assert stats["passed"] == 0
    assert stats["rejected"] == 0


def test_summarize_scores_counts_passed_and_rejected():
    scores = [0.1, 0.2, 0.4, 0.5, 0.9]
    stats = tier0_filter.summarize_scores(scores, min_similarity=0.35)

    assert stats["count"] == 5
    assert stats["passed"] == 3
    assert stats["rejected"] == 2
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert stats["median"] == 0.4


def test_get_resume_embeddings_skips_missing_resume(mocker):
    tier0_filter._resume_embeddings = None  # reset module cache before test

    def fake_get_or_parse_resume(variant):
        if variant == ResumeVariant.PROJECT_MANAGER:
            raise FileNotFoundError("no pm resume")
        return {"raw_text": "fullstack resume text"}

    mocker.patch(
        "jobfinder.ai.tier0_filter.get_or_parse_resume", side_effect=fake_get_or_parse_resume
    )
    model = _FakeModel({"fullstack resume text": [1.0, 0.0]})

    embeddings = tier0_filter.get_resume_embeddings(model=model, force=True)

    assert ResumeVariant.FULLSTACK in embeddings
    assert ResumeVariant.PROJECT_MANAGER not in embeddings
