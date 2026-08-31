from types import SimpleNamespace

import pytest

from jobfinder import config
from jobfinder.ai import relevance
from jobfinder.db.models import ApplyMethod, JobPosting


def _fake_message(score, reason):
    tool_block = SimpleNamespace(
        type="tool_use", name="report_relevance", input={"score": score, "reason": reason}
    )
    return SimpleNamespace(content=[tool_block])


def _sample_job():
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI, PostgreSQL.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def test_score_relevance_calls_tool_and_clamps(mocker):
    mocker.patch(
        "jobfinder.ai.relevance.create_message_with_retry",
        return_value=_fake_message(1.5, "Great match on Python/FastAPI."),
    )

    score, reason = relevance.score_relevance(
        _sample_job(), candidate_profile="Python developer, 5 years FastAPI.", client=object()
    )

    assert score == 1.0  # clamped
    assert "Python" in reason


def test_is_relevant_uses_threshold(mocker):
    mocker.patch(
        "jobfinder.ai.relevance.create_message_with_retry",
        return_value=_fake_message(0.9, "Strong match."),
    )

    passes, score, reason = relevance.is_relevant(
        _sample_job(), candidate_profile="Python developer.", client=object()
    )

    assert passes is True
    assert score == 0.9

    mocker.patch(
        "jobfinder.ai.relevance.create_message_with_retry",
        return_value=_fake_message(config.RELEVANCE_THRESHOLD - 0.1, "Weak match."),
    )
    passes, score, reason = relevance.is_relevant(
        _sample_job(), candidate_profile="Python developer.", client=object()
    )
    assert passes is False


def test_score_relevance_raises_if_tool_not_called(mocker):
    mocker.patch(
        "jobfinder.ai.relevance.create_message_with_retry",
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")]),
    )

    with pytest.raises(ValueError):
        relevance.score_relevance(_sample_job(), candidate_profile="x", client=object())


def test_build_candidate_profile_skips_missing_resumes(mocker):
    mocker.patch(
        "jobfinder.ai.relevance.get_or_parse_resume",
        side_effect=FileNotFoundError("no resume"),
    )

    profile = relevance.build_candidate_profile()

    assert profile == ""
