from types import SimpleNamespace

import pytest

from jobfinder.ai import resume_selector
from jobfinder.db.models import ApplyMethod, JobPosting, ResumeVariant


def _job(title, description=""):
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title=title,
        company="Acme",
        description=description,
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def test_heuristic_selects_fullstack():
    assert resume_selector.heuristic_select(_job("Backend Developer")) == ResumeVariant.FULLSTACK


def test_heuristic_selects_project_manager():
    assert (
        resume_selector.heuristic_select(_job("Senior Project Manager"))
        == ResumeVariant.PROJECT_MANAGER
    )


def test_heuristic_ambiguous_when_no_match():
    assert resume_selector.heuristic_select(_job("Sales Representative")) is None


def test_heuristic_ambiguous_when_both_match():
    assert (
        resume_selector.heuristic_select(
            _job("Technical Project Manager", "Full stack development background required.")
        )
        is None
    )


def test_select_resume_uses_heuristic_without_calling_claude(mocker):
    claude_spy = mocker.patch("jobfinder.ai.resume_selector._claude_select")

    variant, reason = resume_selector.select_resume(_job("Frontend Engineer"))

    assert variant == ResumeVariant.FULLSTACK
    assert "heuristic" in reason.lower()
    claude_spy.assert_not_called()


def test_select_resume_falls_back_to_claude_when_ambiguous(mocker):
    mocker.patch(
        "jobfinder.ai.resume_selector._claude_select",
        return_value=(ResumeVariant.PROJECT_MANAGER, "Claude's reasoning."),
    )

    variant, reason = resume_selector.select_resume(_job("Operations Lead"))

    assert variant == ResumeVariant.PROJECT_MANAGER
    assert reason == "Claude's reasoning."


def test_claude_select_parses_tool_response(mocker):
    mocker.patch(
        "jobfinder.ai.resume_selector.get_or_parse_resume",
        return_value={"raw_text": "some resume text"},
    )
    tool_block = SimpleNamespace(
        type="tool_use",
        name="select_resume",
        input={"variant": "project_manager", "reason": "Better fit."},
    )
    mocker.patch(
        "jobfinder.ai.resume_selector.create_message_with_retry",
        return_value=SimpleNamespace(content=[tool_block]),
    )

    variant, reason = resume_selector._claude_select(_job("Operations Lead"), client=object())

    assert variant == ResumeVariant.PROJECT_MANAGER
    assert reason == "Better fit."


def test_claude_select_raises_if_tool_not_called(mocker):
    mocker.patch(
        "jobfinder.ai.resume_selector.get_or_parse_resume",
        return_value={"raw_text": "some resume text"},
    )
    mocker.patch(
        "jobfinder.ai.resume_selector.create_message_with_retry",
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")]),
    )

    with pytest.raises(ValueError):
        resume_selector._claude_select(_job("Operations Lead"), client=object())
