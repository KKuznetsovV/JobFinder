from types import SimpleNamespace

import pytest

from jobfinder.ai import cover_letter
from jobfinder.db.models import ApplyMethod, CoverLetterRequirement, JobPosting, ResumeVariant


def _job():
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI, PostgreSQL.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def _text_message(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _fact_check_message(approved, issues, corrected_letter):
    tool_block = SimpleNamespace(
        type="tool_use",
        name="report_fact_check",
        input={"approved": approved, "issues": issues, "corrected_letter": corrected_letter},
    )
    return SimpleNamespace(content=[tool_block])


def test_generate_draft_concatenates_text_blocks(mocker):
    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry",
        return_value=_text_message("Dear hiring team, ..."),
    )

    draft = cover_letter._generate_draft(
        _job(), {"raw_text": "Experienced Python developer."}, "casual style", client=object()
    )

    assert draft == "Dear hiring team, ..."


def test_fact_check_returns_corrected_letter_and_issues(mocker):
    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry",
        return_value=_fact_check_message(False, ["Invented 'led a team of 10'."], "Corrected letter text."),
    )

    final_letter, issues = cover_letter._fact_check(
        "draft mentioning leading a team of 10", {"raw_text": "..."}, client=object()
    )

    assert final_letter == "Corrected letter text."
    assert issues == ["Invented 'led a team of 10'."]


def test_fact_check_raises_if_tool_not_called(mocker):
    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry",
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")]),
    )

    with pytest.raises(ValueError):
        cover_letter._fact_check("draft", {"raw_text": "..."}, client=object())


def test_generate_cover_letter_end_to_end(mocker):
    mocker.patch(
        "jobfinder.ai.cover_letter.get_or_parse_resume",
        return_value={"raw_text": "Experienced Python developer."},
    )
    mocker.patch("jobfinder.ai.cover_letter._load_style_reference", return_value="casual style")
    mocker.patch("jobfinder.ai.cover_letter._generate_draft", return_value="Draft letter.")
    mocker.patch(
        "jobfinder.ai.cover_letter._fact_check", return_value=("Final letter.", [])
    )

    final_letter, issues = cover_letter.generate_cover_letter(
        _job(), ResumeVariant.FULLSTACK, client=object()
    )

    assert final_letter == "Final letter."
    assert issues == []


def test_generate_draft_with_revision_feedback_asks_to_revise_previous_letter(mocker):
    capture = {}

    def fake_create_message(client, **kwargs):
        capture["messages"] = kwargs["messages"]
        return _text_message("Shorter revised letter.")

    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry", side_effect=fake_create_message
    )

    draft = cover_letter._generate_draft(
        _job(),
        {"raw_text": "Experienced Python developer."},
        "casual style",
        client=object(),
        revision_feedback="Make it shorter.",
        previous_letter="Dear hiring team, long previous letter...",
    )

    assert draft == "Shorter revised letter."
    user_content = capture["messages"][0]["content"]
    assert "Make it shorter." in user_content
    assert "Dear hiring team, long previous letter..." in user_content


def test_generate_draft_short_note_uses_a_distinctly_different_system_prompt(mocker):
    capture = {}

    def fake_create_message(client, **kwargs):
        capture["system"] = kwargs["system"]
        return _text_message("Great fit, five years of relevant experience.")

    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry", side_effect=fake_create_message
    )

    cover_letter._generate_draft(
        _job(),
        {"raw_text": "Experienced Python developer."},
        "casual style",
        client=object(),
        requirement=CoverLetterRequirement.SHORT_NOTE,
        char_limit=200,
    )

    short_note_system_text = capture["system"][0]["text"]

    mocker.patch(
        "jobfinder.ai.cover_letter.create_message_with_retry", side_effect=fake_create_message
    )
    cover_letter._generate_draft(
        _job(),
        {"raw_text": "Experienced Python developer."},
        "casual style",
        client=object(),
    )
    full_letter_system_text = capture["system"][0]["text"]

    assert short_note_system_text != full_letter_system_text
    assert "40-70 words" in short_note_system_text
    assert "200 characters" in short_note_system_text
    assert "200-300 words" in full_letter_system_text
