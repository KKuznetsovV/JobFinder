from jobfinder.ai import cover_letter_requirement as clr
from jobfinder.db.models import Application, ApplyMethod, CoverLetterRequirement, JobPosting, ResumeVariant


def _job(apply_method=ApplyMethod.LINKEDIN_EASY_APPLY, description="", url="https://example.com/1"):
    return JobPosting(
        source="devjobs",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description=description,
        url=url,
        apply_method=apply_method,
    )


def test_heuristic_defaults_to_full_letter_with_no_signal():
    requirement, char_limit = clr._heuristic_from_text("We're hiring a backend engineer.")

    assert requirement == CoverLetterRequirement.FULL_LETTER
    assert char_limit is None


def test_heuristic_detects_no_cover_letter_needed():
    requirement, char_limit = clr._heuristic_from_text(
        "No cover letter is required for this role, just submit your resume."
    )

    assert requirement == CoverLetterRequirement.NONE
    assert char_limit is None


def test_heuristic_detects_short_note_and_char_limit():
    requirement, char_limit = clr._heuristic_from_text(
        "Please include a brief note (250 characters max) explaining your interest."
    )

    assert requirement == CoverLetterRequirement.SHORT_NOTE
    assert char_limit == 250


def test_classify_for_posting_uses_heuristic_for_linkedin():
    job = _job(apply_method=ApplyMethod.LINKEDIN_EASY_APPLY, description="No cover letter needed.")

    requirement, char_limit = clr.classify_for_posting(job)

    assert requirement == CoverLetterRequirement.NONE
    assert char_limit is None


def test_classify_for_posting_uses_ats_adapter_for_recognized_company_site():
    job = _job(
        apply_method=ApplyMethod.COMPANY_SITE,
        description="No cover letter needed.",
        url="https://boards.greenhouse.io/acme/jobs/1",
    )

    requirement, char_limit = clr.classify_for_posting(job)

    # Greenhouse's static template attribute wins over the heuristic scan of
    # the posting text, since it's free/certain knowledge of the real form.
    assert requirement == CoverLetterRequirement.FULL_LETTER
    assert char_limit is None


def test_classify_for_posting_falls_back_to_heuristic_for_unrecognized_company_site():
    job = _job(
        apply_method=ApplyMethod.COMPANY_SITE,
        description="Cover letter optional.",
        url="https://careers.example.com/jobs/1",
    )

    requirement, char_limit = clr.classify_for_posting(job)

    assert requirement == CoverLetterRequirement.SHORT_NOTE


def test_adapt_cover_letter_locally_none_requirement_clears_letter():
    result = clr.adapt_cover_letter_locally("Dear hiring team. ...", CoverLetterRequirement.NONE)

    assert result is None


def test_adapt_cover_letter_locally_short_note_truncates_to_two_sentences():
    letter = (
        "Dear hiring team, I'm excited about this role. I have five years of "
        "backend experience. I've also led several projects to completion."
    )

    result = clr.adapt_cover_letter_locally(letter, CoverLetterRequirement.SHORT_NOTE)

    assert result == (
        "Dear hiring team, I'm excited about this role. I have five years of "
        "backend experience."
    )


def test_adapt_cover_letter_locally_short_note_respects_char_limit():
    letter = "Dear hiring team, I'm excited about this role. I have five years of backend experience."

    result = clr.adapt_cover_letter_locally(letter, CoverLetterRequirement.SHORT_NOTE, char_limit=20)

    assert len(result) <= 20


def test_adapt_cover_letter_locally_full_letter_unchanged():
    letter = "Dear hiring team, ..."

    result = clr.adapt_cover_letter_locally(letter, CoverLetterRequirement.FULL_LETTER)

    assert result == letter


def _application(requirement):
    return Application(
        job_posting_id=1,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter_requirement=requirement,
    )


def test_log_if_classification_mismatched_logs_on_disagreement():
    application = _application(CoverLetterRequirement.FULL_LETTER)

    clr.log_if_classification_mismatched(application, "no_field")

    events = [entry["event"] for entry in application.apply_log]
    assert "cover_letter_classification_mismatch" in events


def test_log_if_classification_mismatched_no_op_when_matching():
    application = _application(CoverLetterRequirement.FULL_LETTER)

    clr.log_if_classification_mismatched(application, "full_letter_field")

    assert application.apply_log == []


def test_log_if_classification_mismatched_no_op_when_field_found_missing():
    application = _application(CoverLetterRequirement.FULL_LETTER)

    clr.log_if_classification_mismatched(application, None)

    assert application.apply_log == []
