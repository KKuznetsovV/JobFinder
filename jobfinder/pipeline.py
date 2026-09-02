"""End-to-end orchestration: wires together relevance filtering, resume
selection, cover-letter generation, and the Gmail approval notification into
one pipeline for each newly discovered job posting.

This is the glue between the previously independent pieces (relevance
filter, resume selector, cover letter generator, gmail notify) - see
`jobfinder.tracking` for querying the resulting Application lifecycle state.
"""
from __future__ import annotations

import logging

from jobfinder import config
from jobfinder.ai import cover_letter as cover_letter_module
from jobfinder.ai import relevance
from jobfinder.ai import resume_selector
from jobfinder.ai import tier0_filter
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, JobPosting, ResumeVariant
from jobfinder.gmail import notify

logger = logging.getLogger(__name__)


def process_new_posting(
    job: JobPosting,
    gmail_service,
    db_path: str | None = None,
    client=None,
    tier0_result: tier0_filter.Tier0Result | None = None,
) -> Application | None:
    """Run the tier-0 embedding pre-filter, then (if it passes) relevance
    filtering, and if relevant, resume selection + cover letter generation +
    approval-request notification for `job` (which must already be inserted,
    i.e. have an id).

    Postings rejected by the tier-0 filter get an Application row with
    status=REJECTED_TIER0 (for audit/review) but no Claude call is ever made
    for them. Postings rejected by the Claude relevance filter get no
    Application row at all, same as before.

    Returns the created Application, or None if the job was filtered out
    (by either filter) - in both cases the caller should treat it as
    "nothing to notify about".
    """
    if job.id is None:
        raise ValueError("job must be inserted (have an id) before running the pipeline")

    if tier0_result is None:
        tier0_result = tier0_filter.score_posting(job)
    if tier0_result.score < config.TIER0_MIN_SIMILARITY:
        application = Application(
            job_posting_id=job.id,
            resume_variant=tier0_result.resume_hint or ResumeVariant.FULLSTACK,
            status=ApplicationStatus.REJECTED_TIER0,
            tier0_score=tier0_result.score,
            tier0_resume_hint=tier0_result.resume_hint,
        )
        application = store.insert_application(application, db_path=db_path)
        store.append_apply_log(
            application,
            "rejected_tier0",
            f"score={tier0_result.score:.3f} < {config.TIER0_MIN_SIMILARITY}",
            db_path=db_path,
        )
        logger.info(
            "Tier-0 rejected job %s (%s): score=%.3f (LLM calls avoided)",
            job.id,
            job.title,
            tier0_result.score,
        )
        return None

    passes, score, reason = relevance.is_relevant(job, client=client)
    if not passes:
        logger.info("Filtered out job %s (%s): score=%.2f - %s", job.id, job.title, score, reason)
        return None

    variant, resume_reason = resume_selector.select_resume(job, client=client)
    final_cover_letter, fact_check_issues = cover_letter_module.generate_cover_letter(
        job, variant, client=client
    )

    application = Application(
        job_posting_id=job.id,
        resume_variant=variant,
        resume_choice_reason=resume_reason,
        cover_letter=final_cover_letter,
        status=ApplicationStatus.FOUND,
        relevance_score=score,
        tier0_score=tier0_result.score,
        tier0_resume_hint=tier0_result.resume_hint,
    )
    application = store.insert_application(application, db_path=db_path)
    store.append_apply_log(
        application, "relevance_passed", f"score={score:.2f}: {reason}", db_path=db_path
    )
    if fact_check_issues:
        store.append_apply_log(
            application, "fact_check_issues", "; ".join(fact_check_issues), db_path=db_path
        )

    application = notify.send_approval_request(gmail_service, application, job, db_path=db_path)
    return application


def process_revision_request(
    application: Application, job: JobPosting, gmail_service, db_path: str | None = None, client=None
) -> Application:
    """Regenerate the cover letter incorporating the user's latest revision
    feedback (from `application.apply_log`), then re-send the
    approval-request email in the same Gmail thread. Sets the application's
    status back to NOTIFIED so the next reply-check picks up the user's
    response to the revision.
    """
    feedback = ""
    for entry in reversed(application.apply_log):
        if entry.get("event") == "revision_requested":
            feedback = entry.get("detail", "")
            break

    final_cover_letter, fact_check_issues = cover_letter_module.generate_cover_letter(
        job,
        application.resume_variant,
        client=client,
        revision_feedback=feedback,
        previous_letter=application.cover_letter,
    )
    application.cover_letter = final_cover_letter
    store.append_apply_log(application, "revision_applied", feedback, db_path=db_path)
    if fact_check_issues:
        store.append_apply_log(
            application, "fact_check_issues", "; ".join(fact_check_issues), db_path=db_path
        )

    application = notify.send_revision_request(gmail_service, application, job, db_path=db_path)
    return application
