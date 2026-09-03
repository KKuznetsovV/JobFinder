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
from jobfinder.ai import cover_letter_requirement
from jobfinder.ai import tier0_filter
from jobfinder.ai import tier1_classifier
from jobfinder.db import store
from jobfinder.db.models import Application, ApplicationStatus, CoverLetterRequirement, JobPosting, ResumeVariant
from jobfinder.gmail import notify

logger = logging.getLogger(__name__)


def process_new_posting(
    job: JobPosting,
    gmail_service,
    db_path: str | None = None,
    client=None,
    tier0_result: tier0_filter.Tier0Result | None = None,
) -> Application | None:
    """Run the tier-0 embedding pre-filter, then route the relevance +
    resume-selection decision through the tier-1 local classifier (falling
    back to Claude when unconfident or disabled - see
    `jobfinder.ai.tier1_classifier.decide`), and if relevant, send the Stage A
    (job+resume) approval-request notification for `job` (which must already
    be inserted, i.e. have an id). No cover letter is generated here - that
    only happens once Stage A is approved, via `process_stage_a_approval`.

    Postings rejected by the tier-0 filter get an Application row with
    status=REJECTED_TIER0 (for audit/review) but no Claude call is ever made
    for them. Postings rejected by the relevance decision get no Application
    row at all, same as before.

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

    decision = tier1_classifier.decide(job, db_path=db_path, client=client)
    if not decision.passes:
        logger.info(
            "Filtered out job %s (%s): score=%.2f - %s [path=%s]",
            job.id, job.title, decision.score, decision.reason, decision.path,
        )
        return None

    cl_requirement, cl_char_limit = cover_letter_requirement.classify_for_posting(job)
    application = Application(
        job_posting_id=job.id,
        resume_variant=decision.resume_variant,
        resume_choice_reason=decision.resume_reason,
        status=ApplicationStatus.FOUND,
        relevance_score=decision.score,
        tier0_score=tier0_result.score,
        tier0_resume_hint=tier0_result.resume_hint,
        cover_letter_requirement=cl_requirement,
        cover_letter_char_limit=cl_char_limit,
    )
    application = store.insert_application(application, db_path=db_path)
    store.append_apply_log(
        application,
        "relevance_passed",
        f"score={decision.score:.2f} path={decision.path}: {decision.reason}",
        db_path=db_path,
    )
    if decision.agreement is not None:
        store.append_apply_log(
            application, "tier1_agreement_check", f"agreement={decision.agreement}", db_path=db_path
        )

    application = notify.send_stage_a_request(gmail_service, application, job, db_path=db_path)
    return application


def process_stage_a_approval(
    application: Application, job: JobPosting, gmail_service, db_path: str | None = None, client=None
) -> Application:
    """Run once the user approves the Stage A (job+resume) notification:
    generates the cover letter (the expensive Claude call, now deferred until
    this point) and fact-checks it, then sends the Stage B approval email.

    If `application.cover_letter_requirement` is NONE (the posting's form has
    no cover-letter field at all), generation is skipped entirely and a
    lightweight Stage B email is sent instead - see
    `jobfinder.ai.cover_letter_requirement`.
    """
    if application.cover_letter_requirement == CoverLetterRequirement.NONE:
        application.cover_letter = None
        store.append_apply_log(
            application,
            "cover_letter_skipped",
            "No cover-letter field for this application - generation skipped.",
            db_path=db_path,
        )
        application = notify.send_stage_b_no_letter_request(gmail_service, application, job, db_path=db_path)
        return application

    final_cover_letter, fact_check_issues = cover_letter_module.generate_cover_letter(
        job,
        application.resume_variant,
        client=client,
        requirement=application.cover_letter_requirement,
        char_limit=application.cover_letter_char_limit,
    )
    application.cover_letter = final_cover_letter
    store.append_apply_log(application, "cover_letter_generated", "", db_path=db_path)
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
        requirement=application.cover_letter_requirement,
        char_limit=application.cover_letter_char_limit,
    )
    application.cover_letter = final_cover_letter
    store.append_apply_log(application, "revision_applied", feedback, db_path=db_path)
    if fact_check_issues:
        store.append_apply_log(
            application, "fact_check_issues", "; ".join(fact_check_issues), db_path=db_path
        )

    application = notify.send_revision_request(gmail_service, application, job, db_path=db_path)
    return application
