"""Workday (myworkdayjobs.com) adapter.

Workday's apply flow is a multi-step SPA wizard (identity -> experience ->
application questions -> voluntary disclosures -> review -> submit), and its
exact steps/questions vary heavily per employer. This adapter only fills the
first "My Information" step (using Workday's documented `data-automation-id`
testing/automation attributes) and then deliberately raises ATSFormError
rather than pretending to complete a multi-page flow it can't reliably
finish - the local agent marks these applications STUCK for the
`browser-apply-fallback` accessibility-tree loop or manual follow-up instead
of risking a wrong/incomplete auto-submission.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, ATSFormError, fill_if_present, require, split_applicant_name


class WorkdayAdapter(ATSAdapter):
    name = "workday"

    @staticmethod
    def matches(url: str) -> bool:
        return "myworkdayjobs.com" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "input[data-automation-id='legalNameSection_firstName']", first)
        fill_if_present(page, "input[data-automation-id='legalNameSection_lastName']", last)
        require(page, "input[data-automation-id='email']", "email").fill(config.GMAIL_USER_EMAIL)
        raise ATSFormError(
            "Workday's multi-step application wizard isn't fully automated; "
            "identity fields were filled, but this application needs the "
            "browser-apply-fallback flow or manual completion for the "
            "remaining steps."
        )
