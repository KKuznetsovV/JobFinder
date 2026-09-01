"""Lever (jobs.lever.co) adapter.

Field names (name/email/resume/comments) follow Lever's documented default
"Apply for this job" form structure.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, fill_if_present, require, resume_file_for, split_applicant_name


class LeverAdapter(ATSAdapter):
    name = "lever"

    @staticmethod
    def matches(url: str) -> bool:
        return "lever.co" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "input[name='name']", f"{first} {last}".strip())
        require(page, "input[name='email']", "email").fill(config.GMAIL_USER_EMAIL)
        require(page, "input[name='resume']", "resume upload").set_input_files(
            resume_file_for(application)
        )
        fill_if_present(page, "textarea[name='comments']", application.cover_letter or "")
        require(page, "button[type='submit']", "submit button").click()
