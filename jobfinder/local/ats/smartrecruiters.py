"""SmartRecruiters (jobs.smartrecruiters.com) adapter.

Field names follow SmartRecruiters' documented default candidate application
form structure.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, fill_if_present, require, resume_file_for, split_applicant_name


class SmartRecruitersAdapter(ATSAdapter):
    name = "smartrecruiters"

    @staticmethod
    def matches(url: str) -> bool:
        return "smartrecruiters.com" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "input[name='firstName']", first)
        fill_if_present(page, "input[name='lastName']", last)
        require(page, "input[name='email']", "email").fill(config.GMAIL_USER_EMAIL)
        require(page, "input[name='resume']", "resume upload").set_input_files(
            resume_file_for(application)
        )
        fill_if_present(page, "textarea[name='coverLetter']", application.cover_letter or "")
        require(page, "button[type='submit']", "submit button").click()
