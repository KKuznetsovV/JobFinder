"""JazzHR (*.applytojob.com) adapter.

Field names follow JazzHR's documented default candidate application form
structure.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, fill_if_present, require, resume_file_for, split_applicant_name


class JazzHRAdapter(ATSAdapter):
    name = "jazzhr"

    @staticmethod
    def matches(url: str) -> bool:
        return "applytojob.com" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "input[name='first_name']", first)
        fill_if_present(page, "input[name='last_name']", last)
        require(page, "input[name='email']", "email").fill(config.GMAIL_USER_EMAIL)
        require(page, "input[name='resume']", "resume upload").set_input_files(
            resume_file_for(application)
        )
        fill_if_present(page, "textarea[name='cover_letter']", application.cover_letter or "")
        require(page, "input[type='submit'], button[type='submit']", "submit button").click()
