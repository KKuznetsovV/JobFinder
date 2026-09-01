"""Comeet (careers.comeet.com / *.comeet.co) adapter.

Field selectors follow Comeet's documented default candidate application
form structure.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, fill_if_present, require, resume_file_for, split_applicant_name


class ComeetAdapter(ATSAdapter):
    name = "comeet"

    @staticmethod
    def matches(url: str) -> bool:
        return "comeet.com" in url or "comeet.co" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "#name", f"{first} {last}".strip())
        require(page, "#email", "email").fill(config.GMAIL_USER_EMAIL)
        require(page, "input[type='file']", "resume upload").set_input_files(
            resume_file_for(application)
        )
        fill_if_present(page, "#cover_letter", application.cover_letter or "")
        require(page, "button[type='submit']", "submit button").click()
