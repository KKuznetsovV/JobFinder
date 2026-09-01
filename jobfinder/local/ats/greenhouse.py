"""Greenhouse (boards.greenhouse.io / job-boards.greenhouse.io) adapter.

Field ids (#first_name, #last_name, #email, #phone, #resume, #cover_letter,
#submit_app) follow Greenhouse's well-documented, widely-consistent default
application-form template.
"""
from __future__ import annotations

from jobfinder import config
from jobfinder.db.models import Application, JobPosting
from jobfinder.local.ats.base import ATSAdapter, fill_if_present, require, resume_file_for, split_applicant_name


class GreenhouseAdapter(ATSAdapter):
    name = "greenhouse"

    @staticmethod
    def matches(url: str) -> bool:
        return "greenhouse.io" in url

    def fill_and_submit(self, page, application: Application, job: JobPosting) -> None:
        first, last = split_applicant_name()
        fill_if_present(page, "#first_name", first)
        fill_if_present(page, "#last_name", last)
        require(page, "#email", "email").fill(config.GMAIL_USER_EMAIL)
        require(page, "input#resume[type='file']", "resume upload").set_input_files(
            resume_file_for(application)
        )
        fill_if_present(page, "#cover_letter", application.cover_letter or "")
        require(page, "#submit_app", "submit button").click()
