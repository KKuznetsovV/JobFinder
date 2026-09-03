import pytest

from jobfinder import config
from jobfinder.db.models import Application, ApplyMethod, CoverLetterRequirement, JobPosting, ResumeVariant
from jobfinder.local import ats
from jobfinder.local.ats.base import ATSFormError
from jobfinder.local.ats.comeet import ComeetAdapter
from jobfinder.local.ats.greenhouse import GreenhouseAdapter
from jobfinder.local.ats.jazzhr import JazzHRAdapter
from jobfinder.local.ats.lever import LeverAdapter
from jobfinder.local.ats.smartrecruiters import SmartRecruitersAdapter
from jobfinder.local.ats.workday import WorkdayAdapter


class FakeLocator:
    def __init__(self, exists=True):
        self.exists = exists
        self.fill_calls = []
        self.click_calls = 0
        self.set_input_files_calls = []

    def count(self):
        return 1 if self.exists else 0

    @property
    def first(self):
        return self

    def fill(self, value):
        self.fill_calls.append(value)

    def click(self):
        self.click_calls += 1

    def set_input_files(self, path):
        self.set_input_files_calls.append(path)


class FakePage:
    def __init__(self, locators):
        self._locators = locators

    def locator(self, selector):
        return self._locators.get(selector, FakeLocator(exists=False))


def _application():
    return Application(
        job_posting_id=1,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Dear hiring team, ...",
    )


def _job():
    return JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://boards.greenhouse.io/acme/jobs/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def test_greenhouse_fills_and_submits(mocker):
    mocker.patch("jobfinder.local.ats.greenhouse.resume_file_for", return_value="resume.docx")
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "#first_name": FakeLocator(),
        "#last_name": FakeLocator(),
        "#email": FakeLocator(),
        "input#resume[type='file']": FakeLocator(),
        "#cover_letter": FakeLocator(),
        "#submit_app": FakeLocator(),
    }
    page = FakePage(locators)

    GreenhouseAdapter().fill_and_submit(page, _application(), _job())

    assert locators["#email"].fill_calls == ["me@example.com"]
    assert locators["input#resume[type='file']"].set_input_files_calls == ["resume.docx"]
    assert locators["#cover_letter"].fill_calls == ["Dear hiring team, ..."]
    assert locators["#submit_app"].click_calls == 1


def test_greenhouse_raises_when_email_missing():
    page = FakePage({})

    with pytest.raises(ATSFormError):
        GreenhouseAdapter().fill_and_submit(page, _application(), _job())


def test_greenhouse_fills_phone_when_configured(mocker):
    mocker.patch("jobfinder.local.ats.greenhouse.resume_file_for", return_value="resume.docx")
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    mocker.patch.object(config, "APPLICANT_PHONE", "0525655985")
    locators = {
        "#first_name": FakeLocator(),
        "#last_name": FakeLocator(),
        "#email": FakeLocator(),
        "#phone": FakeLocator(),
        "input#resume[type='file']": FakeLocator(),
        "#cover_letter": FakeLocator(),
        "#submit_app": FakeLocator(),
    }
    page = FakePage(locators)

    GreenhouseAdapter().fill_and_submit(page, _application(), _job())

    assert locators["#phone"].fill_calls == ["0525655985"]


def test_lever_fills_and_submits(mocker):
    mocker.patch("jobfinder.local.ats.lever.resume_file_for", return_value="resume.docx")
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "input[name='name']": FakeLocator(),
        "input[name='email']": FakeLocator(),
        "input[name='resume']": FakeLocator(),
        "textarea[name='comments']": FakeLocator(),
        "button[type='submit']": FakeLocator(),
    }
    page = FakePage(locators)

    LeverAdapter().fill_and_submit(page, _application(), _job())

    assert locators["input[name='email']"].fill_calls == ["me@example.com"]
    assert locators["button[type='submit']"].click_calls == 1


def test_workday_fills_identity_then_raises(mocker):
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "input[data-automation-id='legalNameSection_firstName']": FakeLocator(),
        "input[data-automation-id='legalNameSection_lastName']": FakeLocator(),
        "input[data-automation-id='email']": FakeLocator(),
    }
    page = FakePage(locators)

    with pytest.raises(ATSFormError):
        WorkdayAdapter().fill_and_submit(page, _application(), _job())

    assert locators["input[data-automation-id='email']"].fill_calls == ["me@example.com"]


def test_comeet_fills_and_submits(mocker):
    mocker.patch("jobfinder.local.ats.comeet.resume_file_for", return_value="resume.docx")
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "#name": FakeLocator(),
        "#email": FakeLocator(),
        "input[type='file']": FakeLocator(),
        "#cover_letter": FakeLocator(),
        "button[type='submit']": FakeLocator(),
    }
    page = FakePage(locators)

    ComeetAdapter().fill_and_submit(page, _application(), _job())

    assert locators["button[type='submit']"].click_calls == 1


def test_smartrecruiters_fills_and_submits(mocker):
    mocker.patch(
        "jobfinder.local.ats.smartrecruiters.resume_file_for", return_value="resume.docx"
    )
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "input[name='firstName']": FakeLocator(),
        "input[name='lastName']": FakeLocator(),
        "input[name='email']": FakeLocator(),
        "input[name='resume']": FakeLocator(),
        "textarea[name='coverLetter']": FakeLocator(),
        "button[type='submit']": FakeLocator(),
    }
    page = FakePage(locators)

    SmartRecruitersAdapter().fill_and_submit(page, _application(), _job())

    assert locators["button[type='submit']"].click_calls == 1


def test_jazzhr_fills_and_submits(mocker):
    mocker.patch("jobfinder.local.ats.jazzhr.resume_file_for", return_value="resume.docx")
    mocker.patch.object(config, "GMAIL_USER_EMAIL", "me@example.com")
    locators = {
        "input[name='first_name']": FakeLocator(),
        "input[name='last_name']": FakeLocator(),
        "input[name='email']": FakeLocator(),
        "input[name='resume']": FakeLocator(),
        "textarea[name='cover_letter']": FakeLocator(),
        "input[type='submit'], button[type='submit']": FakeLocator(),
    }
    page = FakePage(locators)

    JazzHRAdapter().fill_and_submit(page, _application(), _job())

    assert locators["input[type='submit'], button[type='submit']"].click_calls == 1


@pytest.mark.parametrize(
    "url,expected_name",
    [
        ("https://boards.greenhouse.io/acme/jobs/1", "greenhouse"),
        ("https://jobs.lever.co/acme/1", "lever"),
        ("https://acme.wd1.myworkdayjobs.com/en-US/acme/job/1", "workday"),
        ("https://careers.comeet.com/acme/1", "comeet"),
        ("https://jobs.smartrecruiters.com/acme/1", "smartrecruiters"),
        ("https://acme.applytojob.com/apply/1", "jazzhr"),
    ],
)
def test_get_adapter_for_url_matches_known_platforms(url, expected_name):
    adapter = ats.get_adapter_for_url(url)
    assert adapter is not None
    assert adapter.name == expected_name


def test_get_adapter_for_url_returns_none_for_unknown_platform():
    assert ats.get_adapter_for_url("https://careers.driivz.com/apply/1") is None


@pytest.mark.parametrize(
    "adapter_class,expected_requirement",
    [
        (GreenhouseAdapter, CoverLetterRequirement.FULL_LETTER),
        (LeverAdapter, CoverLetterRequirement.SHORT_NOTE),
        (WorkdayAdapter, CoverLetterRequirement.FULL_LETTER),
        (ComeetAdapter, CoverLetterRequirement.SHORT_NOTE),
        (SmartRecruitersAdapter, CoverLetterRequirement.FULL_LETTER),
        (JazzHRAdapter, CoverLetterRequirement.FULL_LETTER),
    ],
)
def test_adapter_declares_cover_letter_requirement(adapter_class, expected_requirement):
    assert adapter_class.cover_letter_requirement == expected_requirement

