from types import SimpleNamespace

from jobfinder.db.models import Application, ApplicationStatus, ApplyMethod, JobPosting, ResumeVariant
from jobfinder.local import browser_apply_linkedin as linkedin_apply


class FakeElement:
    def __init__(self, tag="input", attrs=None, text=""):
        self.tag = tag
        self.attrs = attrs or {}
        self.text = text
        self.fill_calls = []
        self.click_calls = 0
        self.set_input_files_calls = []

    def evaluate(self, script):
        return self.tag

    def get_attribute(self, name):
        return self.attrs.get(name)

    def inner_text(self):
        return self.text

    def fill(self, value):
        self.fill_calls.append(value)

    def click(self):
        self.click_calls += 1

    def set_input_files(self, path):
        self.set_input_files_calls.append(path)


class FakeLocatorSet:
    def __init__(self, elements):
        self._elements = elements

    def count(self):
        return len(self._elements)

    def nth(self, i):
        return self._elements[i]


class FakePage:
    def __init__(self, elements):
        self._elements = elements

    def locator(self, selector):
        return FakeLocatorSet(self._elements)

    def screenshot(self):
        return b"fake-png-bytes"


def _application():
    return Application(
        job_posting_id=1,
        resume_variant=ResumeVariant.FULLSTACK,
        cover_letter="Dear hiring team, ...",
    )


def _job():
    return JobPosting(
        source="devjobs",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Build things.",
        url="https://il.linkedin.com/jobs/view/1",
        apply_method=ApplyMethod.LINKEDIN_EASY_APPLY,
    )


def _tool_use(name, tool_input, id_="tool-1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=id_)


def test_run_easy_apply_loop_stops_before_submit(mocker):
    email_input = FakeElement(tag="input", attrs={"type": "email"})
    page = FakePage([email_input])

    first_response = SimpleNamespace(content=[_tool_use("list_elements", {}, id_="t1")])
    second_response = SimpleNamespace(
        content=[_tool_use("fill_by_ref", {"ref": "e0", "value": "me@example.com"}, id_="t2")]
    )
    third_response = SimpleNamespace(
        content=[_tool_use("finish", {"status": "ready_for_submit", "note": "done"}, id_="t3")]
    )
    mocker.patch(
        "jobfinder.local.browser_apply_linkedin.create_message_with_retry",
        side_effect=[first_response, second_response, third_response],
    )

    status = linkedin_apply.run_easy_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.AWAITING_MY_CLICK
    assert email_input.fill_calls == ["me@example.com"]


def test_run_easy_apply_loop_refuses_to_click_submit_button(mocker):
    submit_button = FakeElement(tag="button", text="Submit application")
    page = FakePage([submit_button])

    first_response = SimpleNamespace(content=[_tool_use("list_elements", {}, id_="t1")])
    second_response = SimpleNamespace(
        content=[_tool_use("click_by_ref", {"ref": "e0"}, id_="t2")]
    )
    third_response = SimpleNamespace(
        content=[_tool_use("finish", {"status": "ready_for_submit", "note": "done"}, id_="t3")]
    )
    mocker.patch(
        "jobfinder.local.browser_apply_linkedin.create_message_with_retry",
        side_effect=[first_response, second_response, third_response],
    )

    status = linkedin_apply.run_easy_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.AWAITING_MY_CLICK
    assert submit_button.click_calls == 0


def test_run_easy_apply_loop_returns_stuck_on_explicit_finish(mocker):
    page = FakePage([])
    response = SimpleNamespace(
        content=[_tool_use("finish", {"status": "stuck", "note": "captcha"}, id_="t1")]
    )
    mocker.patch(
        "jobfinder.local.browser_apply_linkedin.create_message_with_retry",
        return_value=response,
    )

    status = linkedin_apply.run_easy_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.STUCK


def test_run_easy_apply_loop_exhausts_max_steps(mocker):
    page = FakePage([])
    response = SimpleNamespace(content=[_tool_use("list_elements", {}, id_="t1")])
    mocker.patch(
        "jobfinder.local.browser_apply_linkedin.create_message_with_retry",
        return_value=response,
    )

    status = linkedin_apply.run_easy_apply_loop(page, _application(), _job(), client=object(), max_steps=2)

    assert status == ApplicationStatus.STUCK
