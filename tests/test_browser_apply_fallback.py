from types import SimpleNamespace

from jobfinder.db.models import Application, ApplicationStatus, ApplyMethod, JobPosting, ResumeVariant
from jobfinder.local import browser_apply_fallback as fallback


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
        self.screenshot_calls = 0

    def locator(self, selector):
        return FakeLocatorSet(self._elements)

    def screenshot(self):
        self.screenshot_calls += 1
        return b"fake-png-bytes"


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
        url="https://careers.acme.example/apply/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )


def _tool_use(name, tool_input, id_="tool-1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=id_)


def test_snapshot_elements_builds_ref_map_and_description():
    email_input = FakeElement(tag="input", attrs={"type": "email", "placeholder": "Email"})
    submit_button = FakeElement(tag="button", text="Submit")
    page = FakePage([email_input, submit_button])

    text, ref_map = fallback.snapshot_elements(page)

    assert "e0" in ref_map and ref_map["e0"] is email_input
    assert "e1" in ref_map and ref_map["e1"] is submit_button
    assert "Email" in text
    assert "Submit" in text


def test_run_apply_loop_fills_and_submits(mocker):
    email_input = FakeElement(tag="input", attrs={"type": "email"})
    page = FakePage([email_input])

    first_response = SimpleNamespace(content=[_tool_use("list_elements", {}, id_="t1")])
    second_response = SimpleNamespace(
        content=[
            _tool_use("fill_by_ref", {"ref": "e0", "value": "me@example.com"}, id_="t2"),
        ]
    )
    third_response = SimpleNamespace(
        content=[_tool_use("finish", {"status": "submitted", "note": "done"}, id_="t3")]
    )
    mocker.patch(
        "jobfinder.local.browser_apply_fallback.create_message_with_retry",
        side_effect=[first_response, second_response, third_response],
    )

    status = fallback.run_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.SENT
    assert email_input.fill_calls == ["me@example.com"]


def test_run_apply_loop_returns_stuck_on_explicit_finish(mocker):
    page = FakePage([])
    response = SimpleNamespace(
        content=[_tool_use("finish", {"status": "stuck", "note": "captcha"}, id_="t1")]
    )
    mocker.patch(
        "jobfinder.local.browser_apply_fallback.create_message_with_retry",
        return_value=response,
    )

    status = fallback.run_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.STUCK


def test_run_apply_loop_returns_stuck_when_no_tool_calls(mocker):
    page = FakePage([])
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="I'm not sure what to do.")])
    mocker.patch(
        "jobfinder.local.browser_apply_fallback.create_message_with_retry",
        return_value=response,
    )

    status = fallback.run_apply_loop(page, _application(), _job(), client=object())

    assert status == ApplicationStatus.STUCK


def test_run_apply_loop_exhausts_max_steps(mocker):
    page = FakePage([])
    response = SimpleNamespace(content=[_tool_use("list_elements", {}, id_="t1")])
    mocker.patch(
        "jobfinder.local.browser_apply_fallback.create_message_with_retry",
        return_value=response,
    )

    status = fallback.run_apply_loop(page, _application(), _job(), client=object(), max_steps=2)

    assert status == ApplicationStatus.STUCK


def test_execute_tool_stale_ref_returns_error_without_crashing():
    page = FakePage([])
    result_text, ref_map = fallback._execute_tool(page, {}, "click_by_ref", {"ref": "e5"})

    assert "error" in result_text.lower()
