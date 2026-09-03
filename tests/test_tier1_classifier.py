import math

import pytest

from jobfinder import config
from jobfinder.ai import tier1_classifier
from jobfinder.ai.tier1_classifier import Tier1Decision
from jobfinder.db import store
from jobfinder.db.models import ApplyMethod, JobPosting, ResumeVariant


def _job(db_path):
    posting = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Build things with Python.",
        url="https://example.com/jobs/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )
    return store.insert_job_posting(posting, db_path)


def _confident_decision(relevant=True):
    return Tier1Decision(
        relevant=relevant,
        relevance_confidence=0.95,
        relevance_reason="Strong keyword overlap.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.9,
        resume_reason="Backend-heavy description.",
    )


def test_kill_switch_off_never_invokes_local_model(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", False)
    classify_spy = mocker.patch("jobfinder.ai.tier1_classifier.classify")
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(True, 0.8, "good fit"))
    mocker.patch(
        "jobfinder.ai.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )

    result = tier1_classifier.decide(job, db_path=db_path)

    classify_spy.assert_not_called()
    assert result.path == "claude"
    assert result.passes is True
    assert result.score == 0.8

    decisions = store.summarize_tier1_decisions(db_path=db_path)
    assert decisions["total"] == 1
    assert decisions["local_only"] == 0
    assert decisions["claude_fallback"] == 0


def test_high_confidence_skips_claude_entirely(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_AGREEMENT_CHECK_RATE", 0.0)
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=_confident_decision())
    relevance_spy = mocker.patch("jobfinder.ai.relevance.is_relevant")
    resume_spy = mocker.patch("jobfinder.ai.resume_selector.select_resume")

    result = tier1_classifier.decide(job, db_path=db_path)

    relevance_spy.assert_not_called()
    resume_spy.assert_not_called()
    assert result.path == "tier1"
    assert result.passes is True
    assert result.resume_variant == ResumeVariant.FULLSTACK

    decisions = store.summarize_tier1_decisions(db_path=db_path)
    assert decisions["local_only"] == 1
    assert decisions["claude_fallback"] == 0


def test_low_confidence_falls_back_to_claude(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_RELEVANT_CONFIDENCE_MIN", 0.85)
    low_confidence = Tier1Decision(
        relevant=True,
        relevance_confidence=0.5,  # below threshold
        relevance_reason="Unsure.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.9,
        resume_reason="Backend-heavy.",
    )
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=low_confidence)
    relevance_spy = mocker.patch(
        "jobfinder.ai.relevance.is_relevant", return_value=(True, 0.7, "claude says yes")
    )
    resume_spy = mocker.patch(
        "jobfinder.ai.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )

    result = tier1_classifier.decide(job, db_path=db_path)

    relevance_spy.assert_called_once()
    resume_spy.assert_called_once()
    assert result.path == "claude_fallback"
    assert result.score == 0.7

    decisions = store.summarize_tier1_decisions(db_path=db_path)
    assert decisions["claude_fallback"] == 1
    assert decisions["local_only"] == 0


def test_unparseable_local_output_falls_back_to_claude(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=None)
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(False, 0.2, "not relevant"))

    result = tier1_classifier.decide(job, db_path=db_path)

    assert result.path == "claude_fallback"
    assert result.passes is False


def test_spot_check_logs_agreement(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_AGREEMENT_CHECK_RATE", 1.0)  # always spot-check
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=_confident_decision())
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(True, 0.9, "claude agrees"))
    mocker.patch(
        "jobfinder.ai.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )

    result = tier1_classifier.decide(job, db_path=db_path)

    assert result.path == "tier1_spot_checked"
    assert result.agreement is True

    decisions = store.summarize_tier1_decisions(db_path=db_path)
    assert decisions["spot_checked"] == 1
    assert decisions["agreement_rate"] == 1.0


def test_spot_check_logs_disagreement(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_AGREEMENT_CHECK_RATE", 1.0)
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=_confident_decision())
    # Claude disagrees on relevance this time.
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(False, 0.1, "claude disagrees"))

    result = tier1_classifier.decide(job, db_path=db_path)

    assert result.path == "tier1_spot_checked"
    assert result.agreement is False

    decisions = store.summarize_tier1_decisions(db_path=db_path)
    assert decisions["agreement_rate"] == 0.0


def test_decide_logs_training_example_on_every_claude_call(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", False)
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(True, 0.8, "good fit"))
    mocker.patch(
        "jobfinder.ai.resume_selector.select_resume",
        return_value=(ResumeVariant.FULLSTACK, "keyword match"),
    )

    tier1_classifier.decide(job, db_path=db_path)

    examples = store.export_tier1_training_examples(db_path=db_path)
    assert len(examples) == 1
    assert examples[0].relevant is True
    assert examples[0].relevance_score == 0.8
    assert examples[0].resume_variant == ResumeVariant.FULLSTACK
    assert examples[0].source == "production"


def test_decide_logs_training_example_with_no_resume_when_not_relevant(tmp_path, mocker):
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", False)
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(False, 0.1, "no fit"))
    resume_spy = mocker.patch("jobfinder.ai.resume_selector.select_resume")

    tier1_classifier.decide(job, db_path=db_path)

    resume_spy.assert_not_called()
    examples = store.export_tier1_training_examples(db_path=db_path)
    assert len(examples) == 1
    assert examples[0].relevant is False
    assert examples[0].resume_variant is None
    assert examples[0].resume_reason is None


class _FakeLlm:
    """Stand-in for llama_cpp.Llama: callable, returns the OpenAI-completion-
    style dict shape llama-cpp-python produces (with `logprobs=1` requested).

    Treats each character of `text` as its own token, all at a high-confidence
    logprob (-0.01) by default. `low_confidence_substrings` maps a literal
    substring of `text` (e.g. the value of one JSON field) to the logprob its
    characters should carry instead - used to simulate the model being
    genuinely uncertain about just that field. `include_logprobs=False`
    simulates a caller that didn't request logprobs at all."""

    def __init__(self, text, low_confidence_substrings=None, include_logprobs=True):
        self._text = text
        self._low_confidence_substrings = low_confidence_substrings or {}
        self._include_logprobs = include_logprobs

    def __call__(self, prompt, **kwargs):
        choice = {"text": self._text}
        if self._include_logprobs:
            token_logprobs = [-0.01] * len(self._text)
            for substring, logprob in self._low_confidence_substrings.items():
                start = self._text.index(substring)
                for i in range(start, start + len(substring)):
                    token_logprobs[i] = logprob
            prompt_len = len(prompt)
            choice["logprobs"] = {
                "text_offset": [prompt_len + i for i in range(len(self._text))],
                "token_logprobs": token_logprobs,
            }
        return {"choices": [choice]}


def test_classify_parses_well_formed_json():
    text = (
        '{"relevant": true, "relevance_reason": "Good match.", '
        '"resume_variant": "fullstack", "resume_reason": "Backend focus."}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is not None
    assert decision.relevant is True
    assert decision.resume_variant == ResumeVariant.FULLSTACK


def test_classify_tolerates_markdown_fences():
    text = (
        '```json\n{"relevant": false, "relevance_reason": "Weak.", '
        '"resume_variant": "project_manager", "resume_reason": "Some PM keywords."}\n```'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="PM",
        company="Acme",
        description="Agile.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is not None
    assert decision.relevant is False
    assert decision.resume_variant == ResumeVariant.PROJECT_MANAGER


def test_classify_returns_none_on_garbage_output():
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="x",
        company="Acme",
        description="x",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm("not json at all"))

    assert decision is None


def test_classify_returns_none_on_missing_required_field():
    text = '{"relevant": true, "relevance_reason": "ok"}'  # missing resume fields
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="x",
        company="Acme",
        description="x",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is None


def test_classify_derives_confidence_from_token_logprobs():
    """Confidence is a real signal computed from the model's own output
    token probabilities (see jobfinder.ai.tier1_classifier._field_confidence),
    not a value the model was trained to emit - this is the fix for round 1's
    near-constant placeholder confidence."""
    text = (
        '{"relevant": true, "relevance_reason": "Good match.", '
        '"resume_variant": "fullstack", "resume_reason": "Backend focus."}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is not None
    expected_high_confidence = math.exp(-0.01)
    assert decision.relevance_confidence == pytest.approx(expected_high_confidence, rel=1e-3)
    assert decision.resume_confidence == pytest.approx(expected_high_confidence, rel=1e-3)
    assert decision.is_confident(relevant_threshold=0.95) is True


def test_classify_low_token_confidence_on_relevant_value_lowers_only_that_confidence():
    text = (
        '{"relevant": true, "relevance_reason": "Good match.", '
        '"resume_variant": "fullstack", "resume_reason": "Backend focus."}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(
        job, llm=_FakeLlm(text, low_confidence_substrings={"true": -3.0})
    )

    assert decision is not None
    assert decision.relevance_confidence == pytest.approx(math.exp(-3.0), rel=1e-3)
    assert decision.resume_confidence == pytest.approx(math.exp(-0.01), rel=1e-3)
    assert decision.is_confident(relevant_threshold=0.85) is False


def test_classify_confidence_is_zero_when_logprobs_unavailable():
    text = (
        '{"relevant": true, "relevance_reason": "Good match.", '
        '"resume_variant": "fullstack", "resume_reason": "Backend focus."}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text, include_logprobs=False))

    assert decision is not None
    assert decision.relevance_confidence == 0.0
    assert decision.resume_confidence == 0.0
    assert decision.is_confident() is False


def test_classify_tolerates_out_of_schema_resume_variant_when_not_relevant():
    text = (
        '{"relevant": false, "relevance_reason": "No overlap at all.", '
        '"resume_variant": "none", "resume_reason": ""}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Warehouse Associate",
        company="Acme",
        description="Forklift operation.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is not None
    assert decision.relevant is False
    assert decision.resume_variant == ResumeVariant.FULLSTACK
    assert decision.resume_confidence == 0.0


def test_is_confident_ignores_resume_confidence_when_not_relevant():
    """resume_variant is never acted on downstream for a rejected posting,
    so a low/noisy resume_confidence shouldn't force an otherwise-confident
    "not relevant" decision to fall back to Claude - this was the cause of
    round 2's non-monotonic confidence-calibration table (many correct
    true-negative predictions had a near-zero resume_confidence and got
    lumped into the lowest-confidence bucket despite being reliable)."""
    decision = Tier1Decision(
        relevant=False,
        relevance_confidence=0.95,
        relevance_reason="No overlap.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )

    assert decision.is_confident(not_relevant_threshold=0.85) is True


def test_is_confident_still_requires_resume_confidence_when_relevant():
    decision = Tier1Decision(
        relevant=True,
        relevance_confidence=0.95,
        relevance_reason="Good match.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )

    assert decision.is_confident(relevant_threshold=0.85) is False


def test_is_confident_uses_not_relevant_threshold_for_not_relevant_predictions(mocker):
    """A confidence between the two thresholds should be trusted or not
    purely based on which branch it's in - proves the two thresholds are
    applied independently, not just one overriding the other."""
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    mocker.patch.object(config, "TIER1_RELEVANT_CONFIDENCE_MIN", 0.60)
    decision = Tier1Decision(
        relevant=False,
        relevance_confidence=0.70,  # above the relevant threshold, below the not-relevant one
        relevance_reason="Probably not a fit.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )

    assert decision.is_confident() is False


def test_is_confident_uses_relevant_threshold_for_relevant_predictions(mocker):
    """Same confidence (0.70) as above, but for relevant=True: trusted here
    because the relevant branch gates on the (lower) relevant threshold, not
    the not-relevant one - this is the whole point of asymmetric routing."""
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    mocker.patch.object(config, "TIER1_RELEVANT_CONFIDENCE_MIN", 0.60)
    decision = Tier1Decision(
        relevant=True,
        relevance_confidence=0.70,
        relevance_reason="Good match.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.70,
        resume_reason="Backend-heavy.",
    )

    assert decision.is_confident() is True


def test_is_confident_not_relevant_just_below_its_own_threshold(mocker):
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    decision = Tier1Decision(
        relevant=False,
        relevance_confidence=0.94,
        relevance_reason="Probably not a fit.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )

    assert decision.is_confident() is False


def test_is_confident_not_relevant_at_its_own_threshold(mocker):
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    decision = Tier1Decision(
        relevant=False,
        relevance_confidence=0.95,
        relevance_reason="Probably not a fit.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )

    assert decision.is_confident() is True


def test_decide_asymmetric_thresholds_relevant_trusted_below_not_relevant_threshold(tmp_path, mocker):
    """End-to-end through decide(): a confidence that would fail the
    not-relevant threshold is still enough to trust a relevant prediction,
    since decide() routes relevant=True through the (lower) relevant
    threshold, not the not-relevant one."""
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_AGREEMENT_CHECK_RATE", 0.0)
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    mocker.patch.object(config, "TIER1_RELEVANT_CONFIDENCE_MIN", 0.60)
    decision = Tier1Decision(
        relevant=True,
        relevance_confidence=0.70,
        relevance_reason="Good match.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.70,
        resume_reason="Backend-heavy.",
    )
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=decision)
    relevance_spy = mocker.patch("jobfinder.ai.relevance.is_relevant")

    result = tier1_classifier.decide(job, db_path=db_path)

    relevance_spy.assert_not_called()
    assert result.path == "tier1"


def test_decide_asymmetric_thresholds_not_relevant_falls_back_below_its_threshold(tmp_path, mocker):
    """Same confidence (0.70) as above, but relevant=False: falls back to
    Claude because the not-relevant branch requires the higher not-relevant
    threshold - proves the two thresholds aren't interchangeable."""
    db_path = tmp_path / "test.db"
    store.init_db(db_path)
    job = _job(db_path)
    mocker.patch.object(config, "TIER1_MODEL_ENABLED", True)
    mocker.patch.object(config, "TIER1_NOT_RELEVANT_CONFIDENCE_MIN", 0.95)
    mocker.patch.object(config, "TIER1_RELEVANT_CONFIDENCE_MIN", 0.60)
    decision = Tier1Decision(
        relevant=False,
        relevance_confidence=0.70,
        relevance_reason="Probably not a fit.",
        resume_variant=ResumeVariant.FULLSTACK,
        resume_confidence=0.0,
        resume_reason="",
    )
    mocker.patch("jobfinder.ai.tier1_classifier.classify", return_value=decision)
    mocker.patch("jobfinder.ai.relevance.is_relevant", return_value=(False, 0.2, "claude says no"))
    resume_spy = mocker.patch("jobfinder.ai.resume_selector.select_resume")

    result = tier1_classifier.decide(job, db_path=db_path)

    resume_spy.assert_not_called()
    assert result.path == "claude_fallback"


def test_classify_rejects_out_of_schema_resume_variant_when_relevant():
    """The graceful default above only applies when relevant=False - a
    genuinely relevant posting still needs a real resume choice, so an
    invalid resume_variant here is treated as unparseable (falls back to
    Claude), same as before."""
    text = (
        '{"relevant": true, "relevance_reason": "Good match.", '
        '"resume_variant": "none", "resume_reason": ""}'
    )
    job = JobPosting(
        source="gotfriends",
        external_id="1",
        title="Backend Engineer",
        company="Acme",
        description="Python, FastAPI.",
        url="https://example.com/1",
        apply_method=ApplyMethod.COMPANY_SITE,
    )

    decision = tier1_classifier.classify(job, llm=_FakeLlm(text))

    assert decision is None


def test_parse_completion_used_directly_by_evaluate_script():
    """scripts/tier1/evaluate.py builds a raw llama.cpp completion dict itself
    (rather than going through classify()) and calls parse_completion() on
    it directly - covered here so a change to the shared parsing/confidence
    logic can't silently break the offline eval script."""
    prompt = "### Instruction:\nx\n\n### Input:\ny\n\n### Response:\n"
    text = (
        '{"relevant": false, "relevance_reason": "No match.", '
        '"resume_variant": "project_manager", "resume_reason": "PM keywords."}'
    )
    output = {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "text_offset": [len(prompt) + i for i in range(len(text))],
                    "token_logprobs": [-0.01] * len(text),
                },
            }
        ]
    }

    decision = tier1_classifier.parse_completion(prompt, output)

    assert decision is not None
    assert decision.relevant is False
    assert decision.resume_variant == ResumeVariant.PROJECT_MANAGER
    assert decision.relevance_confidence == pytest.approx(math.exp(-0.01), rel=1e-3)
