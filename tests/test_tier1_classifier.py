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
    mocker.patch.object(config, "TIER1_CONFIDENCE_MIN", 0.85)
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
    style dict shape llama-cpp-python produces."""

    def __init__(self, text):
        self._text = text

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": self._text}]}


def test_classify_parses_well_formed_json():
    text = (
        '{"relevant": true, "relevance_confidence": 0.93, '
        '"relevance_reason": "Good match.", "resume_variant": "fullstack", '
        '"resume_confidence": 0.88, "resume_reason": "Backend focus."}'
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
    assert decision.relevance_confidence == 0.93
    assert decision.resume_variant == ResumeVariant.FULLSTACK
    assert decision.is_confident(threshold=0.85) is True
    assert decision.is_confident(threshold=0.95) is False


def test_classify_tolerates_markdown_fences():
    text = (
        '```json\n{"relevant": false, "relevance_confidence": 0.6, '
        '"relevance_reason": "Weak.", "resume_variant": "project_manager", '
        '"resume_confidence": 0.6, "resume_reason": "Some PM keywords."}\n```'
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
    text = '{"relevant": true, "relevance_confidence": 0.9}'  # missing resume fields
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
