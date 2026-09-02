"""Tier-1 local fine-tuned classifier: replaces the Claude relevance-filter
and resume-selection calls (jobfinder.ai.relevance / jobfinder.ai.resume_selector)
for postings that already passed the tier-0 embedding filter, whenever the
local model is confident enough. Falls back to the real Claude calls
otherwise, and always logs which path was used plus training data for future
fine-tuning rounds - see `decide()`, the single entry point `pipeline.py` uses.

Serves a LoRA-fine-tuned Qwen2.5-1.5B-Instruct checkpoint (quantized to GGUF)
via llama-cpp-python, the same approach used by the SmartServe project this
was adapted from. The model is loaded once per process (module-level cache).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from jobfinder import config
from jobfinder.ai.tier1_schema import INSTRUCTION, PROMPT_TEMPLATE, build_input_text, parse_model_output
from jobfinder.db.models import JobPosting, ResumeVariant, Tier1DecisionLog, Tier1TrainingExample

logger = logging.getLogger(__name__)

_llm = None


@dataclass
class Tier1Decision:
    relevant: bool
    relevance_confidence: float
    relevance_reason: str
    resume_variant: ResumeVariant
    resume_confidence: float
    resume_reason: str

    def is_confident(self, threshold: float | None = None) -> bool:
        threshold = config.TIER1_CONFIDENCE_MIN if threshold is None else threshold
        return self.relevance_confidence >= threshold and self.resume_confidence >= threshold


@dataclass
class RoutedDecision:
    """The result `pipeline.process_new_posting` actually acts on, regardless
    of whether it came from the local model or a real Claude call."""

    passes: bool
    score: float
    reason: str
    resume_variant: ResumeVariant | None
    resume_reason: str | None
    path: str  # "claude" | "claude_fallback" | "tier1" | "tier1_spot_checked"
    resume_confidence: float | None = None
    agreement: bool | None = None


def _get_llm():
    """Lazily load (and cache) the local llama.cpp model. Import is deferred
    so llama-cpp-python is only required when tier1 is actually enabled."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        logger.info("tier1: loading local classifier model from %s", config.TIER1_MODEL_PATH)
        _llm = Llama(model_path=str(config.TIER1_MODEL_PATH), n_ctx=1024, n_threads=4, verbose=False)
    return _llm


def classify(job: JobPosting, llm=None) -> Tier1Decision | None:
    """Run the local model on `job`, returning a Tier1Decision, or None if
    the output couldn't be parsed into the expected shape (callers treat
    that the same as low confidence: fall back to Claude)."""
    llm = llm or _get_llm()
    prompt = PROMPT_TEMPLATE.format(
        instruction=INSTRUCTION,
        input=build_input_text(job.title, job.company, job.description),
        output="",
    )
    output = llm(prompt, max_tokens=300, stop=["###"], temperature=0.0)
    text = output["choices"][0]["text"].strip()
    parsed = parse_model_output(text)
    if parsed is None:
        logger.warning("tier1: could not parse local model output as JSON: %r", text[:200])
        return None

    try:
        return Tier1Decision(
            relevant=bool(parsed["relevant"]),
            relevance_confidence=float(parsed["relevance_confidence"]),
            relevance_reason=str(parsed.get("relevance_reason", "")),
            resume_variant=ResumeVariant(parsed["resume_variant"]),
            resume_confidence=float(parsed["resume_confidence"]),
            resume_reason=str(parsed.get("resume_reason", "")),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("tier1: malformed local model output %r: %s", parsed, exc)
        return None


def decide(job: JobPosting, db_path: str | None = None, client=None, llm=None) -> RoutedDecision:
    """Route `job` (already past tier-0) to either the local model or Claude,
    per config.TIER1_MODEL_ENABLED and the confidence threshold. Always logs
    a tier1_decisions row (for accuracy tracking) and a tier1_training_examples
    row whenever a genuine Claude decision was made (for future fine-tuning)."""
    if not config.TIER1_MODEL_ENABLED:
        result = _decide_with_claude(job, db_path=db_path, client=client)
        _log_decision(job, result, db_path=db_path)
        return result

    decision = None
    try:
        decision = classify(job, llm=llm)
    except Exception:
        logger.exception("tier1: local model inference failed for job %s, falling back to Claude", job.id)

    if decision is None or not decision.is_confident():
        logger.info("tier1: low-confidence or unavailable local decision for job %s, falling back to Claude", job.id)
        result = _decide_with_claude(job, db_path=db_path, client=client)
        result.path = "claude_fallback"
        _log_decision(job, result, db_path=db_path)
        return result

    result = RoutedDecision(
        passes=decision.relevant,
        score=decision.relevance_confidence,
        reason=decision.relevance_reason,
        resume_variant=decision.resume_variant,
        resume_reason=decision.resume_reason,
        resume_confidence=decision.resume_confidence,
        path="tier1",
    )

    if random.random() < config.TIER1_AGREEMENT_CHECK_RATE:
        claude_result = _decide_with_claude(job, db_path=db_path, client=client)
        result.agreement = (claude_result.passes == result.passes) and (
            not result.passes or claude_result.resume_variant == result.resume_variant
        )
        result.path = "tier1_spot_checked"
        logger.info(
            "tier1: spot-check for job %s agreement=%s (local: relevant=%s/%s, claude: relevant=%s/%s)",
            job.id, result.agreement, result.passes, result.resume_variant,
            claude_result.passes, claude_result.resume_variant,
        )

    _log_decision(job, result, db_path=db_path)
    return result


def _decide_with_claude(job: JobPosting, db_path: str | None, client) -> RoutedDecision:
    # Local import: keeps this module free of a hard dependency on the Claude
    # modules unless a Claude call is actually needed.
    from jobfinder.ai import relevance, resume_selector

    passes, score, reason = relevance.is_relevant(job, client=client)
    variant = resume_reason = None
    if passes:
        variant, resume_reason = resume_selector.select_resume(job, client=client)

    _log_training_example(job, passes, score, reason, variant, resume_reason, db_path=db_path)
    return RoutedDecision(
        passes=passes, score=score, reason=reason,
        resume_variant=variant, resume_reason=resume_reason, path="claude",
    )


def _log_training_example(
    job: JobPosting, passes: bool, score: float, reason: str,
    variant: ResumeVariant | None, resume_reason: str | None, db_path: str | None,
) -> None:
    from jobfinder.db import store

    example = Tier1TrainingExample(
        job_posting_id=job.id,
        title=job.title,
        company=job.company,
        description=job.description,
        relevant=passes,
        relevance_score=score,
        relevance_reason=reason,
        resume_variant=variant,
        resume_reason=resume_reason,
        source="production",
    )
    store.insert_tier1_training_example(example, db_path=db_path)


def _log_decision(job: JobPosting, result: RoutedDecision, db_path: str | None) -> None:
    from jobfinder.db import store

    is_local = result.path in ("tier1", "tier1_spot_checked")
    entry = Tier1DecisionLog(
        job_posting_id=job.id,
        path=result.path,
        relevance_confidence=result.score if is_local else None,
        resume_confidence=result.resume_confidence if is_local else None,
        agreement=result.agreement,
    )
    store.insert_tier1_decision_log(entry, db_path=db_path)
