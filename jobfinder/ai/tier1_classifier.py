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
import math
import random
import re
from dataclasses import dataclass

from jobfinder import config
from jobfinder.ai.tier1_schema import INSTRUCTION, PROMPT_TEMPLATE, build_input_text, parse_model_output
from jobfinder.db.models import JobPosting, ResumeVariant, Tier1DecisionLog, Tier1TrainingExample

logger = logging.getLogger(__name__)

_llm = None

# Matches a top-level JSON key's value (quoted string, true/false/null, or a
# number) so we can locate the character span of e.g. the "relevant" or
# "resume_variant" value within the model's raw completion text.
_JSON_VALUE_RE_TEMPLATE = r'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'


@dataclass
class Tier1Decision:
    relevant: bool
    relevance_confidence: float
    relevance_reason: str
    resume_variant: ResumeVariant
    resume_confidence: float
    resume_reason: str

    def is_confident(self, threshold: float | None = None) -> bool:
        """resume_confidence only gates the decision when relevant=True -
        resume_variant is never acted on downstream for a rejected posting,
        so a noisy/low resume-confidence there shouldn't force an otherwise
        confident "not relevant" call to fall back to Claude (this used to
        needlessly break calibration: correct not-relevant predictions with
        a low-signal resume guess looked "unconfident" overall)."""
        threshold = config.TIER1_CONFIDENCE_MIN if threshold is None else threshold
        if not self.relevant:
            return self.relevance_confidence >= threshold
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
    so llama-cpp-python is only required when tier1 is actually enabled.
    `logits_all=True` is required for llama-cpp-python to return per-token
    logprobs, which we use to derive real confidence scores (see
    `_field_confidence`)."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        logger.info("tier1: loading local classifier model from %s", config.TIER1_MODEL_PATH)
        _llm = Llama(
            model_path=str(config.TIER1_MODEL_PATH), n_ctx=1024, n_threads=4, verbose=False, logits_all=True
        )
    return _llm


def _json_value_span(text: str, key: str) -> tuple[int, int] | None:
    """Character range of `key`'s JSON value in `text` (a raw model
    completion), excluding surrounding quotes for string values."""
    match = re.search(_JSON_VALUE_RE_TEMPLATE.format(key=re.escape(key)), text)
    if match is None:
        return None
    start, end = match.span(1)
    if text[start] == '"':
        start, end = start + 1, end - 1
    return start, end


def _field_confidence(prompt: str, text: str, logprobs: dict | None, key: str) -> float | None:
    """Real, calibration-relevant confidence for JSON field `key`'s value:
    the geometric-mean probability (exp of mean logprob) llama.cpp actually
    assigned to the token(s) that make up that value, derived from the
    model's own output distribution - not a number the model was trained to
    emit. Returns None if logprobs weren't requested/available or the field
    couldn't be located in the completion text (callers should treat that as
    "not confident")."""
    if not logprobs:
        return None
    span = _json_value_span(text, key)
    if span is None:
        return None
    start, end = span

    offsets = logprobs.get("text_offset") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    prompt_len = len(prompt)
    picked: list[float] = []
    for i, raw_offset in enumerate(offsets):
        if i >= len(token_logprobs) or token_logprobs[i] is None:
            continue
        tok_start = raw_offset - prompt_len
        tok_end = (offsets[i + 1] - prompt_len) if i + 1 < len(offsets) else len(text)
        if tok_start < end and tok_end > start:  # token overlaps the value's character span
            picked.append(token_logprobs[i])

    if not picked:
        return None
    return math.exp(sum(picked) / len(picked))


def parse_completion(prompt: str, output: dict) -> Tier1Decision | None:
    """Turn a raw llama.cpp completion dict (as returned by `Llama.__call__`,
    with `logprobs=1` requested) into a Tier1Decision, or None if the output
    couldn't be parsed into the expected shape (callers treat that the same
    as low confidence: fall back to Claude). Shared by `classify()` (runtime)
    and scripts/tier1/evaluate.py (offline eval) so both derive confidence
    identically."""
    choice = output["choices"][0]
    text = choice["text"].strip()
    parsed = parse_model_output(text)
    if parsed is None:
        logger.warning("tier1: could not parse local model output as JSON: %r", text[:200])
        return None

    logprobs = choice.get("logprobs")
    try:
        relevant = bool(parsed["relevant"])
        resume_variant_raw = parsed["resume_variant"]
        try:
            resume_variant = ResumeVariant(resume_variant_raw)
            resume_confidence = _field_confidence(prompt, text, logprobs, "resume_variant") or 0.0
        except ValueError:
            if relevant:
                raise  # a genuinely relevant posting needs a real resume choice - treat as unparseable
            # Not relevant: the model sometimes emits an out-of-schema value like "none"/"null"
            # instead of picking one of the two resumes when it judges neither fits - a defensible
            # answer, but not part of ResumeVariant's schema. Default to a placeholder variant with
            # zero confidence (never acted on downstream, since resume_variant is only consumed when
            # relevant=True) rather than discarding an otherwise-correct relevance decision.
            resume_variant = ResumeVariant.FULLSTACK
            resume_confidence = 0.0

        return Tier1Decision(
            relevant=relevant,
            relevance_confidence=_field_confidence(prompt, text, logprobs, "relevant") or 0.0,
            relevance_reason=str(parsed.get("relevance_reason", "")),
            resume_variant=resume_variant,
            resume_confidence=resume_confidence,
            resume_reason=str(parsed.get("resume_reason", "")),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("tier1: malformed local model output %r: %s", parsed, exc)
        return None


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
    output = llm(prompt, max_tokens=300, stop=["###"], temperature=0.0, logprobs=1)
    return parse_completion(prompt, output)


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
