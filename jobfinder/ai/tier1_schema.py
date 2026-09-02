"""Shared prompt template and output schema for the tier-1 local classifier.

Imported by both the runtime inference module (`jobfinder.ai.tier1_classifier`)
and the offline training/eval scripts (`scripts/tier1/`) so the two never
drift apart - the exact instruction text and prompt formatting used at
inference time must match what the model was fine-tuned on.
"""
from __future__ import annotations

import json

OUTPUT_SCHEMA_DESCRIPTION = """\
Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "relevant": <true|false>,
  "relevance_confidence": <float 0-1>,
  "relevance_reason": "<one sentence>",
  "resume_variant": "<fullstack|project_manager>",
  "resume_confidence": <float 0-1>,
  "resume_reason": "<one sentence>"
}
"""

INSTRUCTION = (
    "You are a job-application screening assistant for one specific "
    "candidate. Given a job posting, decide (1) whether it is relevant "
    "enough to the candidate's background to pursue, and (2) which of the "
    "candidate's two resumes (fullstack or project_manager) fits better - "
    "give your best guess for resume_variant even if not relevant. "
    + OUTPUT_SCHEMA_DESCRIPTION
)

# Matches the exact fields already sent to Claude in relevance.py/resume_selector.py,
# so real production examples and synthetic ones share one input format.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"


def build_input_text(title: str, company: str, description: str) -> str:
    return f"Title: {title}\nCompany: {company}\nDescription: {description}"


def parse_model_output(text: str) -> dict | None:
    """Parse a model's JSON response, tolerating ```json ... ``` markdown
    fences around it (seen in practice even when the prompt says not to)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
