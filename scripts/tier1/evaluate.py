"""Evaluate the quantized tier-1 model against the held-out eval split
(scripts/tier1/data/eval.jsonl, produced by split_dataset.py and never seen
during training). Reuses the ground-truth Claude labels already stored in
each eval record's "output" field - no live Claude calls needed.

Reports relevance accuracy, resume-variant accuracy (among relevant examples),
combined accuracy (both correct), a majority-class baseline for comparison,
a full confusion matrix for the relevance decision (plus one for resume_variant
among correctly-relevant examples), and a confidence-calibration table
(accuracy by confidence quartile, using the real logprob-derived confidence
from jobfinder.ai.tier1_classifier - not a model-emitted placeholder).
Record the combined accuracy in the PR description before merging, per the
feature's definition-of-done.

Run:
    python scripts/tier1/evaluate.py --model models/tier1_classifier.q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jobfinder import config  # noqa: E402
from jobfinder.ai.tier1_classifier import parse_completion  # noqa: E402
from jobfinder.ai.tier1_schema import PROMPT_TEMPLATE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/tier1_classifier.q4_k_m.gguf"))
    parser.add_argument("--eval-set", type=Path, default=Path("scripts/tier1/data/eval.jsonl"))
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument(
        "--not-relevant-confidence-min", type=float, default=config.TIER1_NOT_RELEVANT_CONFIDENCE_MIN,
        help="Threshold used for the projected local-vs-fallback routing report (defaults to the "
        "configured TIER1_NOT_RELEVANT_CONFIDENCE_MIN).",
    )
    parser.add_argument(
        "--relevant-confidence-min", type=float, default=config.TIER1_RELEVANT_CONFIDENCE_MIN,
        help="Same, for TIER1_RELEVANT_CONFIDENCE_MIN.",
    )
    args = parser.parse_args()

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(args.model), n_ctx=1024, n_threads=args.n_threads, verbose=False, logits_all=True
    )

    records = [json.loads(line) for line in args.eval_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("Eval set is empty.", file=sys.stderr)
        sys.exit(1)

    relevance_correct = 0
    resume_correct = 0
    resume_total = 0
    both_correct = 0

    # Confusion matrix for the relevance decision (true/false positive/negative).
    tp = fp = tn = fn = 0
    # 2x2 confusion matrix for resume_variant, among examples where the
    # relevance decision was itself correct AND genuinely relevant (the only
    # case a resume choice is actually acted on in production).
    resume_confusion: dict[str, dict[str, int]] = {
        "fullstack": {"fullstack": 0, "project_manager": 0},
        "project_manager": {"fullstack": 0, "project_manager": 0},
    }
    # (routing_confidence, correct) pairs for the calibration table below.
    calibration_points: list[tuple[float, bool]] = []
    # Same, but split by the model's PREDICTED class - the asymmetric routing
    # this feeds into (TIER1_NOT_RELEVANT_CONFIDENCE_MIN / TIER1_RELEVANT_
    # CONFIDENCE_MIN) trusts "not relevant" and "relevant" predictions
    # independently, so their calibration needs to be reported independently
    # too - a single pooled table can hide that one class is much safer to
    # trust than the other at a given confidence level.
    not_relevant_points: list[tuple[float, bool]] = []  # (relevance_confidence, was genuinely not relevant)
    relevant_points: list[tuple[float, bool]] = []  # (resume_confidence, resume choice was correct)
    # Projected routing under the two asymmetric thresholds - the real
    # expected cost savings, more honest than a raw accuracy number since it
    # accounts for how much of the eval set would actually be auto-handled
    # locally vs fall back to Claude.
    local_routed = 0
    local_routed_correct = 0

    for record in records:
        expected = json.loads(record["output"])
        prompt = PROMPT_TEMPLATE.format(
            instruction=record["instruction"], input=record["input"], output=""
        )
        output = llm(prompt, max_tokens=300, stop=["###"], temperature=0.0, logprobs=1)
        decision = parse_completion(prompt, output)

        if decision is None:
            continue

        predicted_relevant = decision.relevant
        expected_relevant = bool(expected["relevant"])
        relevance_ok = predicted_relevant == expected_relevant
        relevance_correct += int(relevance_ok)

        if expected_relevant and predicted_relevant:
            tp += 1
        elif not expected_relevant and predicted_relevant:
            fp += 1
        elif not expected_relevant and not predicted_relevant:
            tn += 1
        else:
            fn += 1

        resume_ok = True
        if expected_relevant:
            resume_total += 1
            resume_ok = decision.resume_variant.value == expected["resume_variant"]
            resume_correct += int(resume_ok)
            if relevance_ok and expected["resume_variant"] in resume_confusion:
                predicted_variant = decision.resume_variant.value
                if predicted_variant in resume_confusion[expected["resume_variant"]]:
                    resume_confusion[expected["resume_variant"]][predicted_variant] += 1

        example_correct = relevance_ok and resume_ok
        both_correct += int(example_correct)
        # Mirrors Tier1Decision.is_confident(): resume_confidence only gates
        # the routing decision when the model predicted relevant=True, since
        # resume_variant is never acted on downstream for a rejected posting.
        routing_confidence = (
            decision.relevance_confidence
            if not decision.relevant
            else min(decision.relevance_confidence, decision.resume_confidence)
        )
        calibration_points.append((routing_confidence, example_correct))

        # Per-predicted-class calibration: the two asymmetric thresholds
        # (TIER1_NOT_RELEVANT_CONFIDENCE_MIN / TIER1_RELEVANT_CONFIDENCE_MIN)
        # each only need to be trustworthy for the class they gate, so their
        # calibration must be measured independently rather than pooled -
        # a "not relevant" call and a "relevant" call have very different
        # base rates and failure costs (see module docstring / PR notes).
        if not predicted_relevant:
            # False-negative risk: of postings called "not relevant", what
            # fraction genuinely were? Bucketed by the same relevance_confidence
            # that TIER1_NOT_RELEVANT_CONFIDENCE_MIN will gate on.
            not_relevant_points.append((decision.relevance_confidence, not expected_relevant))
        else:
            # Of postings called "relevant", what fraction genuinely were?
            # Bucketed by min(relevance_confidence, resume_confidence), the
            # same AND-gated value TIER1_RELEVANT_CONFIDENCE_MIN will gate on
            # (a confident relevance call with a shaky resume pick is still
            # not fully trustworthy for the relevant branch).
            relevant_points.append(
                (min(decision.relevance_confidence, decision.resume_confidence), expected_relevant)
            )

        if decision.is_confident(
            not_relevant_threshold=args.not_relevant_confidence_min,
            relevant_threshold=args.relevant_confidence_min,
        ):
            local_routed += 1
            local_routed_correct += int(example_correct)

    total = len(records)
    relevant_count = sum(1 for r in records if json.loads(r["output"])["relevant"])
    majority_baseline = max(relevant_count, total - relevant_count) / total

    print(f"total examples: {total}")
    print(f"relevance accuracy: {relevance_correct / total:.4f}")
    print(f"majority-class baseline (always predict the more common label): {majority_baseline:.4f}")
    if resume_total:
        print(f"resume-variant accuracy (relevant only): {resume_correct / resume_total:.4f}")
    print(f"combined accuracy: {both_correct / total:.4f}")

    print("\nrelevance confusion matrix (rows=expected, cols=predicted):")
    print(f"                 predicted_relevant  predicted_not_relevant")
    print(f"  expected_relevant       {tp:>6}              {fn:>6}")
    print(f"  expected_not_relevant   {fp:>6}              {tn:>6}")

    print("\nresume-variant confusion matrix (relevant examples with correct relevance decision only,")
    print("rows=expected, cols=predicted):")
    print(f"                         predicted_fullstack  predicted_project_manager")
    for expected_variant in ("fullstack", "project_manager"):
        row = resume_confusion[expected_variant]
        print(
            f"  expected_{expected_variant:<15}   {row['fullstack']:>6}                {row['project_manager']:>6}"
        )

    print("\nconfidence calibration (quartiles of the routing confidence")
    print("tier1_classifier.is_confident() actually gates on):")
    _print_calibration_table(calibration_points)

    print("\nconfidence calibration for PREDICTED 'not relevant' (bucketed by relevance_confidence;")
    print("accuracy = fraction genuinely not relevant - this is the false-negative risk):")
    _print_calibration_table(not_relevant_points)

    print("\nconfidence calibration for PREDICTED 'relevant' (bucketed by")
    print("min(relevance_confidence, resume_confidence); accuracy = fraction genuinely relevant):")
    _print_calibration_table(relevant_points)

    fallback = total - local_routed
    print(
        f"\nprojected routing @ TIER1_NOT_RELEVANT_CONFIDENCE_MIN={args.not_relevant_confidence_min:.2f} "
        f"/ TIER1_RELEVANT_CONFIDENCE_MIN={args.relevant_confidence_min:.2f}:"
    )
    print(f"  auto-handled locally: {local_routed}/{total} ({local_routed / total:.1%})")
    print(f"  falls back to Claude: {fallback}/{total} ({fallback / total:.1%})")
    if local_routed:
        print(f"  accuracy among auto-handled: {local_routed_correct / local_routed:.4f}")


def _print_calibration_table(points: list[tuple[float, bool]], buckets: int = 4) -> None:
    print(f"  {'bucket':<10}{'n':>5}{'avg_confidence':>18}{'accuracy':>12}")
    if not points:
        print("  (no examples)")
        return
    points = sorted(points, key=lambda p: p[0])
    n = len(points)
    bucket_count = min(buckets, n)
    bucket_size = max(1, -(-n // bucket_count))  # ceil division: last bucket may be smaller
    for q in range(bucket_count):
        bucket = points[q * bucket_size : (q + 1) * bucket_size]
        if not bucket:
            continue
        avg_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        print(f"  Q{q + 1:<9}{len(bucket):>5}{avg_confidence:>18.4f}{accuracy:>12.4f}")


if __name__ == "__main__":
    main()
