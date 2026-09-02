"""Generate synthetic job-posting training examples to supplement real,
production-logged ones (see jobfinder.ai.tier1_classifier._log_training_example).

Same "teacher model" data-generation approach used by the sibling SmartServe
project (smartserve-agent-model/data/generate_dataset.py): a deterministic
Python generator invents varied, structurally distinct synthetic job
postings, then a separate step (label_examples.py) asks the real Claude
relevance/resume-selection logic to label them, exactly as production
traffic already does. This script only produces the *unlabeled* postings —
it makes no API calls and costs nothing to run repeatedly.

Run:
    python scripts/tier1/generate_synthetic_postings.py \\
        --out scripts/tier1/data/synthetic_postings.jsonl \\
        --count 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TITLES = [
    "Backend Engineer", "Senior Software Engineer", "Full Stack Developer",
    "Platform Engineer", "Site Reliability Engineer", "Technical Project Manager",
    "Engineering Manager", "Product Manager", "Scrum Master", "Data Engineer",
    "DevOps Engineer", "Frontend Developer", "Mobile Developer (iOS)",
    "QA Automation Engineer", "Solutions Architect", "Sales Development Representative",
    "Marketing Coordinator", "Graphic Designer", "Warehouse Associate", "Executive Assistant",
]

COMPANIES = [
    "Acme Corp", "Northwind Traders", "Globex", "Initech", "Umbrella Labs",
    "Stark Industries", "Wayne Enterprises", "Hooli", "Pied Piper", "Soylent Co",
]

TECH_SNIPPETS = [
    "Python, Django, and PostgreSQL", "Node.js and React", "Go microservices on Kubernetes",
    "Java/Spring backend services", "AWS infrastructure and Terraform", "CI/CD pipelines with Jenkins",
    "GraphQL APIs and TypeScript", "distributed systems at scale",
]

# Closer keyword overlap with the actual fullstack resume on file (React,
# Node/Express, TypeScript, MongoDB/PostgreSQL, Docker, CI/CD, microservices)
# - used by --strong-match to generate a batch skewed toward the relevant
# class, since the default TECH_SNIPPETS bucket under-produces true
# positives (most real postings aren't a fit either, but a training set
# needs enough positive examples to learn from).
STRONG_MATCH_TECH_SNIPPETS = [
    "React, TypeScript, and Redux on the frontend", "Node.js and Express REST APIs",
    "MongoDB and PostgreSQL data layers", "Docker containers and CI/CD via GitHub Actions",
    "a microservices architecture", "JWT-based auth and SPA architecture",
]

PM_SNIPPETS = [
    "leading cross-functional agile teams", "managing product roadmaps and stakeholder alignment",
    "running sprint planning and retrospectives", "coordinating release schedules across teams",
    "gathering requirements from customers and translating them into specs",
]

# Closer overlap with the actual project_manager resume on file (operations/
# venue/GM management: budgets, staff leadership, training programs) rather
# than classic software-team scrum/product management.
STRONG_MATCH_PM_SNIPPETS = [
    "full operational management of a venue or business unit, including budgets and cost analysis",
    "leading and training frontline staff",
    "day-to-day management of a hospitality or retail location",
    "hitting financial and operational targets for a location or division",
]

UNRELATED_SNIPPETS = [
    "operating forklifts and managing warehouse inventory", "cold-calling prospects to book demos",
    "designing marketing collateral and social media graphics",
    "scheduling executive travel and managing calendars",
    "processing payroll and benefits administration",
]


def _fullstack_description(rng: random.Random, strong_match: bool = False) -> str:
    pool = STRONG_MATCH_TECH_SNIPPETS if strong_match else TECH_SNIPPETS
    return (
        f"We're looking for an engineer to build and maintain services using "
        f"{rng.choice(pool)}. You'll work closely with the team on "
        f"{rng.choice(pool)} and help scale our platform."
    )


def _pm_description(rng: random.Random, strong_match: bool = False) -> str:
    pool = STRONG_MATCH_PM_SNIPPETS if strong_match else PM_SNIPPETS
    return (
        f"We're looking for someone experienced in {rng.choice(pool)}. "
        f"You'll partner with engineering and design on {rng.choice(pool)}."
    )


def _unrelated_description(rng: random.Random) -> str:
    return (
        f"This role focuses on {rng.choice(UNRELATED_SNIPPETS)} and "
        f"{rng.choice(UNRELATED_SNIPPETS)}. No software engineering background required."
    )


def generate_posting(rng: random.Random, index: int, strong_match_fraction: float = 0.0) -> dict:
    """Roughly a third each of: fullstack-flavored, PM-flavored, and clearly
    unrelated postings - gives the teacher-labeling step (and eventually the
    fine-tuned model) a mix of both positive classes plus negatives.
    `strong_match_fraction` biases the fullstack/PM buckets toward the
    STRONG_MATCH_* snippet pools (closer resume keyword overlap), for
    generating a supplemental batch when the default mix under-produces
    the relevant class."""
    bucket = rng.choice(["fullstack", "project_manager", "unrelated"])
    strong_match = rng.random() < strong_match_fraction
    if bucket == "fullstack":
        title = rng.choice(TITLES[:5])
        description = _fullstack_description(rng, strong_match=strong_match)
    elif bucket == "project_manager":
        title = rng.choice(TITLES[5:9])
        description = _pm_description(rng, strong_match=strong_match)
    else:
        title = rng.choice(TITLES[9:])
        description = _unrelated_description(rng)

    return {
        "synthetic_id": f"synthetic-{index}",
        "title": title,
        "company": rng.choice(COMPANIES),
        "description": description,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("scripts/tier1/data/synthetic_postings.jsonl"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strong-match-fraction", type=float, default=0.0,
        help="Fraction of fullstack/PM postings biased toward closer resume-keyword overlap.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(json.dumps(generate_posting(rng, i, args.strong_match_fraction), ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} synthetic postings to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
