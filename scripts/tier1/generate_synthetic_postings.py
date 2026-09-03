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

# Junior/entry-level titles for the two relevant buckets, closer to what the
# actual candidate (a career-changer into tech, with a hospitality-management
# background) would realistically be applying to - used when strong_match is
# True, alongside the STRONG_MATCH_* snippet pools below.
JUNIOR_TECH_TITLES = [
    "Junior Software Engineer", "Associate Software Engineer", "Software Engineer I",
    "Junior Full Stack Developer", "Entry-Level Web Developer", "Junior Backend Developer",
]
JUNIOR_PM_TITLES = [
    "Assistant Restaurant Manager", "Shift Manager", "Associate Project Manager",
    "Junior Project Coordinator", "Assistant Operations Manager", "Team Lead",
]

# "Hard positive" titles: round 2's relevant postings all read as obviously
# tech/PM-flavored (STRONG_MATCH_* pools + a junior qualifier sentence),
# which is easy for the model to learn as a shallow keyword shortcut but
# doesn't teach it to recognize genuinely-relevant postings that are worded
# like something else - e.g. a support/QA title that's quietly doing
# fullstack-adjacent work, or a retail/ops supervisor title that's quietly
# doing the same budget/staff-leadership work as the PM resume. These share
# surface vocabulary with the UNRELATED bucket (support tickets, shift
# scheduling, inventory) rather than with the STRONG_MATCH pools, and are
# meant to be genuinely harder for both the model and the Claude teacher to
# call - round 2 had zero examples like this, and its relevant-class recall
# (4/23) suggests it never learned to look past title/vocabulary shortcuts.
HARD_POSITIVE_TECH_TITLES = [
    "IT Support Specialist", "Technical Support Engineer", "QA Tester",
    "Business Systems Analyst", "Junior Data Analyst", "Help Desk Analyst II",
]
HARD_POSITIVE_PM_TITLES = [
    "Shift Lead", "Assistant Store Manager", "Night Shift Supervisor",
    "Retail Operations Supervisor", "Front of House Supervisor", "Store Manager Trainee",
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

# Hard-positive snippets deliberately read like generic support/ops/retail
# work first, with the actual resume-relevant substance folded in almost as
# an afterthought - mirroring how a real posting worded by a non-technical/
# non-PM hiring manager might undersell a genuinely relevant role. Written to
# still contain real overlap with the fullstack/PM resumes on file (scripting/
# automation/dashboards for tech; budgets/staff leadership/P&L for PM), just
# not phrased with the obvious buzzwords the STRONG_MATCH_* pools use.
HARD_POSITIVE_TECH_SNIPPETS = [
    "triaging support tickets, but the role has grown to include writing small Python scripts "
    "to automate repetitive manual reporting tasks",
    "answering internal help-desk requests, plus building and maintaining a few internal tools "
    "and dashboards with Node.js and SQL",
    "manual QA testing of the product, but you'll also write Selenium/JavaScript test automation "
    "scripts to replace repetitive manual test runs",
    "reviewing data entered by other teams and increasingly writing SQL queries and light "
    "reporting automation to replace manual spreadsheet work",
    "supporting internal business users, with a growing part of the job spent building simple "
    "internal web tools and automating recurring data pulls",
]
HARD_POSITIVE_PM_SNIPPETS = [
    "running the daily shift, including opening/closing procedures, cash reconciliation, and "
    "training new hires on store standards",
    "overseeing scheduling, inventory counts, and staff performance reviews for a busy retail "
    "location",
    "coordinating a small team overnight, handling vendor relationships, inventory ordering, "
    "and shift-level budget tracking",
    "managing day-to-day floor operations, resolving customer escalations, and owning "
    "labor-cost and shrink targets for the location",
    "supervising front-of-house staff, building weekly schedules, and reporting sales and "
    "labor numbers up to the general manager",
]

# Appended to strong-match postings to read like realistic junior/entry-level
# listings rather than generic senior-sounding ones - the candidate is a
# career-changer, so postings explicitly open to little experience or a
# transition are the most plausible genuine matches.
JUNIOR_TECH_QUALIFIERS = [
    "0-2 years of professional experience is fine - we're happy to train the right person",
    "recent bootcamp or CS-program graduates are encouraged to apply",
    "this is an entry-level role with mentorship from senior engineers",
    "prior professional software experience is a plus but not required for the right candidate",
]
JUNIOR_PM_QUALIFIERS = [
    "candidates transitioning from hospitality, retail, or restaurant management are strongly encouraged to apply",
    "we care more about your track record leading people and hitting targets than a specific industry background",
    "this is a great fit for someone moving from operations/venue management into a corporate PM track",
    "no formal PM certification required if you've managed a team, budget, or location before",
]


def _fullstack_description(rng: random.Random, strong_match: bool = False) -> str:
    pool = STRONG_MATCH_TECH_SNIPPETS if strong_match else TECH_SNIPPETS
    description = (
        f"We're looking for an engineer to build and maintain services using "
        f"{rng.choice(pool)}. You'll work closely with the team on "
        f"{rng.choice(pool)} and help scale our platform."
    )
    if strong_match:
        description += f" {rng.choice(JUNIOR_TECH_QUALIFIERS)}."
    return description


def _pm_description(rng: random.Random, strong_match: bool = False) -> str:
    pool = STRONG_MATCH_PM_SNIPPETS if strong_match else PM_SNIPPETS
    description = (
        f"We're looking for someone experienced in {rng.choice(pool)}. "
        f"You'll partner with engineering and design on {rng.choice(pool)}."
    )
    if strong_match:
        description += f" {rng.choice(JUNIOR_PM_QUALIFIERS)}."
    return description


def _unrelated_description(rng: random.Random) -> str:
    return (
        f"This role focuses on {rng.choice(UNRELATED_SNIPPETS)} and "
        f"{rng.choice(UNRELATED_SNIPPETS)}. No software engineering background required."
    )


def _hard_positive_fullstack_description(rng: random.Random) -> str:
    return (
        f"This role is primarily {rng.choice(HARD_POSITIVE_TECH_SNIPPETS)}. "
        f"Over time you'll also be doing more of: {rng.choice(HARD_POSITIVE_TECH_SNIPPETS)}. "
        f"{rng.choice(JUNIOR_TECH_QUALIFIERS)}."
    )


def _hard_positive_pm_description(rng: random.Random) -> str:
    return (
        f"This role is primarily {rng.choice(HARD_POSITIVE_PM_SNIPPETS)}. "
        f"You'll also be responsible for {rng.choice(HARD_POSITIVE_PM_SNIPPETS)}. "
        f"{rng.choice(JUNIOR_PM_QUALIFIERS)}."
    )


def generate_posting(
    rng: random.Random,
    index: int,
    strong_match_fraction: float = 0.0,
    target_relevant_fraction: float | None = None,
    hard_positive_fraction: float = 0.0,
) -> dict:
    """Roughly a third each of: fullstack-flavored, PM-flavored, and clearly
    unrelated postings - gives the teacher-labeling step (and eventually the
    fine-tuned model) a mix of both positive classes plus negatives.
    `strong_match_fraction` biases the fullstack/PM buckets toward the
    STRONG_MATCH_* snippet pools (closer resume keyword overlap) and, when
    picked, a junior/entry-level qualifier sentence - for generating a batch
    deliberately skewed toward the relevant class, to fix training-set class
    imbalance. `target_relevant_fraction`, if given, overrides the default
    even 1/3-1/3-1/3 bucket split: that fraction of postings are split evenly
    between fullstack/project_manager, and the rest are unrelated.
    `hard_positive_fraction` replaces that fraction of fullstack/project_manager
    postings with the HARD_POSITIVE_* pools instead of STRONG_MATCH_*/plain
    ones - support/QA/retail-ops-worded postings that should still genuinely
    match a resume, meant to teach recall beyond an obvious-title/buzzword
    shortcut (see HARD_POSITIVE_TECH_TITLES's comment for why this matters)."""
    if target_relevant_fraction is None:
        bucket = rng.choice(["fullstack", "project_manager", "unrelated"])
    elif rng.random() < target_relevant_fraction:
        bucket = rng.choice(["fullstack", "project_manager"])
    else:
        bucket = "unrelated"

    strong_match = rng.random() < strong_match_fraction
    hard_positive = bucket in ("fullstack", "project_manager") and rng.random() < hard_positive_fraction
    if bucket == "fullstack":
        if hard_positive:
            title = rng.choice(HARD_POSITIVE_TECH_TITLES)
            description = _hard_positive_fullstack_description(rng)
        else:
            title = rng.choice(JUNIOR_TECH_TITLES if strong_match else TITLES[:5])
            description = _fullstack_description(rng, strong_match=strong_match)
    elif bucket == "project_manager":
        if hard_positive:
            title = rng.choice(HARD_POSITIVE_PM_TITLES)
            description = _hard_positive_pm_description(rng)
        else:
            title = rng.choice(JUNIOR_PM_TITLES if strong_match else TITLES[5:9])
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
    parser.add_argument(
        "--target-relevant-fraction", type=float, default=None,
        help="Override the default even 1/3-1/3-1/3 bucket split: this fraction of postings are "
        "fullstack/project_manager (split evenly), the rest unrelated. Use with a high "
        "--strong-match-fraction to deliberately generate a relevant-class-heavy supplemental batch.",
    )
    parser.add_argument(
        "--hard-positive-fraction", type=float, default=0.0,
        help="Fraction of fullstack/project_manager postings that use the HARD_POSITIVE_* pools "
        "(support/QA/retail-ops-worded postings that should still genuinely match a resume) "
        "instead of the STRONG_MATCH_*/plain pools - targets relevant-class recall on "
        "borderline/non-obvious postings rather than just adding more clearly-relevant ones.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            posting = generate_posting(
                rng, i, args.strong_match_fraction, args.target_relevant_fraction, args.hard_positive_fraction
            )
            f.write(json.dumps(posting, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} synthetic postings to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
