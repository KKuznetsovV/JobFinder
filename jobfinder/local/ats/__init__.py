"""Registry of deterministic ATS form adapters, matched by job posting URL."""
from __future__ import annotations

from jobfinder.local.ats.base import ATSAdapter, ATSFormError
from jobfinder.local.ats.comeet import ComeetAdapter
from jobfinder.local.ats.greenhouse import GreenhouseAdapter
from jobfinder.local.ats.jazzhr import JazzHRAdapter
from jobfinder.local.ats.lever import LeverAdapter
from jobfinder.local.ats.smartrecruiters import SmartRecruitersAdapter
from jobfinder.local.ats.workday import WorkdayAdapter

ADAPTERS: list[type[ATSAdapter]] = [
    GreenhouseAdapter,
    LeverAdapter,
    WorkdayAdapter,
    ComeetAdapter,
    SmartRecruitersAdapter,
    JazzHRAdapter,
]


def get_adapter_for_url(url: str) -> ATSAdapter | None:
    for adapter_cls in ADAPTERS:
        if adapter_cls.matches(url):
            return adapter_cls()
    return None
