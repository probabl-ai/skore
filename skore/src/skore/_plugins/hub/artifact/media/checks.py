"""Definition of the payload used to associate checks media with a report."""

from __future__ import annotations

from json import loads
from typing import Literal

import pandas as pd

from skore._checks.base import CheckCode, CheckResult
from skore._plugins.hub.artifact.media.media import Media, Report
from skore._plugins.hub.json import dumps


class ChecksSummary(Media[Report]):  # noqa: D101
    name: Literal["checks_summary"] = "checks_summary"
    data_source: None = None
    content_type: Literal["application/vnd.dataframe"] = "application/vnd.dataframe"
    parameters: dict[Literal["fast_mode"], Literal[True]] = {"fast_mode": True}

    def content_to_upload(self) -> bytes:  # noqa: D102
        display = self.report.checks.summarize(fast_mode=True)
        frame = display.frame()

        return dumps(
            frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
        )


def check_results_cache_from_summary(
    report: Report, content: bytes
) -> dict[CheckCode, CheckResult]:
    """Rebuild a checks-results cache from an uploaded fast-mode summary."""
    cache: dict[CheckCode, CheckResult] = {}
    for row in pd.DataFrame.from_dict(loads(content), orient="tight").itertuples(
        index=False
    ):
        check = next(
            (check for check in report._checks_registry if check.code == row.code), None
        )
        if (
            row.section in ("skipped", "ignored")
            or check is None
            or report._report_type not in check.report_types
        ):
            continue
        cache[row.code] = {
            "title": row.title,
            # The summary stores the expanded documentation link. The cache keeps
            # the check's fragment so the link is rebuilt for the running version.
            "docs_url": check.docs_url,
            "explanation": None if row.explanation == "NaN" else row.explanation,
            "section": row.section,
        }
    return cache
