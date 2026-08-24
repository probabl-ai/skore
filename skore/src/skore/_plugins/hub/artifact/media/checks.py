"""Definition of the payload used to associate checks media with a report."""

from __future__ import annotations

from typing import Literal

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
