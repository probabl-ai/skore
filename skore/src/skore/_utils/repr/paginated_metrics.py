"""Paginated HTML tables for metrics summaries."""

from __future__ import annotations

import re

import pandas as pd

from skore._utils.repr.html_repr import render_template

METRICS_HTML_PAGE_SIZE = 10

_TR_OPEN_RE = re.compile(r"<tr\b")


def metrics_summary_html(
    frame: pd.DataFrame | pd.Series, *, inline_assets: bool = False
) -> str:
    """HTML for a metrics summary table.

    Report fragments use the default (``inline_assets=False``): short tables are
    ``DataFrame.to_html``, long tables are paginated markup whose CSS/JS come
    from the report shell.

    Jupyter display/accessor reprs pass ``inline_assets=True``: short tables keep
    pandas ``_repr_html_`` (MultiIndex rowspan), and long tables inline CSS/JS.
    """
    if inline_assets and len(frame) <= METRICS_HTML_PAGE_SIZE:
        html_frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
        return html_frame._repr_html_()

    df = _prepare_metrics_html_frame(frame)
    table_html = df.to_html(index=False)
    if len(df) <= METRICS_HTML_PAGE_SIZE:
        return table_html

    table_html = _annotate_tbody_rows(table_html)
    return render_template(
        "common/paginated_metrics.html.j2",
        {"table_html": table_html, "inline_assets": inline_assets},
    )


def _prepare_metrics_html_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Columnar frame for ``DataFrame.to_html(index=False)``.

    ``summarize().frame()`` uses a named index (metric / label / output).
    Already-reset nameless ``RangeIndex`` frames are left as-is so we do not
    add an extra ``index`` column. ``reset_index`` raises if an index level
    name collides with a column (e.g. a compared estimator named ``"Metric"``).
    """
    df = frame.to_frame() if isinstance(frame, pd.Series) else frame
    if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
        return df
    return df.reset_index()


def _annotate_tbody_rows(table_html: str) -> str:
    tbody_start = table_html.find("<tbody>")
    tbody_end = table_html.find("</tbody>")
    head = table_html[: tbody_start + len("<tbody>")]
    body = table_html[tbody_start + len("<tbody>") : tbody_end]
    tail = table_html[tbody_end:]
    index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal index
        annotated = f'<tr data-page="{index // METRICS_HTML_PAGE_SIZE}"'
        index += 1
        return annotated

    return head + _TR_OPEN_RE.sub(replace, body) + tail
