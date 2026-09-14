"""Paginated HTML tables for metrics summaries."""

from __future__ import annotations

import re

import pandas as pd

from skore._utils.repr.html_repr import render_template

METRICS_HTML_PAGE_SIZE = 10

_TR_OPEN_RE = re.compile(r"<tr\b")


def metrics_summary_html(frame: pd.DataFrame | pd.Series) -> str:
    """HTML for Jupyter display/accessor reprs.

    Short tables keep pandas ``_repr_html_`` (MultiIndex rowspan). Longer tables
    use :func:`paginated_metrics_html`.
    """
    if len(frame) <= METRICS_HTML_PAGE_SIZE:
        html_frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
        return html_frame._repr_html_()
    return paginated_metrics_html(frame, inline_assets=True)


def paginated_metrics_html(
    frame: pd.DataFrame | pd.Series, *, inline_assets: bool = False
) -> str:
    """Render ``frame`` as HTML, paginated when it has more than 10 rows.

    ``DataFrame.to_html`` is called once so numeric formatting is consistent
    across pages.
    """
    df = _prepare_metrics_html_frame(frame)
    table_html = df.to_html(index=False)
    if len(df) <= METRICS_HTML_PAGE_SIZE:
        return table_html

    table_html = _annotate_tbody_rows(table_html, n_rows=len(df))
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


def _annotate_tbody_rows(table_html: str, *, n_rows: int) -> str:
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

    return head + _TR_OPEN_RE.sub(replace, body, count=n_rows) + tail
