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

    Rows are packed by metric group so a metric is not split across pages.
    A group larger than 10 rows occupies its own page (scroll inside a fixed
    height). ``DataFrame.to_html`` is called once so numeric formatting is
    consistent across pages.
    """
    df = _prepare_metrics_html_frame(frame)
    table_html = df.to_html(index=False)
    if len(df) <= METRICS_HTML_PAGE_SIZE:
        return table_html

    pages = _page_indices(df)
    table_html = _annotate_tbody_rows(table_html, pages)
    return render_template(
        "common/paginated_metrics.html.j2",
        {
            "table_html": table_html,
            "inline_assets": inline_assets,
            "page_size": METRICS_HTML_PAGE_SIZE,
        },
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


def _page_indices(df: pd.DataFrame) -> list[int]:
    """Assign a page index to each row, packing metric groups up to 10 rows."""
    for name in ("Metric", "metric"):
        if name in df.columns:
            keys = df[name].astype(str).tolist()
            break
    else:
        keys = [str(i) for i in range(len(df))]

    pages = [0] * len(df)
    page = 0
    used = 0
    i = 0
    while i < len(keys):
        j = i + 1
        while j < len(keys) and keys[j] == keys[i]:
            j += 1
        size = j - i
        if used and used + size > METRICS_HTML_PAGE_SIZE:
            page += 1
            used = 0
        for k in range(i, j):
            pages[k] = page
        used += size
        i = j
    return pages


def _annotate_tbody_rows(table_html: str, pages: list[int]) -> str:
    tbody_start = table_html.find("<tbody>")
    tbody_end = table_html.find("</tbody>")
    head = table_html[: tbody_start + len("<tbody>")]
    body = table_html[tbody_start + len("<tbody>") : tbody_end]
    tail = table_html[tbody_end:]
    index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal index
        annotated = f'<tr data-page="{pages[index]}"'
        index += 1
        return annotated

    return head + _TR_OPEN_RE.sub(replace, body, count=len(pages)) + tail
