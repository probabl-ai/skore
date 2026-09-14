"""Paginated HTML tables for metrics summaries."""

from __future__ import annotations

import uuid

import pandas as pd

METRICS_HTML_PAGE_SIZE = 10


def paginated_dataframe_html(
    frame: pd.DataFrame | pd.Series,
    *,
    page_size: int = METRICS_HTML_PAGE_SIZE,
) -> str:
    """Render ``frame`` as HTML, paginated when it has more than ``page_size`` rows.

    Pagination is display-only: every row is in the DOM, and page switching is
    CSS-only (radio inputs) so it works in Jupyter and in report Shadow DOM.
    """
    df = _prepare_metrics_html_frame(frame)
    n_rows = len(df)
    if n_rows <= page_size:
        return df.to_html(index=False)

    uid = f"skore-metrics-pager-{uuid.uuid4().hex[:8]}"
    n_pages = (n_rows + page_size - 1) // page_size
    pages: list[str] = []
    for page_idx in range(n_pages):
        start = page_idx * page_size
        end = min(start + page_size, n_rows)
        checked = " checked" if page_idx == 0 else ""
        pages.append(
            f'<input type="radio" name="{uid}" id="{uid}-{page_idx}"'
            f' class="skore-metrics-page-input"{checked}>'
            f'<div class="skore-metrics-page">'
            f"{df.iloc[start:end].to_html(index=False)}"
            f'<p class="skore-metrics-pager-status">'
            f"Showing {start + 1}-{end} of {n_rows}</p>"
            f"</div>"
        )
    nav = "".join(
        f'<label for="{uid}-{page_idx}">{page_idx + 1}</label>'
        for page_idx in range(n_pages)
    )
    return (
        f'<div class="skore-metrics-pager" id="{uid}">'
        f"<style>{_pager_css(uid, n_pages)}</style>"
        f"{''.join(pages)}"
        f'<nav class="skore-metrics-pager-nav" aria-label="Metrics pages">'
        f"{nav}</nav>"
        f"</div>"
    )


def _prepare_metrics_html_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    df = frame.to_frame() if isinstance(frame, pd.Series) else frame
    if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
        return df
    return df.reset_index()


def _pager_css(uid: str, n_pages: int) -> str:
    active = ",".join(
        f'#{uid}:has(#{uid}-{i}:checked) label[for="{uid}-{i}"]' for i in range(n_pages)
    )
    return (
        f"#{uid} > input {{"
        "position:absolute;width:1px;height:1px;padding:0;margin:-1px;"
        "overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;"
        "}"
        f"#{uid} > .skore-metrics-page {{display:none;}}"
        f"#{uid} > input:checked + .skore-metrics-page {{display:block;}}"
        f"#{uid} > .skore-metrics-pager-nav {{"
        "display:flex;flex-wrap:nowrap;gap:0.15em;overflow-x:auto;"
        "margin:0.35em 0 0;"
        "}"
        f"#{uid} > .skore-metrics-pager-nav label {{"
        "cursor:pointer;user-select:none;color:var(--color-blue,#3043f0);"
        "font-weight:bold;line-height:1.5em;padding:0.35em 0.65em;"
        "border-bottom:2px solid transparent;flex-shrink:0;"
        "}"
        f"#{uid} .skore-metrics-pager-status {{"
        "margin:0.35em 0 0;font-size:11px;font-style:italic;opacity:0.88;"
        "color:var(--color-text,#000);"
        "}"
        f"{active} {{border-bottom-color:var(--color-blue,#3043f0);}}"
    )
