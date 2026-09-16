"""Tests for paginated metrics HTML tables."""

import re

import pandas as pd

from skore._utils.repr.paginated_metrics import (
    METRICS_HTML_PAGE_SIZE,
    metrics_summary_html,
)


def _td_texts(html: str) -> list[str]:
    return re.findall(r"<td>(.*?)</td>", html, flags=re.DOTALL)


def _row_pages(html: str) -> list[int]:
    return [int(p) for p in re.findall(r'<tr data-page="(\d+)"', html)]


def _long_frame(n_extra: int = 1) -> pd.DataFrame:
    n_rows = METRICS_HTML_PAGE_SIZE + n_extra
    return pd.DataFrame(
        {
            "Metric": [f"m{i}" for i in range(n_rows)],
            "score": range(n_rows),
        }
    )


def test_short_frame_has_no_pager():
    df = pd.DataFrame(
        {
            "Metric": [f"m{i}" for i in range(METRICS_HTML_PAGE_SIZE)],
            "score": range(METRICS_HTML_PAGE_SIZE),
        }
    )
    html = metrics_summary_html(df)
    assert "skore-metrics-pager" not in html
    assert html == df.to_html(index=False)


def test_long_frame_paginates():
    df = _long_frame()
    html = metrics_summary_html(df)
    assert "skore-metrics-pager" in html
    assert "skore-metrics-pager-next" in html
    pages = _row_pages(html)
    assert len(pages) == len(df)
    assert max(pages) >= 1
    assert html.count("<tbody>") == 1


def test_report_fragment_does_not_inline_assets():
    html = metrics_summary_html(_long_frame())
    assert "<style>" not in html
    assert "<script>" not in html


def test_jupyter_long_table_inlines_assets():
    html = metrics_summary_html(_long_frame(), inline_assets=True)
    assert "<style>" in html
    assert "@media print" in html
    assert "skoreInitMetricsPagers" in html
    assert ":not(.is-ready)" in html


def test_jupyter_short_table_uses_pandas_repr():
    index = pd.MultiIndex.from_product([["R²"], [0, 1, 2]], names=["Metric", "Output"])
    df = pd.DataFrame({"Estimator": [1.0, 1.0, 1.0]}, index=index)
    html = metrics_summary_html(df, inline_assets=True)
    assert html == df._repr_html_()
    assert "rowspan" in html
    assert "skore-metrics-pager" not in html


def test_numeric_format_matches_full_to_html():
    df = pd.DataFrame(
        {
            "Metric": ["ones"] * METRICS_HTML_PAGE_SIZE + ["small"],
            "score": [1.0] * METRICS_HTML_PAGE_SIZE + [1 / 3],
        }
    )
    html = metrics_summary_html(df)
    table = html[html.find("<table") : html.find("</table>") + len("</table>")]
    assert _td_texts(table) == _td_texts(df.to_html(index=False))


def test_pages_are_fixed_size():
    df = _long_frame(n_extra=4)
    pages = _row_pages(metrics_summary_html(df))
    assert pages == [0] * METRICS_HTML_PAGE_SIZE + [1] * 4


def test_nameless_range_index_is_not_reset_again():
    df = pd.DataFrame({"Metric": ["accuracy", "r2"], "score": [0.9, 0.8]})
    html = metrics_summary_html(df)
    assert ">index</th>" not in html
    assert ">Metric</th>" in html
    assert ">score</th>" in html


def test_named_index_becomes_a_column():
    series = pd.Series(
        [0.9, 0.8],
        index=pd.Index(["Accuracy", "R²"], name="Metric"),
        name="Estimator",
    )
    html = metrics_summary_html(series)
    assert "Accuracy" in html
    assert "Estimator" in html
    assert ">Metric</th>" in html
