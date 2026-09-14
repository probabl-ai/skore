"""Tests for paginated metrics HTML tables."""

import re

import pandas as pd

from skore._utils.repr.paginated_table import (
    METRICS_HTML_PAGE_SIZE,
    paginated_dataframe_html,
)


def _radio_names(html: str) -> set[str]:
    return set(re.findall(r'type="radio" name="([^"]+)"', html))


def _first_tbody_row_count(html: str) -> int:
    start = html.find("<tbody>")
    end = html.find("</tbody>")
    return html[start:end].count("<tr")


def test_short_frame_has_no_pager():
    df = pd.DataFrame({"Metric": [f"m{i}" for i in range(10)], "score": range(10)})
    html = paginated_dataframe_html(df)
    assert "skore-metrics-pager" not in html
    assert html == df.to_html(index=False)
    assert _first_tbody_row_count(html) == METRICS_HTML_PAGE_SIZE


def test_long_frame_paginates_at_ten_rows():
    df = pd.DataFrame({"Metric": [f"m{i}" for i in range(11)], "score": range(11)})
    html = paginated_dataframe_html(df)
    assert "skore-metrics-pager" in html
    assert html.count('type="radio"') == 2
    assert _first_tbody_row_count(html) == METRICS_HTML_PAGE_SIZE
    second = html[html.find("</tbody>") + len("</tbody>") :]
    assert _first_tbody_row_count(second) == 1
    for i in range(11):
        assert f"m{i}" in html
    assert "Showing 1-10 of 11" in html
    assert "Showing 11-11 of 11" in html


def test_two_calls_use_distinct_radio_names():
    df = pd.DataFrame({"Metric": [f"m{i}" for i in range(11)], "score": range(11)})
    names = _radio_names(paginated_dataframe_html(df)) | _radio_names(
        paginated_dataframe_html(df)
    )
    assert len(names) == 2


def test_nameless_range_index_is_not_reset_again():
    df = pd.DataFrame({"Metric": ["accuracy", "r2"], "score": [0.9, 0.8]})
    html = paginated_dataframe_html(df)
    assert ">index</th>" not in html
    assert ">Metric</th>" in html
    assert ">score</th>" in html


def test_named_index_becomes_a_column():
    series = pd.Series(
        [0.9, 0.8],
        index=pd.Index(["Accuracy", "R²"], name="Metric"),
        name="Estimator",
    )
    html = paginated_dataframe_html(series)
    assert "Accuracy" in html
    assert "Estimator" in html
    assert ">Metric</th>" in html
