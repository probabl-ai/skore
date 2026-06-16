"""Unit tests for ``skore._utils.repr`` helpers."""

from __future__ import annotations

import base64
import re

import pytest
import skrub
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skore._utils.repr.markdown import _markdown_estimator_kind
from skore._utils.repr.utils import (
    figure_to_html,
    repair_estimator_html_for_slotted_host,
)

_SCRIPT_STRIP = re.compile(
    r"<script\b[^>]*>.*?</script\s*[^>]*>", re.DOTALL | re.IGNORECASE
)


def _div_open_close_counts(html: str) -> tuple[int, int]:
    stripped = _SCRIPT_STRIP.sub("", html)
    opens = len(re.findall(r"<div\b", stripped))
    closes = len(re.findall(r"</div>", stripped))
    return opens, closes


class TestRepairEstimatorHtmlForSlottedHost:
    """Closing-div repair keeps fragments parseable before sibling slotted markup."""

    def test_balanced_fragment_unchanged(self):
        html = '<div id="a"><div id="b">x</div></div>'
        assert repair_estimator_html_for_slotted_host(html) is html
        o, c = _div_open_close_counts(html)
        assert o == c

    @pytest.mark.parametrize(
        ("raw", "expected_suffix"),
        [
            ('<div id="outer"><div id="inner">x</div>', "</div>"),
            ("<div><div><div>x", "</div></div></div>"),
        ],
    )
    def test_under_closed_divs_get_suffix(self, raw, expected_suffix):
        out = repair_estimator_html_for_slotted_host(raw)
        assert out == raw + expected_suffix
        o, c = _div_open_close_counts(out)
        assert o == c

    def test_script_body_ignored_for_div_counts(self):
        """``</div>`` inside ``<script>`` must not count as closing markup."""
        raw = '<div id="a"><script>const s = "</div>";</script>'
        out = repair_estimator_html_for_slotted_host(raw)
        assert out.endswith("</div>")
        assert out.count("</div>") == 2
        o, c = _div_open_close_counts(out)
        assert o == c

    def test_repaired_plus_following_markup_balanced(self):
        broken = '<div class="sk"><p>est</p>'
        repaired = repair_estimator_html_for_slotted_host(broken)
        following = '<div slot="table-report"></div>'
        combined = repaired + following
        o, c = _div_open_close_counts(combined)
        assert o == c


def test_figure_to_html_returns_base64_img(pyplot):

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    html = figure_to_html(fig)
    prefix = '<img src="data:image/png;base64,'
    assert html.startswith(prefix)
    png_bytes = base64.b64decode(html[len(prefix) : -2])
    assert png_bytes.startswith(b"\x89PNG")


def _bare_meta_estimator() -> BaseEstimator:
    class _BareMeta(BaseEstimator, MetaEstimatorMixin):
        pass

    return _BareMeta()


def _skrub_data_op():
    X, y = make_classification(n_samples=10, random_state=0)
    return skrub.X(X).skb.apply(LogisticRegression(), y=skrub.y(y))


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (LogisticRegression(), "scikit-learn estimator"),
        (Pipeline([("clf", LogisticRegression())]), "Pipeline"),
        (
            GridSearchCV(LogisticRegression(), param_grid={"C": [1.0]}),
            "meta-estimator GridSearchCV wrapping LogisticRegression",
        ),
        (_bare_meta_estimator(), "meta-estimator _BareMeta"),
        (_skrub_data_op(), "skrub DataOp"),
        (_skrub_data_op().skb.make_learner(), "skrub SkrubLearner"),
        (skrub.SelectCols(0), "skrub estimator"),
    ],
)
def test_markdown_estimator_kind(estimator, expected):
    assert _markdown_estimator_kind(estimator) == expected
