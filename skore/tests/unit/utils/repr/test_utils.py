"""Unit tests for ``skore._utils.repr.utils`` (HTML fragment helpers)."""

from __future__ import annotations

import re

import pytest

from skore._utils.repr.utils import repair_estimator_html_for_slotted_host

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
