"""Miscellaneous helpers for HTML repr and embedding (slots, fragments)."""

from __future__ import annotations

import base64
import re
from io import BytesIO

from matplotlib.figure import Figure

# Strip script bodies when counting <div> so strings like "</div>" in JS do not skew.
_HTML_SCRIPT_RE = re.compile(
    r"<script\b[^>]*>.*?</script\s*[^>]*>", re.DOTALL | re.IGNORECASE
)


def repair_estimator_html_for_slotted_host(html: str) -> str:
    """Append missing closing ``</div>`` tags so sibling slotted nodes stay valid.

    Estimator HTML from ``estimator_._repr_html_()`` is concatenated in the light DOM
    next to ``<div slot="table-report">`` and ``<div slot="checks-summary">`` on the
    same shadow host. Named slots only consider **direct children** of that host; if
    the sklearn fragment leaves ``<div>`` elements unclosed, the HTML parser nests the
    following slotted elements inside the estimator subtree, so they no longer
    receive slot assignment and can appear under the wrong tab.

    This function compares counts of ``<div>`` vs ``</div>`` on a script-stripped copy
    (so ``</div>`` inside ``<script>`` is ignored) and appends the deficit of closing
    ``</div>`` to the original string.
    """
    without_scripts = re.sub(_HTML_SCRIPT_RE, "", html)
    n_open = len(re.findall(r"<div\b", without_scripts))
    n_close = len(re.findall(r"</div>", without_scripts))
    deficit = n_open - n_close
    if deficit > 0:
        return f"{html}{'</div>' * deficit}"
    return html


def figure_to_html(figure: Figure) -> str:
    """Return an HTML ``<img>`` tag embedding a matplotlib figure as base64 PNG."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}">'
