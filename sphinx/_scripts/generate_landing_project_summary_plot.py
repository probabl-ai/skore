# Generate the Plotly HTML fragment used by the custom landing page.
#
# Output:
# - sphinx/_templates/landing/project_summary_plot.html
#
# Executed during the Sphinx build (builder-inited), then included from the custom
# landing template via:
#
#    {% include "landing/project_summary_plot.html" %}

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import uuid

import pandas as pd
import plotly.io as pio

from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve()

# Ensure we import local repo code (skore/src) instead of site-packages.
# File is: repo_root/sphinx/_scripts/generate_landing_project_summary_plot.py
REPO_ROOT = HERE.parents[2]
SKORE_SRC = REPO_ROOT / "skore" / "src"
sys.path.insert(0, str(SKORE_SRC))

# Now it is safe to import local skore/skrub.
import skrub  # noqa: E402
import skore  # noqa: E402
from skore import CrossValidationReport  # noqa: E402


SPHINX_DIR = HERE.parents[1]
OUT_TEMPLATE = SPHINX_DIR / "_templates" / "landing" / "project_summary_plot.html"
OUT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)


def _build_project_summary():
    tmp_dir = tempfile.mkdtemp(prefix="skore-docs-landing-")
    os.chdir(tmp_dir)

    X, y = make_regression(n_samples=800, n_features=8, noise=15.0, random_state=0)
    df_num = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    df = df_num.copy()
    df["division"] = pd.cut(df["f0"], bins=4, labels=["A", "B", "C", "D"]).astype(str)

    report_ridge = CrossValidationReport(Ridge(alpha=1.0), df_num, y)

    report_hgbdt = CrossValidationReport(
        skrub.tabular_pipeline(HistGradientBoostingRegressor(random_state=0)),
        df,
        y,
    )

    project = skore.Project(name=f"adult_census_survey_{uuid.uuid4().hex[:8]}")
    project.put("ridge", report_ridge)
    project.put("hgbdt", report_hgbdt)

    return project.summarize()


def _walk_widgets(root):
    yield root
    kids = getattr(root, "children", None)
    if kids:
        for c in kids:
            yield from _walk_widgets(c)


def _set_checkboxes_by_description(root, descriptions_to_true: set[str]) -> None:
    for w in _walk_widgets(root):
        if hasattr(w, "description") and hasattr(w, "value"):
            if str(getattr(w, "description", "")) in descriptions_to_true:
                try:
                    w.value = True
                except Exception:
                    pass


def _extract_plotly_figure(summary):
    # Widget module location differs across versions.
    from skore.project._widget import ModelExplorerWidget  # type: ignore
    import skore.project._widget as widget_mod  # type: ignore

    # Avoid printing/displaying FigureWidget during Sphinx build.
    widget_mod.display = lambda *args, **kwargs: None

    w = ModelExplorerWidget(dataframe=summary)

    for attr in ("report_type_dropdown", "_report_type_dropdown"):
        if hasattr(w, attr):
            getattr(w, attr).value = "cross-validation"
            break
    for attr in ("task_type_dropdown", "_task_type_dropdown"):
        if hasattr(w, attr):
            getattr(w, attr).value = "regression"
            break
    for attr in ("color_by_dropdown", "_color_by_dropdown"):
        if hasattr(w, attr):
            getattr(w, attr).value = "RMSE"
            break

    _set_checkboxes_by_description(w._layout, {"Fit Time", "Predict Time", "RMSE"})

    w._update_plot()
    fig = getattr(w, "current_fig", None)
    if fig is None:
        raise RuntimeError("Widget did not produce a figure (current_fig is None).")
    return fig


def _write_plot_snippet(fig) -> None:
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )
    OUT_TEMPLATE.write_text(html, encoding="utf-8")
    print(f"[landing] wrote plot snippet to: {OUT_TEMPLATE}")


def main() -> None:
    summary = _build_project_summary()
    fig = _extract_plotly_figure(summary)
    _write_plot_snippet(fig)


if __name__ == "__main__":
    main()