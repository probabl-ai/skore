"""
Sphinx extension that generates the Plotly HTML fragment embedded on the landing page.

This writes a template snippet to sphinx/_templates/landing/project_summary_plot.html
    {% include "landing/project_summary_plot.html" %}
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import uuid

import pandas as pd
import plotly.io as pio
import skrub
import skore
from skore import CrossValidationReport

from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def _build_project_summary():
    """Build a small Project + Summary, fast and deterministic."""
    tmp_dir = tempfile.mkdtemp(prefix="skore-docs-landing-")

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

    tmp_dir = Path(tempfile.mkdtemp(prefix="skore-docs-landing-"))

    project = skore.Project(
        name=f"adult_census_survey_{uuid.uuid4().hex[:8]}",
        workspace=tmp_dir,
    )

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
                w.value = True


def _extract_plotly_figure(summary):
    """Run the widget headlessly and extract its Plotly FigureWidget."""
    # Module name is version-dependent; current docs build uses the private module.
    from skore.project._widget import ModelExplorerWidget  # type: ignore
    import skore.project._widget as widget_mod  # type: ignore

    # Avoid printing/displaying FigureWidget during the docs build.
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


def _write_plot_snippet(sphinx_dir: Path, fig) -> None:
    out = sphinx_dir / "_templates" / "landing" / "project_summary_plot.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )
    out.write_text(html, encoding="utf-8")


def _on_builder_inited(app) -> None:
    sphinx_dir = Path(app.confdir)
    summary = _build_project_summary()
    fig = _extract_plotly_figure(summary)
    _write_plot_snippet(sphinx_dir, fig)


def setup(app):
    app.connect("builder-inited", _on_builder_inited)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": False,
    }
