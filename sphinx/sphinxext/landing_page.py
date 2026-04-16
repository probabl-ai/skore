"""
Sphinx extension for generating the landing page HTML.

On ``builder-inited`` (see :func:`setup`), runs the same Python snippets that are
syntax-highlighted on the page: ``exec`` for statements, ``eval`` for the
summarize ``.frame()`` expression (so we can call ``._repr_html_()`` on the
result). Snippet strings are defined once below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx

DATA_LOADING_CODE = """\
from skrub.datasets import fetch_employee_salaries
dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y"""

REPORT_CREATION_CODE = """\
import sklearn
import skore
import skrub

model = skrub.tabular_pipeline(sklearn.linear_model.Ridge())
report_ridge = skore.evaluate(model, df, y, splitter=5)
report_ridge"""

SUMMARIZE_FRAME_CODE = """\
report_ridge.metrics.summarize().frame(
    aggregate=None
)"""

PLOT_CODE = """\
error = report_ridge.metrics.prediction_error()
error.plot(kind="actual_vs_predicted")"""

PROJECT_CODE = """\
project = skore.Project(
    name="adult_census_survey", mode="local"
)
project.put("ridge", report_ridge)"""


def _code_block(app: Sphinx, code: str) -> str:
    highlighted = app.builder.highlighter.highlight_block(code, "python", force=True)
    return highlighted


def generate_landing_page(app: Sphinx) -> None:
    """Generate the landing page feature sections HTML."""

    template_dir = Path(app.confdir) / "_templates"
    ns: dict = {}

    exec(DATA_LOADING_CODE, ns)
    exec(REPORT_CREATION_CODE, ns)

    report_ridge = ns["report_ridge"]
    ridge_html_path = template_dir / "generated_landing_report_ridge.html"
    ridge_html_path.parent.mkdir(parents=True, exist_ok=True)
    ridge_html_path.write_text(report_ridge._repr_html_(), encoding="utf-8")

    # Expression only; exec would require an extra assignment not shown on the page.
    frame_html = eval(SUMMARIZE_FRAME_CODE, ns)._repr_html_()

    exec(PLOT_CODE, ns)
    plt.close("all")

    exec(PROJECT_CODE, ns)

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("landing.html")

    output = template.render(
        load_data=_code_block(app, DATA_LOADING_CODE),
        create_report=_code_block(app, REPORT_CREATION_CODE),
        frame_html=frame_html,
        plot_error=_code_block(app, PLOT_CODE),
        summarize=_code_block(app, SUMMARIZE_FRAME_CODE),
        put_in_project=_code_block(app, PROJECT_CODE),
    )

    (template_dir / "generated_landing.html").write_text(output)


def setup(app):
    """Sphinx extension to generate landing page HTML."""
    app.connect("builder-inited", generate_landing_page)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
