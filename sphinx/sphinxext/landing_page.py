"""
Sphinx extension for generating the landing page HTML.

All code strings below are both executed (to verify correctness and capture
outputs) and syntax-highlighted for display on the landing page.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from sphinx.application import Sphinx


def _code_block(app: Sphinx, code: str) -> str:
    highlighted = app.builder.highlighter.highlight_block(code, "python", force=True)
    return highlighted


def generate_landing_page(app: Sphinx) -> None:
    """Generate the landing page feature sections HTML."""

    # exec the code blocks in a namespace
    ns = {}

    data_loading_code = """\
from skrub.datasets import fetch_employee_salaries
dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y"""
    exec(data_loading_code, ns)

    report_creation_code = """\
import sklearn
import skore
import skrub

model = skrub.tabular_pipeline(sklearn.linear_model.Ridge())
report_ridge = skore.evaluate(model, df, y, splitter=5)
report_ridge.help()"""
    exec(report_creation_code, ns)
    help_html = ns["report_ridge"]._create_help_html()

    frame_code = """\
report_ridge.metrics.summarize().frame(
    aggregate=None
)"""
    frame_html = eval(frame_code, ns)._repr_html_()

    plot_code = """\
error = report_ridge.metrics.prediction_error()
error.plot(kind="actual_vs_predicted")"""
    exec(plot_code, ns)
    plt.close("all")

    project_code = """\
project = skore.Project(
    name="adult_census_survey", mode="local"
)
project.put("ridge", report_ridge)"""
    exec(project_code, ns)

    template_dir = Path(app.confdir) / "_templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("landing.html")

    output = template.render(
        load_data=_code_block(app, data_loading_code),
        create_report=_code_block(app, report_creation_code),
        help_html=help_html,
        frame_html=frame_html,
        plot_error=_code_block(app, plot_code),
        summarize=_code_block(app, frame_code),
        put_in_project=_code_block(app, project_code),
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
