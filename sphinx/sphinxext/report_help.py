"""
Sphinx extension for generating demo HTML files.

This module generates demo HTML files during the Sphinx build process
to showcase the skore reporting functionality on the landing page.
"""

from typing import Any

from sphinx.application import Sphinx
import sklearn
import skore
import skrub
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


def _code_block(code: str) -> str:
    """Wrap syntax-highlighted Python code in the standard Sphinx code block divs."""
    highlighted = highlight(code, PythonLexer(), HtmlFormatter(nowrap=True))
    return (
        '<div class="doctest highlight-default notranslate">\n'
        '<div class="highlight hl-ipython3 code-margin">'
        f"<pre><span></span>{highlighted}</pre></div>\n"
        "</div>\n"
    )


def generate_demo_files(_app: Sphinx, _config: Any) -> None:
    """Generate all demo HTML files for the landing page."""
    dataset = skrub.datasets.fetch_employee_salaries()
    X, y = dataset.X, dataset.y

    model = skrub.tabular_pipeline(sklearn.linear_model.Ridge())
    report_ridge = skore.evaluate(model, X, y, splitter=5)

    # demo_report_help_generated.html
    with open("_templates/demo_report_help_generated.html", "w", encoding="utf-8") as f:
        f.write(report_ridge._create_help_html())

    # demo_display.html
    frame_html = report_ridge.metrics.frame()._repr_html_()
    with open("_templates/demo_display.html", "w", encoding="utf-8") as f:
        f.write(
            '<div class="container-fluid" style="max-width: 58rem">\n'
            '  <div class="sd-card card-landing">\n'
            '    <div class="card-body">\n'
            '      <div class="output_subarea output_html rendered_html output_result">\n'
            f"        {frame_html}\n"
            "      </div>\n"
            "    </div>\n"
            "  </div>\n"
            "</div>\n"
        )

    # demo_report.html
    data_loading_code = """\
from skrub.datasets import fetch_employee_salaries
dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y
df"""
    report_code = """\
import sklearn
import skore
import skrub

model = skrub.tabular_pipeline(sklearn.linear_model.Ridge())

report_ridge = skore.evaluate(model, df, y, splitter=5)"""
    with open("_templates/demo_report.html", "w", encoding="utf-8") as f:
        data_highlighted = highlight(
            data_loading_code, PythonLexer(), HtmlFormatter(nowrap=True)
        )
        report_highlighted = highlight(
            report_code, PythonLexer(), HtmlFormatter(nowrap=True)
        )
        f.write(
            '<div class="container-fluid" style="max-width: 58rem;">\n'
            '  <div class="sd-card card-landing">\n'
            '    <div class="card-body">\n'
            "      <details>\n"
            "        <summary>Given some data\n"
            '          <code class="docutils literal notranslate">'
            '<span class="pre">df</span></code>: <i>(expand for full code)</i>\n'
            "        </summary>\n"
            '        <div class="doctest highlight-default notranslate">\n'
            '          <div class="highlight">\n'
            f'            <pre style="overflow-x: auto;">{data_highlighted}</pre>\n'
            "          </div>\n"
            "        </div>\n"
            "      </details>\n"
            "    </div>\n"
            '    <div class="highlight">\n'
            f'      <pre style="overflow-x: auto;">{report_highlighted}</pre>\n'
            "    </div>\n"
            "  </div>\n"
            "</div>\n"
        )

    # demo_display_code_frame.html
    with open("_templates/demo_display_code_frame.html", "w", encoding="utf-8") as f:
        f.write(_code_block("report_ridge.metrics.frame()"))

    # demo_display_code_plot.html
    with open("_templates/demo_display_code_plot.html", "w", encoding="utf-8") as f:
        f.write(_code_block(
            'error = report_ridge.metrics.prediction_error()\n'
            'error.plot(kind="actual_vs_predicted")'
        ))

    # demo_project.html
    with open("_templates/demo_project.html", "w", encoding="utf-8") as f:
        f.write(_code_block(
            'project = skore.Project(name="adult_census_survey")\n'
            'project.put("ridge", report_ridge)'
        ))


def setup(app):
    """Sphinx extension to generate demo HTML files for the landing page."""
    app.connect("config-inited", generate_demo_files)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
