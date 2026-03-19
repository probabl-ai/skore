"""
Sphinx extension for generating the landing page HTML.

All code strings below are both executed (to verify correctness and capture
outputs) and syntax-highlighted for display on the landing page.
"""

from typing import Any

import matplotlib.pyplot as plt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from sphinx.application import Sphinx


def _highlight(code: str) -> str:
    return highlight(code, PythonLexer(), HtmlFormatter(nowrap=True))


def _code_block(code: str) -> str:
    return (
        '<div class="doctest highlight-default notranslate">\n'
        '<div class="highlight hl-ipython3 code-margin">'
        f"<pre><span></span>{_highlight(code)}</pre></div>\n"
        "</div>\n"
    )


def generate_landing_page(_app: Sphinx, _config: Any) -> None:
    """Generate the landing page feature sections HTML."""

    # Code strings: shown on the landing page AND executed during the build.
    data_loading_code = """\
from skrub.datasets import fetch_employee_salaries
dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y"""

    report_creation_code = """\
import sklearn
import skore
import skrub

model = skrub.tabular_pipeline(sklearn.linear_model.Ridge())
report_ridge = skore.evaluate(model, df, y, splitter=5)"""

    frame_code = "report_ridge.metrics.summarize().frame()"

    plot_code = """\
error = report_ridge.metrics.prediction_error()
error.plot(kind="actual_vs_predicted")"""

    project_code = """\
project = skore.Project(name="adult_census_survey")
project.put("ridge", report_ridge)"""

    # Execute all code to catch bugs and capture outputs.
    ns: dict = {}
    exec(data_loading_code, ns)
    exec(report_creation_code, ns)
    help_html = ns["report_ridge"]._create_help_html()
    frame_html = eval(frame_code, ns)._repr_html_()
    exec(plot_code, ns)
    plt.close("all")
    exec(project_code, ns)

    with open("_templates/generated_landing.html", "w", encoding="utf-8") as f:
        f.write(f"""\
  <!-- Feature Section 1 -->
  <div class="row mb-0 row-padding-between-features">
    <div class="col-md-5 mb-5">
      <h4 class="feature-title">Reports for your Experiments</h4>
      <p class="feature-text">Create structured reports to quickly evaluate and inspect
        your predictive models by using
        <a class="reference internal"
        href="reference/api/skore.EstimatorReport.html#estimatorreport"
        title="skore.EstimatorReport"><code class="xref py py-class docutils
        literal notranslate sk-landing-code"><span class="pre">EstimatorReport</span></code></a>
        or
        <a class="reference internal"
        href="reference/api/skore.CrossValidationReport.html#crossvalidationreport"
        title="skore.CrossValidationReport"><code class="xref py py-class docutils
        literal notranslate sk-landing-code"><span class="pre">CrossValidationReport</span></code></a>
        and even compare them by using
        <a class="reference internal"
        href="reference/api/skore.ComparisonReport.html#comparisonreport"
        title="skore.ComparisonReport"><code class="xref py py-class docutils
        literal notranslate sk-landing-code"><span class="pre">ComparisonReport</span></code></a>.
      </p>
    </div>

    <div class="col-md-7">
      <div class="container-fluid" style="max-width: 58rem;">
        <div class="sd-card card-landing">
          <div class="card-body">
            <details>
              <summary>Given some data
                <code class="docutils literal notranslate"><span class="pre">df</span></code>: <i>(expand for full code)</i>
              </summary>
              <div class="doctest highlight-default notranslate">
                <div class="highlight">
                  <pre style="overflow-x: auto;">{_highlight(data_loading_code)}</pre>
                </div>
              </div>
            </details>
          </div>
          <div class="highlight">
            <pre style="overflow-x: auto;">{_highlight(report_creation_code)}</pre>
          </div>
        </div>
      </div>
      <div class="container-fluid" style="max-width: 58rem;">
        <div class="sd-card card-landing">
          {help_html}
        </div>
      </div>
    </div>
  </div>

  <!-- Feature Section 2 -->
  <div class="row mb-0 flex-md-row-reverse row-padding-between-features">
    <div class="col-md-5">
      <h4 class="feature-title">Get Insights that Matter</h4>
      <p class="feature-text">Quickly generate beautiful visualizations with
        <code class="highlight python"><span class="n">display</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span></code>
        {_code_block(plot_code)}
        and access the raw data with
        <code class="highlight python"><span class="n">display</span><span class="o">.</span><span class="n">frame</span><span class="p">()</span></code>
        {_code_block(frame_code)}
      </p>
    </div>
    <div class="col-md-7">
      <div class="prediction-error-image"></div>
      <div class="container-fluid" style="max-width: 58rem">
        <div class="sd-card card-landing">
          <div class="card-body">
            <div class="output_subarea output_html rendered_html output_result">
              {frame_html}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Feature Section 3 -->
  <div class="row mb-0 row-padding-between-features">
    <div class="col-md-5">
      <h4 class="feature-title">Store and Retrieve your Reports, Locally or on Skore Hub</h4>
      <p class="feature-text">
        Store your reports by using
        <a class="reference internal"
        href="reference/api/skore.Project.html#project"
        title="skore.Project"><code class="xref py py-class docutils
        literal notranslate sk-landing-code"><span class="pre">Project</span></code></a>.
        Retrieve the most important reports to revisit your insights or compare with
        new experiments later.
      </p>
      {_code_block(project_code)}
      <p class="feature-text">
        When using
        <a class="reference internal"
        href="reference/api/skore.Project.html#project"
        title="skore.Project"><code class="xref py py-class docutils literal notranslate sk-landing-code"><span class="pre">Project</span></code></a>
        with <code>mode="hub"</code>, you can view your reports on
        <a href="https://skore.probabl.ai" target="_blank" rel="noopener noreferrer">Skore Hub</a>.
        For seeing an example of project and reports stored on Skore Hub, view the example
        <a href="{{{{ skore_hub_example_url }}}}" target="_blank" rel="noopener noreferrer">here</a>.
      </p>
    </div>
    <div class="col-md-7 pe-5">
      <div class="project-summary-image"></div>
    </div>
  </div>
""")


def setup(app):
    """Sphinx extension to generate landing page HTML."""
    app.connect("config-inited", generate_landing_page)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
