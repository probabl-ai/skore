import datetime as dt
from pathlib import Path

from IPython.core.display import HTML
from jinja2 import Template
from sklearn.utils import estimator_html_repr


def relevant_versions():
    """Generate tags can be useful for generating versions. Dunno."""
    import sklearn

    versions = [f"scikit-learn {sklearn.__version__}"]
    return versions


def render_model_card(
    cross_validation_report, name, description, intended_use, limitations
):
    """Render a model card from the stats in the cross validation card."""
    jinja_path = Path(__file__).parent / "templates" / "model-card.jinja2"
    temp = Template(jinja_path.read_text())
    return HTML(
        temp.render(
            name=name,
            cv_metrics=cross_validation_report.metrics.report_metrics().to_html(),
            relevant_versions=relevant_versions(),
            timestamp=str(dt.datetime.now),
            model_repr=estimator_html_repr(cross_validation_report.estimator_),
            description=description,
            intended_use=intended_use,
            limitations=limitations,
        )
    )
