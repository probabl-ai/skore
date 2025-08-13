from pydantic import ValidationError
from pytest import raises
from sklearn.utils import estimator_html_repr
from skore_hub_project.media import EstimatorHtmlRepr


def test_estimator_html_repr(monkeypatch, binary_classification):
    media = EstimatorHtmlRepr(report=binary_classification)
    media_dict = media.model_dump()

    assert media_dict == {
        "key": "estimator_html_repr",
        "verbose_name": "Estimator HTML representation",
        "category": "model",
        "attributes": {},
        "parameters": {},
        "representation": {
            "media_type": "text/html",
            "value": estimator_html_repr(binary_classification.estimator_),
        },
    }

    # wrong type
    with raises(ValidationError):
        EstimatorHtmlRepr(report=None)
