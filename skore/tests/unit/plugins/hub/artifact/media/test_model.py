from hashlib import blake2b

from pydantic import ValidationError
from pytest import mark, raises
from sklearn.utils import estimator_html_repr

from skore._plugins.hub.artifact.media import EstimatorHtmlRepr


@mark.respx()
def test_estimator_html_repr(binary_classification, project):
    content = estimator_html_repr(binary_classification.estimator_)
    expected_checksum = f"blake2b-{blake2b(content.encode('utf-8')).hexdigest()}"

    # build media; ``local_plan`` produces content+checksum locally (no network)
    media = EstimatorHtmlRepr(project=project, report=binary_classification)
    plan = media.local_plan()

    assert plan is not None
    assert plan.content_type == "text/html"
    assert plan.payload == content.encode("utf-8")
    assert plan.checksum == expected_checksum

    # wrong type
    with raises(
        ValidationError, match="Input should be an instance of EstimatorReport"
    ):
        EstimatorHtmlRepr(project=project, report=None)
