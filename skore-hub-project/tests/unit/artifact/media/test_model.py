from pydantic import ValidationError
from pytest import mark, raises
from sklearn.utils import estimator_html_repr

from skore_hub_project.artifact.media import EstimatorHtmlRepr
from skore_hub_project.artifact.serializer import Serializer


@mark.respx()
def test_estimator_html_repr(binary_classification, upload_mock, project):
    content = estimator_html_repr(binary_classification.estimator_)

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    # create media
    media = EstimatorHtmlRepr(project=project, report=binary_classification)
    media_dict = media.model_dump()

    # ensure `upload` is well called
    assert upload_mock.called
    assert not upload_mock.call_args.args
    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content": content,
        "content_type": "text/html",
    }

    # ensure payload is well constructed
    assert media_dict == {
        "content_type": "text/html",
        "name": "estimator_html_repr",
        "data_source": None,
        "checksum": checksum,
    }

    # wrong type
    with raises(
        ValidationError, match="Input should be an instance of EstimatorReport"
    ):
        EstimatorHtmlRepr(project=project, report=None)
