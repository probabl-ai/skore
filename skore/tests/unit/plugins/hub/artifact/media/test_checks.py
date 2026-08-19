import pytest
from pydantic import ValidationError

from skore.plugins.hub.artifact.media import ChecksSummary
from skore.plugins.hub.artifact.serializer import Serializer
from skore.plugins.hub.json import dumps


@pytest.mark.parametrize(
    "report",
    (
        pytest.param("binary_classification", id="estimator"),
        pytest.param("cv_binary_classification", id="cross-validation"),
    ),
)
@pytest.mark.respx()
def test_checks_summary(report, upload_mock, request, project):
    report = request.getfixturevalue(report)
    frame = report.checks.summarize(fast_mode=True).frame()
    content = dumps(
        frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
    )

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    assert ChecksSummary(project=project, report=report).model_dump() == {
        "content_type": "application/vnd.dataframe",
        "name": "checks_summary",
        "data_source": None,
        "checksum": checksum,
        "parameters": {"fast_mode": True},
    }

    assert upload_mock.called
    assert not upload_mock.call_args.args
    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content": content,
        "content_type": "application/vnd.dataframe",
    }

    with pytest.raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        ChecksSummary(project=project, report=None)
