from pydantic import ValidationError
from pytest import mark, param, raises

from skore._plugins.hub.artifact.media import ChecksSummary
from skore._plugins.hub.artifact.serializer import Serializer
from skore._plugins.hub.json import dumps


@mark.parametrize(
    "report",
    (
        param("binary_classification", id="estimator"),
        param("cv_binary_classification", id="cross-validation"),
    ),
)
@mark.respx()
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

    with raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        ChecksSummary(project=project, report=None)
