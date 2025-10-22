from io import BytesIO

from matplotlib import pyplot as plt
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project import Project, switch_mpl_backend
from skore_hub_project.artifact.media import (
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
)
from skore_hub_project.artifact.serializer import Serializer


def serialize(display) -> bytes:
    with switch_mpl_backend(), BytesIO() as stream:
        display.plot()
        display.figure_.savefig(stream, format="svg", bbox_inches="tight")
        plt.close(display.figure_)

        figure_bytes = stream.getvalue()

    return figure_bytes


@mark.usefixtures("monkeypatch_artifact_hub_client")
@mark.usefixtures("monkeypatch_upload_routes")
@mark.usefixtures("monkeypatch_upload_with_mock")
@mark.parametrize(
    "Media,report,accessor,data_source",
    (
        param(
            PrecisionRecallTest,
            "binary_classification",
            "precision_recall",
            "test",
            id="PrecisionRecallTest[estimator]",
        ),
        param(
            PrecisionRecallTrain,
            "binary_classification",
            "precision_recall",
            "train",
            id="PrecisionRecallTrain[estimator]",
        ),
        param(
            PrecisionRecallTest,
            "cv_binary_classification",
            "precision_recall",
            "test",
            id="PrecisionRecallTest[cross-validation]",
        ),
        param(
            PrecisionRecallTrain,
            "cv_binary_classification",
            "precision_recall",
            "train",
            id="PrecisionRecallTrain[cross-validation]",
        ),
        param(
            PredictionErrorTest,
            "regression",
            "prediction_error",
            "test",
            id="PredictionErrorTest[estimator]",
        ),
        param(
            PredictionErrorTrain,
            "regression",
            "prediction_error",
            "train",
            id="PredictionErrorTrain[estimator]",
        ),
        param(
            PredictionErrorTest,
            "cv_regression",
            "prediction_error",
            "test",
            id="PredictionErrorTest[cross-validation]",
        ),
        param(
            PredictionErrorTrain,
            "cv_regression",
            "prediction_error",
            "train",
            id="PredictionErrorTrain[cross-validation]",
        ),
        param(
            RocTest,
            "binary_classification",
            "roc",
            "test",
            id="RocTest[estimator]",
        ),
        param(
            RocTrain,
            "binary_classification",
            "roc",
            "train",
            id="RocTrain[estimator]",
        ),
        param(
            RocTest,
            "cv_binary_classification",
            "roc",
            "test",
            id="RocTest[cross-validation]",
        ),
        param(
            RocTrain,
            "cv_binary_classification",
            "roc",
            "train",
            id="RocTrain[cross-validation]",
        ),
    ),
)
def test_performance(
    monkeypatch, Media, report, accessor, data_source, upload_mock, request
):
    project = Project("<tenant>", "<name>")
    report = request.getfixturevalue(report)
    display = getattr(report.metrics, accessor)(data_source=data_source)
    content = serialize(display)

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    # available accessor
    assert Media(project=project, report=report).model_dump() == {
        "content_type": "image/svg+xml",
        "name": accessor,
        "data_source": data_source,
        "checksum": checksum,
    }

    # ensure `upload` is well called
    assert upload_mock.called
    assert not upload_mock.call_args.args
    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content": content,
        "content_type": "image/svg+xml",
    }

    # unavailable accessor
    monkeypatch.delattr(report.metrics.__class__, accessor)
    upload_mock.reset_mock()

    assert Media(project=project, report=report).model_dump() == {
        "content_type": "image/svg+xml",
        "name": accessor,
        "data_source": data_source,
        "checksum": None,
    }

    # ensure `upload` is not called
    assert not upload_mock.called

    # wrong type
    with raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        Media(project=project, report=None)
