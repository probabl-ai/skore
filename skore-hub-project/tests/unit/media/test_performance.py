from io import BytesIO

from matplotlib import pyplot as plt
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project import bytes_to_b64_str, switch_mpl_backend
from skore_hub_project.media import (
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
)


def serialize(display):
    with switch_mpl_backend(), BytesIO() as stream:
        display.plot()
        display.figure_.savefig(stream, format="svg", bbox_inches="tight")

        figure_bytes = stream.getvalue()
        figure_b64_str = bytes_to_b64_str(figure_bytes)

        plt.close(display.figure_)

    return figure_b64_str


@mark.parametrize(
    "Media,report,accessor,verbose_name,data_source",
    (
        param(
            PrecisionRecallTest,
            "binary_classification",
            "precision_recall",
            "Precision Recall",
            "test",
            id="PrecisionRecallTest[estimator]",
        ),
        param(
            PrecisionRecallTrain,
            "binary_classification",
            "precision_recall",
            "Precision Recall",
            "train",
            id="PrecisionRecallTrain[estimator]",
        ),
        param(
            PrecisionRecallTest,
            "cv_binary_classification",
            "precision_recall",
            "Precision Recall",
            "test",
            id="PrecisionRecallTest[cross-validation]",
        ),
        param(
            PrecisionRecallTrain,
            "cv_binary_classification",
            "precision_recall",
            "Precision Recall",
            "train",
            id="PrecisionRecallTrain[cross-validation]",
        ),
        param(
            PredictionErrorTest,
            "regression",
            "prediction_error",
            "Prediction error",
            "test",
            id="PredictionErrorTest[estimator]",
        ),
        param(
            PredictionErrorTrain,
            "regression",
            "prediction_error",
            "Prediction error",
            "train",
            id="PredictionErrorTrain[estimator]",
        ),
        param(
            PredictionErrorTest,
            "cv_regression",
            "prediction_error",
            "Prediction error",
            "test",
            id="PredictionErrorTest[cross-validation]",
        ),
        param(
            PredictionErrorTrain,
            "cv_regression",
            "prediction_error",
            "Prediction error",
            "train",
            id="PredictionErrorTrain[cross-validation]",
        ),
        param(
            RocTest,
            "binary_classification",
            "roc",
            "ROC",
            "test",
            id="RocTest[estimator]",
        ),
        param(
            RocTrain,
            "binary_classification",
            "roc",
            "ROC",
            "train",
            id="RocTrain[estimator]",
        ),
        param(
            RocTest,
            "cv_binary_classification",
            "roc",
            "ROC",
            "test",
            id="RocTest[cross-validation]",
        ),
        param(
            RocTrain,
            "cv_binary_classification",
            "roc",
            "ROC",
            "train",
            id="RocTrain[cross-validation]",
        ),
    ),
)
def test_performance(
    monkeypatch, Media, report, accessor, verbose_name, data_source, request
):
    report = request.getfixturevalue(report)
    display = getattr(report.metrics, accessor)(data_source=data_source)
    display_serialized = serialize(display)

    # available accessor
    assert Media(report=report).model_dump() == {
        "key": accessor,
        "verbose_name": verbose_name,
        "category": "performance",
        "attributes": {"data_source": data_source},
        "parameters": {},
        "representation": {
            "media_type": "image/svg+xml;base64",
            "value": display_serialized,
        },
    }

    # unavailable accessor
    monkeypatch.delattr(report.metrics.__class__, accessor)

    assert Media(report=report).model_dump() == {
        "key": accessor,
        "verbose_name": verbose_name,
        "category": "performance",
        "attributes": {"data_source": data_source},
        "parameters": {},
        "representation": None,
    }

    # wrong type
    with raises(ValidationError):
        Media(report=None)
