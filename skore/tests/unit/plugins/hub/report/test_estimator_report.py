from joblib import hash
from pydantic import ValidationError
from pytest import approx, fixture, mark, raises
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from skore import CrossValidationReport, EstimatorReport, evaluate
from skore._plugins.hub.artifact.media import (
    ConfusionMatrixDataFrameTestAll,
    ConfusionMatrixDataFrameTestNone,
    ConfusionMatrixDataFrameTrainAll,
    ConfusionMatrixDataFrameTrainNone,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    TableReportTest,
    TableReportTrain,
)
from skore._plugins.hub.artifact.serializer import Serializer
from skore._plugins.hub.metric import Metric
from skore._plugins.hub.report import EstimatorReportPayload


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    import io

    import joblib

    reports = [object] + getattr(object, "reports_", [])
    reports_with_cache = [
        (report, report._cache) for report in reports if hasattr(report, "_cache")
    ]
    object.clear_cache()

    try:
        with io.BytesIO() as stream:
            joblib.dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in reports_with_cache:
            report._cache = cache

    with Serializer(pickle_bytes) as serializer:
        checksum = serializer.checksum

    return pickle_bytes, checksum


@fixture
def payload(project, binary_classification):
    # Force the compute of the permutations
    binary_classification.inspection.permutation_importance(
        data_source="train", seed=42
    )
    binary_classification.inspection.permutation_importance(data_source="test", seed=42)

    return EstimatorReportPayload(
        project=project,
        report=binary_classification,
        key="<key>",
    )


class TestEstimatorReportPayload:
    @mark.respx()
    def test_pickle(
        self, binary_classification, project, payload, upload_mock, respx_mock
    ):
        pickle, checksum = serialize(binary_classification)

        # Ensure payload is well constructed
        assert payload.pickle.checksum == checksum

        # Ensure payload is well constructed
        assert payload.pickle.checksum == checksum

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "content": pickle,
            "content_type": "application/octet-stream",
        }

    @mark.respx(assert_all_called=False)
    def test_metrics(self, payload):
        assert [m.model_dump() for m in payload.metrics] == [
            {
                "name": "accuracy",
                "verbose_name": "Accuracy",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "position": None,
            },
            {
                "name": "roc_auc",
                "verbose_name": "ROC AUC",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "position": None,
            },
            {
                "name": "log_loss",
                "verbose_name": "Log loss",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.06911, abs=1e-4),
                "position": None,
            },
            {
                "name": "brier_score",
                "verbose_name": "Brier score",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.00727, abs=1e-4),
                "position": None,
            },
            {
                "name": "fit_time",
                "verbose_name": "Fit time (s)",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "position": None,
            },
            {
                "name": "predict_time",
                "verbose_name": "Predict time (s)",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "position": None,
            },
            {
                "name": "accuracy",
                "verbose_name": "Accuracy",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.9, abs=1e-4),
                "position": None,
            },
            {
                "name": "roc_auc",
                "verbose_name": "ROC AUC",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.98901, abs=1e-4),
                "position": None,
            },
            {
                "name": "log_loss",
                "verbose_name": "Log loss",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.31686, abs=1e-4),
                "position": None,
            },
            {
                "name": "brier_score",
                "verbose_name": "Brier score",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.09025, abs=1e-4),
                "position": None,
            },
            {
                "name": "fit_time",
                "verbose_name": "Fit time (s)",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "position": None,
            },
            {
                "name": "predict_time",
                "verbose_name": "Predict time (s)",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "position": None,
            },
        ]

    @mark.respx(assert_all_called=False)
    def test_metrics_custom(self, project):
        def hello(_estimator, _X, _y):
            return 1

        X, y = make_classification(random_state=42)
        report = evaluate(RandomForestClassifier(random_state=42), X, y)

        report.metrics.add(hello)

        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        assert all(isinstance(m, Metric) for m in payload.metrics)
        assert [m for m in payload.metrics if "hello" in m.name] == [
            Metric(
                name="hello",
                verbose_name="Hello",
                data_source="train",
                greater_is_better=True,
                value=1.0,
            ),
            Metric(
                name="hello",
                verbose_name="Hello",
                data_source="test",
                greater_is_better=True,
                value=1.0,
            ),
        ]

    @mark.respx()
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            ConfusionMatrixDataFrameTestAll,
            ConfusionMatrixDataFrameTestNone,
            ConfusionMatrixDataFrameTrainAll,
            ConfusionMatrixDataFrameTrainNone,
            EstimatorHtmlRepr,
            ImpurityDecrease,
            PermutationImportanceTest,
            PermutationImportanceTrain,
            PrecisionRecallDataFrameTest,
            PrecisionRecallDataFrameTrain,
            RocDataFrameTest,
            RocDataFrameTrain,
            TableReportTest,
            TableReportTrain,
        ]

    @mark.respx()
    def test_model_dump(self, binary_classification, payload):
        binary_classification.cache_predictions()

        _, checksum = serialize(binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("metrics")
        payload_dict.pop("medias")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": hash(binary_classification.y_test),
            "ml_task": "binary-classification",
            "pickle": {
                "checksum": checksum,
                "content_type": "application/octet-stream",
            },
        }

    @mark.respx(assert_all_called=False)
    def test_exception(self, project):
        with raises(ValidationError):
            EstimatorReportPayload(project=project, report=None, key="<key>")
