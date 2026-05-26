from joblib import hash
from pydantic import ValidationError
from pytest import fixture, mark, raises
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
from skore._plugins.hub.metric import EstimatorReportMetric
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
        assert all(isinstance(m, EstimatorReportMetric) for m in payload.metrics)
        assert [(m.name, m.data_source) for m in payload.metrics] == [
            ("score", "train"),
            ("accuracy", "train"),
            ("roc_auc", "train"),
            ("log_loss", "train"),
            ("brier_score", "train"),
            ("fit_time", "train"),
            ("predict_time", "train"),
            ("score", "test"),
            ("accuracy", "test"),
            ("roc_auc", "test"),
            ("log_loss", "test"),
            ("brier_score", "test"),
            ("predict_time", "test"),
        ]

    @mark.respx(assert_all_called=False)
    def test_metrics_custom(self, project):
        def hello(estimator, X, y):
            return 1

        X, y = make_classification(random_state=42)
        report = evaluate(RandomForestClassifier(random_state=42), X, y)

        report.metrics.add(hello)

        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        assert all(isinstance(m, EstimatorReportMetric) for m in payload.metrics)
        assert [m for m in payload.metrics if "hello" in m.name] == [
            EstimatorReportMetric(
                name="hello",
                verbose_name="Hello",
                data_source="train",
                greater_is_better=True,
                position=0,
                value=1.0,
                report=report,
            ),
            EstimatorReportMetric(
                name="hello",
                verbose_name="Hello",
                data_source="test",
                greater_is_better=True,
                position=0,
                value=1.0,
                report=report,
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
