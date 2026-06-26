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
    object._clear_cache()

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
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc",
                "verbose_name": "ROC AUC",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss",
                "verbose_name": "Log loss",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.06911, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score",
                "verbose_name": "Brier score",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.00727, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time",
                "verbose_name": "Fit time (s)",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time",
                "verbose_name": "Predict time (s)",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "accuracy",
                "verbose_name": "Accuracy",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.9, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.77778, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.84615, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc",
                "verbose_name": "ROC AUC",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.98901, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss",
                "verbose_name": "Log loss",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.31686, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score",
                "verbose_name": "Brier score",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.09025, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time",
                "verbose_name": "Fit time (s)",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time",
                "verbose_name": "Predict time (s)",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
        ]

    @mark.respx(assert_all_called=False)
    def test_binary_metrics_excludes_averaged_rows(
        self, project, binary_classification, monkeypatch
    ):
        from unittest.mock import MagicMock

        import pandas as pd

        display = binary_classification.metrics.summarize(data_source="both")
        data = display.data.copy()
        macro_row = (
            data.loc[
                (data["metric_name"] == "precision") & (data["data_source"] == "test")
            ]
            .iloc[0]
            .copy()
        )
        macro_row["label"] = pd.NA
        macro_row["average"] = "macro"
        macro_row["score"] = 0.55
        data = pd.concat([data, pd.DataFrame([macro_row])], ignore_index=True)

        mock_display = MagicMock()
        mock_display.data = data
        monkeypatch.setattr(
            binary_classification.metrics,
            "summarize",
            lambda **kwargs: mock_display,
        )

        payload = EstimatorReportPayload(
            project=project,
            report=binary_classification,
            key="<key>",
        )

        assert not any(m.average == "macro" for m in payload.metrics)
        precision = [
            m
            for m in payload.metrics
            if m.name == "precision" and m.data_source == "test"
        ]
        assert len(precision) == 2
        assert {m.label for m in precision} == {0, 1}
        assert all(m.average is None for m in precision)

    @mark.respx(assert_all_called=False)
    def test_multiclass_metrics_includes_aggregate_averages(
        self, project, forest_multiclass_classification_with_train_test
    ):
        estimator, X_train, X_test, y_train, y_test = (
            forest_multiclass_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        for metric_name in ("precision", "recall", "roc_auc"):
            aggregates = [
                m
                for m in payload.metrics
                if m.name == metric_name
                and m.data_source == "test"
                and m.label is None
                and m.average is not None
            ]
            assert {m.average for m in aggregates} == {"macro"}

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
        binary_classification._cache_predictions()

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
