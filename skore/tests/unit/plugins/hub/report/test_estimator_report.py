from joblib import hash
from pydantic import ValidationError
from pytest import approx, fixture, mark, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, precision_score, r2_score

from skore import CrossValidationReport, EstimatorReport, evaluate
from skore._plugins.hub.artifact.media import (
    ChecksSummary,
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
    reports_with_check_results_cache = [
        (report, report._check_results_cache)
        for report in reports
        if hasattr(report, "_check_results_cache")
    ]
    object._clear_cache()
    for report, _ in reports_with_check_results_cache:
        del report._check_results_cache

    try:
        with io.BytesIO() as stream:
            joblib.dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in reports_with_cache:
            report._cache = cache
        for report, check_results_cache in reports_with_check_results_cache:
            report._check_results_cache = check_results_cache

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
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "0",
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "1",
                "output": None,
                "average": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "0",
                "output": None,
                "average": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "1",
                "output": None,
                "average": None,
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
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "0",
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.77778, abs=1e-4),
                "label": "1",
                "output": None,
                "average": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.84615, abs=1e-4),
                "label": "0",
                "output": None,
                "average": None,
            },
            {
                "name": "recall",
                "verbose_name": "Recall",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": "1",
                "output": None,
                "average": None,
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
            },
        ]

    @mark.respx(assert_all_called=False)
    def test_binary_metrics_includes_averaged_rows(self, project):
        X, y = make_classification(random_state=42)
        report = evaluate(RandomForestClassifier(random_state=42), X, y)

        report.metrics.add(make_scorer(precision_score, average="macro"), name="xxx")

        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        precision = [
            m
            for m in payload.metrics
            if m.name == "precision" and m.data_source == "test"
        ]
        assert len(precision) == 2
        assert {m.label for m in precision} == {"0", "1"}
        assert all(m.average is None for m in precision)

        custom = [
            m for m in payload.metrics if m.name == "xxx" and m.data_source == "test"
        ]
        assert len(custom) == 1
        assert custom[0].average == "macro"
        assert custom[0].label is None
        assert custom[0].value is not None

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

        for metric_name in ("precision_avg", "recall_avg", "roc_auc_avg"):
            macro_metrics = [
                m
                for m in payload.metrics
                if m.name == metric_name
                and m.average == "macro"
                and m.data_source == "test"
            ]
            assert len(macro_metrics) == 1

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

    @mark.respx(assert_all_called=False)
    def test_metrics_multioutput_regression(self, project):
        X, y = make_regression(n_targets=2, random_state=42)
        report = evaluate(Ridge(random_state=42), X, y)
        report.metrics.add(
            make_scorer(r2_score, multioutput="uniform_average"),
            name="r2_uniform",
        )

        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        r2_rows = [
            m for m in payload.metrics if m.name == "r2" and m.data_source == "test"
        ]
        r2_uniform_rows = [
            m
            for m in payload.metrics
            if m.name == "r2_uniform" and m.data_source == "test"
        ]

        assert {m.output for m in r2_rows} == {0, 1}
        assert all(m.average is None for m in r2_rows)
        assert len(r2_uniform_rows) == 1
        assert r2_uniform_rows[0].average == "uniform_average"
        assert r2_uniform_rows[0].output is None

    @mark.respx(assert_all_called=False)
    def test_metrics_multimetric_scorer(self, project):
        def my_multi_scorer(_estimator, _X, _y):
            return {"score_a_1": 1.0, "score_b_1": 2.0, "score_c_1": 3.0}

        X, y = make_classification(random_state=42)
        report = evaluate(RandomForestClassifier(random_state=42), X, y)
        report.metrics.add(my_multi_scorer)

        payload = EstimatorReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        custom = [m for m in payload.metrics if m.name.startswith("score_")]
        assert {m.name for m in custom} == {"score_a_1", "score_b_1", "score_c_1"}
        assert {m.verbose_name for m in custom} == {
            "score_a_1",
            "score_b_1",
            "score_c_1",
        }
        # train + test for each submetric
        assert len(custom) == 6
        assert len({m.name for m in custom}) == 3

    @mark.respx()
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            ChecksSummary,
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
        payload_dict.pop("environment")

        assert payload_dict == {
            "key": "<key>",
            "canonical_report_id": str(binary_classification.id),
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
