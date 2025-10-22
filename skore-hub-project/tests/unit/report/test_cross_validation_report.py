import numpy as np
from pydantic import ValidationError
from pytest import fixture, mark, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from skore import CrossValidationReport, EstimatorReport
from skore_hub_project import Project
from skore_hub_project.artifact.media.data import TableReport
from skore_hub_project.artifact.media.model import EstimatorHtmlRepr
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.metric import (
    AccuracyTestMean,
    AccuracyTestStd,
    AccuracyTrainMean,
    AccuracyTrainStd,
    BrierScoreTestMean,
    BrierScoreTestStd,
    BrierScoreTrainMean,
    BrierScoreTrainStd,
    FitTimeMean,
    FitTimeStd,
    LogLossTestMean,
    LogLossTestStd,
    LogLossTrainMean,
    LogLossTrainStd,
    PrecisionTestMean,
    PrecisionTestStd,
    PrecisionTrainMean,
    PrecisionTrainStd,
    PredictTimeTestMean,
    PredictTimeTestStd,
    PredictTimeTrainMean,
    PredictTimeTrainStd,
    RecallTestMean,
    RecallTestStd,
    RecallTrainMean,
    RecallTrainStd,
    RocAucTestMean,
    RocAucTestStd,
    RocAucTrainMean,
    RocAucTrainStd,
)
from skore_hub_project.report import (
    CrossValidationReportPayload,
    EstimatorReportPayload,
)


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    import io

    import joblib

    reports = [object] + getattr(object, "estimator_reports_", [])
    caches = [report_to_clear._cache for report_to_clear in reports]

    object.clear_cache()

    try:
        with io.BytesIO() as stream:
            joblib.dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in zip(reports, caches, strict=True):
            report._cache = cache

    with Serializer(pickle_bytes) as serializer:
        checksum = serializer.checksum

    return pickle_bytes, checksum


@fixture
def project():
    return Project("<tenant>", "<name>")


@fixture
def payload(project, small_cv_binary_classification):
    return CrossValidationReportPayload(
        project=project,
        report=small_cv_binary_classification,
        key="<key>",
    )


class TestCrossValidationReportPayload:
    def test_dataset_size(self, payload):
        assert payload.dataset_size == 10

    def test_splitting_strategy_name(self, payload):
        assert payload.splitting_strategy_name == "StratifiedKFold"

    def test_splits_test_samples_density(self, payload):
        assert payload.splits == [
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        ]

    def test_splits_test_samples_density_many_rows(self):
        X, y = make_regression(random_state=42, n_samples=10_000)
        cvr = CrossValidationReport(
            LinearRegression(),
            X,
            y,
            splitter=ShuffleSplit(random_state=42, n_splits=7),
        )
        payload = CrossValidationReportPayload(
            project=Project("<tenant>", "<name>"),
            report=cvr,
            key="<key>",
        )
        splits = payload.splits
        assert len(splits) == 7
        assert all(len(s) == 200 for s in splits)
        for s in splits:
            assert all(bucket >= 0 and bucket <= 1 for bucket in s)

    def test_class_names(self, payload):
        assert payload.class_names == ["1", "0"]

    def test_classes(self, payload):
        X, y = make_classification(
            random_state=42,
            n_samples=10_000,
            n_classes=2,
        )
        cvr = CrossValidationReport(
            LogisticRegression(),
            X,
            y,
            splitter=ShuffleSplit(random_state=42, n_splits=7),
        )
        payload = CrossValidationReportPayload(
            project=Project("<tenant>", "<name>"),
            report=cvr,
            key="<key>",
        )
        classes = payload.classes
        assert len(classes) == 200
        assert np.unique(classes).tolist() == [0, 1]
        assert np.sum(classes) == 93

    def test_classes_many_rows(self, payload):
        assert payload.classes == [0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    def test_estimators(self, project, payload, upload_mock):
        assert len(payload.estimators) == len(payload.report.estimator_reports_)

        for i, estimator in enumerate(payload.estimators):
            # Ensure payload is well constructed
            assert isinstance(estimator, EstimatorReportPayload)
            assert estimator.project == project
            assert estimator.report == payload.report.estimator_reports_[i]

            # ensure `upload` is well called
            pickle, checksum = serialize(payload.report.estimator_reports_[i])

            estimator.model_dump()

            assert upload_mock.called
            assert not upload_mock.call_args.args
            assert upload_mock.call_args.kwargs == {
                "project": project,
                "content": pickle,
                "content_type": "application/octet-stream",
            }

            upload_mock.reset_mock()

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    def test_pickle(
        self, small_cv_binary_classification, project, payload, upload_mock, respx_mock
    ):
        pickle, checksum = serialize(small_cv_binary_classification)

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

    def test_metrics(self, payload):
        assert list(map(type, payload.metrics)) == [
            AccuracyTestMean,
            AccuracyTestStd,
            AccuracyTrainMean,
            AccuracyTrainStd,
            BrierScoreTestMean,
            BrierScoreTestStd,
            BrierScoreTrainMean,
            BrierScoreTrainStd,
            LogLossTestMean,
            LogLossTestStd,
            LogLossTrainMean,
            LogLossTrainStd,
            PrecisionTestMean,
            PrecisionTestStd,
            PrecisionTrainMean,
            PrecisionTrainStd,
            RecallTestMean,
            RecallTestStd,
            RecallTrainMean,
            RecallTrainStd,
            RocAucTestMean,
            RocAucTestStd,
            RocAucTrainMean,
            RocAucTrainStd,
            FitTimeMean,
            FitTimeStd,
            PredictTimeTestMean,
            PredictTimeTestStd,
            PredictTimeTrainMean,
            PredictTimeTrainStd,
        ]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            EstimatorHtmlRepr,
            TableReport,
        ]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_model_dump(self, small_cv_binary_classification, payload):
        _, checksum = serialize(small_cv_binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("estimators")
        payload_dict.pop("metrics")
        payload_dict.pop("medias")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": "cffe9686d06a56d0afe0c3a29d3ac6bf",
            "ml_task": "binary-classification",
            "groups": None,
            "pickle": {
                "checksum": checksum,
                "content_type": "application/octet-stream",
            },
            "dataset_size": 10,
            "splitting_strategy_name": "StratifiedKFold",
            "splits": [[1, 1, 1, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]],
            "class_names": ["1", "0"],
            "classes": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        }

    def test_exception(self):
        with raises(ValidationError):
            CrossValidationReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
