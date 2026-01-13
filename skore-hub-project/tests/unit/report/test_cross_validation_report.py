from io import BytesIO

from joblib import dump, hash
from pydantic import ValidationError
from pytest import fixture, mark, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project import Project
from skore_hub_project.artifact.media import (
    EstimatorHtmlRepr,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    RocTest,
    RocTrain,
)
from skore_hub_project.artifact.media.data import TableReport
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
    reports = [object] + getattr(object, "estimator_reports_", [])
    caches = [report_to_clear._cache for report_to_clear in reports]

    object.clear_cache()

    try:
        with BytesIO() as stream:
            dump(object, stream)
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

    def test_splits(self, payload):
        assert payload.splitting_strategy == {
            "repeat_count": 1,
            "seed": "None",
            "splits": [
                {
                    "test": {
                        "target_distribution": [3, 2],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [2, 3],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                },
                {
                    "test": {
                        "target_distribution": [2, 3],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [3, 2],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                },
            ],
            "strategy_name": "StratifiedKFold",
        }

    def test_splits_with_repetitions(self, project):
        X, y = make_classification(random_state=42, n_samples=10)
        payload = CrossValidationReportPayload(
            project=project,
            report=CrossValidationReport(
                RandomForestClassifier(random_state=42),
                X,
                y,
                splitter=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42),
            ),
            key="<key>",
        )
        assert payload.splitting_strategy == {
            "repeat_count": 2,
            "seed": "42",
            "splits": [
                {
                    "test": {
                        "target_distribution": [4, 1],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [1, 4],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                },
                {
                    "test": {
                        "target_distribution": [1, 4],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [4, 1],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                },
                {
                    "test": {
                        "target_distribution": [4, 1],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [1, 4],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                },
                {
                    "test": {
                        "target_distribution": [1, 4],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train": {
                        "target_distribution": [4, 1],
                        "groups": None,
                        "sample_count": 5,
                    },
                    "train_test_distribution": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                },
            ],
            "strategy_name": "RepeatedKFold",
        }

    def test_splits_for_regression(self, project):
        X, y = make_regression(random_state=42, n_samples=10)
        payload = CrossValidationReportPayload(
            project=project,
            report=CrossValidationReport(
                LinearRegression(),
                X,
                y,
                splitter=KFold(n_splits=2),
            ),
            key="<key>",
        )

        assert payload.splitting_strategy["repeat_count"] == 1
        assert payload.splitting_strategy["seed"] == "None"
        splits = payload.splitting_strategy["splits"]
        assert len(splits) == 2
        for split in splits:
            train_target_distribution = split["train"]["target_distribution"]
            test_target_distribution = split["test"]["target_distribution"]
            assert len(train_target_distribution) == 100
            assert len(test_target_distribution) == 100
            assert all(isinstance(value, float) for value in train_target_distribution)
            assert all(isinstance(value, float) for value in test_target_distribution)

    def test_class_names(self, payload):
        assert payload.class_names == ["1", "0"]

    def test_target_range_regression(self, project):
        X, y = make_regression(random_state=42, n_samples=10)
        payload = CrossValidationReportPayload(
            project=project,
            report=CrossValidationReport(
                LinearRegression(),
                X,
                y,
                splitter=KFold(n_splits=2),
            ),
            key="<key>",
        )
        assert payload.target_range == [float(y.min()), float(y.max())]

    def test_target_range_classification(self, payload):
        assert payload.target_range is None

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined*:sklearn.exceptions.UndefinedMetricWarning"
    )
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

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning"
    )
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

    @mark.filterwarnings(
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn`, which is a dependency of `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning:seaborn",
    )
    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            EstimatorHtmlRepr,
            PrecisionRecallTest,
            PrecisionRecallTrain,
            RocTest,
            RocTrain,
            TableReport,
        ]

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn`, which is a dependency of `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning:seaborn",
    )
    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_model_dump_classification(self, small_cv_binary_classification, payload):
        _, checksum = serialize(small_cv_binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("estimators")
        payload_dict.pop("metrics")
        payload_dict.pop("medias")
        payload_dict.pop("splitting_strategy")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": hash(small_cv_binary_classification.y),
            "ml_task": "binary-classification",
            "pickle": {
                "checksum": checksum,
                "content_type": "application/octet-stream",
            },
            "dataset_size": 10,
            "class_names": ["1", "0"],
            "groups": None,
            "target_range": None,
        }

    def test_exception(self):
        with raises(ValidationError):
            CrossValidationReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
