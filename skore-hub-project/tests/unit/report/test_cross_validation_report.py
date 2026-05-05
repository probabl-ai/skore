from io import BytesIO

from joblib import dump, hash
from pydantic import ValidationError
from pytest import fixture, mark, param, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project.artifact.media import (
    ConfusionMatrixDataFrameTest,
    ConfusionMatrixDataFrameTrain,
    ConfusionMatrixSVGTest,
    ConfusionMatrixSVGTrain,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PrecisionRecallSVGTest,
    PrecisionRecallSVGTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    RocSVGTest,
    RocSVGTrain,
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
    reports_with_cache = [
        (report, report._cache) for report in reports if hasattr(report, "_cache")
    ]
    object.clear_cache()

    try:
        with BytesIO() as stream:
            dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in reports_with_cache:
            report._cache = cache

    with Serializer(pickle_bytes) as serializer:
        checksum = serializer.checksum

    return pickle_bytes, checksum


@fixture
def payload(project, small_cv_binary_classification):
    # Force the compute of the permutations
    small_cv_binary_classification.inspection.permutation_importance(
        data_source="train", seed=42
    )
    small_cv_binary_classification.inspection.permutation_importance(
        data_source="test", seed=42
    )

    return CrossValidationReportPayload(
        project=project,
        report=small_cv_binary_classification,
        key="<key>",
    )


class TestCrossValidationReportPayload:
    @mark.respx(assert_all_called=False)
    def test_dataset_size(self, payload):
        assert payload.dataset_size == 10

    @mark.parametrize(
        ("splitter", "metadata", "expected_splits"),
        [
            param(
                KFold(n_splits=2, shuffle=False),
                {
                    "type": "KFold",
                    "n_splits": 2,
                    "n_repeats": None,
                    "shuffle": False,
                    "random_state": None,
                },
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ],
                id="KFold",
            ),
            param(
                RepeatedKFold(n_splits=2, n_repeats=2, random_state=0),
                {
                    "type": "RepeatedKFold",
                    "n_splits": 2,
                    "n_repeats": 2,
                    "shuffle": True,
                    "random_state": 0,
                },
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ],
                id="RepeatedKFold",
            ),
            param(
                ShuffleSplit(n_splits=2, test_size=0.25, random_state=0),
                {
                    "type": "ShuffleSplit",
                    "n_splits": 2,
                    "n_repeats": None,
                    "shuffle": True,
                    "random_state": 0,
                },
                [
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0],
                ],
                id="ShuffleSplit",
            ),
            param(
                TimeSeriesSplit(n_splits=2),
                {
                    "type": "TimeSeriesSplit",
                    "n_splits": 2,
                    "n_repeats": None,
                    "shuffle": False,
                    "random_state": None,
                },
                [
                    [0, 0, 0, 0, 1, 1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],
                id="TimeSeriesSplit",
            ),
        ],
    )
    def test_regression_splitting_strategy(
        self, project, splitter, metadata, expected_splits, monkeypatch
    ):
        monkeypatch.setattr(
            "skore_hub_project.report.cross_validation_report.TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT",
            10,
        )

        X, y = make_regression(random_state=42, n_samples=8)
        estimator = LinearRegression()

        report = CrossValidationReport(estimator, X, y, splitter=splitter)
        payload = CrossValidationReportPayload(
            project=project, report=report, key="<key>"
        )

        train_target_distributions = payload.splitting_strategy.pop(
            "train_target_distributions"
        )
        test_target_distributions = payload.splitting_strategy.pop(
            "test_target_distributions"
        )
        for train_distribution, test_distribution in zip(
            train_target_distributions, test_target_distributions, strict=True
        ):
            assert len(train_distribution) == len(test_distribution) == 10
            assert all(0 <= distribution <= 1 for distribution in train_distribution)
            assert all(0 <= distribution <= 1 for distribution in test_distribution)

        train_target_distributions_sample_counts = payload.splitting_strategy.pop(
            "train_target_distributions_sample_counts"
        )
        test_target_distributions_sample_counts = payload.splitting_strategy.pop(
            "test_target_distributions_sample_counts"
        )
        assert len(train_target_distributions_sample_counts) == len(
            test_target_distributions_sample_counts
        )
        assert len(train_target_distributions_sample_counts) == len(
            test_target_distributions_sample_counts
        )

        assert payload.splitting_strategy == {
            "splitter": metadata,
            "splits": expected_splits,
        }

    @mark.filterwarnings(
        # ignore deprecation warning due to `scikit-learn` misusing `scipy` arguments,
        # raised by `scipy`
        (
            "ignore:scipy.optimize.*The `disp` and `iprint` options of the L-BFGS-B "
            "solver are deprecated:DeprecationWarning"
        ),
    )
    @mark.parametrize(
        ("splitter", "metadata", "expected_splits"),
        [
            param(
                StratifiedKFold(n_splits=2, shuffle=False),
                {
                    "type": "StratifiedKFold",
                    "n_splits": 2,
                    "n_repeats": None,
                    "shuffle": False,
                    "random_state": None,
                },
                [
                    [1, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 1, 1],
                ],
                id="StratifiedKFold",
            ),
            param(
                RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=0),
                {
                    "type": "RepeatedStratifiedKFold",
                    "n_splits": 2,
                    "shuffle": True,
                    "n_repeats": 2,
                    "random_state": 0,
                },
                [
                    [1, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 1, 1],
                ],
                id="RepeatedStratifiedKFold",
            ),
            param(
                StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=0),
                {
                    "type": "StratifiedShuffleSplit",
                    "n_splits": 2,
                    "n_repeats": None,
                    "shuffle": True,
                    "random_state": 0,
                },
                [
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0],
                ],
                id="StratifiedShuffleSplit",
            ),
        ],
    )
    def test_classification_splitting_strategy(
        self,
        project,
        splitter,
        metadata,
        expected_splits,
        monkeypatch,
    ):
        X, y = make_classification(random_state=42, n_samples=8, n_classes=2)
        estimator = LogisticRegression(random_state=42)

        report = CrossValidationReport(estimator, X, y, splitter=splitter)
        payload = CrossValidationReportPayload(
            project=project, report=report, key="<key>"
        )

        train_target_distributions = payload.splitting_strategy.pop(
            "train_target_distributions"
        )
        test_target_distributions = payload.splitting_strategy.pop(
            "test_target_distributions"
        )

        for train_distribution, test_distribution in zip(
            train_target_distributions, test_target_distributions, strict=True
        ):
            assert len(train_distribution) == len(test_distribution) == 2

        train_target_distributions_sample_counts = payload.splitting_strategy.pop(
            "train_target_distributions_sample_counts"
        )
        test_target_distributions_sample_counts = payload.splitting_strategy.pop(
            "test_target_distributions_sample_counts"
        )
        assert len(train_target_distributions_sample_counts) == len(
            test_target_distributions
        )
        assert len(train_target_distributions_sample_counts) == len(
            test_target_distributions_sample_counts
        )

        assert payload.splitting_strategy == {
            "splitter": metadata,
            "splits": expected_splits,
        }

    @mark.respx(assert_all_called=False)
    def test_class_names(self, payload):
        assert payload.class_names == ["1", "0"]

    @mark.respx(assert_all_called=False)
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

    @mark.respx(assert_all_called=False)
    def test_target_range_classification(self, payload):
        assert payload.target_range is None

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined*:sklearn.exceptions.UndefinedMetricWarning"
    )
    @mark.respx()
    def test_estimators(self, project, payload, upload_mock):
        payload.report.cache_predictions()
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

    @mark.respx()
    def test_pickle(
        self, small_cv_binary_classification, project, payload, upload_mock
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
    @mark.respx(assert_all_called=False)
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

    @mark.respx(assert_all_called=False)
    def test_metrics_raises_exception(self, monkeypatch, payload):
        """
        Since metrics compute is multi-threaded, ensure that any exceptions thrown in a
        sub-thread are also thrown in the main thread.
        """

        def raise_exception(_):
            raise Exception("test_metrics_raises_exception")

        monkeypatch.setattr(
            "skore_hub_project.report.cross_validation_report.CrossValidationReportPayload.METRICS",
            [AccuracyTestMean],
        )
        monkeypatch.setattr(
            "skore_hub_project.metric.AccuracyTestMean.compute",
            raise_exception,
        )

        with raises(Exception, match="test_metrics_raises_exception"):
            list(map(type, payload.metrics))

    @mark.filterwarnings(
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn` and `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    @mark.respx()
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            ConfusionMatrixDataFrameTest,
            ConfusionMatrixDataFrameTrain,
            ConfusionMatrixSVGTest,
            ConfusionMatrixSVGTrain,
            EstimatorHtmlRepr,
            ImpurityDecrease,
            PermutationImportanceTest,
            PermutationImportanceTrain,
            PrecisionRecallDataFrameTest,
            PrecisionRecallDataFrameTrain,
            PrecisionRecallSVGTest,
            PrecisionRecallSVGTrain,
            RocDataFrameTest,
            RocDataFrameTrain,
            RocSVGTest,
            RocSVGTrain,
            TableReport,
        ]

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn` and `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    @mark.respx()
    def test_model_dump_classification(self, small_cv_binary_classification, payload):
        small_cv_binary_classification.cache_predictions()

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

    @mark.respx(assert_all_called=False)
    def test_exception(self, project):
        with raises(ValidationError):
            CrossValidationReportPayload(project=project, report=None, key="<key>")
