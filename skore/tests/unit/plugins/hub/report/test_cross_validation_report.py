from io import BytesIO

from joblib import dump, hash
from numpy import array
from pydantic import ValidationError
from pytest import fixture, mark, param, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
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
)
from skore._plugins.hub.artifact.media.data import TableReport
from skore._plugins.hub.artifact.serializer import Serializer
from skore._plugins.hub.metric import Metric
from skore._plugins.hub.report import (
    CrossValidationReportPayload,
    EstimatorReportPayload,
)


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    reports = [object] + getattr(object, "reports_", [])
    reports_with_cache = [
        (report, report._cache) for report in reports if hasattr(report, "_cache")
    ]
    object._clear_cache()

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
            "skore._plugins.hub.report.cross_validation_report.TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT",
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
        (
            "ignore:scipy.optimize.*The `disp` and `iprint` options of the L-BFGS-B "
            "solver are deprecated:DeprecationWarning"
        ),
    )
    def test_multioutput_regression_splitting_strategy(self, project, monkeypatch):
        """Regression for https://github.com/probabl-ai/skore/issues/3021."""
        monkeypatch.setattr(
            "skore._plugins.hub.report.cross_validation_report.TARGET_DISTRIBUTION_REPR_SAMPLE_COUNT",
            10,
        )

        X, y = make_regression(
            n_samples=100, n_features=10, n_targets=2, random_state=0
        )

        report = CrossValidationReport(
            DummyRegressor(), X, y, splitter=KFold(3, shuffle=False)
        )
        payload = CrossValidationReportPayload(
            project=project, report=report, key="<key>"
        )

        train_target_distributions = payload.splitting_strategy[
            "train_target_distributions"
        ]
        test_target_distributions = payload.splitting_strategy[
            "test_target_distributions"
        ]

        for train_distribution, test_distribution in zip(
            train_target_distributions, test_target_distributions, strict=True
        ):
            assert len(train_distribution) == len(test_distribution) == 20
            assert all(distribution >= 0 for distribution in train_distribution)
            assert all(distribution >= 0 for distribution in test_distribution)

        assert payload.ml_task == "multioutput-regression"
        assert payload.target_names == ["Target 0", "Target 1"]
        assert payload.target_range == [
            [float(y[:, 0].min()), float(y[:, 0].max())],
            [float(y[:, 1].min()), float(y[:, 1].max())],
        ]
        assert len(payload.target_range) == len(payload.target_names)

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

    def test_regression_splitting_do_not_call_get_n_splits(self, project):
        X = array([0, 1, 2, 3, 4])
        y = array([5, 6, 7, 8, 9])

        class Splitter:
            def split(self, X, y=None, groups=None):
                yield array([0, 1]), array([2, 3, 4])

            def get_n_splits(self, X, y=None, groups=None):
                raise Exception

        report = CrossValidationReport(DummyRegressor(), X, y, splitter=Splitter())
        payload = CrossValidationReportPayload(
            project=project, report=report, key="<key>"
        )

        assert payload.splitting_strategy["splits"] == [[0, 0, 1, 1, 1]]
        assert payload.splitting_strategy["splitter"] == {
            "type": "Splitter",
            "n_splits": 1,
            "n_repeats": None,
            "shuffle": False,
            "random_state": None,
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
        payload.report._cache_predictions()
        assert len(payload.estimators) == len(payload.report.reports_)

        for i, estimator in enumerate(payload.estimators):
            # Ensure payload is well constructed
            assert isinstance(estimator, EstimatorReportPayload)
            assert estimator.project == project
            assert estimator.report == payload.report.reports_[i]

            # ensure `upload` is well called
            pickle, _ = serialize(payload.report.reports_[i])

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
        # `small_cv_binary_classification` has too few labels
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning"
    )
    @mark.respx(assert_all_called=False)
    def test_metrics(self, payload):
        from pytest import approx

        # Float `value`s are wrapped in `approx` to absorb rounding noise on CI.
        # `fit_time` / `predict_time` are wall-clock measurements, so they use an
        # open-ended `approx` (infinite tolerance) that accepts any float.
        assert [m.model_dump() for m in payload.metrics] == [
            {
                "name": "accuracy_mean",
                "verbose_name": "Accuracy - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.4, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "accuracy_std",
                "verbose_name": "Accuracy - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "accuracy_mean",
                "verbose_name": "Accuracy - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "accuracy_std",
                "verbose_name": "Accuracy - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score_mean",
                "verbose_name": "Brier score - MEAN",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.32946, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score_std",
                "verbose_name": "Brier score - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.03320, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score_mean",
                "verbose_name": "Brier score - MEAN",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.02895, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "brier_score_std",
                "verbose_name": "Brier score - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.00128, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time_mean",
                "verbose_name": "Fit time (s) - MEAN",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time_std",
                "verbose_name": "Fit time (s) - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time_mean",
                "verbose_name": "Fit time (s) - MEAN",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "fit_time_std",
                "verbose_name": "Fit time (s) - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss_mean",
                "verbose_name": "Log loss - MEAN",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.90003, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss_std",
                "verbose_name": "Log loss - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.04497, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss_mean",
                "verbose_name": "Log loss - MEAN",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.17775, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "log_loss_std",
                "verbose_name": "Log loss - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.00232, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_mean",
                "verbose_name": "Precision - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.2, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_std",
                "verbose_name": "Precision - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.28284, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_mean",
                "verbose_name": "Precision - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.2, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_std",
                "verbose_name": "Precision - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.28284, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_mean",
                "verbose_name": "Precision - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_std",
                "verbose_name": "Precision - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_mean",
                "verbose_name": "Precision - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "precision_std",
                "verbose_name": "Precision - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time_mean",
                "verbose_name": "Predict time (s) - MEAN",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time_std",
                "verbose_name": "Predict time (s) - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time_mean",
                "verbose_name": "Predict time (s) - MEAN",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "predict_time_std",
                "verbose_name": "Predict time (s) - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=float("inf")),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_mean",
                "verbose_name": "Recall - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.5, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_std",
                "verbose_name": "Recall - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.70711, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_mean",
                "verbose_name": "Recall - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.5, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_std",
                "verbose_name": "Recall - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.70711, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_mean",
                "verbose_name": "Recall - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_std",
                "verbose_name": "Recall - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": 0,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_mean",
                "verbose_name": "Recall - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "recall_std",
                "verbose_name": "Recall - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": 1,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc_mean",
                "verbose_name": "ROC AUC - MEAN",
                "data_source": "test",
                "greater_is_better": True,
                "value": approx(0.45833, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc_std",
                "verbose_name": "ROC AUC - STD",
                "data_source": "test",
                "greater_is_better": False,
                "value": approx(0.29462, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc_mean",
                "verbose_name": "ROC AUC - MEAN",
                "data_source": "train",
                "greater_is_better": True,
                "value": approx(1.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
            {
                "name": "roc_auc_std",
                "verbose_name": "ROC AUC - STD",
                "data_source": "train",
                "greater_is_better": False,
                "value": approx(0.0, abs=1e-4),
                "label": None,
                "output": None,
                "average": None,
                "position": None,
            },
        ]

    @mark.filterwarnings(
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning"
    )
    @mark.respx(assert_all_called=False)
    def test_binary_metrics_excludes_averaged_rows(
        self, project, small_cv_binary_classification, monkeypatch
    ):
        from unittest.mock import MagicMock

        import pandas as pd

        from skore._plugins.hub.report import CrossValidationReportPayload

        display = small_cv_binary_classification.metrics.summarize(data_source="both")
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
            small_cv_binary_classification.metrics,
            "summarize",
            lambda **kwargs: mock_display,
        )

        payload = CrossValidationReportPayload(
            project=project,
            report=small_cv_binary_classification,
            key="<key>",
        )

        assert not any(m.average == "macro" for m in payload.metrics)
        precision = [
            m
            for m in payload.metrics
            if m.name == "precision_mean" and m.data_source == "test"
        ]
        assert len(precision) >= 1
        assert all(m.average is None for m in precision)
        assert all(m.label is not None for m in precision)

    @mark.filterwarnings(
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning"
    )
    @mark.respx(assert_all_called=False)
    def test_multiclass_metrics_includes_aggregate_averages(
        self, project, cross_validation_report_multiclass_classification
    ):
        from skore._plugins.hub.report import CrossValidationReportPayload

        payload = CrossValidationReportPayload(
            project=project,
            report=cross_validation_report_multiclass_classification,
            key="<key>",
        )

        for metric_name in ("precision_mean", "recall_mean", "roc_auc_mean"):
            aggregates = [
                m
                for m in payload.metrics
                if m.name == metric_name
                and m.data_source == "test"
                and m.label is None
                and m.average is not None
            ]
            assert {m.average for m in aggregates} == {"macro", "micro", "weighted"}

    @mark.filterwarnings(
        # `small_cv_binary_classification` has too few labels
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning"
    )
    @mark.respx(assert_all_called=False)
    def test_metrics_custom(self, project):
        def hello(_estimator, _X, _y):
            return 1

        X, y = make_classification(random_state=42, n_samples=10)
        report = evaluate(RandomForestClassifier(random_state=42), X, y, splitter=2)

        report.metrics.add(hello)

        payload = CrossValidationReportPayload(
            project=project,
            report=report,
            key="<key>",
        )

        assert all(isinstance(m, Metric) for m in payload.metrics)
        assert [m for m in payload.metrics if "hello" in m.name] == [
            Metric(
                name="hello_mean",
                verbose_name="Hello - MEAN",
                data_source="test",
                greater_is_better=True,
                value=1.0,
            ),
            Metric(
                name="hello_std",
                verbose_name="Hello - STD",
                data_source="test",
                greater_is_better=False,
                value=0.0,
            ),
            Metric(
                name="hello_mean",
                verbose_name="Hello - MEAN",
                data_source="train",
                greater_is_better=True,
                value=1.0,
            ),
            Metric(
                name="hello_std",
                verbose_name="Hello - STD",
                data_source="train",
                greater_is_better=False,
                value=0.0,
            ),
        ]

    @mark.filterwarnings(
        # seaborn's use of pandas
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
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
        small_cv_binary_classification._cache_predictions()

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
            "target_names": None,
        }

    @mark.respx(assert_all_called=False)
    def test_exception(self, project):
        with raises(ValidationError):
            CrossValidationReportPayload(project=project, report=None, key="<key>")
