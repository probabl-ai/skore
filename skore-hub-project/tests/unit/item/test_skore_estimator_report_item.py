import json
from io import BytesIO

import joblib
from numpy.testing import assert_array_equal
from pytest import fixture, raises
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore_hub_project.item import PickleItem, SkoreEstimatorReportItem
from skore_hub_project.item.item import ItemTypeError, bytes_to_b64_str


@fixture
def report():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture
def report_b64_str(report):
    with BytesIO() as stream:
        joblib.dump(report, stream)

        pickle_bytes = stream.getvalue()
        pickle_b64_str = bytes_to_b64_str(pickle_bytes)

    return pickle_b64_str


class TestSkoreEstimatorReportItem:
    def test_factory(self, report, report_b64_str):
        item = SkoreEstimatorReportItem.factory(report)

        assert isinstance(item, SkoreEstimatorReportItem)
        assert isinstance(item, PickleItem)
        assert item.pickle_b64_str == report_b64_str

    def test_factory_exception(self):
        with raises(ItemTypeError):
            SkoreEstimatorReportItem.factory(None)

    def test_parameters(self, report, report_b64_str):
        item = SkoreEstimatorReportItem.factory(report)
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "SkoreEstimatorReportItem",
                "parameters": {
                    "pickle_b64_str": report_b64_str,
                },
            }
        }

        # Ensure parameters are JSONable
        json.dumps(item_parameters)

    def test_metadata(self, monkeypatch, report, report_b64_str):
        metadata = {
            "estimator_class_name": "LinearRegression",
            "estimator_hyper_params": {
                "copy_X": True,
                "fit_intercept": True,
                "n_jobs": None,
                "positive": False,
            },
            "dataset_fingerprint": joblib.hash(report.y_test),
            "ml_task": "regression",
            "metrics": [
                {
                    "name": "r2",
                    "value": float(hash("r2-train")),
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                    "verbose_name": "R²",
                },
                {
                    "name": "r2",
                    "value": float(hash("r2-test")),
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                    "verbose_name": "R²",
                },
                {
                    "name": "rmse",
                    "value": float(hash("rmse-train")),
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 3,
                    "verbose_name": "RMSE",
                },
                {
                    "name": "rmse",
                    "value": float(hash("rmse-test")),
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 3,
                    "verbose_name": "RMSE",
                },
                {
                    "name": "fit_time",
                    "value": float(hash("fit_time")),
                    "data_source": None,
                    "greater_is_better": False,
                    "position": 1,
                    "verbose_name": "Fit time (s)",
                },
                {
                    "name": "predict_time",
                    "value": float(hash("predict_time_train")),
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 2,
                    "verbose_name": "Predict time (s)",
                },
                {
                    "name": "predict_time",
                    "value": float(hash("predict_time_test")),
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 2,
                    "verbose_name": "Predict time (s)",
                },
            ],
        }

        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        monkeypatch.setattr(
            "skore.sklearn._estimator.metrics_accessor._MetricsAccessor.r2",
            lambda self, data_source: hash(f"r2-{data_source}"),
        )
        monkeypatch.setattr(
            "skore.sklearn._estimator.metrics_accessor._MetricsAccessor.rmse",
            lambda self, data_source: hash(f"rmse-{data_source}"),
        )
        monkeypatch.setattr(
            "skore.sklearn._estimator.metrics_accessor._MetricsAccessor.timings",
            lambda self: {
                "fit_time": hash("fit_time"),
                "predict_time_test": hash("predict_time_test"),
                "predict_time_train": hash("predict_time_train"),
            },
        )

        assert item1.__metadata__ == metadata
        assert item2.__metadata__ == metadata

        # Ensure metadata is JSONable
        json.dumps(item1.__metadata__)
        json.dumps(item2.__metadata__)

    def test_raw(self, report, report_b64_str):
        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        def compare_report(a, b):
            assert a.estimator_.__class__ == b.estimator_.__class__
            assert a.estimator_.get_params() == b.estimator_.get_params()
            assert_array_equal(a.X_train, b.X_train)
            assert_array_equal(a.y_train, b.y_train)
            assert_array_equal(a.X_test, b.X_test)
            assert_array_equal(a.y_test, b.y_test)

        compare_report(item1.__raw__, report)
        compare_report(item2.__raw__, report)

    def test_representation(self, monkeypatch, report, report_b64_str):
        representation = {
            "related_items": [
                {
                    "key": "prediction_error",
                    "category": "performance",
                    "attributes": {"data_source": "train"},
                    "parameters": {},
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": None,
                    },
                },
                {
                    "key": "prediction_error",
                    "category": "performance",
                    "attributes": {"data_source": "test"},
                    "parameters": {},
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": None,
                    },
                },
                {
                    "key": "permutation",
                    "category": "feature_importance",
                    "attributes": {"data_source": "train", "method": "permutation"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "permutation",
                    "category": "feature_importance",
                    "attributes": {"data_source": "test", "method": "permutation"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "coefficients",
                    "category": "feature_importance",
                    "attributes": {"method": "coefficients"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "estimator_html_repr",
                    "category": "model",
                    "attributes": {},
                    "parameters": {},
                    "representation": {"media_type": "text/html", "value": None},
                },
            ]
        }

        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        def clean(representation):
            for item in representation["related_items"]:
                item["representation"]["value"] = None
            return representation

        assert clean(item1.__representation__) == representation
        assert clean(item2.__representation__) == representation

        # Ensure representation is JSONable
        json.dumps(item1.__representation__)
        json.dumps(item2.__representation__)
