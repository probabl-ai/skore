from io import BytesIO
from json import dumps

from joblib import dump, hash
from numpy.testing import assert_array_equal
from pytest import fixture, raises
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore_remote_project.item import PickleItem, SkoreEstimatorReportItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str


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
        dump(report, stream)

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
        dumps(item_parameters)

    def test_metadata(self, report, report_b64_str):
        metadata = {
            "estimator_class_name": "LinearRegression",
            "estimator_hyper_params": {
                "copy_X": True,
                "fit_intercept": True,
                "n_jobs": None,
                "positive": False,
            },
            "dataset_fingerprint": hash(
                (report.X_train, report.y_train, report.X_test, report.y_test)
            ),
            "ml_task": "regression",
            "metrics": [
                {
                    "name": "r2",
                    "value": 0.6688617330074285,
                    "data_source": "test",
                    "greater_is_better": None,
                },
                {
                    "name": "r2",
                    "value": 1.0,
                    "data_source": "train",
                    "greater_is_better": None,
                },
                {
                    "name": "rmse",
                    "value": 73.92385036768957,
                    "data_source": "test",
                    "greater_is_better": None,
                },
                {
                    "name": "rmse",
                    "value": 2.091309874874337e-13,
                    "data_source": "train",
                    "greater_is_better": None,
                },
            ],
        }

        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        assert item1.__metadata__ == metadata
        assert item2.__metadata__ == metadata

        # Ensure metadata is JSONable
        dumps(item1.__metadata__)
        dumps(item2.__metadata__)

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
                    "data_source": "test",
                    "parameters": {
                        "class": "MatplotlibFigureItem",
                        "parameters": {
                            "figure_b64_str": bytes_to_b64_str(b"<raw-figure>"),
                        },
                    },
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": bytes_to_b64_str(b"<svg-figure>"),
                    },
                },
                {
                    "key": "prediction_error",
                    "data_source": "train",
                    "parameters": {
                        "class": "MatplotlibFigureItem",
                        "parameters": {
                            "figure_b64_str": bytes_to_b64_str(b"<raw-figure>"),
                        },
                    },
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": bytes_to_b64_str(b"<svg-figure>"),
                    },
                },
            ]
        }

        def savefig(self, stream, *args, **kwargs):
            stream.write(b"<svg-figure>")

        monkeypatch.setattr("matplotlib.figure.Figure.savefig", savefig)

        def dump(_, stream):
            stream.write(b"<raw-figure>")

        monkeypatch.setattr(
            "skore_remote_project.item.matplotlib_figure_item.dump", dump
        )

        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        assert item1.__representation__ == representation
        assert item2.__representation__ == representation

        # Ensure representation is JSONable
        dumps(item1.__representation__)
        dumps(item2.__representation__)
