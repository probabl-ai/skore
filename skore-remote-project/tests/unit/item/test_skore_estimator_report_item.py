from json import dumps, loads
from io import BytesIO

from altair import Chart
from joblib import dump
from pytest import fixture, raises
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skore import EstimatorReport
from skore_remote_project.item import PickleItem, SkoreEstimatorReportItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str


class TestSkoreEstimatorReportItem:
    @fixture(scope="class")
    def report(self):
        X, y = make_classification(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return EstimatorReport(
            SVC(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    @fixture(scope="class")
    def report_b64_str(self, report):
        with BytesIO() as stream:
            dump(report, stream)

            pickle_bytes = stream.getvalue()
            pickle_b64_str = bytes_to_b64_str(pickle_bytes)

        return pickle_b64_str

    def test_factory(self, report, report_b64_str):
        item = SkoreEstimatorReportItem.factory(report)

        assert isinstance(item, SkoreEstimatorReportItem)
        assert isinstance(item, PickleItem)
        assert item.pickle_b64_str == report_b64_str

    def test_factory_exception(self):
        with raises(ItemTypeError):
            SkoreEstimatorReportItem.factory(None)

    # def test_parameters(self):
    #     chart = Chart().mark_point()
    #     item = AltairChartItem.factory(chart)
    #     item_parameters = item.__parameters__

    #     assert item_parameters == {"chart_json_str": chart.to_json()}

    #     # Ensure parameters are JSONable
    #     dumps(item_parameters)

    # def test_metadata(self): ...

    def test_raw(self, report, report_b64_str):
        item1 = SkoreEstimatorReportItem.factory(report)
        item2 = SkoreEstimatorReportItem(report_b64_str)

        assert item1.__raw__ == report
        assert item2.__raw__ == report

    # def test_representation(self):
    #     chart = Chart().mark_point()
    #     chart_json_str = chart.to_json()
    #     representation = Representation(
    #         media_type="application/vnd.vega.v5+json",
    #         value=loads(chart_json_str),
    #     )

    #     item1 = AltairChartItem.factory(chart)
    #     item2 = AltairChartItem(chart_json_str)

    #     assert item1.__representation__ == representation
    #     assert item2.__representation__ == representation

    #     # Ensure representation is JSONable
    #     dumps(item1.__representation__.__dict__)
    #     dumps(item2.__representation__.__dict__)
