import pytest
from skore.persistence.in_memory_storage import InMemoryStorage
from skore.report.report import LayoutItem, LayoutItemSize, Report
from skore.report.report_repository import ReportRepository


@pytest.fixture
def report_repository():
    return ReportRepository(InMemoryStorage())


def test_get(report_repository):
    report = Report(
        layout=[
            LayoutItem(key="key1", size=LayoutItemSize.LARGE),
            LayoutItem(key="key2", size=LayoutItemSize.SMALL),
        ]
    )

    report_repository.put_report("report", report)

    assert report_repository.get_report("report") == report


def test_get_with_no_put(report_repository):
    with pytest.raises(KeyError):
        report_repository.get_report("report")


def test_delete(report_repository):
    report_repository.put_report("report", Report(layout=[]))

    report_repository.delete_report("report")

    with pytest.raises(KeyError):
        report_repository.get_report("report")


def test_delete_with_no_put(report_repository):
    with pytest.raises(KeyError):
        report_repository.delete_report("report")
