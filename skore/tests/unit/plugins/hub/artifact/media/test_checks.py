import pandas as pd
import pytest
from pydantic import ValidationError

from skore._plugins.hub.artifact.media import ChecksSummary
from skore._plugins.hub.artifact.media.checks import check_results_cache_from_summary
from skore._plugins.hub.artifact.serializer import Serializer
from skore._plugins.hub.json import dumps


@pytest.mark.parametrize(
    "report",
    (
        pytest.param("binary_classification", id="estimator"),
        pytest.param("cv_binary_classification", id="cross-validation"),
    ),
)
@pytest.mark.respx()
def test_checks_summary(report, upload_mock, request, project):
    report = request.getfixturevalue(report)
    frame = report.checks.summarize(fast_mode=True).frame()
    content = dumps(
        frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
    )

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    assert ChecksSummary(project=project, report=report).model_dump() == {
        "content_type": "application/vnd.dataframe",
        "name": "checks_summary",
        "data_source": None,
        "checksum": checksum,
        "parameters": {"fast_mode": True},
    }

    assert upload_mock.called
    assert not upload_mock.call_args.args
    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content": content,
        "content_type": "application/vnd.dataframe",
    }

    with pytest.raises(
        ValidationError,
        match=f"Input should be an instance of {report.__class__.__name__}",
    ):
        ChecksSummary(project=project, report=None)


def test_check_results_cache_from_summary(binary_classification):
    """Uploaded fast-mode rows that were computed become cache entries."""
    report = binary_classification
    registry = {check.code: check for check in report._checks_registry}
    frame = pd.DataFrame(
        [
            {
                "code": "SKD001",
                "title": "uploaded title",
                "section": "issue",
                "explanation": "Train score is far above test score.",
                "documentation_url": "https://docs.example/skd001",
            },
            {
                "code": "SKD002",
                "title": "uploaded title",
                "section": "passed",
                "explanation": None,
                "documentation_url": None,
            },
            {
                "code": "SKD006",
                "title": "uploaded title",
                "section": "tip",
                "explanation": "Coefficients are not on a comparable scale.",
                "documentation_url": "https://docs.example/skd006",
            },
            {
                "code": "SKD004",
                "title": "uploaded title",
                "section": "not_applicable",
                "explanation": "ML task is not binary classification.",
                "documentation_url": None,
            },
            {
                "code": "SKD009",
                "title": "uploaded title",
                "section": "skipped",
                "explanation": None,
                "documentation_url": None,
            },
            {
                "code": "SKD010",
                "title": "uploaded title",
                "section": "ignored",
                "explanation": None,
                "documentation_url": None,
            },
            {
                "code": "SKD003",
                "title": "uploaded title",
                "section": "not_applicable",
                "explanation": "Not applicable to estimator reports.",
                "documentation_url": None,
            },
        ]
    )
    content = dumps(
        frame.astype(object).where(frame.notna(), "NaN").to_dict(orient="tight")
    )

    cache = check_results_cache_from_summary(report, content)

    assert set(cache) == {"SKD001", "SKD002", "SKD004", "SKD006"}
    assert cache["SKD001"] == {
        "title": "uploaded title",
        "docs_url": registry["SKD001"].docs_url,
        "explanation": "Train score is far above test score.",
        "section": "issue",
    }
    assert cache["SKD002"] == {
        "title": "uploaded title",
        "docs_url": registry["SKD002"].docs_url,
        "explanation": None,
        "section": "passed",
    }
    assert cache["SKD006"] == {
        "title": "uploaded title",
        "docs_url": registry["SKD006"].docs_url,
        "explanation": "Coefficients are not on a comparable scale.",
        "section": "tip",
    }
    assert cache["SKD004"] == {
        "title": "uploaded title",
        "docs_url": registry["SKD004"].docs_url,
        "explanation": "ML task is not binary classification.",
        "section": "not_applicable",
    }
