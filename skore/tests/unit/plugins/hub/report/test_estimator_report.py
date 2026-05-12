from hashlib import blake2b

from httpx import Client as HTTPXClient
from joblib import hash
from pydantic import ValidationError
from pytest import fixture, mark, raises

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.hub.artifact.media import (
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
    TableReportTest,
    TableReportTrain,
)
from skore._plugins.hub.artifact.pickle import Pickle
from skore._plugins.hub.artifact.upload import upload_artifacts
from skore._plugins.hub.metric import (
    AccuracyTest,
    AccuracyTrain,
    BrierScoreTest,
    BrierScoreTrain,
    FitTime,
    LogLossTest,
    LogLossTrain,
    PrecisionTest,
    PrecisionTrain,
    PredictTimeTest,
    PredictTimeTrain,
    RecallTest,
    RecallTrain,
    RocAucTest,
    RocAucTrain,
)
from skore._plugins.hub.report import EstimatorReportPayload


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    import io

    import joblib

    reports = [object] + getattr(object, "estimator_reports_", [])
    reports_with_cache = [
        (report, report._cache) for report in reports if hasattr(report, "_cache")
    ]
    object.clear_cache()

    try:
        with io.BytesIO() as stream:
            joblib.dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in reports_with_cache:
            report._cache = cache

    checksum = f"blake2b-{blake2b(pickle_bytes).hexdigest()}"
    return pickle_bytes, checksum


def _run_orchestrator(payload):
    """Execute the batched upload pipeline and return ``(medias, pickle_plan)``.

    ``medias`` is a list of ``(media_artifact, plan)`` pairs filtered to those
    that produced content. Late import of ``HubClient`` so the
    ``monkeypatch_artifact_hub_client`` fixture's swap to ``FakeClient`` is
    observable.
    """
    from skore._plugins.hub.artifact.upload import HubClient

    media_artifacts = [
        media_cls(project=payload.project, report=payload.report)
        for media_cls in payload.MEDIAS
    ]
    pickle_artifact = Pickle(project=payload.project, report=payload.report)

    with HubClient() as hub_client, HTTPXClient() as storage_client:
        plans = upload_artifacts(
            hub_client=hub_client,
            storage_client=storage_client,
            workspace=payload.project.workspace,
            project_name=payload.project.name,
            artifacts=[*media_artifacts, pickle_artifact],
        )

    *media_plans, pickle_plan = plans
    medias = [
        (m, p)
        for m, p in zip(media_artifacts, media_plans, strict=True)
        if p is not None
    ]
    return medias, pickle_plan


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
    def test_pickle(self, binary_classification, project, payload, respx_mock):
        # The orchestrator's metric/media compute memoizes
        # ``EstimatorReport._can_skip_predict`` (a ``@cached_property``) into
        # the report's ``__dict__``; ``clear_cache`` only resets ``_cache``, so
        # this attribute survives the pickle's clear-cache step and shifts the
        # checksum. Compute the expected checksum after the orchestrator runs
        # so both pickles are taken from the same state.
        _, pickle_plan = _run_orchestrator(payload)
        _, checksum = serialize(binary_classification)
        assert pickle_plan.checksum == checksum

    @mark.respx(assert_all_called=False)
    def test_metrics(self, payload):
        assert list(map(type, payload.metrics)) == [
            AccuracyTest,
            AccuracyTrain,
            BrierScoreTest,
            BrierScoreTrain,
            LogLossTest,
            LogLossTrain,
            PrecisionTest,
            PrecisionTrain,
            RecallTest,
            RecallTrain,
            RocAucTest,
            RocAucTrain,
            FitTime,
            PredictTimeTest,
            PredictTimeTrain,
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
            "skore._plugins.hub.report.estimator_report.EstimatorReportPayload.METRICS",
            [AccuracyTest],
        )
        monkeypatch.setattr(
            "skore._plugins.hub.metric.AccuracyTest.compute", raise_exception
        )

        with raises(Exception, match="test_metrics_raises_exception"):
            list(map(type, payload.metrics))

    @mark.respx()
    def test_medias(self, payload):
        medias, _ = _run_orchestrator(payload)
        assert [type(m) for m, _ in medias] == [
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
            TableReportTest,
            TableReportTrain,
        ]

    @mark.respx()
    def test_model_dump(self, binary_classification, payload):
        binary_classification.cache_predictions()

        _, checksum = serialize(binary_classification)
        _, pickle_plan = _run_orchestrator(payload)

        payload_dict = payload.model_dump()
        payload_dict.pop("metrics")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": hash(binary_classification.y_test),
            "ml_task": "binary-classification",
        }
        assert pickle_plan.checksum == checksum

    @mark.respx(assert_all_called=False)
    def test_exception(self, project):
        with raises(ValidationError):
            EstimatorReportPayload(project=project, report=None, key="<key>")
