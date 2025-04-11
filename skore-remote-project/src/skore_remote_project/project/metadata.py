import pandas as pd

from .. import item as item_module
from ..client.client import AuthenticatedClient


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    def __init__(self, project, /):
        self.project = project

        with AuthenticatedClient(raises=True) as client:
            response = client.get(
                "/".join(
                    (
                        "projects",
                        self.project.tenant,
                        self.project.name,
                        "experiments",
                        "estimator-reports",
                    )
                )
            )

            summaries = response.json()["experiments"]["estimator_reports"]
            indexes = [summary.pop("id") for summary in summaries]

            def dto(summary):
                return {
                    "run_id": summary["run_id"],
                    "ml_task": summary["ml_task"],
                    "estimator_class_name": summary["estimator_class_name"],
                    "dataset_fingerprint": summary["estimator_class_name"],
                    "date": summary["created_at"],
                    **(
                        (
                            (
                                f'{metric["name"]}_{metric["data_source"]}'
                                if metric["data_source"] is not None
                                else metric["name"]
                            ),
                            metric["value"],
                        )
                        for metric in summary["metrics"]
                    ),
                }

            # transform summaries, nested metrics, remove useless columns etc
            # [
            #     {
            #         "id": 0,
            #         "project_id": 0,
            #         "creator_id": "string",
            #         "run_id": 0,
            #         "key": "string",
            #         "ml_task": "regression",
            #         "estimator_class_name": "string",
            #         "dataset_fingerprint": "string",
            #         "metrics": [
            #             {
            #                 "report_id": 0,
            #                 "name": "string",
            #                 "data_source": "string",
            #                 "value": 0,
            #                 "greater_is_better": true,
            #             }
            #         ],
            #         "created_at": "2025-04-11T14:31:07.004Z",
            #         "updated_at": "2025-04-11T14:31:07.004Z",
            #     }
            # ]

            breakpoint()

        super().__init__(
            data=pd.DataFrame(
                map(dto, summaries),
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(len(summaries)),
                        pd.Index(indexes, name="id"),
                    ]
                ),
            ),
            copy=False,
        )

    @property
    def _constructor(self):
        def _constructor_with_fallback(cls, *args, **kwargs):
            metadata = cls(*args, *kwargs)
            # Metadata.__init__(metadata, *args, **kwargs)

            return metadata

        return _constructor_with_fallback

    def reports(self):
        if not hasattr(self, "project") or "id" not in self.index.names:
            raise Exception

        def dto(report: dict):
            item_class = getattr(item_module, report["class"])
            item = item_class(**report["parameters"])
            return item.__raw__

        # retrieve reports by ids
        # get:/reports/{id}

        ids = list(self.index.get_level_values("id"))
        reports = []

        return list(map(dto, reports))
