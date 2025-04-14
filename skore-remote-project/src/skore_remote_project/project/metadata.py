import pandas as pd

from .. import item as item_module
from ..client.client import AuthenticatedClient


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

        def dto(summary):
            return dict(
                (
                    ("run_id", summary["run_id"]),
                    ("ml_task", summary["ml_task"]),
                    ("estimator_class_name", summary["estimator_class_name"]),
                    ("dataset_fingerprint", summary["estimator_class_name"]),
                    ("date", summary["created_at"]),
                    *(
                        (
                            (
                                "_".join(
                                    filter(
                                        None,
                                        (
                                            metric["name"],
                                            metric["data_source"],
                                        ),
                                    )
                                )
                            ),
                            metric["value"],
                        )
                        for metric in summary["metrics"]
                    ),
                )
            )

        summaries = response.json()[:10]  # /!\
        indexes = [summary.pop("id") for summary in summaries]

        super().__init__(
            data=pd.DataFrame(
                map(dto, summaries),
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(len(summaries)),
                        pd.Index(indexes, name="id", dtype=str),
                    ]
                ),
            ),
            copy=False,
        )

    @property
    def _constructor(self):
        return pd.DataFrame

        # def _constructor_with_fallback(cls, *args, **kwargs):
        #     metadata = cls(*args, *kwargs)
        #     # Metadata.__init__(metadata, *args, **kwargs)

        #     return metadata

        # return _constructor_with_fallback

    @property
    def reports(self):
        if not hasattr(self, "project") or "id" not in self.index.names:
            raise Exception

        ids = list(self.index.get_level_values("id"))

        def dto(response):
            report = response.json()

            breakpoint()

            item_class_name = report["parameters"]["class"]
            item_class = getattr(item_module, item_class_name)
            item_parameters = report["parameters"]["parameters"]
            item = item_class(**item_parameters)
            return item.__raw__

        with AuthenticatedClient(raises=True) as client:
            return [
                dto(
                    client.get(
                        "/".join(
                            (
                                "projects",
                                self.project.tenant,
                                self.project.name,
                                "experiments",
                                "estimator-reports",
                                id,
                            )
                        )
                    )
                )
                for id in ids
            ]
