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

        summaries = response.json()
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

    def reports(self):
        if not hasattr(self, "project") or "id" not in self.index.names:
            raise Exception

        def dto(response):
            report = response.json()
            item_class_name = report["raw"]["class"]
            item_class = getattr(item_module, item_class_name)
            item_parameters = report["raw"]["parameters"]
            item = item_class(**item_parameters)
            return item.__raw__

        ids = list(self.index.get_level_values("id"))

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
