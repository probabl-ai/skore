import pandas as pd

from .. import item as item_module
from ..client.client import AuthenticatedClient


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
        def dto(summary):
            return dict(
                (
                    ("run_id", summary["run_id"]),
                    ("ml_task", summary["ml_task"]),
                    ("learner", summary["estimator_class_name"]),
                    ("dataset", summary["dataset_fingerprint"]),
                    ("date", summary["created_at"]),
                    *(
                        (metric["name"], metric["value"])
                        for metric in summary["metrics"]
                        if metric["data_source"] in (None, "test")
                    ),
                )
            )

        with AuthenticatedClient(raises=True) as client:
            response = client.get(
                "/".join(
                    (
                        "projects",
                        project.tenant,
                        project.name,
                        "experiments",
                        "estimator-reports",
                    )
                )
            )

        summaries = response.json()
        summaries = pd.DataFrame(
            data=pd.DataFrame(
                map(dto, summaries),
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(len(summaries)),
                        pd.Index(
                            (summary.pop("id") for summary in summaries),
                            name="id",
                            dtype=str,
                        ),
                    ]
                ),
            ),
            copy=False,
        )

        metadata = Metadata(summaries)
        metadata.project = project

        return metadata

    @property
    def _constructor(self):
        return Metadata

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
