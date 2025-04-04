import pandas as pd

from .. import item as item_module


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    def fill(self, project):
        self.project = project

        # retrieve reports summaries
        # get:/reports/

        kwargs = dict(copy=False)
        args = [
            pd.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": [7, 8, 9],
                },
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(3),
                        pd.Index(["RID-1", "RID-2", "RID-3"], name="id"),
                    ]
                ),
            )
        ]

        super().__init__(*args, **kwargs)

        return self

    @property
    def _constructor(self):
        return Metadata

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
