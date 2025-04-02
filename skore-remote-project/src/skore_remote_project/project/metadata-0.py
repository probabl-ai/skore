import pandas as pd


class Metadata(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        if not (args or kwargs):  # retrieve metadata from hub
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

    @property
    def _constructor(self):
        return Metadata

    def reports(self):
        return [f"REPORT {id}" for id in self.index.get_level_values("id")]


df = Metadata()
df.iloc[1:2].reports()
