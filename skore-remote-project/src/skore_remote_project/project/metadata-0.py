import pandas as pd


class Metadata(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.index, pd.RangeIndex):
            self.set_index("report_id", inplace=True)

        self.index.name = None

    @property
    def _constructor(self):
        return Metadata

    def reports(self):
        return [f"REPORT {index}" for index in self.index]


df = Metadata(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "report_id": ["RID-1", "RID-2", "RID-3"],
    }
)

df.iloc[1:2].reports()
