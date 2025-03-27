import pandas as pd


class Metadata:
    def __init__(self):
        self.__query = self.__matrix = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [9, 8, 7],
            },
            index=pd.Index(["RID-1", "RID-2", "RID-3"]),
        )

    def query(self, *args, **kwargs):
        """Reset query on the original matrix at each call."""
        self.__query = self.__matrix.query(*args, **kwargs)
        return self

    def sort_values(self, *args, **kwargs):
        """Sort the result of a query."""
        self.__query = self.__query.sort_values(*args, **kwargs)
        return self

    def reports(self):
        return [f"REPORT {index}" for index in self.__query.index]

    def __repr__(self):
        return self.__query.to_string(index=False)


metadata = Metadata()

metadata.query("A > 1")
metadata.sort_values(by="C")
print(repr(metadata))
print(repr(metadata.reports()))

metadata.query("A >= 1")
metadata.sort_values(by="C")
print(repr(metadata))
print(repr(metadata.reports()))
