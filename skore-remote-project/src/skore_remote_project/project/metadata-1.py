import pandas as pd


class Metadata:
    def query(self, *args, **kwargs):
        metadata = Metadata()
        metadata.__matrix = self.__matrix.query(*args, **kwargs)

        return metadata

    def sort_values(self, *args, **kwargs):
        metadata = Metadata()
        metadata.__matrix = self.__matrix.sort_values(*args, **kwargs)

        return metadata

    def reports(self):
        return [f"REPORT {index}" for index in self.__matrix.index]

    def __repr__(self):
        return self.__matrix.to_string(index=False)


class Project:
    @property
    def metadata(self):
        project_metadata = Metadata()
        project_metadata._Metadata__matrix = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [9, 8, 7],
            },
            index=pd.Index(["RID-1", "RID-2", "RID-3"]),
        )

        return project_metadata


metadata = Project().metadata

filtered_metadata = metadata.query("A > 1")
sorted_metadata = filtered_metadata.sort_values(by="C")
print(repr(sorted_metadata))
print(repr(sorted_metadata.reports()))

filtered_metadata = metadata.query("A >= 1")
sorted_metadata = filtered_metadata.sort_values(by="C")
print(repr(sorted_metadata))
print(repr(sorted_metadata.reports()))
