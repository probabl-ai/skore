import json
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Lasso
from skore.create_project import create_project
from skore.project import Project, load
from skore.storage import FileSystem
from skore.ui.app import create_app


class TestApiApp:
    @pytest.fixture
    def client(self):
        return TestClient(app=create_app(Project(self.storage)))

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_ROOT", str(tmp_path))

        self.storage = FileSystem(directory=tmp_path)

        Project(self.storage).put("root/key", "value")
        Project(self.storage).put("root/subroot1/key", "value")
        Project(self.storage).put("root/subroot2/key", "value")
        Project(self.storage).put("root/subroot2/subsubroot1/key", "value")
        Project(self.storage).put("root/subroot2/subsubroot2/key1", "value1")
        Project(self.storage).put("root/subroot2/subsubroot2/key2", "value2")

    def test_get_report(self, tmp_path):
        create_project(tmp_path / "test.skore")
        project = load(tmp_path / "test.skore")
        project.put("Math", "$\\sum_{x=1}^{3} x_i$")
        project.put("Array", np.random.randint(0, 1000, size=50).tolist())
        project.put("Inline code", """`x = 4`""")
        project.put("lasso", Lasso())

        def create_fake_dataframe(num_rows: int) -> pd.DataFrame:
            """Create a fake dataframe with specified number of rows.

            Args:
            num_rows: Number of rows in the dataframe.

            Returns
            -------
            A pandas DataFrame with columns 'id', 'date', 'random_string',
            and 'random_float'.
            """
            data = []
            start_date = datetime(2023, 1, 1)

            for i in range(num_rows):
                random_date = start_date + timedelta(days=random.randint(0, 365))
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=10)
                )
                random_float = random.uniform(0, 100)
                data.append(
                    [i, random_date.strftime("%Y-%m-%d"), random_string, random_float]
                )

            df = pd.DataFrame(
                data, columns=["id", "date", "random_string", "random_float"]
            )
            return df

        project.put("Dataframe", create_fake_dataframe(100))

        serialized = {}
        for key in project.list_keys():
            item = project.get_item(key)
            serialized[key] = {
                "item_type": str(item.item_type),
                "media_type": item.media_type,
                "serialized": json.loads(item.serialized),
            }

        json.dumps(serialized)
