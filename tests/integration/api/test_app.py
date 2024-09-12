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
from skore.project import load
from skore.storage import FileSystem
from skore.store import Store
from skore.ui.app import create_app


class TestApiApp:
    @pytest.fixture
    def client(self):
        return TestClient(app=create_app())

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_ROOT", str(tmp_path))

        self.storage = FileSystem(directory=tmp_path)

        Store("root", storage=self.storage).insert("key", "value")
        Store("root/subroot1", storage=self.storage).insert("key", "value")
        Store("root/subroot2", storage=self.storage).insert("key", "value")
        Store("root/subroot2/subsubroot1", storage=self.storage).insert("key", "value")
        Store("root/subroot2/subsubroot2", storage=self.storage).insert(
            "key1", "value1"
        )
        Store("root/subroot2/subsubroot2", storage=self.storage).insert(
            "key2", "value2"
        )

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

    def test_list_stores(self, client):
        routes = [
            "/api/skores",
            "/api/skores/",
            "/api/stores",
            "/api/stores/",
        ]

        for route in routes:
            response = client.get(route)

            assert response.status_code == 200
            assert response.json() == [
                "/root",
                "/root/subroot1",
                "/root/subroot2",
                "/root/subroot2/subsubroot1",
                "/root/subroot2/subsubroot2",
            ]

    def test_get_store_by_uri(self, client):
        response = client.get("/api/skores/root/subroot2/subsubroot3")

        assert response.status_code == 404

        routes = [
            "/api/skores/root/subroot2/subsubroot2",
            "/api/stores/root/subroot2/subsubroot2",
        ]

        s = Store("root/subroot2/subsubroot2")
        v1, m1 = s.read("key1", metadata=True)
        v2, m2 = s.read("key2", metadata=True)
        for route in routes:
            response = client.get(route)

            assert response.status_code == 200
            assert response.json() == {
                "schema": "schema:dashboard:v0",
                "uri": "/root/subroot2/subsubroot2",
                "payload": {
                    "key1": {"type": "markdown", "data": v1, "metadata": m1},
                    "key2": {"type": "markdown", "data": v2, "metadata": m2},
                },
                "layout": [],
            }

    def test_put_layout(self, client):
        layout = [{"key": "key", "size": "small"}]
        s = Store("root", storage=self.storage)
        value, metadata = s.read("key", metadata=True)
        response = client.put(f"/api/skores/{s.uri}/layout", json=layout)

        assert response.status_code == 201
        assert response.json() == {
            "schema": "schema:dashboard:v0",
            "uri": "/root",
            "payload": {
                "key": {"type": "markdown", "data": value, "metadata": metadata},
            },
            "layout": layout,
        }

    def test_share(self, client):
        s = Store("root", storage=self.storage)
        response = client.get(f"/api/skores/share/{s.uri}")

        assert response.is_success
        assert '<script id="skore-data" type="application/json">' in response.text
        assert response.text.count("</script>") >= 3
        assert response.text.count("</style>") >= 1
