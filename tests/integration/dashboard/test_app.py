import pytest
from fastapi.testclient import TestClient
from mandr.dashboard import create_dashboard_app


class TestDashboardApp:
    @pytest.fixture
    def client(self):
        return TestClient(app=create_dashboard_app())

    def test_frontend_index(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.content == (
            b'<!DOCTYPE html>\n<html lang="en">\n\n<head>\n  <meta charset="UTF-8">\n  '
            b'<link rel="icon" href="/favicon.ico" sizes="any">\n  <link rel="icon" hre'
            b'f="/favicon.svg" type="image/svg+xml">\n  <link rel="apple-touch-icon" hr'
            b'ef="/apple-touch-icon.png">\n  <meta name="theme-color" content="#1E22AA"'
            b'>\n  <meta name="viewport" content="width=device-width, initial-scale=1.0'
            b'">\n  <title>:mandr.</title>\n  <script type="module" crossorigin src='
            b'"/assets/index-C7ihIfh-.js"></script>\n  <link rel="stylesheet" crossorig'
            b'in href="/assets/index-B2ZdhJBO.css">\n</head>\n\n<body>\n  <div id="app'
            b'"></div>\n</body>\n\n</html>\n'
        )
