# from urllib.parse import urljoin
# from httpx import Client
# from pytest import fixture

# class FakeClient(Client):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def request(self, method, url, **kwargs):
#         response = super().request(method, urljoin("http://localhost", url), **kwargs)
#         response.raise_for_status()

#         return response


# @fixture(autouse=True)
# def monkeypatch_client(monkeypatch):
#     monkeypatch.setattr(
#         "skore_hub_project.project.artefact.HUBClient",
#         FakeClient,
#     )
