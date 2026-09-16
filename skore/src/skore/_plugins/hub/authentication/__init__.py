"""Module to manage ``skore hub`` authentication."""

# from collections.abc import Callable
# from functools import cache

# from skore._plugins.hub.authentication.api_key import API_key
# from skore._plugins.hub.authentication.token import token


# @cache
# def factory(self, *, host: str, workspace: str) -> Callable[[], dict[str, str]]:
#     if API_key.available():
#         return API_key()

#     if token is not None:
#         return token

#     raise NotImplementedError
