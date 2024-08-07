"""The definition of API routes to interact with stores."""

from fastapi import APIRouter

from mandr.api.routes.fake_stores import FAKE_STORES_ROUTER
from mandr.api.routes.stores import MANDRS_ROUTER, STORES_ROUTER

__all__ = ["ROOT_ROUTER"]


ROOT_ROUTER = APIRouter(prefix="/api")
SUBROUTERS = [
    FAKE_STORES_ROUTER,
    MANDRS_ROUTER,
    STORES_ROUTER,
]


for router in SUBROUTERS:
    ROOT_ROUTER.include_router(router)
