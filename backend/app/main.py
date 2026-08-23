"""FastAPI app entrypoint. Minimal by design: app instantiation, router
registration, and a startup hook that caches the trained detector -- no
business logic lives here (CLAUDE.md §7).

Not in Day 6 (final)'s ALLOWED_TO_TOUCH list explicitly, but required to
run any of the endpoints; flagged and approved before creation (planning
turn) as a genuine oversight in the phase note rather than an intentional
exclusion.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.endpoints import initialize_app_state, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state()
    yield


app = FastAPI(title="Sentinel-X API", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
