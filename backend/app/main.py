"""FastAPI app entrypoint. Minimal by design: app instantiation, router
registration, CORS, and a startup hook that caches the trained detector --
no business logic lives here (CLAUDE.md §7).

Not in Day 6 (final)'s or Day 7's ALLOWED_TO_TOUCH lists explicitly, but
required for the API to be usable at all; flagged and approved before each
touch (Day 6 final: file creation; Day 7: CORS) as genuine oversights in
the phase notes rather than intentional exclusions.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import initialize_app_state, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state()
    yield


app = FastAPI(title="Sentinel-X API", lifespan=lifespan)

# Dev-only CORS: the frontend (localhost:3000) and backend (localhost:8000)
# are different origins, so the browser blocks fetch() without this --
# confirmed via a simulated preflight returning 405 before this was added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
