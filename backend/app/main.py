"""FastAPI app entrypoint. Minimal by design: app instantiation, router
registration, CORS, and a startup hook that caches the trained detector --
no business logic lives here (CLAUDE.md §7).

Not in Day 6 (final)'s or Day 7's ALLOWED_TO_TOUCH lists explicitly, but
required for the API to be usable at all; flagged and approved before each
touch (Day 6 final: file creation; Day 7: CORS) as genuine oversights in
the phase notes rather than intentional exclusions.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import initialize_app_state, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state()
    yield


app = FastAPI(title="Sentinel-X API", lifespan=lifespan)

# Local dev origins (frontend/backend are different ports, so the browser
# blocks fetch() without this -- confirmed via a simulated preflight
# returning 405 before this was added), plus one optional deployed frontend
# origin read from an environment variable. Never a wildcard: unset means
# local dev behaves exactly as before, set means exactly one additional
# real origin is allowed -- same "env var with a safe local default"
# pattern the frontend already uses for its own API base URL.
_allow_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
_deployed_frontend_origin = os.environ.get("FRONTEND_ORIGIN")
if _deployed_frontend_origin:
    _allow_origins.append(_deployed_frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
