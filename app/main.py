from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.storage.database import init_db
from app.api import health, telemetry, readings, events, overview, audit, metrics

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables on startup
    init_db()
    yield


app = FastAPI(
    title="FusionHeal — Industrial Data-Quality Gateway",
    description=(
        "An intelligent inline pre-analytics gate for autonomous anomaly detection, "
        "evidence-governed telemetry reconstruction, and continuous MLES audit compliance."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for dashboards / frontend tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health.router)
app.include_router(telemetry.router)
app.include_router(readings.router)
app.include_router(events.router)
app.include_router(overview.router)
app.include_router(audit.router)
app.include_router(metrics.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
