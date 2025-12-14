from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

# Import routers
from routes import train_model, forecast_multiple, data_input, dashboard, backtesting
# from routes import forecast  # Disabled
from utils.logger import setup_logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Set LOG_FILE empty to disable file logging
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log") or None

# Uvicorn expects lowercase log levels (e.g. "debug"), while Python logging often uses
# uppercase names (e.g. "DEBUG"). Normalize to avoid KeyError in uvicorn's LOG_LEVELS.
_UVICORN_LOG_LEVEL_MAP = {
    "WARN": "warning",
    "WARNING": "warning",
    "FATAL": "critical",
}

# Configure logging on import (helps for non-uvicorn runs), but we also re-apply it on
# FastAPI startup because uvicorn's own log config may override logging at process start.
setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler (replaces deprecated on_event startup/shutdown)."""
    # Ensure desired log level is active (even under uvicorn CLI).
    setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)
    logger.info("DPDC OpenSTEF application started successfully")
    yield
    logger.info("DPDC OpenSTEF application shutting down")


app = FastAPI(title="DPDC OpenSTEF", lifespan=lifespan)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(train_model.router, tags=["Train Model"])
# app.include_router(forecast.router, tags=["Forecast"])  # Disabled
app.include_router(forecast_multiple.router, tags=["Forecast Multiple"])
app.include_router(backtesting.router, tags=["Backtesting"])
app.include_router(data_input.router, tags=["Data Input"])
app.include_router(dashboard.router, tags=["Dashboard"])


if __name__ == "__main__":
    import uvicorn
    uvicorn_log_level = _UVICORN_LOG_LEVEL_MAP.get(LOG_LEVEL.upper(), LOG_LEVEL).lower()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_config=None,
        log_level=uvicorn_log_level,
    )

