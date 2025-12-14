from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import logging
import os

# Import routers
from routes import train_model, forecast_multiple, data_input, dashboard, backtesting
# from routes import forecast  # Disabled
from utils.logger import setup_logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Set LOG_FILE empty to disable file logging
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log") or None

# Configure logging on import (helps for non-uvicorn runs), but we also re-apply it on
# FastAPI startup because uvicorn's own log config may override logging at process start.
setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)

logger = logging.getLogger(__name__)

app = FastAPI(title="DPDC OpenSTEF")


@app.on_event("startup")
async def startup_event():
    """Ensure desired log level is active (even under uvicorn CLI)."""
    setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)
    logger.info("DPDC OpenSTEF application started successfully")


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Log application shutdown"""
#     logger.info("DPDC OpenSTEF application shutting down")


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
    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=None, log_level=LOG_LEVEL)

