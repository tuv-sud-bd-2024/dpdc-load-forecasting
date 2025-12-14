"""Centralized logging configuration for the application"""
import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import TimedRotatingFileHandler


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the entire application.
    Call this once at app startup.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    
    Example:
        setup_logging(log_level="INFO", log_file="logs/app.log")
    """
    # Create logs directory if logging to file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Roll logs weekly (Monday 00:00 by default), keep a limited history.
        # If you prefer Sunday rollovers, change when="W6".
        handlers.append(
            TimedRotatingFileHandler(
                log_file,
                when="W0",
                interval=1,
                backupCount=8,  # keep ~8 weeks of history
                encoding="utf-8",
                utc=True,
            )
        )
    
    # Resolve level safely (fallback to INFO on typos)
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Optionally set different levels for specific modules
    # Reduce noise from uvicorn access logs
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    # Keep uvicorn error logs aligned with app verbosity
    logging.getLogger('uvicorn.error').setLevel(level)
    
    # Example: Set debug level for specific module
    # logging.getLogger('services.model_service').setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

