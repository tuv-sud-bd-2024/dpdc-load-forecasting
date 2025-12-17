"""Forecast Multiple Models routes"""
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
import logging
import csv
from pathlib import Path
from services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

HOLIDAY_CODES_CSV_PATH = Path(__file__).resolve().parent.parent / "static" / "config" / "Holiday_Codes.csv"
NATIONAL_EVENTS_CSV_PATH = Path(__file__).resolve().parent.parent / "static" / "National_Events.csv"
NATIONAL_EVENTS_FALLBACK_CSV_PATH = (
    Path(__file__).resolve().parent.parent / "static" / "config" / "National_Event_Codes.csv"
)

def _load_holiday_type_options() -> List[Dict[str, Any]]:
    """
    Load holiday type options from Holiday_Codes.csv.

    Returns list entries shaped like:
      {"code_int": 1, "code_str": "01", "holiday_name": "National holiday"}
    """
    if not HOLIDAY_CODES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Holiday codes CSV not found at expected path: {HOLIDAY_CODES_CSV_PATH}"
        )

    options: List[Dict[str, Any]] = []
    with HOLIDAY_CODES_CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_str = (row.get("Code") or "").strip()
            holiday_name = (row.get("Holiday_Name") or "").strip()
            if not code_str or not holiday_name:
                continue
            try:
                code_int = int(code_str)
            except ValueError:
                logger.warning(f"Skipping invalid holiday code: {code_str!r}")
                continue
            options.append(
                {
                    "code_int": code_int,
                    "code_str": code_str,
                    "holiday_name": holiday_name,
                }
            )

    options.sort(key=lambda x: x["code_int"])
    return options


def _load_national_event_options() -> List[Dict[str, Any]]:
    """
    Load national event options from National_Events.csv (preferred) or fallback CSV.

    Returns list entries shaped like:
      {"code_int": 0, "code_str": "0", "national_event_name": "No Event"}
    """
    csv_path = NATIONAL_EVENTS_CSV_PATH if NATIONAL_EVENTS_CSV_PATH.exists() else NATIONAL_EVENTS_FALLBACK_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"National events CSV not found at expected paths: {NATIONAL_EVENTS_CSV_PATH} or {NATIONAL_EVENTS_FALLBACK_CSV_PATH}"
        )

    options: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_str = (row.get("Code") or "").strip()
            event_name = (row.get("National_Event_Name") or "").strip()
            if not code_str or not event_name:
                continue
            try:
                code_int = int(code_str)
            except ValueError:
                logger.warning(f"Skipping invalid national event code: {code_str!r}")
                continue
            options.append(
                {
                    "code_int": code_int,
                    "code_str": code_str,
                    "national_event_name": event_name,
                }
            )

    options.sort(key=lambda x: x["code_int"])
    return options


@router.get("/forecast-multiple", response_class=HTMLResponse)
async def forecast_multiple_page(request: Request):
    """Forecast Multiple Models page"""
    return templates.TemplateResponse(
        "forecast_multiple.html", 
        {
            "request": request, 
            "active_page": "forecast-multiple", 
            "available_models": ModelService.get_trained_models(),
            "holiday_type_options": _load_holiday_type_options(),
            "national_event_options": _load_national_event_options(),
        }
    )


@router.post("/api/generate-forecast")
async def generate_forecast(
    date: str = Form(...),
    model_names: str = Form(...),  # Comma-separated list of model names
    holiday: int = Form(...),
    holiday_type: int = Form(...),
    nation_event: int = Form(...)
):
    """API endpoint for generating real-time forecasts from current hour to end of day"""
    # Parse the comma-separated model names
    model_names_list = [name.strip() for name in model_names.split(',') if name.strip()]
    
    logger.info(f"Generate Forecast request - Models: {model_names_list}, Date: {date}")
    logger.debug(f"Holiday: {holiday}, Holiday Type: {holiday_type}, Nation Event: {nation_event}")

    try:
        # Get real-time forecast results from multiple models
        forecast_result = await ModelService.generate_realtime_forecast(
            model_names_list, 
            date, 
            holiday, 
            holiday_type, 
            nation_event
        )
        
        logger.info(f"Real-time forecast completed successfully for {len(model_names_list)} models")
        
        return JSONResponse(forecast_result)
    
    except ValueError as e:
        # Handle validation errors (e.g., wrong date, empty model list)
        logger.error(f"Validation error in generate_forecast: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    
    except FileNotFoundError as e:
        # Handle missing files
        logger.error(f"File not found error in generate_forecast: {str(e)}")
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Unexpected error in generate_forecast: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )
