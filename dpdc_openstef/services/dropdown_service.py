"""
Dropdown data loading helpers.

This module centralizes loading of dropdown options (holiday types, national events)
from the static config CSVs so multiple routes can reuse the same logic.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Keep paths consistent with other services (e.g. TRAINING_DATA_PATH in model_service.py)
HOLIDAY_CODES_CSV_PATH = "./static/config/Holiday_Codes.csv"
NATIONAL_EVENTS_CSV_PATH = "./static/config/National_Event_Codes.csv"


def _load_holiday_type_options() -> List[Dict[str, Any]]:
    """
    Load holiday type options from Holiday_Codes.csv.

    Returns list entries shaped like:
      {"code_int": 1, "code_str": "01", "holiday_name": "National holiday"}
    """
    holiday_codes_path = Path(HOLIDAY_CODES_CSV_PATH)
    if not holiday_codes_path.exists():
        raise FileNotFoundError(
            f"Holiday codes CSV not found at expected path: {holiday_codes_path}"
        )

    options: List[Dict[str, Any]] = []
    with holiday_codes_path.open(newline="", encoding="utf-8") as f:
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
    Load national event options from National_Event_Codes.csv.

    Returns list entries shaped like:
      {"code_int": 0, "code_str": "0", "national_event_name": "No Event"}
    """
    national_events_path = Path(NATIONAL_EVENTS_CSV_PATH)
    if not national_events_path.exists():
        raise FileNotFoundError(
            f"National events CSV not found at expected path: {national_events_path}"
        )

    options: List[Dict[str, Any]] = []
    with national_events_path.open(newline="", encoding="utf-8") as f:
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

