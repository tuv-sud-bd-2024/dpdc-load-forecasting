"""Train Model routes"""
from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import json
import logging
from typing import Optional
from services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def train_model_page(request: Request):
    """Train Model page"""
    return templates.TemplateResponse("train_model.html", {"request": request, "active_page": "train"})


@router.post("/api/train")
async def train_model(
    model: str = Form(...),
    custom_name: str = Form(...),
    training_data_start_date: str = Form(...),
    training_data_end_date: str = Form(...),
    hyperparams: str = Form(...),
    training_data_file: Optional[UploadFile] = File(None)
):
    """API endpoint for training model"""
    hyperparams_dict = json.loads(hyperparams)
    logger.debug(f"Training request received - Model: {model}, Custom Name: {custom_name}")
    logger.debug(f"Training data period: {training_data_start_date} to {training_data_end_date}")
    logger.debug(f"Hyperparameters: {hyperparams_dict}")
    
    if training_data_file:
        logger.debug(f"Training data file uploaded: {training_data_file.filename}")
    
    # Call the new train_model_with_hyperparams function
    await ModelService.train_model_with_hyperparams(
        model=model,
        custom_name=custom_name,
        training_data_start_date=training_data_start_date,
        training_data_end_date=training_data_end_date,
        hyperparams_dict=hyperparams_dict,
        training_data_file=training_data_file
    )
    
    logger.info(f"Training initiated successfully for {model} model with name '{custom_name}' using data from {training_data_start_date} to {training_data_end_date}")
    
    return JSONResponse({
        "status": "success",
        "message": f"Training completed for {model} model",
        "model": model,
        "custom_name": custom_name,
        "training_data_start_date": training_data_start_date,
        "training_data_end_date": training_data_end_date,
        "hyperparameters": hyperparams_dict
    })

