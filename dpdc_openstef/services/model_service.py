"""Service class for model training and forecasting operations"""
import numpy as np
import pandas as pd
import pickle
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.pipeline.train_model import train_model_pipeline
from openstef.pipeline.create_forecast import create_forecast_pipeline
from utils.dateutils import create_utc_datetime
from datetime import datetime, timedelta, timezone

# Get logger for this module (configuration is done in main.py)
logger = logging.getLogger(__name__)

PARENT_DIR = "trained_models"
TRAINING_DATA_PATH = "./static/master_data_with_forecasted.csv"

class ModelService:
    """Service class for handling model training and forecasting operations"""
    
    @staticmethod
    def get_trained_models() -> List[str]:
        """Get list of trained model directories"""
        path = Path(PARENT_DIR)
        dirs = [d.name for d in path.iterdir() if d.is_dir()]
        logger.debug(f"Found trained model directories: {dirs}")
        return dirs
    
    @staticmethod
    async def train_model(model: str, custom_name: str, training_data_start_date: str, training_data_end_date: str, hyperparams_dict: Dict[str, Any]) -> str:
        """
        Train a model with given hyperparameters
        
        Args:
            model: Model type (e.g., 'xgb')
            custom_name: Custom name for the model
            training_data_start_date: Start date for training data
            training_data_end_date: End date for training data
            hyperparams_dict: Dictionary of hyperparameters
            
        Returns:
            Status message
        """
        pd.options.plotting.backend = 'plotly'
        pj = dict(
            id=101,
            model='xgb',
            forecast_type="demand",
            horizon_minutes=120,
            resolution_minutes=60,
            name="xgb_poc_1",
            save_train_forecasts=True,
            ignore_existing_models=True,
            model_kwargs={
                "learning_rate": float(hyperparams_dict['learning_rate']),
                "early_stopping_rounds": int(hyperparams_dict['early_stopping_rounds']),
                "n_estimators": int(hyperparams_dict['n_estimators'])
            },
            quantiles=[0.1, 0.5, 0.9]
        )

        pj = PredictionJobDataClass(**pj)

        input_data = pd.read_csv(TRAINING_DATA_PATH, index_col=0, parse_dates=True)

        # dropping columns as we want
        input_data = input_data.drop(columns=["date_time_com", "forecasted_load"])

        pd.options.display.max_columns = None
        logger.debug(f"Input data head:\n{input_data.head()}")

        # Filter data based on provided date range
        start_date = create_utc_datetime(training_data_start_date, 0)
        end_date = create_utc_datetime(training_data_end_date, 23)
        
        # Filter the input data to the specified date range
        train_data = input_data[(input_data.index >= start_date) & (input_data.index <= end_date)]

        logger.info(f"Training data starting hour: {train_data.head(1).index}")
        logger.info(f"Training data ending hour: {train_data.tail(1).index}")
        logger.info(f"Training data filtered from {training_data_start_date} to {training_data_end_date}")

        train_data = train_data[~train_data.index.duplicated(keep='first')]

        # Remove rows with NaT in the index
        train_data = train_data[train_data.index.notna()]

        path_to_create = f"./{PARENT_DIR}/{custom_name}/"

        try:
            os.makedirs(path_to_create, exist_ok=True)
            logger.info(f"Directory structure '{path_to_create}' created successfully.")
        except OSError as e:
            logger.error(f"Error creating directory structure: {e}")
        
        # storing pj for using later
        dictionary_path = f"./{PARENT_DIR}/{custom_name}/pj.pkl"
        with open(dictionary_path, "wb") as file:
            pickle.dump(pj, file, protocol=pickle.HIGHEST_PROTOCOL) 

        mlflow_tracking_uri = f"{PARENT_DIR}/{custom_name}/mlflow_trained_models"

        train_data, validation_data, test_data = train_model_pipeline(
            pj,
            train_data,
            check_old_model_age=False,
            mlflow_tracking_uri=mlflow_tracking_uri,
            artifact_folder=f"{PARENT_DIR}/{custom_name}/mlflow_artifacts",
        )
        return "hello"
    
    @staticmethod
    async def train_model_with_hyperparams(
        model: str, 
        custom_name: str, 
        training_data_start_date: str, 
        training_data_end_date: str, 
        hyperparams_dict: Dict[str, Any],
        training_data_file: Optional[UploadFile] = None
    ) -> str:
        """
        Train a model with comprehensive hyperparameters
        
        Args:
            model: Model type ('xgb' or 'lgb')
            custom_name: Custom name for the model
            training_data_start_date: Start date for training data
            training_data_end_date: End date for training data
            hyperparams_dict: Dictionary of hyperparameters
            training_data_file: Optional uploaded training data file (ignored as per requirements)
            
        Returns:
            Status message
        """
        pd.options.plotting.backend = 'plotly'
        
        # Create PredictionJobDataClass with proper model type and hyperparameters
        pj_dict = dict(
            id=101,
            model=model,  # Use the actual model type from parameter
            forecast_type="demand",
            horizon_minutes=120,
            resolution_minutes=60,
            name=custom_name,  # Use the custom name
            save_train_forecasts=True,
            ignore_existing_models=True,
            model_kwargs=hyperparams_dict,  # Use all hyperparameters from the dictionary
            quantiles=[0.1, 0.5, 0.9]
        )
        
        logger.info(f"Creating PredictionJobDataClass with model={model}, name={custom_name}")
        logger.debug(f"Model kwargs: {hyperparams_dict}")
        
        pj = PredictionJobDataClass(**pj_dict)
        
        # Load training data from default path (file upload is ignored as per requirements)
        input_data = pd.read_csv(TRAINING_DATA_PATH, index_col=0, parse_dates=True)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = []
        if "date_time_com" in input_data.columns:
            columns_to_drop.append("date_time_com")
        if "forecasted_load" in input_data.columns:
            columns_to_drop.append("forecasted_load")
        
        if columns_to_drop:
            input_data = input_data.drop(columns=columns_to_drop)
        
        pd.options.display.max_columns = None
        logger.debug(f"Input data head:\n{input_data.head()}")
        
        # Filter data based on provided date range
        start_date = create_utc_datetime(training_data_start_date, 0)
        end_date = create_utc_datetime(training_data_end_date, 23)
        
        # Filter the input data to the specified date range
        train_data = input_data[(input_data.index >= start_date) & (input_data.index <= end_date)]
        
        logger.info(f"Training data starting hour: {train_data.head(1).index}")
        logger.info(f"Training data ending hour: {train_data.tail(1).index}")
        logger.info(f"Training data filtered from {training_data_start_date} to {training_data_end_date}")
        
        # Remove duplicate index values
        train_data = train_data[~train_data.index.duplicated(keep='first')]
        
        # Remove rows with NaT in the index
        train_data = train_data[train_data.index.notna()]
        
        # Create directory structure for saving the model
        path_to_create = f"./{PARENT_DIR}/{custom_name}/"
        
        try:
            os.makedirs(path_to_create, exist_ok=True)
            logger.info(f"Directory structure '{path_to_create}' created successfully.")
        except OSError as e:
            logger.error(f"Error creating directory structure: {e}")
            raise
        
        # Store PredictionJob for later use
        dictionary_path = f"./{PARENT_DIR}/{custom_name}/pj.pkl"
        with open(dictionary_path, "wb") as file:
            pickle.dump(pj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"PredictionJob saved to {dictionary_path}")
        
        # Save training metadata to JSON file
        metadata = {
            "model": model,
            "custom_name": custom_name,
            "training_data_start_date": training_data_start_date,
            "training_data_end_date": training_data_end_date,
            "hyperparameters": hyperparams_dict,
            "trained_at": datetime.now(timezone.utc).isoformat()
        }
        
        metadata_path = f"./{PARENT_DIR}/{custom_name}/training_metadata.json"
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)
        
        logger.info(f"Training metadata saved to {metadata_path}")
        
        # Set up MLflow tracking
        mlflow_tracking_uri = f"{PARENT_DIR}/{custom_name}/mlflow_trained_models"
        
        # Train the model
        logger.info(f"Starting model training for {model} with custom name '{custom_name}'")
        train_data, validation_data, test_data = train_model_pipeline(
            pj,
            train_data,
            check_old_model_age=False,
            mlflow_tracking_uri=mlflow_tracking_uri,
            artifact_folder=f"{PARENT_DIR}/{custom_name}/mlflow_artifacts",
        )
        
        logger.info(f"Model training completed successfully for '{custom_name}'")
        return "Training completed successfully"
    
    @staticmethod
    async def forecast_from_model(custom_name: str, date: str, hour: int) -> Dict[str, Any]:
        """
        Create forecast from a trained model
        
        Args:
            custom_name: Name of the trained model
            date: Date string in format 'YYYY-MM-DD'
            hour: Hour value (0-23)
            
        Returns:
            Dict containing timestamp, forecast value, and custom_name
        """
        input_data = pd.read_csv(TRAINING_DATA_PATH, index_col=0, parse_dates=True)

        
        traing_data_last_index = input_data.index.get_loc(calculate_previous_hr_of_forecast(date, hour))
        # checking if the limit of test data matches our expectation
        test_data = input_data.iloc[traing_data_last_index+1:traing_data_last_index+25]
        logger.info(f"Test data starting hour: {test_data.head(1).index}")
        logger.info(f"Test data ending hour: {test_data.tail(1).index}")

        # Prepare data to make the forecast.
        realised = input_data.loc[test_data.index, 'load'].copy(deep=True)
        to_forecast_data = input_data.copy(deep=True)
        to_forecast_data.loc[test_data.index, 'load'] = np.nan  # clear the load data for the part you want to forecast    
        
        # Remove duplicate index values from train_data
        to_forecast_data = to_forecast_data[~to_forecast_data.index.duplicated(keep='first')]
        # Remove rows with NaT in the index
        to_forecast_data = to_forecast_data[to_forecast_data.index.notna()]
        
        # storing pj for using later
        dictionary_path = f"./{PARENT_DIR}/{custom_name}/pj.pkl"
        with open(dictionary_path, "rb") as file:
            pj = pickle.load(file)

        mlflow_tracking_uri = f"{PARENT_DIR}/{custom_name}/mlflow_trained_models"

        forecast = create_forecast_pipeline(
            pj,
            to_forecast_data,
            mlflow_tracking_uri,
        )

        logger.info(f"Forecast results:\n{forecast}")

        # Create the time index using the utility method
        forecast_timestamp = create_utc_datetime(date, hour)
        
        forecast_value = forecast.loc[forecast_timestamp, 'forecast']
        
        # Return the required JSON structure
        result = {
            "timestamp": create_utc_datetime(date, hour, timezone(timedelta(hours=6))).isoformat(),
            "forecast": float(forecast_value),
            "custom_name": custom_name
        }
        
        logger.info(f"Returning forecast result: {result}")
        return result
    
    @staticmethod
    async def forecast_from_mulitple_models(custom_names: List[str], date: str) -> Dict[str, Any]:
        """
        Create forecasts from multiple trained models for 24 hours (0-23)
        
        Args:
            custom_names: List of trained model names
            date: Date string in format 'YYYY-MM-DD'
            
        Returns:
            Dict with 'all_forecasts' key containing list of model forecasts
        """
        # Load input data and prepare dataframe with NaN for 24 hours
        input_data = pd.read_csv(TRAINING_DATA_PATH, index_col=0, parse_dates=True)
        
        # Calculate the start of the 24-hour forecast period (hour 0 of the given date)
        forecast_start_datetime = create_utc_datetime(date, 0)
        
        # Get the index of the hour before the forecast period starts (robust to missing timestamps)
        traing_data_last_index = get_training_data_last_index(input_data, date)
        
        # Get the test data for the forecast date (only available timestamps)
        test_data = get_test_data_for_date(input_data, date)
        
        # Prepare data to make the forecast - set load values to NaN for the 24 hours
        to_forecast_data = input_data.copy(deep=True)
        to_forecast_data.loc[test_data.index, 'load'] = np.nan
        # Drop all data points after the last test_data timestamp
        if len(test_data) > 0:
            last_test_timestamp = test_data.index[-1]
            to_forecast_data = to_forecast_data[to_forecast_data.index <= last_test_timestamp]
            logger.info(f"Dropped data points after {last_test_timestamp}")
        
        # Remove duplicate index values and NaT
        to_forecast_data = to_forecast_data[~to_forecast_data.index.duplicated(keep='first')]
        to_forecast_data = to_forecast_data[to_forecast_data.index.notna()]
        
        all_forecasts = []
        
        # Extract actual load data for the 24 hours (if available)
        actual_loads = []
        for hour in range(24):
            forecast_timestamp = create_utc_datetime(date, hour)
            # Check if this timestamp exists in the input data
            if forecast_timestamp in input_data.index:
                actual_load = input_data.loc[forecast_timestamp, 'load']
                actual_loads.append({
                    "timestamp": create_utc_datetime(date, hour, timezone(timedelta(hours=6))).isoformat(),
                    "load": float(actual_load) if pd.notna(actual_load) else None
                })
            else:
                actual_loads.append({
                    "timestamp": create_utc_datetime(date, hour, timezone(timedelta(hours=6))).isoformat(),
                    "load": None
                })
        
        # Loop through each model sequentially
        for custom_name in custom_names:
            logger.info(f"Starting forecast for model: {custom_name}")
            
            # Get 24-hour forecasts from this model
            forecast_df = _forecast_24_hours(custom_name, to_forecast_data)
            
            # Format the forecasts for all 24 hours
            model_forecasts = []
            for hour in range(24):
                forecast_timestamp = create_utc_datetime(date, hour)
                
                # Safely access forecast value with error handling
                try:
                    if forecast_timestamp in forecast_df.index and 'forecast' in forecast_df.columns:
                        forecast_value = forecast_df.loc[forecast_timestamp, 'forecast']
                        # Check if the value is valid (not NaN)
                        if pd.notna(forecast_value):
                            forecast_value = float(forecast_value)
                        else:
                            forecast_value = None
                            logger.warning(f"Forecast value is NaN for {custom_name} at {forecast_timestamp}")
                    else:
                        forecast_value = None
                        logger.warning(f"Forecast timestamp {forecast_timestamp} not found in forecast_df for {custom_name}")
                except Exception as e:
                    forecast_value = None
                    logger.error(f"Error accessing forecast value for {custom_name} at {forecast_timestamp}: {e}")
                
                forecast_result = {
                    "timestamp": create_utc_datetime(date, hour, timezone(timedelta(hours=6))).isoformat(),
                    "forecast": forecast_value
                }
                model_forecasts.append(forecast_result)
            
            # Store the model name and its 24-hour forecasts
            model_result = {
                "custom_name": custom_name,
                "model_forecasts": model_forecasts
            }
            all_forecasts.append(model_result)
            logger.info(f"Completed forecast for model: {custom_name}")
        
        logger.info(f"Completed forecasts for all {len(custom_names)} models")
        return {
            "all_forecasts": all_forecasts,
            "actual_loads": actual_loads
        }

def _forecast_24_hours(custom_name: str, to_forecast_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 24-hour forecast for a given model using pre-prepared data with NaN values
    
    Args:
        custom_name: Name of the trained model
        to_forecast_data: DataFrame with NaN values for hours to be predicted
        
    Returns:
        DataFrame containing forecast results for 24 hours
    """
    # Load the prediction job configuration
    dictionary_path = f"./{PARENT_DIR}/{custom_name}/pj.pkl"
    with open(dictionary_path, "rb") as file:
        pj = pickle.load(file)
    
    # Set up MLflow tracking URI
    mlflow_tracking_uri = f"{PARENT_DIR}/{custom_name}/mlflow_trained_models"
    
    # Create forecast pipeline
    forecast = create_forecast_pipeline(
        pj,
        to_forecast_data,
        mlflow_tracking_uri,
    )
    
    logger.info(f"Forecast results for {custom_name}:\n{forecast}")
    
    return forecast

def calculate_previous_hr_of_forecast(date: str, hour: int) -> datetime:
    # Create UTC datetime from date and hour parameters
    # Handle hour adjustment logic: if hour > 0, subtract 1; if hour == 0, go to previous date at hour 23
    if hour > 0:
        adjusted_hour = hour - 1
        adjusted_date = date
    else:
        # hour == 0, so we need previous date and hour 23    
        given_date = datetime.strptime(date, '%Y-%m-%d')
        previous_date = given_date - timedelta(days=1)
        adjusted_date = previous_date.strftime('%Y-%m-%d')
        adjusted_hour = 23
    
    return create_utc_datetime(adjusted_date, adjusted_hour)


def get_training_data_last_index(input_data: pd.DataFrame, date: str) -> int:
    """
    Robustly find the index position of the last training data point before forecast period.
    
    Uses searchsorted to handle cases where the exact timestamp doesn't exist in the data.
    
    Args:
        input_data: DataFrame with datetime index containing training data
        date: Forecast date in 'YYYY-MM-DD' format
        
    Returns:
        Integer index position of the last training data point
        
    Raises:
        ValueError: If forecast date is before available training data
    """
    previous_hour = calculate_previous_hr_of_forecast(date, 0)
    
    # Check if the exact timestamp exists
    if previous_hour in input_data.index:
        training_data_last_index = input_data.index.get_loc(previous_hour)
        logger.info(f"Found training data ending at: {previous_hour}")
        return training_data_last_index
    
    # Exact timestamp not found - use searchsorted to find closest position
    insert_pos = input_data.index.searchsorted(previous_hour)
    
    if insert_pos == 0:
        # Requested date is before all available data
        logger.error(f"Forecast date {date} is before available training data")
        raise ValueError(
            f"No training data available before {date}. "
            f"Earliest available data: {input_data.index.min().strftime('%Y-%m-%d')}. "
            f"Please select a later date."
        )
    elif insert_pos >= len(input_data):
        # Requested date is after all available data - use last available index
        logger.warning(f"Previous hour {previous_hour} is after all data. Using last available timestamp.")
        training_data_last_index = len(input_data) - 1
    else:
        # Use the index just before the insertion point
        training_data_last_index = insert_pos - 1
        actual_timestamp = input_data.index[training_data_last_index]
        logger.warning(
            f"Previous hour {previous_hour} not found in data. "
            f"Using closest earlier timestamp: {actual_timestamp}"
        )
    
    return training_data_last_index


def get_test_data_for_date(input_data: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Extract test data for a specific forecast date, handling missing timestamps gracefully.
    
    Instead of assuming 24 consecutive hours exist, this filters by date range to get
    only the timestamps that actually exist in the data.
    
    Args:
        input_data: DataFrame with datetime index containing all data
        date: Forecast date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing only the available timestamps for the forecast date
    """
    # Define the 24-hour forecast period
    forecast_start = create_utc_datetime(date, 0)
    forecast_end = create_utc_datetime(date, 23)
    
    # Filter to get only timestamps within the forecast period that exist in the data
    test_data = input_data[(input_data.index >= forecast_start) & (input_data.index <= forecast_end)]
    
    # Log information about data availability
    logger.info(f"Test data contains {len(test_data)} hours out of 24 possible for date {date}")
    
    if len(test_data) > 0:
        logger.info(f"Test data starting hour: {test_data.head(1).index[0]}")
        logger.info(f"Test data ending hour: {test_data.tail(1).index[0]}")
        
        # Log any missing hours for debugging
        expected_hours = set(range(24))
        available_hours = set(test_data.index.hour)
        missing_hours = expected_hours - available_hours
        
        if missing_hours:
            logger.warning(f"Missing hours in test data: {sorted(missing_hours)}")
    else:
        logger.warning(f"No data found for forecast date {date}")
    
    return test_data

