# DPDC OpenSTEF - Load Forecasting Application

A professional FastAPI-based web application for electrical load forecasting with multiple machine learning models.

## Features

- **Train Model**: Configure and train XGBoost or LightGBM models with custom hyperparameters
- **Forecast**: Generate load forecasts with weather data integration and visualization
- **Data Input**: Manage predicted and actual load data by date and hour
- **Dashboard**: Visualize forecast performance with interactive charts

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

The application will be available at: http://localhost:8080

## Pages

### Train Model (/)
- Select model type (XGBoost or LightGBM)
- Configure hyperparameters dynamically based on model selection
- Submit training jobs

### Forecast (/forecast)
- Input forecast parameters (date, hour, holiday info)
- Fetch weather data automatically
- Generate forecasts from multiple models
- View 24-hour forecast charts

### Data Input (/data-input)
- Select date to view/edit data
- Update predicted and actual values for all 24 hours
- Submit batch updates

### Dashboard (/dashboard)
- View daily forecast vs actual comparison
- Compare model performance metrics
- Analyze hourly load patterns
- Key statistics at a glance

## Technology Stack

- **Backend**: FastAPI
- **Frontend**: Bootstrap 5.3, jQuery
- **Charts**: Plotly.js
- **Date Picker**: Flatpickr
- **Select Components**: Select2

## API Endpoints

- `POST /api/train` - Submit model training request
- `POST /api/forecast` - Generate load forecast
- `GET /api/weather` - Fetch weather data
- `GET /api/forecast-chart` - Get 24-hour forecast chart data
- `GET /api/data-input` - Fetch hourly data for a date
- `POST /api/data-input` - Update hourly data
- `GET /api/dashboard-data` - Get dashboard statistics and charts

## Project Structure

```
dpdc_openstef/
├── main.py                 # FastAPI application
├── templates/              # Jinja2 templates
│   ├── base.html          # Base template with navigation
│   ├── train_model.html   # Train model page
│   ├── forecast.html      # Forecast page
│   ├── data_input.html    # Data input page
│   └── dashboard.html     # Dashboard page
├── static/                 # Static files (CSS, JS, images)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Notes

- The application currently uses mock data for demonstration purposes
- All API endpoints are designed to be easily integrated with actual ML models and databases
- The UI is fully responsive and works on desktop screens of various sizes

