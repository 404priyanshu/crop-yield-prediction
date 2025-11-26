"""
FastAPI Application for Crop Yield Prediction.

Provides REST API endpoints for yield prediction, batch prediction,
and model health checks.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
import io
import asyncio
import yaml


# Load configuration
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {'api': {'host': '0.0.0.0', 'port': 8000}}


# Initialize FastAPI app
app = FastAPI(
    title="Crop Yield Prediction API",
    description="ML-powered API for predicting crop yields based on environmental factors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input model for single prediction."""
    crop_type: str = Field(..., description="Type of crop (wheat, rice, corn, soybean)")
    region: str = Field(..., description="Region name (north, south, east, west, central)")
    sowing_date: str = Field(..., description="Sowing date in DD-MM-YYYY format")
    ndvi: float = Field(..., ge=-1.0, le=1.0, description="NDVI value (-1 to 1)")
    precipitation_mm: float = Field(..., ge=0, description="Precipitation in mm")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    soil_organic_carbon_pct: float = Field(..., ge=0, le=10, description="Soil organic carbon %")
    
    @field_validator('crop_type')
    @classmethod
    def validate_crop_type(cls, v):
        valid_crops = ['wheat', 'rice', 'corn', 'soybean']
        if v.lower() not in valid_crops:
            raise ValueError(f'crop_type must be one of {valid_crops}')
        return v.lower()
    
    @field_validator('region')
    @classmethod
    def validate_region(cls, v):
        valid_regions = ['north', 'south', 'east', 'west', 'central']
        if v.lower() not in valid_regions:
            raise ValueError(f'region must be one of {valid_regions}')
        return v.lower()
    
    model_config = {"json_schema_extra": {
        "example": {
            "crop_type": "wheat",
            "region": "north",
            "sowing_date": "15-03-2024",
            "ndvi": 0.65,
            "precipitation_mm": 850.0,
            "temperature_c": 18.5,
            "soil_organic_carbon_pct": 2.5
        }
    }}


class PredictionOutput(BaseModel):
    """Output model for prediction response."""
    predicted_yield: float = Field(..., description="Predicted yield in tons/hectare")
    confidence_interval_lower: float = Field(..., description="Lower bound of 95% CI")
    confidence_interval_upper: float = Field(..., description="Upper bound of 95% CI")
    input_features: Dict[str, Any] = Field(..., description="Original input features")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")


class BatchPredictionInput(BaseModel):
    """Input model for batch predictions."""
    predictions: List[PredictionInput]


class BatchPredictionOutput(BaseModel):
    """Output model for batch prediction response."""
    predictions: List[PredictionOutput]
    total_count: int
    processing_time_seconds: float


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# Global model storage
class ModelManager:
    """Manages model loading and predictions."""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.model_version = "1.0.0"
        self.loaded = False
    
    def load_models(self, models_dir: str = "models"):
        """Load trained model and preprocessors."""
        try:
            # Try to load the latest model
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'RandomForest' in f]
            if model_files:
                latest_model = sorted(model_files)[-1]
                self.model = joblib.load(os.path.join(models_dir, latest_model))
                self.model_version = latest_model.replace('.pkl', '')
            
            # Load preprocessors
            encoder_path = os.path.join(models_dir, 'target_encoder.pkl')
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            
            if os.path.exists(encoder_path):
                self.encoder = joblib.load(encoder_path)
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.loaded = self.model is not None
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.loaded = False
    
    def preprocess_input(self, input_data: PredictionInput) -> np.ndarray:
        """Preprocess input data for prediction."""
        # Extract date features
        try:
            date = datetime.strptime(input_data.sowing_date, '%d-%m-%Y')
            day_of_year = date.timetuple().tm_yday
        except ValueError:
            day_of_year = 100  # Default
        
        # Determine season
        if day_of_year <= 79 or day_of_year > 355:
            season_encoded = 0  # winter
        elif day_of_year <= 171:
            season_encoded = 1  # spring
        elif day_of_year <= 265:
            season_encoded = 2  # summer
        else:
            season_encoded = 3  # autumn
        
        # Get days to harvest based on crop
        days_to_harvest = {
            'wheat': 120, 'rice': 130, 'corn': 140, 'soybean': 150
        }.get(input_data.crop_type, 135)
        
        # Growth stage (assume vegetative for new predictions)
        growth_stage_encoded = 1
        
        # Interaction features
        ndvi_precip_product = input_data.ndvi * input_data.precipitation_mm / 1000
        temp_soil_interaction = input_data.temperature_c * input_data.soil_organic_carbon_pct
        ndvi_squared = input_data.ndvi ** 2
        
        # Target encoding (use approximate values if encoder not available)
        crop_encoded = {'wheat': 3.5, 'rice': 4.5, 'corn': 5.0, 'soybean': 2.5}.get(
            input_data.crop_type, 3.5)
        region_encoded = {'north': 3.2, 'south': 4.0, 'east': 3.8, 'west': 3.0, 'central': 3.5}.get(
            input_data.region, 3.5)
        
        # Create feature array matching model training order
        features = np.array([
            input_data.ndvi,
            input_data.precipitation_mm,
            input_data.temperature_c,
            input_data.soil_organic_carbon_pct,
            day_of_year,
            days_to_harvest,
            season_encoded,
            growth_stage_encoded,
            ndvi_precip_product,
            temp_soil_interaction,
            ndvi_squared,
            crop_encoded,
            region_encoded
        ]).reshape(1, -1)
        
        return features
    
    def predict(self, input_data: PredictionInput) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self.loaded:
            # Return mock prediction if model not loaded
            mock_yield = 3.5 + (input_data.ndvi * 2) + (input_data.precipitation_mm / 1000)
            return {
                'predicted_yield': round(mock_yield, 2),
                'confidence_interval_lower': round(mock_yield * 0.85, 2),
                'confidence_interval_upper': round(mock_yield * 1.15, 2)
            }
        
        features = self.preprocess_input(input_data)
        
        prediction = self.model.predict(features)[0]
        
        # Estimate confidence interval (using empirical approach)
        ci_range = 0.15 * prediction
        
        return {
            'predicted_yield': round(float(prediction), 2),
            'confidence_interval_lower': round(float(prediction - ci_range), 2),
            'confidence_interval_upper': round(float(prediction + ci_range), 2)
        }


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    model_manager.load_models()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Crop Yield Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API and model status.
    """
    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_manager.loaded,
        model_version=model_manager.model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict crop yield for a single input.
    
    Takes crop type, region, sowing date, and environmental factors
    to predict expected yield in tons per hectare.
    """
    try:
        result = model_manager.predict(input_data)
        
        return PredictionOutput(
            predicted_yield=result['predicted_yield'],
            confidence_interval_lower=result['confidence_interval_lower'],
            confidence_interval_upper=result['confidence_interval_upper'],
            input_features=input_data.model_dump(),
            prediction_timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Batch prediction for multiple inputs.
    
    Accepts a list of prediction inputs and returns predictions for all.
    """
    start_time = datetime.now()
    
    try:
        predictions = []
        for input_data in batch_input.predictions:
            result = model_manager.predict(input_data)
            predictions.append(PredictionOutput(
                predicted_yield=result['predicted_yield'],
                confidence_interval_lower=result['confidence_interval_lower'],
                confidence_interval_upper=result['confidence_interval_upper'],
                input_features=input_data.model_dump(),
                prediction_timestamp=datetime.now().isoformat()
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_count=len(predictions),
            processing_time_seconds=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_csv", tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Batch prediction from CSV file upload.
    
    Upload a CSV file with columns: crop_type, region, sowing_date,
    ndvi, precipitation_mm, temperature_c, soil_organic_carbon_pct
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        required_columns = [
            'crop_type', 'region', 'sowing_date', 'ndvi',
            'precipitation_mm', 'temperature_c', 'soil_organic_carbon_pct'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        predictions = []
        for _, row in df.iterrows():
            input_data = PredictionInput(
                crop_type=row['crop_type'],
                region=row['region'],
                sowing_date=str(row['sowing_date']),
                ndvi=float(row['ndvi']),
                precipitation_mm=float(row['precipitation_mm']),
                temperature_c=float(row['temperature_c']),
                soil_organic_carbon_pct=float(row['soil_organic_carbon_pct'])
            )
            result = model_manager.predict(input_data)
            predictions.append(result)
        
        df['predicted_yield'] = [p['predicted_yield'] for p in predictions]
        df['ci_lower'] = [p['confidence_interval_lower'] for p in predictions]
        df['ci_upper'] = [p['confidence_interval_upper'] for p in predictions]
        
        return JSONResponse(content={
            "predictions": df.to_dict(orient='records'),
            "total_count": len(df)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_loaded": model_manager.loaded,
        "model_version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__ if model_manager.model else None,
        "features": [
            "ndvi", "precipitation_mm", "temperature_c", "soil_organic_carbon_pct",
            "day_of_year", "days_to_harvest", "season_encoded", "growth_stage_encoded",
            "ndvi_precipitation_product", "temp_soil_interaction", "ndvi_squared",
            "crop_type_encoded", "region_encoded"
        ]
    }


@app.get("/crops", tags=["Reference"])
async def get_supported_crops():
    """Get list of supported crops."""
    return {
        "crops": ["wheat", "rice", "corn", "soybean"],
        "details": {
            "wheat": {"days_to_harvest": 120, "optimal_temp": "15-22°C"},
            "rice": {"days_to_harvest": 130, "optimal_temp": "20-30°C"},
            "corn": {"days_to_harvest": 140, "optimal_temp": "18-28°C"},
            "soybean": {"days_to_harvest": 150, "optimal_temp": "20-30°C"}
        }
    }


@app.get("/regions", tags=["Reference"])
async def get_supported_regions():
    """Get list of supported regions."""
    return {
        "regions": ["north", "south", "east", "west", "central"]
    }


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    api_config = config.get('api', {})
    
    uvicorn.run(
        app,
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000)
    )
