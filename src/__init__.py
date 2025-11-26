"""
Crop Yield Prediction System - Source Package.

This package contains modules for data generation, preprocessing,
model training, and interpretation for crop yield prediction.
"""

from .data_generation import generate_synthetic_dataset
from .preprocessing import preprocess_pipeline, prepare_model_data
from .models import ModelTrainer, train_and_evaluate

__version__ = "1.0.0"
__all__ = [
    "generate_synthetic_dataset",
    "preprocess_pipeline",
    "prepare_model_data",
    "ModelTrainer",
    "train_and_evaluate"
]
