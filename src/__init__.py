"""
Crop Yield Prediction System - Source Package.

This package contains modules for data generation, preprocessing,
model training, and interpretation for crop yield prediction.
"""

from .data_generation import generate_synthetic_dataset
from .preprocessing import preprocess_pipeline, prepare_model_data

__version__ = "1.0.0"
__all__ = [
    "generate_synthetic_dataset",
    "preprocess_pipeline",
    "prepare_model_data"
]

# Optional imports (may fail if dependencies not installed)
try:
    from .models import ModelTrainer, train_and_evaluate
    __all__.extend(["ModelTrainer", "train_and_evaluate"])
except ImportError:
    pass
