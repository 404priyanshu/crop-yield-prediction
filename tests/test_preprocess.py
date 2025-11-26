"""
Unit Tests for Crop Yield Prediction System.

Tests preprocessing functions, data validation, and model performance.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataGeneration:
    """Tests for data generation module."""
    
    def test_generate_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=100, 
                                         output_path="/tmp/test_dataset.csv")
        
        assert df is not None
        assert len(df) == 100
        assert 'crop_type' in df.columns
        assert 'region' in df.columns
        assert 'ndvi' in df.columns
        assert 'actual_yield_tons_per_ha' in df.columns
    
    def test_ndvi_range(self):
        """Test that NDVI values are in valid range."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=1000, 
                                         output_path="/tmp/test_ndvi.csv")
        
        # NDVI should be between -1 and 1
        valid_ndvi = df['ndvi'].dropna()
        assert valid_ndvi.min() >= -1.0
        assert valid_ndvi.max() <= 1.0
    
    def test_precipitation_positive(self):
        """Test that precipitation is non-negative."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=1000, 
                                         output_path="/tmp/test_precip.csv")
        
        valid_precip = df['precipitation_mm'].dropna()
        assert valid_precip.min() >= 0
    
    def test_crop_types(self):
        """Test that only valid crop types are generated."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=1000, 
                                         output_path="/tmp/test_crops.csv")
        
        valid_crops = {'wheat', 'rice', 'corn', 'soybean'}
        assert set(df['crop_type'].unique()).issubset(valid_crops)
    
    def test_sowing_date_format(self):
        """Test that sowing dates are in correct format."""
        from src.data_generation import generate_synthetic_dataset
        from datetime import datetime
        
        df = generate_synthetic_dataset(n_samples=100, 
                                         output_path="/tmp/test_dates.csv")
        
        # Try parsing all dates
        for date_str in df['sowing_date']:
            try:
                datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        from src.data_generation import generate_synthetic_dataset
        return generate_synthetic_dataset(n_samples=500, 
                                          output_path="/tmp/test_preprocess.csv")
    
    def test_impute_missing_values(self, sample_data):
        """Test missing value imputation."""
        from src.preprocessing import impute_missing_values, load_config
        
        config = load_config()
        df_imputed = impute_missing_values(sample_data, config)
        
        # No missing values in numeric columns
        numeric_cols = ['ndvi', 'precipitation_mm', 'temperature_c', 'soil_organic_carbon_pct']
        for col in numeric_cols:
            if col in df_imputed.columns:
                assert df_imputed[col].isnull().sum() == 0
    
    def test_extract_date_features(self):
        """Test date feature extraction."""
        from src.preprocessing import extract_date_features, load_config
        
        config = load_config()
        
        # Test spring date
        features = extract_date_features('15-04-2024', 'wheat', config)
        assert features['season'] == 'spring'
        assert features['day_of_year'] == 106
        
        # Test summer date
        features = extract_date_features('15-07-2024', 'corn', config)
        assert features['season'] == 'summer'
    
    def test_target_encoder(self, sample_data):
        """Test target encoding."""
        from src.preprocessing import TargetEncoder
        
        encoder = TargetEncoder(smoothing=1.0)
        
        y = sample_data['actual_yield_tons_per_ha']
        df_encoded = encoder.fit_transform(sample_data, y, ['crop_type', 'region'])
        
        assert 'crop_type_encoded' in df_encoded.columns
        assert 'region_encoded' in df_encoded.columns
    
    def test_engineer_features(self, sample_data):
        """Test feature engineering."""
        from src.preprocessing import engineer_features, load_config
        
        config = load_config()
        df_eng = engineer_features(sample_data, config)
        
        # Check new features exist
        assert 'day_of_year' in df_eng.columns
        assert 'season' in df_eng.columns
        assert 'days_to_harvest' in df_eng.columns
        assert 'ndvi_precipitation_product' in df_eng.columns
        assert 'temp_soil_interaction' in df_eng.columns
        assert 'ndvi_squared' in df_eng.columns
    
    def test_preprocess_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        from src.preprocessing import preprocess_pipeline
        
        df_processed, encoder, scaler = preprocess_pipeline(sample_data)
        
        assert df_processed is not None
        assert encoder is not None
        assert scaler is not None
        assert len(df_processed) == len(sample_data)


class TestModels:
    """Tests for model training module."""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare data for model testing."""
        from src.data_generation import generate_synthetic_dataset
        from src.preprocessing import preprocess_pipeline, prepare_model_data
        
        df = generate_synthetic_dataset(n_samples=500, 
                                         output_path="/tmp/test_model_data.csv")
        df_processed, _, _ = preprocess_pipeline(df)
        X, y = prepare_model_data(df_processed)
        
        return X.values, y.values
    
    def test_random_forest_model(self, prepared_data):
        """Test Random Forest model training."""
        from src.models import RandomForestModel, load_config
        from sklearn.model_selection import train_test_split
        
        X, y = prepared_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        config = load_config()
        model = RandomForestModel(config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_calculate_metrics(self):
        """Test metric calculations."""
        from src.models import calculate_metrics
        
        y_true = np.array([3.0, 4.0, 5.0, 3.5, 4.5])
        y_pred = np.array([3.1, 3.9, 5.2, 3.4, 4.6])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0
    
    def test_model_r2_performance(self, prepared_data):
        """Test that model achieves acceptable R² score."""
        from src.models import RandomForestModel, load_config, calculate_metrics
        from sklearn.model_selection import train_test_split
        
        X, y = prepared_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        config = load_config()
        model = RandomForestModel(config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        
        # Model should achieve R² > 0.5 on synthetic data
        assert metrics['r2'] > 0.5, f"R² score {metrics['r2']} is below threshold"


class TestDataValidation:
    """Tests for data validation."""
    
    def test_ndvi_range_validation(self):
        """Test NDVI range violations detection."""
        # Valid NDVI
        valid_ndvi = [0.5, 0.7, -0.2, 0.0, 0.9]
        for ndvi in valid_ndvi:
            assert -1.0 <= ndvi <= 1.0
        
        # Invalid NDVI
        invalid_ndvi = [1.5, -1.5, 2.0]
        for ndvi in invalid_ndvi:
            assert not (-1.0 <= ndvi <= 1.0)
    
    def test_temperature_range(self):
        """Test temperature values are reasonable."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=1000, 
                                         output_path="/tmp/test_temp.csv")
        
        valid_temp = df['temperature_c'].dropna()
        
        # Temperature should be in reasonable range (-30 to 50°C)
        assert valid_temp.min() >= -30
        assert valid_temp.max() <= 50
    
    def test_soil_carbon_range(self):
        """Test soil organic carbon values."""
        from src.data_generation import generate_synthetic_dataset
        
        df = generate_synthetic_dataset(n_samples=1000, 
                                         output_path="/tmp/test_soc.csv")
        
        valid_soc = df['soil_organic_carbon_pct'].dropna()
        
        # SOC should be positive and below 10%
        assert valid_soc.min() >= 0
        assert valid_soc.max() <= 10


class TestAPI:
    """Tests for API endpoints."""
    
    def test_prediction_input_validation(self):
        """Test input validation for prediction."""
        # Import API model
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from api import PredictionInput
        
        # Valid input
        valid_input = PredictionInput(
            crop_type="wheat",
            region="north",
            sowing_date="15-03-2024",
            ndvi=0.65,
            precipitation_mm=850.0,
            temperature_c=22.0,
            soil_organic_carbon_pct=2.5
        )
        assert valid_input.crop_type == "wheat"
    
    def test_invalid_crop_type(self):
        """Test that invalid crop type raises error."""
        from api import PredictionInput
        
        with pytest.raises(ValueError):
            PredictionInput(
                crop_type="invalid_crop",
                region="north",
                sowing_date="15-03-2024",
                ndvi=0.65,
                precipitation_mm=850.0,
                temperature_c=22.0,
                soil_organic_carbon_pct=2.5
            )
    
    def test_ndvi_out_of_range(self):
        """Test that NDVI out of range raises error."""
        from api import PredictionInput
        
        with pytest.raises(ValueError):
            PredictionInput(
                crop_type="wheat",
                region="north",
                sowing_date="15-03-2024",
                ndvi=1.5,  # Invalid: > 1
                precipitation_mm=850.0,
                temperature_c=22.0,
                soil_organic_carbon_pct=2.5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
