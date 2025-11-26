"""
Data Preprocessing Module for Crop Yield Prediction System.

Handles missing value imputation, feature engineering, encoding,
and scaling for the crop yield dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
import yaml
import os
import joblib


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dictionary containing configuration settings.
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'random_seed': 42,
        'preprocessing': {
            'knn_imputer_neighbors': 5,
            'robust_scaler_quantile_range': [25.0, 75.0],
            'ndvi_poly_degree': 2
        },
        'crop_growth': {
            'wheat': {'days_to_harvest': 120},
            'rice': {'days_to_harvest': 130},
            'corn': {'days_to_harvest': 140},
            'soybean': {'days_to_harvest': 150}
        }
    }


class TargetEncoder:
    """
    Target encoder for categorical variables.
    
    Encodes categorical features using the mean of the target variable,
    with smoothing to prevent overfitting.
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the target encoder.
        
        Args:
            smoothing: Smoothing parameter for regularization.
        """
        self.smoothing = smoothing
        self.encodings: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 0.0
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            columns: List[str]) -> 'TargetEncoder':
        """
        Fit the encoder to the training data.
        
        Args:
            X: Input DataFrame with categorical columns.
            y: Target variable series.
            columns: List of column names to encode.
            
        Returns:
            Self reference for method chaining.
        """
        self.global_mean = y.mean()
        
        for col in columns:
            self.encodings[col] = {}
            col_data = X[col].astype(str)
            
            for category in col_data.unique():
                mask = col_data == category
                n = mask.sum()
                target_mean = y[mask].mean()
                
                # Apply smoothing
                smoothed = (n * target_mean + self.smoothing * self.global_mean) / (n + self.smoothing)
                self.encodings[col][category] = smoothed
                
        return self
    
    def transform(self, X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform categorical columns using learned encodings.
        
        Args:
            X: Input DataFrame with categorical columns.
            columns: List of column names to transform.
            
        Returns:
            DataFrame with encoded columns.
        """
        X_encoded = X.copy()
        
        for col in columns:
            if col in self.encodings:
                encoded_col = f"{col}_encoded"
                X_encoded[encoded_col] = X_encoded[col].astype(str).map(
                    lambda x: self.encodings[col].get(x, self.global_mean)
                )
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                      columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input DataFrame with categorical columns.
            y: Target variable series.
            columns: List of column names to encode.
            
        Returns:
            DataFrame with encoded columns.
        """
        self.fit(X, y, columns)
        return self.transform(X, columns)
    
    def save(self, path: str) -> None:
        """Save encoder to file."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'TargetEncoder':
        """Load encoder from file."""
        return joblib.load(path)


def extract_date_features(sowing_date: str, crop_type: str, 
                          config: dict) -> Dict[str, Any]:
    """
    Extract features from sowing date.
    
    Args:
        sowing_date: Date string in DD-MM-YYYY format.
        crop_type: Type of crop for days_to_harvest calculation.
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing extracted date features.
    """
    try:
        date = datetime.strptime(str(sowing_date), '%d-%m-%Y')
    except (ValueError, TypeError):
        # Default values for invalid dates
        return {
            'day_of_year': 182,
            'season': 'summer',
            'days_to_harvest': 135,
            'growth_stage': 'vegetative'
        }
    
    # Extract day of year
    day_of_year = date.timetuple().tm_yday
    
    # Determine season
    if day_of_year <= 79 or day_of_year > 355:
        season = 'winter'
    elif day_of_year <= 171:
        season = 'spring'
    elif day_of_year <= 265:
        season = 'summer'
    else:
        season = 'autumn'
    
    # Get days to harvest from config
    crop_config = config.get('crop_growth', {})
    crop_settings = crop_config.get(str(crop_type), {})
    days_to_harvest = crop_settings.get('days_to_harvest', 135)
    
    # Calculate growth stage (assume current observation is mid-season)
    # Random stage assignment based on typical distribution
    np.random.seed(hash(sowing_date) % (2**32))
    random_progress = np.random.uniform(0.1, 0.9)
    
    if random_progress < 0.15:
        growth_stage = 'germination'
    elif random_progress < 0.50:
        growth_stage = 'vegetative'
    elif random_progress < 0.80:
        growth_stage = 'reproductive'
    else:
        growth_stage = 'maturity'
    
    return {
        'day_of_year': day_of_year,
        'season': season,
        'days_to_harvest': days_to_harvest,
        'growth_stage': growth_stage
    }


def impute_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Impute missing values in the dataset.
    
    Uses KNN imputation for NDVI and linear interpolation for weather data.
    
    Args:
        df: Input DataFrame with missing values.
        config: Configuration dictionary.
        
    Returns:
        DataFrame with imputed values.
    """
    df_imputed = df.copy()
    
    # Get preprocessing config
    preprocess_config = config.get('preprocessing', {})
    n_neighbors = preprocess_config.get('knn_imputer_neighbors', 5)
    
    # Identify numeric columns for imputation
    numeric_cols = ['ndvi', 'precipitation_mm', 'temperature_c', 'soil_organic_carbon_pct']
    
    # Check which columns exist and have missing values
    cols_to_impute = [col for col in numeric_cols if col in df_imputed.columns]
    
    if len(cols_to_impute) > 0:
        # Use KNN imputation for all numeric columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        # Only impute if there are missing values
        if df_imputed[cols_to_impute].isnull().any().any():
            df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])
    
    return df_imputed


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create engineered features from the dataset.
    
    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
        
    Returns:
        DataFrame with new engineered features.
    """
    df_eng = df.copy()
    
    # Extract date features
    date_features = []
    for idx, row in df_eng.iterrows():
        features = extract_date_features(
            row['sowing_date'], 
            row['crop_type'], 
            config
        )
        date_features.append(features)
    
    date_df = pd.DataFrame(date_features)
    df_eng = pd.concat([df_eng.reset_index(drop=True), date_df], axis=1)
    
    # Create interaction features
    df_eng['ndvi_precipitation_product'] = df_eng['ndvi'] * df_eng['precipitation_mm'] / 1000
    df_eng['temp_soil_interaction'] = df_eng['temperature_c'] * df_eng['soil_organic_carbon_pct']
    
    # Create polynomial features for NDVI
    preprocess_config = config.get('preprocessing', {})
    poly_degree = preprocess_config.get('ndvi_poly_degree', 2)
    
    if poly_degree >= 2:
        df_eng['ndvi_squared'] = df_eng['ndvi'] ** 2
    
    # Encode season and growth_stage as numeric
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    growth_stage_map = {'germination': 0, 'vegetative': 1, 'reproductive': 2, 'maturity': 3}
    
    df_eng['season_encoded'] = df_eng['season'].map(season_map)
    df_eng['growth_stage_encoded'] = df_eng['growth_stage'].map(growth_stage_map)
    
    return df_eng


def scale_features(df: pd.DataFrame, columns: List[str], 
                   config: dict, scaler: Optional[RobustScaler] = None,
                   fit: bool = True) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Scale numeric features using RobustScaler.
    
    Args:
        df: Input DataFrame.
        columns: List of column names to scale.
        config: Configuration dictionary.
        scaler: Pre-fitted scaler (optional).
        fit: Whether to fit the scaler on this data.
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    df_scaled = df.copy()
    
    # Get scaler config
    preprocess_config = config.get('preprocessing', {})
    quantile_range = tuple(preprocess_config.get('robust_scaler_quantile_range', [25.0, 75.0]))
    
    if scaler is None:
        scaler = RobustScaler(quantile_range=quantile_range)
    
    # Filter to columns that exist
    existing_cols = [col for col in columns if col in df_scaled.columns]
    
    if len(existing_cols) > 0:
        if fit:
            df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
        else:
            df_scaled[existing_cols] = scaler.transform(df_scaled[existing_cols])
    
    return df_scaled, scaler


def create_time_series_splits(df: pd.DataFrame, n_splits: int = 5) -> TimeSeriesSplit:
    """
    Create time series cross-validation splits.
    
    Args:
        df: Input DataFrame with date information.
        n_splits: Number of splits for cross-validation.
        
    Returns:
        TimeSeriesSplit object.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def preprocess_pipeline(df: pd.DataFrame, 
                        config_path: str = "config.yaml",
                        target_col: str = 'actual_yield_tons_per_ha',
                        fit_encoders: bool = True,
                        encoder: Optional[TargetEncoder] = None,
                        scaler: Optional[RobustScaler] = None
                        ) -> Tuple[pd.DataFrame, TargetEncoder, RobustScaler]:
    """
    Complete preprocessing pipeline for crop yield data.
    
    Args:
        df: Input DataFrame.
        config_path: Path to configuration file.
        target_col: Name of target variable column.
        fit_encoders: Whether to fit encoders on this data.
        encoder: Pre-fitted target encoder (optional).
        scaler: Pre-fitted scaler (optional).
        
    Returns:
        Tuple of (preprocessed DataFrame, target encoder, scaler).
    """
    config = load_config(config_path)
    
    print("Starting preprocessing pipeline...")
    
    # Step 1: Impute missing values
    print("Step 1: Imputing missing values...")
    df_processed = impute_missing_values(df, config)
    
    # Step 2: Engineer features
    print("Step 2: Engineering features...")
    df_processed = engineer_features(df_processed, config)
    
    # Step 3: Target encoding for categorical variables
    print("Step 3: Applying target encoding...")
    categorical_cols = ['crop_type', 'region']
    
    if fit_encoders:
        if encoder is None:
            encoder = TargetEncoder(smoothing=1.0)
        encoder.fit(df_processed, df_processed[target_col], categorical_cols)
    
    df_processed = encoder.transform(df_processed, categorical_cols)
    
    # Step 4: Scale numeric features
    print("Step 4: Scaling features...")
    numeric_cols_to_scale = [
        'ndvi', 'precipitation_mm', 'temperature_c', 'soil_organic_carbon_pct',
        'day_of_year', 'days_to_harvest', 'ndvi_precipitation_product',
        'temp_soil_interaction', 'ndvi_squared', 'crop_type_encoded', 'region_encoded'
    ]
    
    # Filter to existing columns
    numeric_cols_to_scale = [col for col in numeric_cols_to_scale if col in df_processed.columns]
    
    df_processed, scaler = scale_features(
        df_processed, numeric_cols_to_scale, config, 
        scaler=scaler, fit=fit_encoders
    )
    
    print("Preprocessing complete!")
    print(f"Final shape: {df_processed.shape}")
    
    return df_processed, encoder, scaler


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names for model input.
    
    Returns:
        List of feature column names.
    """
    return [
        'ndvi', 'precipitation_mm', 'temperature_c', 'soil_organic_carbon_pct',
        'day_of_year', 'days_to_harvest', 'season_encoded', 'growth_stage_encoded',
        'ndvi_precipitation_product', 'temp_soil_interaction', 'ndvi_squared',
        'crop_type_encoded', 'region_encoded'
    ]


def prepare_model_data(df: pd.DataFrame, 
                       target_col: str = 'actual_yield_tons_per_ha'
                       ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        df: Preprocessed DataFrame.
        target_col: Name of target variable column.
        
    Returns:
        Tuple of (features DataFrame, target Series).
    """
    feature_cols = get_feature_columns()
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def save_preprocessors(encoder: TargetEncoder, scaler: RobustScaler, 
                       output_dir: str = "models") -> None:
    """
    Save preprocessing objects for later use.
    
    Args:
        encoder: Fitted target encoder.
        scaler: Fitted scaler.
        output_dir: Directory to save the objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(encoder, os.path.join(output_dir, 'target_encoder.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"Preprocessors saved to {output_dir}")


def load_preprocessors(input_dir: str = "models") -> Tuple[TargetEncoder, RobustScaler]:
    """
    Load saved preprocessing objects.
    
    Args:
        input_dir: Directory containing saved objects.
        
    Returns:
        Tuple of (target encoder, scaler).
    """
    encoder = joblib.load(os.path.join(input_dir, 'target_encoder.pkl'))
    scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
    return encoder, scaler


if __name__ == "__main__":
    # Test the preprocessing pipeline
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load sample data
    from data_generation import generate_synthetic_dataset
    
    df = generate_synthetic_dataset(n_samples=1000, output_path="data/test_dataset.csv")
    
    # Run preprocessing
    df_processed, encoder, scaler = preprocess_pipeline(df)
    
    # Prepare for modeling
    X, y = prepare_model_data(df_processed)
    
    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFeature columns:", list(X.columns))
    print("\nSample features:")
    print(X.head())
