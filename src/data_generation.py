"""
Data Generation Module for Crop Yield Prediction System.

Generates realistic synthetic training data with 10,000+ records covering
multiple crops and regions with realistic correlations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os
from typing import Tuple, Optional


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
    return get_default_config()


def get_default_config() -> dict:
    """
    Get default configuration if config file is not available.
    
    Returns:
        Dictionary with default configuration settings.
    """
    return {
        'random_seed': 42,
        'data_generation': {
            'n_samples': 10000,
            'crops': ['wheat', 'rice', 'corn', 'soybean'],
            'regions': ['north', 'south', 'east', 'west', 'central'],
            'precipitation_optimal': [800, 1200],
            'soil_organic_carbon_range': [0.5, 5.0]
        },
        'crop_growth': {
            'wheat': {'days_to_harvest': 120, 'optimal_temp': [15, 22], 'base_yield': 3.5},
            'rice': {'days_to_harvest': 130, 'optimal_temp': [20, 30], 'base_yield': 4.5},
            'corn': {'days_to_harvest': 140, 'optimal_temp': [18, 28], 'base_yield': 5.0},
            'soybean': {'days_to_harvest': 150, 'optimal_temp': [20, 30], 'base_yield': 2.5}
        }
    }


def generate_sowing_date(n_samples: int, seed: int = 42) -> pd.Series:
    """
    Generate realistic sowing dates for crops.
    
    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        Series of sowing dates in DD-MM-YYYY format.
    """
    np.random.seed(seed)
    
    # Define sowing windows for different seasons
    sowing_windows = [
        (datetime(2020, 1, 15), datetime(2020, 3, 15)),   # Winter/early spring
        (datetime(2020, 4, 1), datetime(2020, 5, 31)),    # Late spring
        (datetime(2020, 6, 15), datetime(2020, 8, 15)),   # Summer
        (datetime(2020, 9, 1), datetime(2020, 11, 15))    # Fall
    ]
    
    dates = []
    for _ in range(n_samples):
        window_idx = np.random.choice(len(sowing_windows))
        start, end = sowing_windows[window_idx]
        days_range = (end - start).days
        random_days = np.random.randint(0, days_range + 1)
        date = start + timedelta(days=random_days)
        dates.append(date.strftime('%d-%m-%Y'))
    
    return pd.Series(dates)


def generate_ndvi(n_samples: int, growth_stages: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Generate realistic NDVI values that peak mid-season.
    
    Args:
        n_samples: Number of samples to generate.
        growth_stages: Array of growth stage indicators (0-1).
        seed: Random seed for reproducibility.
        
    Returns:
        Array of NDVI values in range [-1, 1].
    """
    np.random.seed(seed)
    
    # NDVI follows a bell curve peaking mid-season
    # Use growth stage to create realistic pattern
    peak_ndvi = 0.3 + 0.5 * np.sin(np.pi * growth_stages)  # Peak around 0.8
    noise = np.random.normal(0, 0.1, n_samples)
    ndvi = peak_ndvi + noise
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi


def generate_precipitation(n_samples: int, regions: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Generate precipitation data with regional variation.
    
    Args:
        n_samples: Number of samples to generate.
        regions: Array of region names.
        seed: Random seed for reproducibility.
        
    Returns:
        Array of precipitation values in mm.
    """
    np.random.seed(seed)
    
    # Regional precipitation patterns
    region_precip = {
        'north': (600, 200),
        'south': (1200, 300),
        'east': (1000, 250),
        'west': (500, 150),
        'central': (800, 200)
    }
    
    precipitation = []
    for region in regions:
        mean, std = region_precip.get(str(region), (800, 200))
        precip = np.random.normal(mean, std)
        precipitation.append(max(0, precip))  # No negative precipitation
    
    return np.array(precipitation)


def generate_temperature(n_samples: int, sowing_dates: pd.Series, 
                         regions: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Generate temperature data based on season and region.
    
    Args:
        n_samples: Number of samples to generate.
        sowing_dates: Series of sowing dates.
        regions: Array of region names.
        seed: Random seed for reproducibility.
        
    Returns:
        Array of temperature values in Celsius.
    """
    np.random.seed(seed)
    
    # Base temperature by region
    region_temp_base = {
        'north': 12,
        'south': 28,
        'east': 22,
        'west': 18,
        'central': 20
    }
    
    temperatures = []
    for i, date_str in enumerate(sowing_dates):
        try:
            date = datetime.strptime(date_str, '%d-%m-%Y')
        except ValueError:
            date = datetime(2020, 6, 1)
        
        # Seasonal adjustment
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        region = str(regions[i]) if i < len(regions) else 'central'
        base_temp = region_temp_base.get(region, 20)
        
        temp = base_temp + seasonal_factor + np.random.normal(0, 3)
        temperatures.append(temp)
    
    return np.array(temperatures)


def generate_soil_organic_carbon(n_samples: int, regions: np.ndarray, 
                                  seed: int = 42) -> np.ndarray:
    """
    Generate soil organic carbon percentage values.
    
    Args:
        n_samples: Number of samples to generate.
        regions: Array of region names.
        seed: Random seed for reproducibility.
        
    Returns:
        Array of soil organic carbon percentage values.
    """
    np.random.seed(seed)
    
    # Regional SOC patterns
    region_soc = {
        'north': (2.5, 0.8),
        'south': (1.8, 0.6),
        'east': (3.0, 1.0),
        'west': (1.5, 0.5),
        'central': (2.2, 0.7)
    }
    
    soc = []
    for region in regions:
        mean, std = region_soc.get(str(region), (2.0, 0.7))
        soc_value = np.random.normal(mean, std)
        soc.append(max(0.1, min(6.0, soc_value)))  # Clip to realistic range
    
    return np.array(soc)


def calculate_yield(crop_types: np.ndarray, ndvi: np.ndarray, 
                    precipitation: np.ndarray, temperature: np.ndarray,
                    soil_organic_carbon: np.ndarray, config: dict,
                    seed: int = 42) -> np.ndarray:
    """
    Calculate realistic crop yield based on all features.
    
    Args:
        crop_types: Array of crop type names.
        ndvi: Array of NDVI values.
        precipitation: Array of precipitation values in mm.
        temperature: Array of temperature values in Celsius.
        soil_organic_carbon: Array of SOC percentage values.
        config: Configuration dictionary.
        seed: Random seed for reproducibility.
        
    Returns:
        Array of yield values in tons per hectare.
    """
    np.random.seed(seed)
    
    crop_config = config.get('crop_growth', {})
    yields = []
    
    for i in range(len(crop_types)):
        crop = str(crop_types[i])
        crop_settings = crop_config.get(crop, {'base_yield': 3.0, 'optimal_temp': [20, 25]})
        
        base_yield = crop_settings.get('base_yield', 3.0)
        optimal_temp = crop_settings.get('optimal_temp', [20, 25])
        
        # NDVI factor (higher NDVI = higher yield)
        ndvi_factor = 0.5 + 0.5 * max(0, ndvi[i])
        
        # Precipitation factor (optimal range: 800-1200mm)
        precip = precipitation[i]
        if 800 <= precip <= 1200:
            precip_factor = 1.0
        elif precip < 800:
            precip_factor = 0.5 + 0.5 * (precip / 800)
        else:
            precip_factor = 1.0 - 0.3 * min(1, (precip - 1200) / 800)
        
        # Temperature factor
        temp = temperature[i]
        temp_min, temp_max = optimal_temp
        if temp_min <= temp <= temp_max:
            temp_factor = 1.0
        else:
            deviation = min(abs(temp - temp_min), abs(temp - temp_max))
            temp_factor = max(0.4, 1.0 - 0.05 * deviation)
        
        # Soil organic carbon factor
        soc = soil_organic_carbon[i]
        soc_factor = 0.7 + 0.3 * min(1, soc / 3.0)
        
        # Calculate final yield with noise
        yield_value = (base_yield * ndvi_factor * precip_factor * 
                       temp_factor * soc_factor)
        noise = np.random.normal(0, 0.3)
        yield_value = max(0.5, yield_value + noise)
        
        yields.append(yield_value)
    
    return np.array(yields)


def generate_synthetic_dataset(n_samples: int = 10000, 
                               output_path: str = "data/crop_yield_dataset.csv",
                               config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Generate complete synthetic crop yield dataset.
    
    Args:
        n_samples: Number of samples to generate.
        output_path: Path to save the generated CSV file.
        config_path: Path to configuration file.
        
    Returns:
        DataFrame containing the generated dataset.
    """
    config = load_config(config_path)
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    
    print(f"Generating synthetic dataset with {n_samples} samples...")
    
    # Get crop and region lists from config
    data_config = config.get('data_generation', {})
    crops = data_config.get('crops', ['wheat', 'rice', 'corn', 'soybean'])
    regions = data_config.get('regions', ['north', 'south', 'east', 'west', 'central'])
    
    # Generate basic categorical variables
    crop_types = np.random.choice(crops, n_samples)
    region_names = np.random.choice(regions, n_samples)
    
    # Generate sowing dates
    sowing_dates = generate_sowing_date(n_samples, seed)
    
    # Generate growth stages (0-1, random for now)
    growth_stages = np.random.uniform(0.2, 0.8, n_samples)
    
    # Generate features
    ndvi = generate_ndvi(n_samples, growth_stages, seed)
    precipitation = generate_precipitation(n_samples, region_names, seed)
    temperature = generate_temperature(n_samples, sowing_dates, region_names, seed)
    soil_organic_carbon = generate_soil_organic_carbon(n_samples, region_names, seed)
    
    # Calculate yield
    actual_yield = calculate_yield(
        crop_types, ndvi, precipitation, temperature, 
        soil_organic_carbon, config, seed
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'crop_type': crop_types,
        'region': region_names,
        'sowing_date': sowing_dates,
        'ndvi': ndvi,
        'precipitation_mm': precipitation,
        'temperature_c': temperature,
        'soil_organic_carbon_pct': soil_organic_carbon,
        'actual_yield_tons_per_ha': actual_yield
    })
    
    # Add some missing values to simulate real data (about 2%)
    np.random.seed(seed + 1)
    missing_mask = np.random.random(n_samples) < 0.02
    df.loc[missing_mask, 'ndvi'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.01
    df.loc[missing_mask, 'precipitation_mm'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.01
    df.loc[missing_mask, 'temperature_c'] = np.nan
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print("\nSample data:")
    print(df.head(10))
    print("\nStatistics:")
    print(df.describe())
