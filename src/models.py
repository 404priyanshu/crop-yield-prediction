"""
Model Training Module for Crop Yield Prediction System.

Implements various ML models including Random Forest, XGBoost, SVR, MLP,
LSTM, CNN, and Ensemble methods for crop yield prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import yaml
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, TimeSeriesSplit
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Install with: pip install xgboost")

# TensorFlow/Keras for deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten,
        BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. Install with: pip install tensorflow")


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
    """Get default model configuration."""
    return {
        'random_seed': 42,
        'models': {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'svr': {
                'kernel': 'rbf',
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'mlp': {
                'hidden_layer_sizes': [128, 64, 32],
                'activation': 'relu',
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42
            },
            'lstm': {
                'units': 128,
                'dropout': 0.3,
                'dense_units': 64,
                'epochs': 100,
                'batch_size': 32,
                'patience': 10
            }
        },
        'training': {
            'cv_folds': 5,
            'test_size': 0.2
        }
    }


def rmsle_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error loss function.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        RMSLE value.
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        MAPE value as percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of metric names and values.
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'rmsle': rmsle_loss(y_true, y_pred)
    }


class RandomForestModel:
    """Random Forest Regressor wrapper."""
    
    def __init__(self, config: dict):
        """
        Initialize Random Forest model.
        
        Args:
            config: Model configuration dictionary.
        """
        model_config = config.get('models', {}).get('random_forest', {})
        self.model = RandomForestRegressor(
            n_estimators=model_config.get('n_estimators', 200),
            max_depth=model_config.get('max_depth', 20),
            random_state=model_config.get('random_state', 42),
            n_jobs=model_config.get('n_jobs', -1)
        )
        self.name = "RandomForest"
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_


class XGBoostModel:
    """XGBoost Regressor wrapper."""
    
    def __init__(self, config: dict):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed")
        
        model_config = config.get('models', {}).get('xgboost', {})
        self.model = xgb.XGBRegressor(
            n_estimators=model_config.get('n_estimators', 200),
            max_depth=model_config.get('max_depth', 8),
            learning_rate=model_config.get('learning_rate', 0.1),
            subsample=model_config.get('subsample', 0.8),
            colsample_bytree=model_config.get('colsample_bytree', 0.8),
            random_state=model_config.get('random_state', 42),
            verbosity=0
        )
        self.name = "XGBoost"
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[List] = None) -> 'XGBoostModel':
        """Train the model."""
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_


class SVRModel:
    """Support Vector Regression wrapper with GridSearch."""
    
    def __init__(self, config: dict):
        """
        Initialize SVR model.
        
        Args:
            config: Model configuration dictionary.
        """
        model_config = config.get('models', {}).get('svr', {})
        
        # Set up GridSearchCV for hyperparameter tuning
        self.model = GridSearchCV(
            SVR(kernel=model_config.get('kernel', 'rbf')),
            param_grid={
                'C': model_config.get('C', [0.1, 1, 10]),
                'gamma': model_config.get('gamma', ['scale', 'auto'])
            },
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        self.name = "SVR"
        self.best_params_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRModel':
        """Train the model with GridSearchCV."""
        self.model.fit(X, y)
        self.best_params_ = self.model.best_params_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class MLPModel:
    """Multi-layer Perceptron Regressor wrapper."""
    
    def __init__(self, config: dict):
        """
        Initialize MLP model.
        
        Args:
            config: Model configuration dictionary.
        """
        model_config = config.get('models', {}).get('mlp', {})
        
        hidden_layers = model_config.get('hidden_layer_sizes', [128, 64, 32])
        
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation=model_config.get('activation', 'relu'),
            max_iter=model_config.get('max_iter', 500),
            early_stopping=model_config.get('early_stopping', True),
            validation_fraction=model_config.get('validation_fraction', 0.1),
            random_state=model_config.get('random_state', 42)
        )
        self.name = "MLP"
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPModel':
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class LSTMModel:
    """LSTM Neural Network for temporal sequence prediction."""
    
    def __init__(self, config: dict, input_shape: Tuple[int, int] = (12, 5)):
        """
        Initialize LSTM model.
        
        Args:
            config: Model configuration dictionary.
            input_shape: Shape of input (timesteps, features).
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is not installed")
        
        model_config = config.get('models', {}).get('lstm', {})
        
        self.units = model_config.get('units', 128)
        self.dropout = model_config.get('dropout', 0.3)
        self.dense_units = model_config.get('dense_units', 64)
        self.epochs = model_config.get('epochs', 100)
        self.batch_size = model_config.get('batch_size', 32)
        self.patience = model_config.get('patience', 10)
        self.input_shape = input_shape
        
        self.model = self._build_model()
        self.name = "LSTM"
        self.history = None
    
    def _build_model(self) -> keras.Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.units, input_shape=self.input_shape, return_sequences=False),
            Dropout(self.dropout),
            Dense(self.dense_units, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple] = None) -> 'LSTMModel':
        """
        Train the LSTM model.
        
        Args:
            X: Input features with shape (samples, timesteps, features).
            y: Target values.
            validation_data: Optional validation tuple (X_val, y_val).
        """
        callbacks = [
            EarlyStopping(patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save the model."""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load a saved model."""
        instance = cls.__new__(cls)
        instance.model = keras.models.load_model(path)
        instance.name = "LSTM"
        return instance


class CNNModel:
    """CNN model for spatial NDVI pattern recognition."""
    
    def __init__(self, config: dict, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        """
        Initialize CNN model.
        
        Args:
            config: Model configuration dictionary.
            input_shape: Shape of input images (height, width, channels).
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is not installed")
        
        model_config = config.get('models', {}).get('cnn', {})
        
        self.conv1_filters = model_config.get('conv1_filters', 32)
        self.conv2_filters = model_config.get('conv2_filters', 64)
        self.kernel_size = model_config.get('kernel_size', 3)
        self.pool_size = model_config.get('pool_size', 2)
        self.input_shape = input_shape
        
        self.model = self._build_model()
        self.name = "CNN"
    
    def _build_model(self) -> keras.Model:
        """Build CNN model architecture."""
        model = Sequential([
            Conv2D(self.conv1_filters, (self.kernel_size, self.kernel_size),
                   activation='relu', input_shape=self.input_shape),
            MaxPooling2D((self.pool_size, self.pool_size)),
            Conv2D(self.conv2_filters, (self.kernel_size, self.kernel_size),
                   activation='relu'),
            MaxPooling2D((self.pool_size, self.pool_size)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple] = None,
            epochs: int = 50, batch_size: int = 32) -> 'CNNModel':
        """Train the CNN model."""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0).flatten()


class EnsembleModel:
    """Stacking ensemble model combining multiple base models."""
    
    def __init__(self, config: dict):
        """
        Initialize ensemble model.
        
        Args:
            config: Model configuration dictionary.
        """
        model_config = config.get('models', {})
        
        # Base estimators
        estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=model_config.get('random_forest', {}).get('n_estimators', 100),
                max_depth=model_config.get('random_forest', {}).get('max_depth', 15),
                random_state=42,
                n_jobs=-1
            )),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42
            ))
        ]
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            estimators.append(
                ('xgb', xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ))
            )
        
        # Meta-learner
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1
        )
        self.name = "Ensemble"
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """Train the ensemble model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class ModelTrainer:
    """
    Main class for training and evaluating all models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
    
    def initialize_models(self, 
                          include_deep_learning: bool = True) -> Dict[str, Any]:
        """
        Initialize all models.
        
        Args:
            include_deep_learning: Whether to include LSTM and CNN models.
            
        Returns:
            Dictionary of initialized models.
        """
        print("Initializing models...")
        
        # Traditional ML models
        self.models['RandomForest'] = RandomForestModel(self.config)
        
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBoostModel(self.config)
        
        self.models['SVR'] = SVRModel(self.config)
        self.models['MLP'] = MLPModel(self.config)
        
        # Ensemble
        self.models['Ensemble'] = EnsembleModel(self.config)
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
        return self.models
    
    def train_model(self, model_name: str, X_train: np.ndarray, 
                    y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            
        Returns:
            Trained model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        
        if hasattr(model, 'fit'):
            if model_name == 'XGBoost' and X_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            
        Returns:
            Dictionary of trained models.
        """
        for name in self.models:
            try:
                self.train_model(name, X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        return self.models
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a specific model.
        
        Args:
            model_name: Name of the model to evaluate.
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary of metrics.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, 
                            y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            DataFrame with all model metrics.
        """
        results = []
        
        for name in self.models:
            try:
                metrics = self.evaluate_model(name, X_test, y_test)
                metrics['model'] = name
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return pd.DataFrame(results).set_index('model')
    
    def cross_validate(self, model_name: str, X: np.ndarray, 
                       y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model.
            X: Feature matrix.
            y: Target vector.
            cv: Number of cross-validation folds.
            
        Returns:
            Dictionary with cross-validation scores.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name].model
        
        tscv = TimeSeriesSplit(n_splits=cv)
        
        scores = cross_val_score(model, X, y, cv=tscv, 
                                  scoring='neg_mean_squared_error')
        
        return {
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(-scores).std()
        }
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on R² score.
        
        Returns:
            Tuple of (model name, model object).
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        best_name = max(self.results.keys(), 
                        key=lambda x: self.results[x]['metrics']['r2'])
        
        return best_name, self.models[best_name]
    
    def save_models(self, output_dir: str = "models") -> None:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            if hasattr(model, 'model'):
                path = os.path.join(output_dir, f"{name}_{timestamp}.pkl")
                joblib.dump(model.model, path)
                print(f"Saved {name} to {path}")
    
    def load_model(self, path: str, model_name: str) -> Any:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model.
            model_name: Name to assign to the loaded model.
            
        Returns:
            Loaded model.
        """
        model = joblib.load(path)
        self.models[model_name] = model
        return model


def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       test_size: float = 0.2,
                       config_path: str = "config.yaml") -> Tuple[ModelTrainer, pd.DataFrame]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of data for testing.
        config_path: Path to configuration file.
        
    Returns:
        Tuple of (trainer object, results DataFrame).
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Create validation set from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(config_path)
    trainer.initialize_models(include_deep_learning=False)
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate all models
    results = trainer.evaluate_all_models(X_test, y_test)
    
    print("\nModel Comparison Results:")
    print(results.round(4))
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest model: {best_name}")
    
    # Save models
    trainer.save_models()
    
    return trainer, results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load and preprocess data
    from data_generation import generate_synthetic_dataset
    from preprocessing import preprocess_pipeline, prepare_model_data
    
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=5000, output_path="data/train_dataset.csv")
    
    # Preprocess
    df_processed, encoder, scaler = preprocess_pipeline(df)
    
    # Prepare for modeling
    X, y = prepare_model_data(df_processed)
    
    # Train and evaluate
    trainer, results = train_and_evaluate(X.values, y.values)
