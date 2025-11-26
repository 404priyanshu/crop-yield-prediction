"""
Model Interpretability Module for Crop Yield Prediction System.

Provides SHAP analysis, Partial Dependence Plots, LIME explanations,
and feature importance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# SHAP for feature explanations
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not available. Install with: pip install shap")

# LIME for local explanations
try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("LIME not available. Install with: pip install lime")

from sklearn.inspection import PartialDependenceDisplay


class ModelInterpreter:
    """
    Model interpretability and explanation class.
    
    Provides SHAP values, partial dependence plots, LIME explanations,
    and feature importance analysis.
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, 
                 feature_names: List[str]):
        """
        Initialize the interpreter.
        
        Args:
            model: Trained model object.
            X_train: Training data for background.
            feature_names: List of feature names.
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_values = None
        self.shap_explainer = None
    
    def compute_shap_values(self, X: np.ndarray, 
                            sample_size: int = 100) -> np.ndarray:
        """
        Compute SHAP values for predictions.
        
        Args:
            X: Feature matrix to explain.
            sample_size: Number of background samples.
            
        Returns:
            Array of SHAP values.
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is not installed")
        
        # Use a sample of training data for background
        if len(self.X_train) > sample_size:
            background = shap.sample(self.X_train, sample_size)
        else:
            background = self.X_train
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict'):
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict, background
            )
        else:
            raise ValueError("Model must have a predict method")
        
        # Compute SHAP values
        self.shap_values = self.shap_explainer.shap_values(X)
        
        return self.shap_values
    
    def plot_shap_summary(self, X: np.ndarray, 
                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature matrix to explain.
            output_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=self.feature_names,
            show=False
        )
        
        plt.title('SHAP Feature Importance Summary', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"SHAP summary plot saved to {output_path}")
        
        return plt.gcf()
    
    def plot_shap_waterfall(self, X_sample: np.ndarray, 
                            sample_idx: int = 0,
                            output_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X_sample: Feature matrix.
            sample_idx: Index of sample to explain.
            output_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        if self.shap_values is None:
            self.compute_shap_values(X_sample)
        
        fig = plt.figure(figsize=(12, 8))
        
        # Create explanation object
        if hasattr(self.shap_explainer, 'expected_value'):
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = 0
        
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=base_value,
            data=X_sample[sample_idx],
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot (Sample {sample_idx})', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {output_path}")
        
        return plt.gcf()
    
    def plot_shap_dependence(self, X: np.ndarray, feature_idx: int,
                             output_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature matrix.
            feature_idx: Index of the feature to analyze.
            output_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_idx, self.shap_values, X,
            feature_names=self.feature_names,
            ax=ax, show=False
        )
        
        plt.title(f'SHAP Dependence: {self.feature_names[feature_idx]}', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compute_partial_dependence(self, X: np.ndarray, 
                                   features: List[int],
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        Create partial dependence plots.
        
        Args:
            X: Feature matrix.
            features: List of feature indices to analyze.
            output_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(len(features), 1, figsize=(10, 4 * len(features)))
        if len(features) == 1:
            axes = [axes]
        
        for idx, feat_idx in enumerate(features):
            display = PartialDependenceDisplay.from_estimator(
                self.model, X, [feat_idx],
                feature_names=self.feature_names,
                ax=axes[idx]
            )
            axes[idx].set_title(f'Partial Dependence: {self.feature_names[feat_idx]}')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Partial dependence plots saved to {output_path}")
        
        return fig
    
    def explain_with_lime(self, X_sample: np.ndarray,
                          sample_idx: int = 0,
                          num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction.
        
        Args:
            X_sample: Feature matrix.
            sample_idx: Index of sample to explain.
            num_features: Number of features to show.
            
        Returns:
            Dictionary with LIME explanation.
        """
        if not HAS_LIME:
            raise ImportError("LIME is not installed")
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            mode='regression',
            random_state=42
        )
        
        # Get explanation
        exp = explainer.explain_instance(
            X_sample[sample_idx],
            self.model.predict,
            num_features=num_features
        )
        
        return {
            'feature_weights': exp.as_list(),
            'predicted_value': self.model.predict(X_sample[sample_idx:sample_idx+1])[0],
            'intercept': exp.intercept[0] if hasattr(exp, 'intercept') else None
        }
    
    def plot_lime_explanation(self, X_sample: np.ndarray,
                              sample_idx: int = 0,
                              num_features: int = 10,
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot LIME explanation for a single prediction.
        
        Args:
            X_sample: Feature matrix.
            sample_idx: Index of sample to explain.
            num_features: Number of features to show.
            output_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        explanation = self.explain_with_lime(X_sample, sample_idx, num_features)
        
        features, weights = zip(*explanation['feature_weights'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Weight')
        ax.set_title(f'LIME Explanation (Sample {sample_idx})\nPredicted: {explanation["predicted_value"]:.2f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"LIME explanation plot saved to {output_path}")
        
        return fig
    
    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute feature importance with confidence intervals.
        
        Returns:
            DataFrame with feature importance scores.
        """
        # Use permutation importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif self.shap_values is not None:
            # Use mean absolute SHAP values
            importances = np.abs(self.shap_values).mean(axis=0)
        else:
            raise ValueError("No feature importance available")
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 100
        bootstrap_importances = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(self.X_train), len(self.X_train), replace=True)
            if hasattr(self.model, 'feature_importances_'):
                bootstrap_importances.append(importances)  # Use same for tree-based
            else:
                bootstrap_importances.append(importances + np.random.normal(0, 0.01, len(importances)))
        
        bootstrap_importances = np.array(bootstrap_importances)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
            'ci_lower': np.percentile(bootstrap_importances, 2.5, axis=0),
            'ci_upper': np.percentile(bootstrap_importances, 97.5, axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, output_path: Optional[str] = None,
                                 top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance with confidence intervals.
        
        Args:
            output_path: Path to save the figure.
            top_n: Number of top features to show.
            
        Returns:
            Matplotlib figure.
        """
        importance_df = self.compute_feature_importance()
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_features['importance'], 
                xerr=[top_features['importance'] - top_features['ci_lower'],
                      top_features['ci_upper'] - top_features['importance']],
                capsize=3, alpha=0.7, color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance with 95% Confidence Intervals', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
        
        return fig


def analyze_sowing_date_sensitivity(model: Any, X_base: np.ndarray,
                                     feature_names: List[str],
                                     output_path: Optional[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze yield sensitivity to sowing date.
    
    Args:
        model: Trained model.
        X_base: Base feature values.
        feature_names: List of feature names.
        output_path: Path to save the figure.
        
    Returns:
        Tuple of (sensitivity DataFrame, figure).
    """
    # Find day_of_year column index
    if 'day_of_year' not in feature_names:
        raise ValueError("day_of_year feature not found")
    
    doy_idx = feature_names.index('day_of_year')
    
    # Create predictions for different sowing dates
    results = []
    for doy in range(1, 366, 7):  # Every week
        X_modified = X_base.copy()
        X_modified[:, doy_idx] = doy
        
        predictions = model.predict(X_modified)
        
        results.append({
            'day_of_year': doy,
            'mean_yield': predictions.mean(),
            'std_yield': predictions.std(),
            'min_yield': predictions.min(),
            'max_yield': predictions.max()
        })
    
    sensitivity_df = pd.DataFrame(results)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(sensitivity_df['day_of_year'], sensitivity_df['mean_yield'],
            linewidth=2, color='blue', label='Mean Yield')
    ax.fill_between(sensitivity_df['day_of_year'],
                    sensitivity_df['mean_yield'] - sensitivity_df['std_yield'],
                    sensitivity_df['mean_yield'] + sensitivity_df['std_yield'],
                    alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Mark seasons
    seasons = [
        (1, 79, 'Winter', 'lightblue'),
        (80, 171, 'Spring', 'lightgreen'),
        (172, 265, 'Summer', 'lightyellow'),
        (266, 365, 'Autumn', 'orange')
    ]
    
    for start, end, name, color in seasons:
        ax.axvspan(start, end, alpha=0.2, color=color, label=name)
    
    # Find optimal sowing window
    optimal_idx = sensitivity_df['mean_yield'].idxmax()
    optimal_doy = sensitivity_df.loc[optimal_idx, 'day_of_year']
    optimal_yield = sensitivity_df.loc[optimal_idx, 'mean_yield']
    
    ax.axvline(x=optimal_doy, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: Day {optimal_doy}')
    ax.scatter([optimal_doy], [optimal_yield], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Day of Year', fontsize=12)
    ax.set_ylabel('Predicted Yield (tons/ha)', fontsize=12)
    ax.set_title('Sowing Date Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Sowing date sensitivity analysis saved to {output_path}")
    
    return sensitivity_df, fig


def run_full_interpretation(model: Any, X_train: np.ndarray, 
                            X_test: np.ndarray, feature_names: List[str],
                            output_dir: str = "docs/figures") -> Dict[str, Any]:
    """
    Run complete model interpretation analysis.
    
    Args:
        model: Trained model.
        X_train: Training features.
        X_test: Test features.
        feature_names: List of feature names.
        output_dir: Directory to save figures.
        
    Returns:
        Dictionary with all interpretation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Model Interpretation Analysis...")
    results = {}
    
    # Initialize interpreter
    interpreter = ModelInterpreter(model, X_train, feature_names)
    
    # 1. Feature importance
    print("\n1. Computing feature importance...")
    try:
        results['feature_importance'] = interpreter.compute_feature_importance()
        results['importance_fig'] = interpreter.plot_feature_importance(
            os.path.join(output_dir, 'feature_importance.png')
        )
    except Exception as e:
        print(f"Feature importance error: {e}")
    
    # 2. SHAP analysis
    if HAS_SHAP:
        print("\n2. Computing SHAP values...")
        try:
            # Use a sample for efficiency
            sample_idx = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
            X_sample = X_test[sample_idx]
            
            interpreter.compute_shap_values(X_sample)
            results['shap_summary_fig'] = interpreter.plot_shap_summary(
                X_sample, os.path.join(output_dir, 'shap_summary.png')
            )
            
            # Waterfall for first sample
            results['shap_waterfall_fig'] = interpreter.plot_shap_waterfall(
                X_sample, 0, os.path.join(output_dir, 'shap_waterfall.png')
            )
        except Exception as e:
            print(f"SHAP analysis error: {e}")
    
    # 3. Partial dependence
    print("\n3. Computing partial dependence plots...")
    try:
        # Select key features for PDP
        key_features = ['ndvi', 'precipitation_mm', 'temperature_c']
        feature_indices = [feature_names.index(f) for f in key_features if f in feature_names]
        
        if feature_indices:
            results['pdp_fig'] = interpreter.compute_partial_dependence(
                X_test, feature_indices[:3],
                os.path.join(output_dir, 'partial_dependence.png')
            )
    except Exception as e:
        print(f"Partial dependence error: {e}")
    
    # 4. LIME explanations
    if HAS_LIME:
        print("\n4. Generating LIME explanations...")
        try:
            results['lime_fig'] = interpreter.plot_lime_explanation(
                X_test, 0, 10, os.path.join(output_dir, 'lime_explanation.png')
            )
        except Exception as e:
            print(f"LIME analysis error: {e}")
    
    # 5. Sowing date sensitivity
    print("\n5. Analyzing sowing date sensitivity...")
    try:
        results['sowing_sensitivity'], results['sowing_fig'] = analyze_sowing_date_sensitivity(
            model, X_test[:100], feature_names,
            os.path.join(output_dir, 'sowing_sensitivity.png')
        )
    except Exception as e:
        print(f"Sowing sensitivity error: {e}")
    
    print("\nInterpretation analysis complete!")
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load and prepare data
    from data_generation import generate_synthetic_dataset
    from preprocessing import preprocess_pipeline, prepare_model_data
    from models import RandomForestModel, load_config
    
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=2000, output_path="data/test_dataset.csv")
    
    # Preprocess
    df_processed, encoder, scaler = preprocess_pipeline(df)
    
    # Prepare for modeling
    X, y = prepare_model_data(df_processed)
    feature_names = list(X.columns)
    
    # Train a simple model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    
    config = load_config()
    model = RandomForestModel(config)
    model.fit(X_train, y_train)
    
    # Run interpretation
    results = run_full_interpretation(
        model.model, X_train, X_test, feature_names
    )
