"""
Exploratory Data Analysis Module for Crop Yield Prediction System.

Generates visualizations and statistical analysis of the crop yield dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import os
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def set_plot_style() -> None:
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12


def plot_correlation_heatmap(df: pd.DataFrame, 
                             output_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Generate correlation heatmap for numeric features.
    
    Args:
        df: Input DataFrame.
        output_path: Path to save the figure.
        figsize: Figure size tuple.
        
    Returns:
        Matplotlib figure object.
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Correlation heatmap saved to {output_path}")
    
    return fig


def plot_ndvi_time_series(df: pd.DataFrame, 
                          output_path: Optional[str] = None) -> plt.Figure:
    """
    Create NDVI time series plots across growth stages for different crops.
    
    Args:
        df: Input DataFrame with growth_stage and crop_type columns.
        output_path: Path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    crops = df['crop_type'].unique()[:4]
    growth_order = ['germination', 'vegetative', 'reproductive', 'maturity']
    
    for idx, crop in enumerate(crops):
        ax = axes[idx]
        crop_data = df[df['crop_type'] == crop]
        
        # Group by growth stage and calculate mean NDVI
        if 'growth_stage' in df.columns:
            stage_means = []
            stage_stds = []
            for stage in growth_order:
                stage_data = crop_data[crop_data['growth_stage'] == stage]['ndvi']
                stage_means.append(stage_data.mean())
                stage_stds.append(stage_data.std())
            
            x = range(len(growth_order))
            ax.errorbar(x, stage_means, yerr=stage_stds, marker='o', 
                       capsize=5, linewidth=2, markersize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(growth_order, rotation=45)
        else:
            ax.hist(crop_data['ndvi'], bins=30, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Growth Stage')
        ax.set_ylabel('NDVI')
        ax.set_title(f'{crop.capitalize()} NDVI by Growth Stage')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('NDVI Time Series Across Growth Stages', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"NDVI time series plot saved to {output_path}")
    
    return fig


def plot_yield_distribution(df: pd.DataFrame, 
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot yield distribution by crop type and region.
    
    Args:
        df: Input DataFrame.
        output_path: Path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Yield by crop type
    ax1 = axes[0]
    crops = df['crop_type'].unique()
    crop_yields = [df[df['crop_type'] == crop]['actual_yield_tons_per_ha'] for crop in crops]
    
    bp = ax1.boxplot(crop_yields, labels=crops, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(crops)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xlabel('Crop Type')
    ax1.set_ylabel('Yield (tons/ha)')
    ax1.set_title('Yield Distribution by Crop Type')
    ax1.grid(True, alpha=0.3)
    
    # Yield by region
    ax2 = axes[1]
    regions = df['region'].unique()
    region_yields = [df[df['region'] == region]['actual_yield_tons_per_ha'] for region in regions]
    
    bp2 = ax2.boxplot(region_yields, labels=regions, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Yield (tons/ha)')
    ax2.set_title('Yield Distribution by Region')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Crop Yield Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Yield distribution plot saved to {output_path}")
    
    return fig


def plot_precipitation_vs_yield(df: pd.DataFrame, 
                                 output_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter plot of precipitation vs yield with temperature color coding.
    
    Args:
        df: Input DataFrame.
        output_path: Path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df['precipitation_mm'], 
                         df['actual_yield_tons_per_ha'],
                         c=df['temperature_c'], 
                         cmap='RdYlBu_r',
                         alpha=0.6, 
                         s=30,
                         edgecolors='gray',
                         linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add optimal precipitation range
    ax.axvspan(800, 1200, alpha=0.2, color='green', label='Optimal Range (800-1200mm)')
    
    ax.set_xlabel('Precipitation (mm)', fontsize=12)
    ax.set_ylabel('Yield (tons/ha)', fontsize=12)
    ax.set_title('Precipitation vs Yield with Temperature', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Precipitation vs yield plot saved to {output_path}")
    
    return fig


def plot_feature_pairplot(df: pd.DataFrame, 
                          output_path: Optional[str] = None) -> plt.Figure:
    """
    Create pairplot for key features.
    
    Args:
        df: Input DataFrame.
        output_path: Path to save the figure.
        
    Returns:
        Matplotlib figure object.
    """
    key_features = ['ndvi', 'precipitation_mm', 'temperature_c', 
                    'soil_organic_carbon_pct', 'actual_yield_tons_per_ha']
    
    # Filter to existing columns
    available_features = [f for f in key_features if f in df.columns]
    
    g = sns.pairplot(df[available_features], 
                     diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 20},
                     diag_kws={'linewidth': 2})
    
    g.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16, fontweight='bold')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        g.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Pairplot saved to {output_path}")
    
    return g.fig


def calculate_mutual_information(X: pd.DataFrame, 
                                  y: pd.Series,
                                  output_path: Optional[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Calculate and plot mutual information scores for features.
    
    Args:
        X: Feature DataFrame.
        y: Target series.
        output_path: Path to save the figure.
        
    Returns:
        Tuple of (MI scores DataFrame, figure).
    """
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Create DataFrame
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mi_df)))
    bars = ax.barh(mi_df['Feature'], mi_df['MI_Score'], color=colors)
    
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance (Mutual Information)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, score in zip(bars, mi_df['MI_Score']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Mutual information plot saved to {output_path}")
    
    return mi_df, fig


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with summary statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = df[numeric_cols].describe().T
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    
    return stats


def run_full_eda(df: pd.DataFrame, 
                 output_dir: str = "docs/figures",
                 show_plots: bool = False) -> dict:
    """
    Run complete exploratory data analysis.
    
    Args:
        df: Input DataFrame.
        output_dir: Directory to save figures.
        show_plots: Whether to display plots.
        
    Returns:
        Dictionary with analysis results.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Exploratory Data Analysis...")
    results = {}
    
    # Summary statistics
    print("\n1. Computing summary statistics...")
    results['summary_stats'] = generate_summary_statistics(df)
    print(results['summary_stats'])
    
    # Correlation heatmap
    print("\n2. Generating correlation heatmap...")
    results['correlation_fig'] = plot_correlation_heatmap(
        df, os.path.join(output_dir, 'correlation_heatmap.png')
    )
    
    # NDVI time series
    print("\n3. Generating NDVI time series plots...")
    results['ndvi_fig'] = plot_ndvi_time_series(
        df, os.path.join(output_dir, 'ndvi_time_series.png')
    )
    
    # Yield distribution
    print("\n4. Generating yield distribution plots...")
    results['yield_dist_fig'] = plot_yield_distribution(
        df, os.path.join(output_dir, 'yield_distribution.png')
    )
    
    # Precipitation vs yield
    print("\n5. Generating precipitation vs yield scatter...")
    results['precip_yield_fig'] = plot_precipitation_vs_yield(
        df, os.path.join(output_dir, 'precipitation_vs_yield.png')
    )
    
    # Pairplot
    print("\n6. Generating feature pairplot...")
    results['pairplot_fig'] = plot_feature_pairplot(
        df, os.path.join(output_dir, 'feature_pairplot.png')
    )
    
    # Mutual information (only for numeric features)
    print("\n7. Computing mutual information scores...")
    numeric_features = ['ndvi', 'precipitation_mm', 'temperature_c', 
                        'soil_organic_carbon_pct']
    available_features = [f for f in numeric_features if f in df.columns]
    
    if len(available_features) > 0 and 'actual_yield_tons_per_ha' in df.columns:
        X_numeric = df[available_features].dropna()
        y_numeric = df.loc[X_numeric.index, 'actual_yield_tons_per_ha']
        
        results['mi_scores'], results['mi_fig'] = calculate_mutual_information(
            X_numeric, y_numeric, os.path.join(output_dir, 'mutual_information.png')
        )
    
    print("\nEDA complete! Figures saved to", output_dir)
    
    if show_plots:
        plt.show()
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load sample data
    from data_generation import generate_synthetic_dataset
    from preprocessing import preprocess_pipeline
    
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=1000, output_path="data/test_dataset.csv")
    
    # Preprocess for additional features
    df_processed, _, _ = preprocess_pipeline(df)
    
    # Run EDA
    results = run_full_eda(df_processed)
