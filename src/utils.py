"""
Utility functions for spatial transcriptomics prediction
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from scipy.stats import pearsonr


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: Dict) -> None:
    """Create necessary directories if they don't exist"""
    dirs = [
        config['paths']['data_dir'],
        config['paths']['output_dir'],
        config['paths']['checkpoints_dir'],
        config['paths']['logs_dir'],
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def calculate_pearson_correlation(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation for each gene
    
    Args:
        pred: Predictions (n_samples, n_genes)
        target: Ground truth (n_samples, n_genes)
    
    Returns:
        correlations: Array of correlations (n_genes,)
    """
    n_genes = pred.shape[1]
    correlations = np.zeros(n_genes)
    
    for i in range(n_genes):
        if np.std(pred[:, i]) > 0 and np.std(target[:, i]) > 0:
            correlations[i], _ = pearsonr(pred[:, i], target[:, i])
        else:
            correlations[i] = 0.0
    
    return correlations


def plot_gene_expression_heatmap(
    expression: np.ndarray,
    coordinates: np.ndarray,
    gene_names: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot spatial gene expression heatmap
    
    Args:
        expression: Gene expression values (n_spots, n_genes)
        coordinates: Spatial coordinates (n_spots, 2)
        gene_names: List of gene names
        output_path: Path to save figure
        figsize: Figure size
    """
    n_genes = min(len(gene_names), expression.shape[1])
    n_cols = 5
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_genes > 1 else [axes]
    
    for i in range(n_genes):
        ax = axes[i]
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=expression[:, i],
            cmap='viridis',
            s=10,
            alpha=0.7
        )
        ax.set_title(gene_names[i], fontsize=10)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax)
    
    # Hide unused subplots
    for i in range(n_genes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_interactive_heatmap(
    expression: np.ndarray,
    coordinates: np.ndarray,
    gene_name: str,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive plotly heatmap for single gene
    
    Args:
        expression: Gene expression values (n_spots,)
        coordinates: Spatial coordinates (n_spots, 2)
        gene_name: Gene name
        output_path: Path to save HTML
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Scatter(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=expression,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Expression")
        ),
        text=[f"Expression: {val:.3f}" for val in expression],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Spatial Expression: {gene_name}",
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        hovermode='closest',
        width=800,
        height=600
    )
    
    if output_path:
        fig.write_html(output_path)
    
    return fig


def plot_correlation_distribution(
    correlations: np.ndarray,
    gene_names: List[str],
    output_path: Optional[str] = None
) -> None:
    """
    Plot distribution of prediction correlations
    
    Args:
        correlations: Correlation values per gene
        gene_names: List of gene names
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(correlations, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(correlations), color='red', linestyle='--', 
                    label=f'Median: {np.median(correlations):.3f}')
    axes[0].axvline(np.mean(correlations), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(correlations):.3f}')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Gene Prediction Correlations')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Top genes bar plot
    top_n = min(20, len(correlations))
    top_indices = np.argsort(correlations)[-top_n:][::-1]
    top_corrs = correlations[top_indices]
    top_genes = [gene_names[i] for i in top_indices]
    
    axes[1].barh(range(top_n), top_corrs, alpha=0.7)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_genes, fontsize=8)
    axes[1].set_xlabel('Pearson Correlation')
    axes[1].set_title(f'Top {top_n} Predicted Genes')
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def save_predictions_csv(
    predictions: np.ndarray,
    coordinates: np.ndarray,
    gene_names: List[str],
    output_path: str
) -> None:
    """
    Save predictions to CSV file
    
    Args:
        predictions: Predicted expression (n_spots, n_genes)
        coordinates: Spatial coordinates (n_spots, 2)
        gene_names: List of gene names
        output_path: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame(predictions, columns=gene_names)
    df.insert(0, 'x', coordinates[:, 0])
    df.insert(1, 'y', coordinates[:, 1])
    
    df.to_csv(output_path, index=False)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
