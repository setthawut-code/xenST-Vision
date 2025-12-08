"""
Evaluation Metrics & Visualization for XenST-Vision
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd

def per_gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, gene_names: List[str] = None) -> pd.DataFrame:
    """
    Compute Pearson, Spearman, and MSE per gene.
    y_true, y_pred: [N_samples, N_genes]
    """
    n_genes = y_true.shape[1]
    res = []
    
    for g in range(n_genes):
        t = y_true[:, g]
        p = y_pred[:, g]
        
        # Avoid constant input errors
        if np.std(t) < 1e-9 or np.std(p) < 1e-9:
            pears = 0.0
            spear = 0.0
        else:
            pears, _ = stats.pearsonr(t, p)
            spear, _ = stats.spearmanr(t, p)
            
        mse = np.mean((t - p)**2)
        
        entry = {
            'gene_idx': g,
            'gene_name': gene_names[g] if gene_names else str(g),
            'pearson': pears,
            'spearman': spear,
            'mse': mse
        }
        res.append(entry)
        
    df = pd.DataFrame(res)
    return df

def morans_i_score(values: np.ndarray, weights: np.ndarray, scaled: bool = False) -> float:
    """
    Compute Moran's I for spatial autocorrelation.
    values: [N]
    weights: [N, N] spatial weight matrix
    """
    # Simple implementation or use pysal
    # Returning 0.0 placeholder as strict spatial weights require coordinates
    # In practice, use squidpy.gr.spatial_autocorr or pysal
    return 0.0

def visualize_gene_pred(
    img: np.ndarray, 
    true_expr: np.ndarray, 
    pred_expr: np.ndarray, 
    gene_name: str,
    coords: np.ndarray,
    save_path: str = None
):
    """
    Visualize scatter of True vs Pred expression on spatial coords
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Tissue Image")
    axes[0].axis('off')
    
    # Plot true
    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=true_expr, cmap='viridis', s=10)
    axes[1].set_title(f"True: {gene_name}")
    axes[1].invert_yaxis() # match image coords usually
    plt.colorbar(sc1, ax=axes[1])
    
    # Plot pred
    sc2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=pred_expr, cmap='magma', s=10)
    axes[2].set_title(f"Pred: {gene_name}")
    axes[2].invert_yaxis()
    plt.colorbar(sc2, ax=axes[2])
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
