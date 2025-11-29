"""
Loss functions for spatial transcriptomics prediction
Includes Negative Binomial loss for count data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeBinomialLoss(nn.Module):
    """
    Negative Binomial loss for count data (gene expression)
    
    Parameterization: mu (mean), theta (dispersion)
    More appropriate than MSE for over-dispersed count data
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, mu, theta, target):
        """
        Args:
            mu: Predicted mean (B, G) - must be positive
            theta: Dispersion parameter (B, G) - must be positive
            target: Ground truth counts (B, G) - non-negative
        
        Returns:
            Negative log-likelihood (scalar)
        """
        # Ensure numerical stability
        mu = mu.clamp(min=self.eps)
        theta = theta.clamp(min=self.eps)
        
        # Negative Binomial log-likelihood
        # log P(y|μ,θ) = log Γ(θ+y) - log Γ(θ) - log Γ(y+1) 
        #                + θ·log(θ) - θ·log(θ+μ) + y·log(μ) - y·log(θ+μ)
        
        t1 = torch.lgamma(theta + target) - torch.lgamma(theta) - torch.lgamma(target + 1.0)
        t2 = theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))
        t3 = target * (torch.log(mu + self.eps) - torch.log(theta + mu + self.eps))
        
        # Negative log-likelihood
        nll = -(t1 + t2 + t3)
        
        return nll.mean()


class MSELog1pLoss(nn.Module):
    """
    MSE loss on log1p transformed values
    Baseline/fallback option for comparison
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted values (B, G)
            target: Ground truth values (B, G)
        
        Returns:
            MSE on log1p transformed values
        """
        pred_log = torch.log1p(pred)
        target_log = torch.log1p(target)
        return self.mse(pred_log, target_log)


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Negative Binomial and MSE
    Useful for balancing biological accuracy (NB) with training stability (MSE)
    """
    
    def __init__(self, nb_weight=1.0, mse_weight=0.1, eps=1e-8):
        super().__init__()
        self.nb_loss = NegativeBinomialLoss(eps=eps)
        self.mse_loss = MSELog1pLoss()
        self.nb_weight = nb_weight
        self.mse_weight = mse_weight
    
    def forward(self, mu, theta, target):
        """
        Args:
            mu: Predicted mean (B, G)
            theta: Dispersion parameter (B, G)
            target: Ground truth counts (B, G)
        
        Returns:
            Weighted combination of NB and MSE losses
        """
        nb_loss = self.nb_loss(mu, theta, target)
        mse_loss = self.mse_loss(mu, target)
        
        total_loss = self.nb_weight * nb_loss + self.mse_weight * mse_loss
        
        return total_loss


def get_loss_function(loss_config):
    """
    Factory function to create loss function from config
    
    Args:
        loss_config: Dictionary with loss configuration
            {
                'type': 'negative_binomial' | 'mse' | 'hybrid',
                'nb_weight': 1.0,
                'mse_weight': 0.1
            }
    
    Returns:
        Loss function instance
    """
    loss_type = loss_config.get('type', 'negative_binomial')
    
    if loss_type == 'negative_binomial':
        return NegativeBinomialLoss()
    elif loss_type == 'mse':
        return MSELog1pLoss()
    elif loss_type == 'hybrid':
        nb_weight = loss_config.get('nb_weight', 1.0)
        mse_weight = loss_config.get('mse_weight', 0.1)
        return HybridLoss(nb_weight=nb_weight, mse_weight=mse_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Per-gene metrics helpers
def compute_per_gene_metrics(predictions, targets):
    """
    Compute per-gene metrics for validation
    
    Args:
        predictions: Predicted values (N, G) numpy array
        targets: Ground truth values (N, G) numpy array
    
    Returns:
        Dictionary with per-gene metrics
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
    n_genes = predictions.shape[1]
    
    pearson_scores = []
    spearman_scores = []
    rmse_scores = []
    
    for g in range(n_genes):
        pred_g = predictions[:, g]
        target_g = targets[:, g]
        
        # Pearson correlation
        try:
            r_pearson, _ = pearsonr(pred_g, target_g)
        except:
            r_pearson = np.nan
        pearson_scores.append(r_pearson)
        
        # Spearman correlation (rank-based, more robust)
        try:
            r_spearman, _ = spearmanr(pred_g, target_g)
        except:
            r_spearman = np.nan
        spearman_scores.append(r_spearman)
        
        # RMSE on log1p
        pred_log = np.log1p(pred_g)
        target_log = np.log1p(target_g)
        rmse = np.sqrt(np.mean((pred_log - target_log) ** 2))
        rmse_scores.append(rmse)
    
    return {
        'pearson_per_gene': np.array(pearson_scores),
        'pearson_median': np.nanmedian(pearson_scores),
        'pearson_mean': np.nanmean(pearson_scores),
        'spearman_per_gene': np.array(spearman_scores),
        'spearman_median': np.nanmedian(spearman_scores),
        'rmse_per_gene': np.array(rmse_scores),
        'rmse_mean': np.mean(rmse_scores)
    }
