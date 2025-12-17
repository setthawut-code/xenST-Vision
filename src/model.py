"""
Spatial Transcriptomics Prediction Model (Wave A)
Integrates Multi-Scale Transformer Encoder (timm) and optional Spatial GNN (PyG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model as timm_create_model
from einops import rearrange

# Optional: simple GNN head using torch_geometric
try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder: extract embeddings from multiple patch sizes.
    Uses a backbone from `timm` (e.g., Swin Transformer).
    """
    def __init__(self, backbone_name='swin_base_patch4_window7_224', pretrained=True, out_dim=768):
        super().__init__()
        # create_model with global_pool='token' or similar depending on model type; 
        # swin transformers usually output (B, H*W, C) or (B, C) with pool. 
        # Here we use num_classes=0 to get features.
        self.backbone = timm_create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.out_dim = out_dim
        
        # We need to determine the feature dimension dynamically or default to 1024/768/etc
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            # swin output might be (B, H, W, C) or (B, HW, C). Let's standardize to (B, C) via pooling if not already
            self.feat_dim = feats.shape[-1]
            
        self.proj = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.backbone(x) 
        # feats could be (B, H, W, C). Global average pool
        if feats.ndim == 4:
            feats = feats.mean(dim=(1, 2))
        elif feats.ndim == 3:
             # (B, L, C) -> mean over L
            feats = feats.mean(dim=1)
            
        return self.proj(feats)

class SpatialGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError('torch_geometric not available but use_gnn=True')
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, in_dim], edge_index: [2, E]
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

class XenSTModel(nn.Module):
    def __init__(self, encoder_cfg, embed_dim=768, num_genes=50, use_gnn=False):
        super().__init__()
        self.encoder = MultiScaleEncoder(**encoder_cfg)
        self.use_gnn = use_gnn
        self.embed_dim = embed_dim
        
        gnn_out_dim = embed_dim
        if use_gnn:
            self.gnn = SpatialGNN(in_dim=embed_dim, out_dim=embed_dim)
            gnn_out_dim = embed_dim # We keep dim same for simplicity or concatenate
        
        # Regression head for gene expression (per-spot)
        self.reg_head = nn.Sequential(
            nn.Linear(gnn_out_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_genes)
        )
        
        # Optional: tissue classifier head for multi-task
        self.cls_head = nn.Sequential(
            nn.Linear(gnn_out_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10) # Assuming 10 tissue classes
        )

    def forward(self, x, graph=None, return_feats=False):
        # x: [B, C, H, W]
        feats = self.encoder(x)  # [B, D]
        
        if self.use_gnn and graph is not None:
            # graph object from PyG DataLoader batch or explicit edge_index
            # standard usage: feats = self.gnn(feats, graph.edge_index)
            # Here we assume 'graph' is edge_index for simplicity or a Batch object
            edge_index = graph.edge_index if hasattr(graph, 'edge_index') else graph
            feats = self.gnn(feats, edge_index)
            
        preds = self.reg_head(feats)
        cls_logits = self.cls_head(feats)
        
        if return_feats:
            return preds, cls_logits, feats
        return preds, cls_logits

def create_model(config):
    """Factory function to create XenSTModel from config"""
    model_cfg = config['model']
    
    # Handle config key differences
    embed_dim = model_cfg.get('out_dim', model_cfg.get('embed_dim', 512))
    
    encoder_cfg = {
        'backbone_name': model_cfg.get('backbone_name', 'resnet18'),
        'pretrained': model_cfg.get('pretrained', True),
        'out_dim': embed_dim
    }
    
    model = XenSTModel(
        encoder_cfg=encoder_cfg,
        embed_dim=embed_dim,
        num_genes=model_cfg.get('num_genes', 50),
        use_gnn=model_cfg.get('use_gnn', False)
    )
    return model

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model instantiation
    cfg = {'backbone_name': 'resnet18', 'pretrained': False, 'out_dim': 512}
    model = XenSTModel(encoder_cfg=cfg, embed_dim=512, num_genes=100)
    print("Model created successfully")
    
    x = torch.randn(2, 3, 224, 224)
    p, c = model(x)
    print(f"Output shape: {p.shape}, Cls shape: {c.shape}")
