"""
Advanced model architectures with flexible backbone selection
Supports ResNet, ConvNeXt, Swin via timm
"""

import torch
import torch.nn as nn
import timm


class AdvancedSpatialPredictor(nn.Module):
    """
    Advanced spatial transcriptomics predictor with flexible backbone
    
    - Supports multiple architectures via timm
    - Dual-head prediction: mu (mean) and theta (dispersion)
    - Learnable per-gene dispersion parameters
    """
    
    def __init__(
        self,
        backbone_name='convnext_base',
        num_genes=50,
        pretrained=True,
        hidden_dim=512,
        dropout=0.3
    ):
        """
        Args:
            backbone_name: timm model name (e.g., 'resnet50', 'convnext_base', 'swin_base_patch4_window7_224')
            num_genes: Number of genes to predict
            pretrained: Use ImageNet pretrained weights
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_genes = num_genes
        
        # Create backbone encoder
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Get feature dimension
        self.feat_dim = self.encoder.num_features
        
        # Prediction head for mu (mean expression)
        self.fc_mu = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
            nn.Softplus()  # Ensures positive mu
        )
        
        # Learnable per-gene dispersion parameter (theta)
        # Initialize to 1.0 (moderate dispersion)
        self.theta_param = nn.Parameter(torch.ones(num_genes) * 1.0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            mu: Predicted mean expression (B, G) - positive
            theta: Dispersion parameters (B, G) - positive
        """
        # Extract features
        features = self.encoder(x)  # (B, feat_dim)
        
        # Predict mu (mean)
        mu = self.fc_mu(features)  # (B, G), positive via Softplus
        
        # Expand theta to batch size
        theta = torch.relu(self.theta_param) + 1e-6  # Ensure > 0
        theta = theta.unsqueeze(0).expand(mu.shape[0], -1)  # (B, G)
        
        return mu, theta
    
    def predict_mean(self, x):
        """Predict only mean (for inference)"""
        mu, _ = self.forward(x)
        return mu


def create_advanced_model(config):
    """
    Factory function to create advanced model from config
    
    Args:
        config: Configuration dictionary with:
            model:
                backbone: 'resnet50' | 'convnext_base' | 'swin_base_patch4_window7_224'
                num_genes: 50
                pretrained: true
                dropout: 0.3
    
    Returns:
        Model instance
    """
    model_config = config['model']
    
    model = AdvancedSpatialPredictor(
        backbone_name=model_config.get('backbone', 'convnext_base'),
        num_genes=model_config['num_genes'],
        pretrained=model_config.get('pretrained', True),
        dropout=model_config.get('dropout', 0.3)
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Available backbones with descriptions
AVAILABLE_BACKBONES = {
    # ResNet family (baseline)
    'resnet50': {
        'params': '~25M',
        'speed': 'Fast',
        'description': 'Classic CNN, good baseline'
    },
    'resnet101': {
        'params': '~44M',
        'speed': 'Medium',
        'description': 'Deeper ResNet, better features'
    },
    
    # ConvNeXt family (recommended)
    'convnext_tiny': {
        'params': '~28M',
        'speed': 'Fast',
        'description': 'Efficient modern CNN'
    },
    'convnext_small': {
        'params': '~50M',
        'speed': 'Medium',
        'description': 'Good balance'
    },
    'convnext_base': {
        'params': '~88M',
        'speed': 'Medium',
        'description': 'Strong performance (recommended)'
    },
    
    # Swin Transformer (heavy but powerful)
    'swin_tiny_patch4_window7_224': {
        'params': '~28M',
        'speed': 'Slow',
        'description': 'Transformer, good for complex patterns'
    },
    'swin_small_patch4_window7_224': {
        'params': '~49M',
        'speed': 'Slow',
        'description': 'Stronger transformer'
    },
    'swin_base_patch4_window7_224': {
        'params': '~87M',
        'speed': 'Very Slow',
        'description': 'Best quality, expensive'
    },
    
    # EfficientNet (efficient alternative)
    'efficientnet_b3': {
        'params': '~12M',
        'speed': 'Fast',
        'description': 'Very efficient'
    },
    'efficientnet_b5': {
        'params': '~30M',
        'speed': 'Medium',
        'description': 'Good efficiency/performance'
    },
}


def print_backbone_options():
    """Print available backbone options"""
    print("\n" + "="*60)
    print("AVAILABLE BACKBONES")
    print("="*60)
    
    for name, info in AVAILABLE_BACKBONES.items():
        print(f"\n{name}")
        print(f"  Parameters: {info['params']}")
        print(f"  Speed: {info['speed']}")
        print(f"  Description: {info['description']}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Start with 'convnext_base'")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing Advanced Spatial Predictor...")
    
    # Create sample config
    config = {
        'model': {
            'backbone': 'convnext_base',
            'num_genes': 50,
            'pretrained': False,  # False for faster testing
            'dropout': 0.3
        }
    }
    
    model = create_advanced_model(config)
    print(f"✓ Model created: {config['model']['backbone']}")
    print(f"✓ Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    mu, theta = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Theta shape: {theta.shape}")
    print(f"  Mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  Theta range: [{theta.min():.4f}, {theta.max():.4f}]")
    
    print("\n✅ All tests passed!")
    
    # Show available backbones
    print_backbone_options()
