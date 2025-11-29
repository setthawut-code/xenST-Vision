"""
Spatial Transcriptomics Prediction Model
Baseline: ResNet50 + Regression Head
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class SpatialTranscriptomicsPredictor(nn.Module):
    """
    Baseline model for predicting spatial gene expression from H&E images
    
    Architecture:
        - ResNet50 backbone (pretrained on ImageNet)
        - Custom regression head for gene expression prediction
    """
    
    def __init__(self, num_genes: int = 50, pretrained: bool = True, dropout: float = 0.3):
        """
        Args:
            num_genes: Number of genes to predict
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in regression head
        """
        super().__init__()
        
        self.num_genes = num_genes
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50 outputs 2048 features
        encoder_dim = 2048
        
        # Regression head for gene expression prediction
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, num_genes)
        )
        
        # Initialize weights
        self._init_regression_head()
    
    def _init_regression_head(self):
        """Initialize regression head weights"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            Gene expression predictions (batch_size, num_genes)
        """
        # Extract features
        features = self.encoder(x)  # (batch_size, 2048, 1, 1)
        
        # Predict gene expression
        predictions = self.regression_head(features)  # (batch_size, num_genes)
        
        return predictions
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without prediction
        Useful for visualization/analysis
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            Features (batch_size, 2048)
        """
        with torch.no_grad():
            features = self.encoder(x)
            features = features.flatten(1)
        return features


def create_model(config: Dict) -> SpatialTranscriptomicsPredictor:
    """
    Factory function to create model from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Model instance
    """
    model = SpatialTranscriptomicsPredictor(
        num_genes=config['model']['num_genes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SpatialTranscriptomicsPredictor(num_genes=50)
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 50), "Output shape mismatch!"
    print("✓ Model test passed!")
