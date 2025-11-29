"""
Inference script for spatial transcriptomics prediction
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from model import SpatialTranscriptomicsPredictor
from data_preprocessing import get_transforms
from utils import (
    load_config,
    plot_gene_expression_heatmap,
    plot_interactive_heatmap,
    save_predictions_csv
)


class SpatialInference:
    """Inference engine for spatial transcriptomics prediction"""
    
    def __init__(self, checkpoint_path: str, config: Dict, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = SpatialTranscriptomicsPredictor(
            num_genes=config['model']['num_genes'],
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Gene names from checkpoint
        self.gene_names = checkpoint.get('gene_names', [f"Gene_{i}" for i in range(config['model']['num_genes'])])
        
        # Transforms
        self.transform = get_transforms(config, is_training=False)
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Genes: {len(self.gene_names)}")
    
    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: int = 224,
        stride: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches from whole image
        
        Args:
            image: Input image (H, W, 3)
            patch_size: Size of patches
            stride: Stride for patch extraction (default: patch_size, no overlap)
        
        Returns:
            patches: List of image patches
            coordinates: List of (x, y) coordinates for each patch
        """
        if stride is None:
            stride = patch_size
        
        h, w = image.shape[:2]
        patches = []
        coordinates = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coordinates.append((x + patch_size//2, y + patch_size//2))
        
        return patches, coordinates
    
    def predict_patches(
        self,
        patches: List[np.ndarray],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Predict gene expression for list of patches
        
        Args:
            patches: List of image patches
            batch_size: Batch size for prediction
        
        Returns:
            predictions: Gene expression predictions (n_patches, n_genes)
        """
        all_predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(patches), batch_size), desc="Predicting"):
                batch_patches = patches[i:i+batch_size]
                
                # Convert to tensors
                batch_tensors = []
                for patch in batch_patches:
                    patch_pil = Image.fromarray(patch.astype(np.uint8))
                    patch_tensor = self.transform(patch_pil)
                    batch_tensors.append(patch_tensor)
                
                batch = torch.stack(batch_tensors).to(self.device)
                
                # Predict
                outputs = self.model(batch)
                all_predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0)
    
    def predict_image(
        self,
        image_path: str,
        output_dir: str,
        patch_size: int = 224,
        stride: Optional[int] = None
    ) -> Dict:
        """
        Run full prediction pipeline on an image
        
        Args:
            image_path: Path to H&E image
            output_dir: Directory to save results
            patch_size: Size of patches
            stride: Stride for patch extraction
        
        Returns:
            results: Dictionary with predictions and metadata
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image size: {image.shape}")
        
        # Extract patches
        print("Extracting patches...")
        patches, coordinates = self.extract_patches(image, patch_size, stride)
        print(f"Extracted {len(patches)} patches")
        
        # Predict
        predictions = self.predict_patches(patches, self.config['inference']['batch_size'])
        print(f"Predictions shape: {predictions.shape}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        print("Saving results...")
        
        # 1. CSV with all predictions
        csv_path = output_path / 'predictions.csv'
        save_predictions_csv(
            predictions,
            np.array(coordinates),
            self.gene_names,
            str(csv_path)
        )
        print(f"✓ Saved predictions to {csv_path}")
        
        # 2. Spatial heatmaps
        heatmap_path = output_path / 'spatial_heatmaps.png'
        plot_gene_expression_heatmap(
            predictions,
            np.array(coordinates),
            self.gene_names,
            str(heatmap_path)
        )
        print(f"✓ Saved heatmaps to {heatmap_path}")
        
        # 3. Interactive plots for top genes
        top_var_genes = np.argsort(np.var(predictions, axis=0))[-5:][::-1]
        for gene_idx in top_var_genes:
            gene_name = self.gene_names[gene_idx]
            interactive_path = output_path / f'interactive_{gene_name}.html'
            plot_interactive_heatmap(
                predictions[:, gene_idx],
                np.array(coordinates),
                gene_name,
                str(interactive_path)
            )
        print(f"✓ Saved interactive plots")
        
        results = {
            'predictions': predictions,
            'coordinates': coordinates,
            'gene_names': self.gene_names,
            'output_dir': str(output_path)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Spatial Transcriptomics Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to H&E image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='output/predictions',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    from utils import load_config
    config = load_config(args.config)
    
    # Create inference engine
    inference = SpatialInference(args.checkpoint, config, args.device)
    
    # Run prediction
    results = inference.predict_image(
        args.image,
        args.output,
        patch_size=config['data']['patch_size']
    )
    
    print(f"\n{'='*50}")
    print(f"✓ Inference completed successfully!")
    print(f"  Results saved to: {results['output_dir']}")
    print(f"  Total predictions: {len(results['predictions'])}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
