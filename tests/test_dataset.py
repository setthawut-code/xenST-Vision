
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dataset import XeniumDataset
import torch
import numpy as np

def test_dataset():
    print("Testing XeniumDataset...")
    
    # Paths (relative to project root, assuming run from root)
    image_path = "Xenium dataset file/morphology_mip.ome.tif"
    transcripts_path = "Xenium dataset file/transcripts_full.csv"
    metadata_path = "Xenium dataset file/metadata.json"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    # Initialize dataset
    dataset = XeniumDataset(
        image_path=image_path,
        transcripts_path=transcripts_path,
        metadata_path=metadata_path,
        patch_size=256,
        stride=256,
        min_counts=5, # Lower threshold for testing
        preload_image=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: Dataset empty. Check coordinate mapping or paths.")
        return

    # Check first item
    img, target = dataset[0]
    
    print(f"Sample 0 image shape: {img.shape}")
    print(f"Sample 0 target shape: {target.shape}")
    print(f"Sample 0 target sum: {target.sum()}")
    
    assert img.shape == (1, 256, 256) or img.shape == (3, 256, 256), f"Unexpected image shape: {img.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert isinstance(target, torch.Tensor), "Target is not a tensor"
    assert img.dtype == torch.float32, f"Image dtype is {img.dtype}"
    assert target.dtype == torch.float32, f"Target dtype is {target.dtype}"
    
    print("\n✓ Dataset test passed!")

if __name__ == "__main__":
    test_dataset()
