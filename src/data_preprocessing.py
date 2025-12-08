"""
Data preprocessing for Xenium spatial transcriptomics data
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm


class XeniumDataProcessor:
    """Process Xenium spatial transcriptomics data"""
    
    def __init__(self, data_path: str, config: Dict):
        """
        Args:
            data_path: Path to Xenium output directory
            config: Configuration dictionary
        """
        self.data_path = Path(data_path)
        self.config = config
        self.adata = None
        self.gene_names = None
        self.scaler = None
    
    def load_xenium_data(self) -> ad.AnnData:
        """
        Load Xenium data using Scanpy
        
        Returns:
            AnnData object with spatial information
        """
        print("Loading Xenium data...")
        
        # Load cell feature matrix
        matrix_path = self.data_path / "cell_feature_matrix.h5"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Cell feature matrix not found at {matrix_path}")
        
        # Read with scanpy
        adata = sc.read_10x_h5(matrix_path)
        
        # Load cell positions
        cells_path = self.data_path / "cells.csv.gz"
        if cells_path.exists():
            cells_df = pd.read_csv(cells_path)
            # Add spatial coordinates
            adata.obs['x_centroid'] = cells_df['x_centroid'].values
            adata.obs['y_centroid'] = cells_df['y_centroid'].values
            
            # Store in obsm for spatial plotting
            adata.obsm['spatial'] = cells_df[['x_centroid', 'y_centroid']].values
        
        print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
        
        self.adata = adata
        return adata
    
    def select_top_variable_genes(self, n_genes: int = 50) -> List[str]:
        """
        Select top variable genes for prediction
        
        Args:
            n_genes: Number of genes to select
        
        Returns:
            List of gene names
        """
        print(f"Selecting top {n_genes} variable genes...")
        
        # Normalize and log-transform
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_genes * 2)
        
        # Get gene variance
        gene_var = np.var(self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X, axis=0)
        top_gene_indices = np.argsort(gene_var)[-n_genes:]
        
        self.gene_names = [self.adata.var_names[i] for i in top_gene_indices]
        print(f"Selected genes: {self.gene_names[:10]}... (showing first 10)")
        
        return self.gene_names
    
    def extract_gene_expression(self, gene_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract and normalize gene expression
        
        Args:
            gene_names: Genes to extract (default: use self.gene_names)
        
        Returns:
            Normalized expression matrix (n_cells, n_genes)
        """
        if gene_names is None:
            gene_names = self.gene_names
        
        # Subset to selected genes
        adata_subset = self.adata[:, gene_names]
        
        # Get expression matrix
        expression = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
        
        # Apply scaling
        scale_method = self.config['data']['scale_method']
        if scale_method == 'standard':
            self.scaler = StandardScaler()
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scale method: {scale_method}")
        
        expression_scaled = self.scaler.fit_transform(expression)
        
        return expression_scaled
    
    def extract_image_patches(
        self,
        image_path: str,
        coordinates: np.ndarray,
        patch_size: int = 224
    ) -> List[np.ndarray]:
        """
        Extract image patches around spatial coordinates
        
        Args:
            image_path: Path to H&E image
            coordinates: Cell coordinates (n_cells, 2)
            patch_size: Size of patches to extract
        
        Returns:
            List of image patches
        """
        print(f"Extracting {len(coordinates)} image patches...")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        patches = []
        half_size = patch_size // 2
        
        for x, y in tqdm(coordinates):
            x, y = int(x), int(y)
            
            # Extract patch
            x_min = max(0, x - half_size)
            x_max = min(image.shape[1], x + half_size)
            y_min = max(0, y - half_size)
            y_max = min(image.shape[0], y + half_size)
            
            patch = image[y_min:y_max, x_min:x_max]
            
            # Resize if necessary
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
        
        return patches


class SpatialTranscriptomicsDataset(Dataset):
    """PyTorch Dataset for spatial transcriptomics"""
    
    def __init__(
        self,
        image_patches: List[np.ndarray],
        gene_expression: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_patches: List of image patches (H, W, 3)
            gene_expression: Gene expression matrix (n_samples, n_genes)
            transform: Image transformations
        """
        self.image_patches = image_patches
        self.gene_expression = gene_expression
        self.transform = transform
        
        assert len(image_patches) == len(gene_expression), "Mismatch in data length"
    
    def __len__(self) -> int:
        return len(self.image_patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image and expression
        image = self.image_patches[idx]
        expression = self.gene_expression[idx]
        
        # Convert to PIL for transforms
        image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert expression to tensor
        expression = torch.FloatTensor(expression)
        
        return image, expression


def get_transforms(config: Dict, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms
    
    Args:
        config: Configuration dictionary
        is_training: Whether for training (with augmentation)
    
    Returns:
        Composed transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
    
    if is_training and config['data']['augmentation']:
        transform_list = [
            transforms.RandomRotation(config['data']['rotation_range']),
        ]
        
        if config['data']['flip_horizontal']:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if config['data']['flip_vertical']:
            transform_list.append(transforms.RandomVerticalFlip())
        
        if config['data']['color_jitter'] > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config['data']['color_jitter'],
                    contrast=config['data']['color_jitter'],
                    saturation=config['data']['color_jitter'],
                    hue=config['data']['color_jitter'] / 2
                )
            )
        
        transform_list.extend([transforms.ToTensor(), normalize])
    else:
        transform_list = [transforms.ToTensor(), normalize]
    
    return transforms.Compose(transform_list)


def create_data_splits(
    dataset: SpatialTranscriptomicsDataset,
    config: Dict,
    random_state: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test
    
    Args:
        dataset: Full dataset
        config: Configuration dictionary
        random_state: Random seed
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    # Train/temp split
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Val/test split
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    def create_graph_dataset(self, config: Dict) -> 'list[torch_geometric.data.Data]':
        """
        Create Graph Dataset for Spatial GNN
        Returns list of PyG Data objects (one per ROI or batch of single items not recommended for large graphs)
        For this simplified version: we assume one large graph or per-tile graphs.
        Let's implement a 'per-sample' graph builder if you have multiple WSIs.
        
        Realistically for Xenium: We have spots. We can build a graph over all spots.
        """
        # Placeholder for full graph implementation
        # This usually requires re-loading all spots and building a massive graph or tiling graphs
        pass

# Add standalone functions for graph and stain norm

def stain_normalization(image: np.ndarray, target_img: np.ndarray = None) -> np.ndarray:
    """
    Macenko Stain Normalization (Simplified)
    For production use, consider using 'torchstain' or 'histomicstk'.
    This is a placeholder to remind the user to integrate a robust normalizer.
    """
    # implementation omitted for brevity, returning original image
    # Todo: Integrate real Macenko normalization
    return image

def build_spot_graph(spot_xy: np.ndarray, k: int = 6):
    """
    Build kNN graph from spot coordinates.
    Returns edge_index [2, E]
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(spot_xy)
    distances, indices = nbrs.kneighbors(spot_xy)
    
    src = []
    dst = []
    for i, neighbors in enumerate(indices):
        for n in neighbors[1:]: # skip self
             src.append(i)
             dst.append(n)
             
    edge_index = np.array([src, dst], dtype=np.long)
    return torch.tensor(edge_index, dtype=torch.long)

