
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json
from tqdm import tqdm

class XeniumDataset(Dataset):
    """
    Dataset for Xenium spatial transcriptomics data.
    Loads abundant OME-TIFF image and transcript CSV, and generates 
    image patches with corresponding gene expression counts.
    """
    
    def __init__(
        self,
        image_path: str,
        transcripts_path: str,
        metadata_path: str = None,
        patch_size: int = 256,
        stride: int = 128,
        min_counts: int = 10,
        gene_list: Optional[List[str]] = None,
        preload_image: bool = False,
        force_rebuild_index: bool = False
    ):
        """
        Args:
            image_path: Path to morphology OME-TIFF file
            transcripts_path: Path to transcripts CSV file
            metadata_path: Path to metadata JSON (for pixel size)
            patch_size: Size of image patches (pixels)
            stride: Stride for patch generation (pixels)
            min_counts: Minimum number of transcripts to include a patch
            gene_list: Optional list of specific genes to track
            preload_image: Whether to load full image into memory (WARNING: High RAM usage for Xenium data)
        """
        self.image_path = Path(image_path)
        self.transcripts_path = Path(transcripts_path)
        self.patch_size = patch_size
        self.stride = stride
        self.min_counts = min_counts
        self.preload_image = preload_image
        self.config_force_rebuild = force_rebuild_index
        
        # Load metadata for coordinate conversion
        self.pixel_size = 0.2125 # Default for Xenium
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                self.pixel_size = meta.get('micron_per_pixel', self.pixel_size)
        
        print(f"Loading dataset from {self.image_path.parent}...")
        print(f"Pixel scale: {self.pixel_size} microns/pixel")
        
        # Initialize image handling
        self.memmap_mode = False
        self.image = None
        
        if self.preload_image:
            print("⚠️ WARNING: Loading full image into memory. This may consume 50GB+ RAM.")
            try:
                self.image = tifffile.imread(self.image_path)
                self._process_loaded_image()
            except Exception as e:
                print(f"❌ Failed to preload image: {e}. Falling back to memory mapping.")
                self.preload_image = False
                self._read_image_metadata()
        else:
            self._read_image_metadata()
        
        # Load transcripts
        print("Loading transcripts...")
        df = pd.read_csv(self.transcripts_path)
        
        # Filter genes if list provided
        if gene_list:
            df = df[df['gene'].isin(gene_list)]
            self.gene_names = sorted(gene_list)
        else:
            self.gene_names = sorted(df['gene'].unique())
            
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_names)}
        self.num_genes = len(self.gene_names)
        
        # Convert coords to pixels
        df['px'] = df['x'] / self.pixel_size
        df['py'] = df['y'] / self.pixel_size
        
        # Create patch index
        print("Generating patch index...")
        self.patches = []
        
        # Check for cached patches
        cache_name = f"patches_{self.image_path.stem}_{patch_size}_{stride}_{min_counts}.npy"
        cache_path = self.image_path.parent / cache_name
        
        if cache_path.exists() and not self.config_force_rebuild:
            print(f"Loading cached patches from {cache_path}...")
            try:
                self.patches = np.load(cache_path, allow_pickle=True).tolist()
                print(f"Loaded {len(self.patches)} patches.")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}")

        # Simple grid generation
        x_steps = (self.img_width - patch_size) // stride + 1
        y_steps = (self.img_height - patch_size) // stride + 1
        
        valid_patches = 0
        
        # Convert df to numpy for speed
        coords = df[['px', 'py']].values
        gene_indices = df['gene'].map(self.gene_to_idx).values
        
        # Loop over patches
        for y_idx in tqdm(range(y_steps), desc="Indexing patches"):
            y_start = y_idx * stride
            y_end = y_start + patch_size
            
            # Pre-filter by Y band
            y_mask = (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
            if not np.any(y_mask):
                continue
                
            band_coords = coords[y_mask]
            band_genes = gene_indices[y_mask]
            
            for x_idx in range(x_steps):
                x_start = x_idx * stride
                x_end = x_start + patch_size
                
                # Check bounds
                if x_end > self.img_width or y_end > self.img_height:
                    continue
                
                # Filter x
                mask = (band_coords[:, 0] >= x_start) & (band_coords[:, 0] < x_end)
                
                if np.sum(mask) >= self.min_counts:
                    patch_genes = band_genes[mask]
                    counts = np.bincount(patch_genes, minlength=self.num_genes)
                    
                    self.patches.append({
                        'x': x_start,
                        'y': y_start,
                        'counts': counts.astype(np.float32)
                    })
                    valid_patches += 1
        
        print(f"Created {len(self.patches)} valid patches from {x_steps * y_steps} candidates")
        
        # Save cache
        try:
            print(f"Saving patch index to {cache_path}...")
            np.save(cache_path, np.array(self.patches))
        except Exception as e:
            print(f"Warning: Could not save patch cache: {e}")

    def _process_loaded_image(self):
        """Process preloaded numpy array"""
        if len(self.image.shape) == 3 and self.image.shape[0] < 10:
            # Handle (C, H, W) -> (H, W, C)
            self.image = np.moveaxis(self.image, 0, -1)
        # Handle grayscale (H, W) -> (H, W, 1)
        if len(self.image.shape) == 2:
            self.image = self.image[:, :, np.newaxis]
            
        self.img_height, self.img_width = self.image.shape[:2]

    def _read_image_metadata(self):
        """Read image dimensions without loading the whole file"""
        try:
            with tifffile.TiffFile(self.image_path) as tif:
                shape = tif.pages[0].shape
                if len(shape) == 2: # HW
                    self.img_height, self.img_width = shape
                elif len(shape) == 3 and shape[0] < 10: # CHW
                    self.img_height, self.img_width = shape[1], shape[2]
                else: # HWC or large 3D
                    self.img_height, self.img_width = shape[0], shape[1]
            print(f"[OK] Image metadata: {self.img_height}x{self.img_width}")
        except Exception as e:
            print(f"Error reading metadata: {e}")
            # Fallback
            self.img_height, self.img_width = 10000, 10000

    def _ensure_image_loaded(self):
        """Lazy loader for worker processes"""
        if self.image is not None:
            return

        # Attempt to use Zarr for efficient tiled access
        try:
            import zarr
            # Open as Zarr store (works for compressed/tiled OME-TIFF)
            store = tifffile.imread(self.image_path, aszarr=True)
            z = zarr.open(store, mode='r')
            
            # Handle OME-TIFF structure (Group vs Array)
            if isinstance(z, zarr.Group):
                # Usually the highest resolution is at key '0'
                if '0' in z:
                    self.image = z['0']
                else:
                    # Fallback to first key
                    key = list(z.keys())[0]
                    self.image = z[key]
            else:
                self.image = z
                
            self.memmap_mode = False # It's zarr backed
            
            # Verify dimensions
            self.img_height, self.img_width = self.image.shape[:2]
            
        except ImportError:
            print("⚠️ Zarr not installed. Falling back to slow reading.")
            self.image = None
        except Exception as e:
            print(f"Zarr open failed ({e}). Falling back to slow reading.")
            self.image = None

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        x, y = patch_info['x'], patch_info['y']
        
        # Ensure image is ready (unique per worker)
        if not self.preload_image:
            self._ensure_image_loaded()

        # Get image patch
        img_patch = None
        
        if self.image is not None:
            # Zarr or Memmap access
            try:
                # Zarr supports slicing efficiently even with compression
                img_patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
            except Exception as e:
                # Fallback if access fails
                pass
        
        if img_patch is None:
            # Last resort: one-off read (very slow)
            try:
                img_patch = tifffile.imread(self.image_path)[y:y+self.patch_size, x:x+self.patch_size]
            except Exception as e:
                 raise RuntimeError(f"Failed to load patch at {x},{y}: {e}")
            
        # Normalize and formatting logic
        
        # Ensure it's a numpy array
        img_patch = np.array(img_patch)
        
        # Handle normalization
        if img_patch.dtype == np.uint16:
            img_patch = img_patch.astype(np.float32) / 65535.0
        elif img_patch.dtype == np.uint8:
            img_patch = img_patch.astype(np.float32) / 255.0
            
        # Ensure dimensions
        if len(img_patch.shape) == 2:
            img_patch = img_patch[:, :, np.newaxis]
        
        # Ensure (C, H, W)
        if img_patch.shape[-1] < 10: # HWC -> CHW
            img_patch = np.moveaxis(img_patch, -1, 0)
            
        # Replicate channels if grayscale
        if img_patch.shape[0] == 1:
            img_patch = np.repeat(img_patch, 3, axis=0)
            
        # Get target
        target = patch_info['counts']
        
        return torch.from_numpy(img_patch), torch.from_numpy(target)

if __name__ == "__main__":
    # Simple test
    dataset = XeniumDataset(
        image_path="Xenium dataset file/morphology_mip.ome.tif",
        transcripts_path="Xenium dataset file/transcripts_full.csv",
        metadata_path="Xenium dataset file/metadata.json",
        patch_size=256,
        stride=256,
        preload_image=False
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}, Target shape: {target.shape}")
