
import tifffile
import sys
from pathlib import Path

def inspect_tiff(path):
    print(f"Inspecting {path}...")
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            print(f"Shape: {page.shape}")
            print(f"Dtype: {page.dtype}")
            print(f"Compression: {page.compression}")
            print(f"Is Tiled: {page.is_tiled}")
            if page.is_tiled:
                print(f"Tile shape: {page.tilewidth}x{page.tilelength}")
            print(f"Is Contiguous: {page.is_contiguous}")
            print(f"Tags: {page.tags}")
    except Exception as e:
        print(f"Error inspecting: {e}")

if __name__ == "__main__":
    # Default to the path seen in previous logs if argument not provided
    default_path = "Xenium dataset file/morphology_mip.ome.tif" 
    
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # Try to find the file if the default path is a placeholder
    if not Path(path).exists():
        # Look in likely directories
        potential_paths = [
            "data/morphology_mip.ome.tif",
            "../data/morphology_mip.ome.tif",
            # Add the path from the user's config if known
        ]
        for p in potential_paths:
            if Path(p).exists():
                path = p
                break
    
    inspect_tiff(path)
