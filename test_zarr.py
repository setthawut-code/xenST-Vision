
import tifffile
import numpy as np
import time
import sys

def test_zarr_read(path):
    print(f"Testing zarr read on {path}...")
    try:
        import zarr
    except ImportError:
        print("Zarr not installed. Please install it: pip install zarr")
        return

    try:
        # Open as Zarr store
        store = tifffile.imread(path, aszarr=True)
        z = zarr.open(store, mode='r')
        
        print(f"Zarr object type: {type(z)}")
        if isinstance(z, zarr.Group):
            print(f"Keys: {list(z.keys())}")
            # Try to get the first array (usually '0' or similar for OME-TIFF)
            # OME-TIFF usually maps series to keys '0', '1', etc.
            if '0' in z:
                z = z['0']
            else:
                 # Try first key
                 key = list(z.keys())[0]
                 z = z[key]
        
        print(f"Zarr array shape: {z.shape}")
        print(f"Zarr chunks: {z.chunks}")
        
        # Benchmarking random access
        start = time.time()
        # Read a 256x256 patch from middle
        y, x = 10000, 10000
        # Ensure bounds
        y = min(y, z.shape[0] - 256)
        x = min(x, z.shape[1] - 256)
        
        patch = z[y:y+256, x:x+256]
        print(f"Read 256x256 patch in {time.time() - start:.4f}s")
        print(f"Patch shape: {patch.shape}")
        
    except Exception as e:
        print(f"Error reading as zarr: {e}")

if __name__ == "__main__":
    path = "Xenium dataset file/morphology_mip.ome.tif"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    test_zarr_read(path)
