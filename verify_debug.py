
import os
import torch
from torch.utils.data import DataLoader
from src.dataset import XeniumDataset
import time
import multiprocessing

def main():
    print(f"Main process: {os.getpid()}")
    try:
        # Note: This will only work if you have the actual data paths configured in your environment
        # or if the default paths in dataset.py exist. 
        # Since I am not sure of the exact paths on your machine (they were 'Xenium dataset file' in the code I saw),
        # I'll rely on the default logic or you might need to edit this.
        
        # Assuming the user's config points to valid data, we'll try to instantiate the dataset 
        # with the same params as training, but minimal.
        
        dataset = XeniumDataset(
            image_path="Xenium dataset file/morphology_mip.ome.tif", # Verify this path!
            transcripts_path="Xenium dataset file/transcripts_full.csv",
            force_rebuild_index=False
        )
        
        print(f"Dataset loaded. Length: {len(dataset)}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=8, # Small batch
            shuffle=False,
            num_workers=4, # Test parallel
            multiprocessing_context='spawn' # Enforce spawn like macOS default
        )
        
        print("Starting iteration...")
        start = time.time()
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} loaded in {time.time() - start:.2f}s")
            start = time.time()
            if i >= 2: break
            
        print("Iteration successful.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
