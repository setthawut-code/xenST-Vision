# Run Training Script
# NOTE: Ensure you are using Python 3.10-3.12 for GPU support. Python 3.14 is currently CPU-only.

$ErrorActionPreference = "Stop"

Write-Host "Activating Virtual Environment..."
.\venv_gpu\Scripts\Activate.ps1

Write-Host "Checking for CUDA..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

Write-Host "Starting Training..."
python src/train.py --config configs/config_low_vram.yaml

Write-Host "Training process finished."
