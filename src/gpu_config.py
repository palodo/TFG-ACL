"""
GPU Configuration for ACL Classifier
Handles device setup and multi-GPU distribution.
"""

import torch

# ==========================================
# GPU CONFIGURATION (SHARED FOR ALL PLANES)
# ==========================================
VIT_GPU_IDS = [0, 1, 2, 3]  # ViT usará 4 GPUs (GPU física 1,2,3,4 -> lógica 0,1,2,3) con DataParallel
CNN_GPU_ID = 0  # CNN Selector usará GPU 0 durante pre-procesamiento (GPU física 1 -> lógica 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print GPU info
if torch.cuda.is_available():
    print(f"{'='*80}")
    print(f"GPU CONFIGURATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"GPUs disponibles: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"\n Asignación de GPUs:")
    print(f"  ViT Training: GPUs {VIT_GPU_IDS}")
    print(f"  CNN Pre-processing: GPU {CNN_GPU_ID}")
    print(f"{'='*80}\n")
else:
    print("  GPU no disponible, usando CPU (procesamiento muy lento)")
