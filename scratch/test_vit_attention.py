import sys
from pathlib import Path
import torch
import torch.nn as nn
import math
import numpy as np

# Add parent workspace dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import ViTMultiSliceClassifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize a ViT model for Sagittal plane
    model = ViTMultiSliceClassifier(
        model_name="google/vit-base-patch16-224",
        num_classes=1,
        pretrained=False,
        pooling_mode='attention',
        dropout_input=0.3,
        dropout_dense=0.2
    ).to(device)
    
    # Load sagittal checkpoint to verify
    checkpoint_path = "/home/palodo2/acl_classifier/checkpoints/vit_multiseed_base/best_sagittal_multiseed_final.pth"
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Clean state_dict keys (remove 'module.' if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Checkpoint loaded successfully!")
    
    # Create a dummy batch of shape (1, 5, 3, 224, 224)
    # 1 patient, 5 slices, 3 channels, 224x224
    dummy_input = torch.randn(1, 5, 3, 224, 224).to(device)
    
    # Register forward hook on self-attention of the last encoder layer
    attention_inputs = []
    def hook_fn(module, input, output):
        attention_inputs.append(input[0])
        
    self_attn = model.vit.encoder.layer[-1].attention.attention
    handle = self_attn.register_forward_hook(hook_fn)
    
    # Run inference
    with torch.no_grad():
        logits, slice_weights = model(dummy_input)
        
    handle.remove()
    
    print(f"Logits shape: {logits.shape}")
    print(f"Slice weights shape: {slice_weights.shape}")
    print(f"Captured attention inputs count: {len(attention_inputs)}")
    
    if len(attention_inputs) > 0:
        hidden_states = attention_inputs[0]
        print(f"Hidden states shape entering attention: {hidden_states.shape}")
        
        # Manually compute attention scores for CLS token
        with torch.no_grad():
            mixed_query_layer = self_attn.query(hidden_states)
            key_layer = self_attn.transpose_for_scores(self_attn.key(hidden_states))
            query_layer = self_attn.transpose_for_scores(mixed_query_layer)
            
            q_cls = query_layer[:, :, 0:1, :] # CLS query
            k_all = key_layer # All keys
            
            scores = torch.matmul(q_cls, k_all.transpose(-1, -2)) / math.sqrt(self_attn.attention_head_size)
            probs = torch.softmax(scores, dim=-1) # (batch*slices, num_heads, 1, 197)
            
            # Average across heads
            attn_map = probs.mean(dim=1) # (batch*slices, 1, 197)
            
            # Extract for the first slice, excluding CLS token
            first_slice_map = attn_map[0, 0, 1:] # (196,)
            grid_map = first_slice_map.view(14, 14).cpu().numpy()
            
            print(f"Grid map shape: {grid_map.shape}")
            print(f"Grid map mean: {grid_map.mean():.6f}, max: {grid_map.max():.6f}")
            print("✓ Manual attention extraction verified!")

if __name__ == '__main__':
    main()
