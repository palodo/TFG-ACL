#!/usr/bin/env python3
"""
predict_patient.py

Unified diagnostic script for patient DICOM processing.
Extracts ZIP securely (Zip Slip protection), segments planes, selects top slices,
runs Swin ensemble, computes saliency heatmaps, and outputs detailed JSON.
"""

import os
import sys
import json
import zipfile
import tempfile
import shutil
import base64
import io
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
from PIL import Image
from torchvision import transforms

# Suppress warnings
warnings.filterwarnings('ignore')


def select_device(min_free_gb=3.0):
    """Pick CUDA only if it exists AND has enough free memory.

    This keeps the app usable while a training job occupies the GPU: if there is
    not enough free VRAM (or ACL_FORCE_CPU is set), inference falls back to CPU
    instead of hanging or stealing memory from a running training process.
    """
    if os.environ.get('ACL_FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
        return torch.device('cpu')
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < min_free_gb * (1024 ** 3):
            print(json.dumps({
                'type': 'progress',
                'message': f'GPU ocupada ({free_bytes / 1024**3:.1f} GB libres); '
                           f'ejecutando en CPU (puede tardar más).',
                'step_index': 0
            }), flush=True)
            return torch.device('cpu')
    except Exception:
        return torch.device('cpu')
    return torch.device('cuda')


# Set device (GPU-aware: cae a CPU si la GPU no tiene memoria suficiente)
device = select_device()

# Add parent workspace dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import SwinMultiSliceClassifier, ViTSmallMultiSliceClassifier, CNNMultiSliceClassifier


class SimpleCNNSelector(nn.Module):
    """CNN selector based on MobileNetV2, matching the reference notebook."""
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        import timm

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[clean_key] = value

        self.device = device
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)

        if 'backbone.classifier.1.weight' in new_state_dict:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 2),
            )
            self.model = self.backbone
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1280, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
            self.model = nn.Sequential(self.backbone, self.classifier)

        self.load_state_dict(new_state_dict, strict=False)
        self.to(device)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if hasattr(self, 'classifier') and self.classifier is not None:
            features = self.backbone(x)
            return self.classifier(features)
        return self.backbone(x)

    def get_acl_probability(self, slice_2d):
        """Return ACL probability for a single 2D slice."""
        slice_tensor = torch.from_numpy(slice_2d).float()
        slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0)
        slice_tensor = slice_tensor.repeat(1, 3, 1, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        slice_tensor = (slice_tensor - mean) / std
        slice_tensor = slice_tensor.to(self.device)

        with torch.no_grad():
            logits = self.forward(slice_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs[0, 1].item()

    def select_top_k_slices(self, volume, k=5):
        """Select the k slices with highest ACL probability."""
        acl_probs = []
        for i in range(volume.shape[0]):
            prob = self.get_acl_probability(volume[i])
            acl_probs.append(prob)
        indices = np.argsort(acl_probs)[-k:]
        indices = sorted(indices.tolist())
        return indices, acl_probs


def get_plane_from_dicom(ds) -> str:
    """Determine anatomical plane from Image Orientation or description."""
    try:
        orientation = ds.ImageOrientationPatient
        row_orient = np.array(orientation[:3], dtype=np.float32)
        col_orient = np.array(orientation[3:], dtype=np.float32)
        normal = np.abs(np.cross(row_orient, col_orient))
        max_idx = int(np.argmax(normal))
        if max_idx == 0:
            return 'sagittal'
        if max_idx == 1:
            return 'coronal'
        return 'axial'
    except Exception:
        try:
            desc = str(getattr(ds, 'SeriesDescription', '')).lower()
            if 'sag' in desc:
                return 'sagittal'
            if 'cor' in desc:
                return 'coronal'
            if 'ax' in desc or 'tra' in desc:
                return 'axial'
        except Exception:
            pass
    return 'sagittal'


def secure_zip_extract(zip_path, extract_dir):
    """Extract zip archive safely, protecting against Zip Slip path traversal."""
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Strip potential leading/trailing slashes or relative segments
            filename = os.path.normpath(member.filename)
            if filename.startswith('..') or '/..' in filename or '\\..' in filename:
                raise ValueError(f"Path traversal detected in ZIP filename: {member.filename}")
                
            member_path = Path(extract_dir / filename).resolve()
            # Enforce trailing slash constraint or direct inclusion check
            if not member_path.as_posix().startswith(extract_dir.as_posix() + '/'):
                # Handle direct file match under exact directory name
                if member_path.as_posix() != extract_dir.as_posix():
                    raise ValueError(f"Zip extraction boundary violated by member: {member.filename}")
            
            # Recreate folder if directories are stored as members
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as source, open(member_path, 'wb') as target:
                    shutil.copyfileobj(source, target)


def load_dicom_from_dir(temp_dir):
    """Scan and load valid DICOM slices grouped by anatomical plane."""
    temp_dir = Path(temp_dir)
    # Weasis ZIPs can contain files without extensions, so we look recursively at all files
    all_files = [f for f in temp_dir.rglob('*') if f.is_file()]

    planes_data = {'sagittal': {}, 'coronal': {}, 'axial': {}}
    
    patient_info = {}

    def series_priority(series_desc: str) -> int:
        """Prefer PD/fat-sat-like protocols when multiple series tie on slice count."""
        desc = (series_desc or '').lower()
        score = 0
        preferred_tokens = ['pd', 'spair', 'fs', 'fat', 'stir']
        for token in preferred_tokens:
            if token in desc:
                score += 1
        return score

    def choose_best_series(series_dict):
        """Pick the most representative series for a plane to avoid mixing resolutions."""
        best_key = None
        best_score = None
        for key, payload in series_dict.items():
            series_desc = payload.get('series_description', '')
            slices_count = len(payload.get('slices', []))
            height = int(payload.get('shape', (0, 0))[0])
            width = int(payload.get('shape', (0, 0))[1])
            current_score = (
                slices_count,
                series_priority(series_desc),
                height * width,
            )
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_key = key
        return best_key

    for item in all_files:
        try:
            ds = pydicom.dcmread(str(item), force=True)
            if not hasattr(ds, 'pixel_array'):
                continue
                
            # Extract key metadata on first valid file
            if not patient_info:
                try:
                    patient_info = {
                        'patient_name': str(getattr(ds, 'PatientName', 'Paciente Anónimo')),
                        'patient_id': str(getattr(ds, 'PatientID', 'ANON-ID')),
                        'patient_age': str(getattr(ds, 'PatientAge', 'N/A')),
                        'patient_sex': str(getattr(ds, 'PatientSex', 'N/A')),
                        'study_date': str(getattr(ds, 'StudyDate', 'N/A')),
                        'modality': str(getattr(ds, 'Modality', 'MR')),
                        'body_part': str(getattr(ds, 'BodyPartExamined', 'KNEE')),
                    }
                except Exception:
                    pass

            image = ds.pixel_array.astype(np.float32)
            
            # Apply windowing if available
            try:
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    center = ds.WindowCenter[0] if isinstance(ds.WindowCenter, (list, tuple)) else ds.WindowCenter
                    width = ds.WindowWidth[0] if isinstance(ds.WindowWidth, (list, tuple)) else ds.WindowWidth
                    lower = center - width / 2
                    upper = center + width / 2
                    image = np.clip(image, lower, upper)
            except Exception:
                pass

            # Min-Max Normalization to 0-255 uint8 range
            v_min = image.min()
            v_max = image.max()
            if v_max > v_min:
                image_norm = ((image - v_min) / (v_max - v_min) * 255).astype(np.uint8)
            else:
                image_norm = np.zeros_like(image, dtype=np.uint8)

            plane = get_plane_from_dicom(ds)
            instance_number = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0
            slice_location = float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None

            record = {
                'image': image_norm,
                'instance_number': instance_number,
                'slice_location': slice_location
            }
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown_series_uid'))
            series_desc = str(getattr(ds, 'SeriesDescription', ''))
            shape = tuple(image_norm.shape)
            series_key = (series_uid, shape)

            if series_key not in planes_data[plane]:
                planes_data[plane][series_key] = {
                    'series_description': series_desc,
                    'shape': shape,
                    'slices': []
                }

            planes_data[plane][series_key]['slices'].append(record)
        except Exception:
            pass

    # Select one coherent series per plane, sort slices, then stack into volumes
    planes_volumes = {}
    for plane, series_dict in planes_data.items():
        if not series_dict:
            continue

        selected_key = choose_best_series(series_dict)
        if selected_key is None:
            continue

        slices = series_dict[selected_key]['slices']
        slices.sort(key=lambda x: (
            x['slice_location'] if x['slice_location'] is not None else 0,
            x['instance_number'] if x['instance_number'] is not None else 0,
        ))
        planes_volumes[plane] = np.stack([s['image'] for s in slices], axis=0)
        
    return planes_volumes, patient_info


def center_consecutive_indices(num_slices_total, k=5):
    """Return consecutive indices focused at the center of the volume."""
    start_idx = max(0, (num_slices_total - k) // 2)
    end_idx = start_idx + k
    indices = list(range(start_idx, min(end_idx, num_slices_total)))
    if len(indices) < k and indices:
        indices += [indices[-1]] * (k - len(indices))
    return indices


def image_to_base64(img_array):
    """Convert a 2D uint8 numpy array to a base64 grayscale JPEG string."""
    img = Image.fromarray(img_array).convert('L')
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def robust_unit_scale(arr, low_q=2, high_q=98):
    """Robust scaling using percentiles to eliminate extreme outliers."""
    low = np.percentile(arr, low_q)
    high = np.percentile(arr, high_q)
    if high - low < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0, 1)


def generate_saliency_overlay(model, slice_2d, model_transform, device, colormap='viridis'):
    """Compute Grad-CAM/Saliency map and generate a transparent RGBA PNG overlay."""
    model.eval()
    
    # Pre-process single slice
    img = Image.fromarray(slice_2d.astype(np.uint8)).convert('RGB')
    slice_rgb = model_transform(img)
    
    x = slice_rgb.unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Reset model gradients
    model.backbone.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    # Forward pass reproducing notebook saliency hook / Grad-CAM
    if isinstance(model, CNNMultiSliceClassifier):
        # Propagate manually through backbone to layer4 output (child 7)
        x_in = x
        for i in range(8):  # conv1 to layer4 inclusive
            x_in = model.backbone[i](x_in)
            
        x_in.retain_grad()
        feature_maps = x_in
        
        pooled = model.backbone[8](feature_maps)  # avgpool
        features = pooled.view(1, model.feature_dim)
        features = features.unsqueeze(1)
        
        if model.pooling_mode == 'attention':
            pooled_features, _ = model.pooling(features)
        elif model.pooling_mode == 'max':
            pooled_features = model.pooling(features.transpose(1, 2)).squeeze(-1)
        elif model.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        logits = model.classifier(pooled_features)
        
        score = logits.squeeze()
        score.backward()
        
        # Grad-CAM on layer4 features
        weights = torch.mean(x_in.grad, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * x_in, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=slice_rgb.shape[-2:], mode='bilinear', align_corners=False)
        saliency_2d = grad_cam[0, 0].detach().cpu().numpy()
    else:
        feat_map = model.backbone.forward_features(x)
        feat_vec = model.backbone.forward_head(feat_map, pre_logits=True)
        pooled_features, _ = model.pooling(feat_vec.unsqueeze(1))
        logits = model.classifier(pooled_features)
        
        score = logits.squeeze()
        score.backward()
        
        # Saliency maps post-processing for other architectures
        saliency = x.grad.detach().abs().mean(dim=1, keepdim=True)
        saliency = F.avg_pool2d(saliency, kernel_size=3, stride=1, padding=1)
        saliency = F.interpolate(saliency, size=slice_rgb.shape[-2:], mode='bilinear', align_corners=False)
        saliency_2d = saliency[0, 0].cpu().numpy()
    
    # Scale robustly and enhance contrast via gamma correction
    saliency_scaled = robust_unit_scale(saliency_2d, low_q=1, high_q=99)
    saliency_enhanced = np.power(saliency_scaled, 0.55)

    # Colorize saliency map using matplotlib colormaps
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(colormap)
    rgba = cm(saliency_enhanced)  # shape (H, W, 4)
    
    # Set dynamic transparency: highly active regions are opaque, inactive are fully transparent
    rgba[..., 3] = np.clip(saliency_enhanced * 1.6, 0.0, 1.0)
    
    # Mask out extremely low activations (noise filter)
    rgba[rgba[..., 3] < 0.15, 3] = 0.0

    # Resize to match output resolution and base64 encode
    rgba_img = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba_img = rgba_img.resize((256, 256), Image.Resampling.BILINEAR)
    
    buffered = io.BytesIO()
    rgba_img.save(buffered, format="PNG")
    
    pred_prob = float(torch.sigmoid(score).detach().cpu().item())
    return base64.b64encode(buffered.getvalue()).decode('utf-8'), pred_prob


import math

def generate_vit_attention_overlay(model, slice_2d, model_transform, device, colormap='viridis'):
    """Extract ViT spatial self-attention maps and generate a transparent RGBA PNG overlay."""
    model.eval()
    
    # Pre-process single slice
    img = Image.fromarray(slice_2d.astype(np.uint8)).convert('RGB')
    slice_rgb = model_transform(img)
    
    # Model expects (batch_size, num_slices, 3, 224, 224)
    x = slice_rgb.unsqueeze(0).unsqueeze(0).to(device)
    
    # Register forward hook on self-attention of the last encoder layer
    attention_inputs = []
    def hook_fn(module, input, output):
        attention_inputs.append(input[0])
        
    self_attn = model.vit.encoder.layer[-1].attention.attention
    handle = self_attn.register_forward_hook(hook_fn)
    
    # Forward pass to activate hook
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
            
    handle.remove()
    
    if len(attention_inputs) > 0:
        hidden_states = attention_inputs[0] # Shape (1, 197, 768)
        
        # Manually compute attention scores for CLS token
        with torch.no_grad():
            mixed_query_layer = self_attn.query(hidden_states)
            key_layer = self_attn.transpose_for_scores(self_attn.key(hidden_states))
            query_layer = self_attn.transpose_for_scores(mixed_query_layer)
            
            q_cls = query_layer[:, :, 0:1, :] # CLS query -> (1, num_heads, 1, head_dim)
            k_all = key_layer # All keys -> (1, num_heads, 197, head_dim)
            
            scores = torch.matmul(q_cls, k_all.transpose(-1, -2)) / math.sqrt(self_attn.attention_head_size)
            probs = torch.softmax(scores, dim=-1) # (1, num_heads, 1, 197)
            
            # Average across heads
            attn_map = probs.mean(dim=1) # (1, 1, 197)
            
            # Extract for the slice, excluding CLS token
            slice_map = attn_map[0, 0, 1:] # Shape (196,)
            grid_map = slice_map.view(14, 14) # Shape (14, 14)
            
            # Interpolate grid_map to 224x224
            grid_map_4d = grid_map.unsqueeze(0).unsqueeze(0) # (1, 1, 14, 14)
            grid_map_resized = F.interpolate(grid_map_4d, size=(224, 224), mode='bilinear', align_corners=False)
            attn_2d = grid_map_resized[0, 0].cpu().numpy()
    else:
        attn_2d = np.zeros((224, 224), dtype=np.float32)
        
    # Scale robustly and enhance contrast via gamma correction
    attn_scaled = robust_unit_scale(attn_2d, low_q=1, high_q=99)
    attn_enhanced = np.power(attn_scaled, 0.55)
    
    # Colorize saliency map using matplotlib colormaps
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(colormap)
    rgba = cm(attn_enhanced) # shape (H, W, 4)
    
    # Set dynamic transparency: highly active regions are opaque, inactive are fully transparent
    rgba[..., 3] = np.clip(attn_enhanced * 1.6, 0.0, 1.0)
    
    # Mask out extremely low activations (noise filter)
    rgba[rgba[..., 3] < 0.15, 3] = 0.0
    
    # Resize to match output resolution and base64 encode
    rgba_img = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba_img = rgba_img.resize((256, 256), Image.Resampling.BILINEAR)
    
    buffered = io.BytesIO()
    rgba_img.save(buffered, format="PNG")
    
    pred_prob = float(torch.sigmoid(logits).squeeze().item())
    return base64.b64encode(buffered.getvalue()).decode('utf-8'), pred_prob


def print_progress(message, step_index=None):
    """Helper to stream progress status updates to the Node BFF in real-time."""
    print(json.dumps({
        'type': 'progress',
        'message': message,
        'step_index': step_index
    }))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="ACL Injury Diagnostic API Pipeline")
    parser.add_argument("--zip", type=str, required=True, help="Path to patient DICOM series ZIP file")
    parser.add_argument("--k", type=int, default=5, help="Number of slices per plane")
    parser.add_argument("--model", type=str, default="swin", choices=["swin", "vit", "cnn"], help="AI Model Architecture to use (swin, vit, or cnn)")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(json.dumps({'error': f"Input file not found: {args.zip}"}))
        sys.exit(1)

    # Create temporary directory for DICOM extraction
    temp_dir = tempfile.mkdtemp(prefix='diag_extract_')
    
    try:
        # 1. Secure zip extraction
        print_progress("Descomprimiendo serie DICOM y verificando integridad de firmas de cabecera...", 0)
        secure_zip_extract(zip_path, temp_dir)
        
        # 2. Parse DICOM files and stack planes
        print_progress("Cargando y ordenando cortes DICOM espacialmente en base a coordenadas 3D...", 0)
        planes_volumes, patient_info = load_dicom_from_dir(temp_dir)
        
        if not planes_volumes:
            raise ValueError("No valid DICOM series could be extracted from the provided archive.")

        # Stream patient info update instantly
        print(json.dumps({
            'type': 'patient_info',
            'patient_info': patient_info
        }))
        sys.stdout.flush()
        
        patient_name = patient_info.get('patient_name', 'Paciente Anónimo')
        print_progress(f"Ficha clínica de metadatos extraída con éxito: {patient_name}.", 1)

        sag_count = planes_volumes['sagittal'].shape[0] if 'sagittal' in planes_volumes else 0
        cor_count = planes_volumes['coronal'].shape[0] if 'coronal' in planes_volumes else 0
        ax_count = planes_volumes['axial'].shape[0] if 'axial' in planes_volumes else 0
        print_progress(f"Volúmenes 3D leídos - Sagital: {sag_count} cortes | Coronal: {cor_count} cortes | Axial: {ax_count} cortes.", 1)

        # Set paths to model checkpoints
        checkpoints_dir = Path('/home/palodo2/acl_classifier/checkpoints')
        
        # Load CNN selectors
        print_progress("Iniciando localizadores MobileNetV2 y cargando pesos en GPU...", 1)
        selector_sag_path = checkpoints_dir / 'acl_slice_classifier_v3' / 'best_model_f1.pth'
        selector_cor_path = checkpoints_dir / 'acl_slice_classifier_coronal' / 'best_model_f1.pth'
        
        selector_sag = SimpleCNNSelector(selector_sag_path, device=device).eval()
        selector_cor = SimpleCNNSelector(selector_cor_path, device=device).eval()

        # Create model transforms
        model_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        selected_indices = {}
        selector_probs = {}

        # Define dynamic K values per plane matching model training calibrations
        plane_k = {
            'sagittal': 5,
            'coronal': 10,
            'axial': 10
        }

        # Sagittal Selection
        if 'sagittal' in planes_volumes:
            print_progress("Analizando Plano Sagital: Seleccionando los 5 cortes óptimos que contienen el LCA...", 2)
            vol = planes_volumes['sagittal']
            norm_vol = (vol.astype(np.float32) - vol.min()) / (vol.max() - vol.min() + 1e-8)
            idxs, probs = selector_sag.select_top_k_slices(norm_vol, k=plane_k['sagittal'])
            selected_indices['sagittal'] = idxs
            selector_probs['sagittal'] = [probs[i] for i in idxs]
            print_progress(f"Plano Sagital: Localizados 5 cortes de alto interés (Indices: {idxs}).", 2)
        else:
            selected_indices['sagittal'] = []
            selector_probs['sagittal'] = []

        # Coronal Selection
        if 'coronal' in planes_volumes:
            print_progress("Analizando Plano Coronal: Seleccionando los 10 cortes de mayor relevancia...", 3)
            vol = planes_volumes['coronal']
            norm_vol = (vol.astype(np.float32) - vol.min()) / (vol.max() - vol.min() + 1e-8)
            idxs, probs = selector_cor.select_top_k_slices(norm_vol, k=plane_k['coronal'])
            selected_indices['coronal'] = idxs
            selector_probs['coronal'] = [probs[i] for i in idxs]
            print_progress(f"Plano Coronal: Localizados 10 cortes de alto interés (Indices: {idxs}).", 3)
        else:
            selected_indices['coronal'] = []
            selector_probs['coronal'] = []

        # Axial Selection (central slices)
        if 'axial' in planes_volumes:
            print_progress("Analizando Plano Axial: Seleccionando los 10 cortes concéntricos del plano de inserción...", 4)
            vol = planes_volumes['axial']
            k_val = plane_k['axial']
            selected_indices['axial'] = center_consecutive_indices(vol.shape[0], k=k_val)
            selector_probs['axial'] = [0.5] * k_val # No selector model used
            print_progress(f"Plano Axial: Seleccionados 10 cortes de referencia central (Indices: {selected_indices['axial']}).", 4)
        else:
            selected_indices['axial'] = []
            selector_probs['axial'] = []

        # Swin Classifiers Loading
        swin_dir = checkpoints_dir / 'swin_small_multiseed'
        swin_models = {
            'sagittal': swin_dir / 'best_sagittal_multiseed_final.pth',
            'coronal': swin_dir / 'best_coronal_multiseed_final.pth',
            'axial': swin_dir / 'best_axial_multiseed_final.pth'
        }

        # ViT Classifiers Loading (ViT-Small: 22M params, mejor que ViT-Base en test)
        vit_dir = checkpoints_dir / 'vit_small_multiseed'
        vit_models = {
            'sagittal': vit_dir / 'best_sagittal_multiseed_final.pth',
            'coronal': vit_dir / 'best_coronal_multiseed_final.pth',
            'axial': vit_dir / 'best_axial_multiseed_final.pth'
        }

        vit_configs = {
            'sagittal': {'pooling_mode': 'attention', 'dropout_input': 0.2, 'dropout_dense': 0.15},
            'coronal': {'pooling_mode': 'max', 'dropout_input': 0.25, 'dropout_dense': 0.15},
            'axial': {'pooling_mode': 'attention', 'dropout_input': 0.25, 'dropout_dense': 0.15}
        }

        # CNN Classifiers Loading
        cnn_dir = checkpoints_dir / 'cnn_multiseed_resnet50'
        cnn_models = {
            'sagittal': cnn_dir / 'best_sagittal_multiseed_final.pth',
            'coronal': cnn_dir / 'best_coronal_multiseed_final.pth',
            'axial': cnn_dir / 'best_axial_multiseed_final.pth'
        }

        cnn_configs = {
            'sagittal': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32},
            'coronal': {'pooling_mode': 'max', 'dropout_input': 0.47, 'dropout_dense': 0.37},
            'axial': {'pooling_mode': 'attention', 'dropout_input': 0.42, 'dropout_dense': 0.32}
        }

        # Inference results structure
        planes_results = {}
        plane_scores = {}

        for plane in ['sagittal', 'coronal', 'axial']:
            if plane not in planes_volumes or not selected_indices[plane]:
                continue
                
            plane_name_es = "Sagital" if plane == "sagittal" else "Coronal" if plane == "coronal" else "Axial"
            step_idx = 2 if plane == "sagittal" else 3 if plane == "coronal" else 4
            
            # Load specialized model based on selected architecture
            if args.model == 'vit':
                print_progress(f"Plano {plane_name_es}: Cargando clasificador ViT-Small Transformer Deep Ensemble...", step_idx)
                model_path = vit_models[plane]
                cfg = vit_configs[plane]
                # pretrained=True para instanciar la arquitectura real de ViT-Small
                # (hidden 384); los pesos de ImageNet se sobrescriben con el checkpoint.
                model = ViTSmallMultiSliceClassifier(
                    num_classes=1,
                    pretrained=True,
                    pooling_mode=cfg['pooling_mode'],
                    dropout_input=cfg['dropout_input'],
                    dropout_dense=cfg['dropout_dense']
                ).to(device)
            elif args.model == 'cnn':
                print_progress(f"Plano {plane_name_es}: Cargando clasificador KneeCNN-Net ResNet50 Deep Ensemble...", step_idx)
                model_path = cnn_models[plane]
                cfg = cnn_configs[plane]
                model = CNNMultiSliceClassifier(
                    num_classes=1,
                    pretrained=False,
                    pooling_mode=cfg['pooling_mode'],
                    dropout_input=cfg['dropout_input'],
                    dropout_dense=cfg['dropout_dense']
                ).to(device)
            else:
                print_progress(f"Plano {plane_name_es}: Cargando clasificador Swin Transformer Deep Ensemble...", step_idx)
                model_path = swin_models[plane]
                model = SwinMultiSliceClassifier().to(device)

            checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Clean state_dict keys (remove 'module.' if present)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            vol = planes_volumes[plane]
            indices = selected_indices[plane]
            
            print_progress(f"Plano {plane_name_es}: Ejecutando inferencia en lote en GPU...", step_idx)
            
            # Package into multislice input shape (1, K, 3, 224, 224)
            slice_tensors = []
            for idx in indices:
                img_array = vol[idx]
                img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
                slice_tensors.append(model_transform(img))
            
            batch_x = torch.stack(slice_tensors, dim=0).unsqueeze(0).to(device)
            
            # Forward pass to get attention weights and global prediction
            with torch.no_grad():
                output = model(batch_x)
                if isinstance(output, tuple):
                    logits, attention_weights = output
                    if attention_weights is not None:
                        attention_weights = attention_weights.squeeze(0).cpu().numpy().tolist()
                    else:
                        attention_weights = [1.0 / len(indices)] * len(indices)
                else:
                    logits = output
                    attention_weights = [1.0 / len(indices)] * len(indices)
                
                # Compute global prediction probability
                if logits.shape[-1] == 1:
                    global_prob = torch.sigmoid(logits).squeeze().item()
                else:
                    global_prob = torch.softmax(logits, dim=1)[:, 1].squeeze().item()

            plane_scores[plane] = global_prob

            if args.model == 'vit':
                print_progress(f"Plano {plane_name_es}: Generando mapas transparentes de auto-atención de ViT...", step_idx)
            else:
                print_progress(f"Plano {plane_name_es}: Generando mapas transparentes de salienca Grad-CAM...", step_idx)

            # Calculate individual slice base64 frames + transparent saliency overlays
            slices_data = []
            
            # Map selected indexes to their sequence positions
            idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
            
            # Process EVERY slice in the volume for reference scrolling!
            for idx in range(vol.shape[0]):
                slice_2d = vol[idx]
                base_img_b64 = image_to_base64(slice_2d)
                
                if idx in idx_to_pos:
                    pos = idx_to_pos[idx]
                    # Map overlay based on active architecture
                    if args.model == 'vit':
                        heatmap_b64, slice_pred_prob = generate_vit_attention_overlay(
                            model, slice_2d, model_transform, device, colormap='viridis'
                        )
                    else:
                        heatmap_b64, slice_pred_prob = generate_saliency_overlay(
                            model, slice_2d, model_transform, device, colormap='viridis'
                        )
                    
                    slices_data.append({
                        'slice_index': idx,
                        'raw_image': base_img_b64,
                        'heatmap_overlay': heatmap_b64,
                        'selector_prob': selector_probs[plane][pos],
                        'swin_attention': attention_weights[pos],
                        'slice_probability': slice_pred_prob,
                        'has_prediction': True
                    })
                else:
                    slices_data.append({
                        'slice_index': idx,
                        'raw_image': base_img_b64,
                        'heatmap_overlay': None,
                        'selector_prob': 0.0,
                        'swin_attention': 0.0,
                        'slice_probability': 0.0,
                        'has_prediction': False
                    })
            
            planes_results[plane] = {
                'average_probability': global_prob,
                'slices': slices_data
            }

            # Stream the plane result update instantly!
            print(json.dumps({
                'type': 'plane_result',
                'plane': plane,
                'average_probability': global_prob,
                'slices': slices_data
            }))
            sys.stdout.flush()
            
            print_progress(f"✓ Plano {plane_name_es} completado (Inferencia del plano: {round(global_prob*100)}%).", step_idx)

        # Global Ensemble Formulation
        print_progress("Consolidando predicciones de planos anatómicos (Calculando Ensemble multi-plano)...", 5)
        ensemble_prob = np.mean(list(plane_scores.values()))
        
        # Calibrated score relative to threshold (model-specific on MRNet validation splits)
        thresholds = {
            'swin': 0.3734,
            'vit': 0.3804,   # ViT-Small ensemble, calibrado en validación
            'cnn': 0.5455
        }
        THRESHOLD = thresholds.get(args.model, 0.3734)
        if ensemble_prob >= THRESHOLD:
            calibrated_confidence = 0.5 + 0.5 * (ensemble_prob - THRESHOLD) / (1.0 - THRESHOLD + 1e-8)
        else:
            calibrated_confidence = 0.5 * ensemble_prob / (THRESHOLD + 1e-8)

        predicted_class = 'ACL INJURY' if ensemble_prob >= THRESHOLD else 'HEALTHY'
        confidence_label = 'strong' if calibrated_confidence >= 0.75 else 'moderate' if calibrated_confidence >= 0.60 else 'weak'

        # Stream final ensemble diagnosis update instantly!
        print(json.dumps({
            'type': 'diagnosis',
            'diagnosis': {
                'prediction': predicted_class,
                'ensemble_probability': float(ensemble_prob),
                'calibrated_confidence': float(calibrated_confidence),
                'confidence_level': confidence_label,
                'threshold': THRESHOLD
            }
        }))
        sys.stdout.flush()
        
        print_progress("✓ Análisis clínico completado con éxito.", 6)
        
    except Exception as e:
        import traceback
        error_details = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_details))
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
