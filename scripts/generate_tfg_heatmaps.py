#!/usr/bin/env python3
"""
Genera las figuras de explicabilidad del TFG con el MISMO algoritmo que la app
(scripts/predict_patient.py::generate_saliency_overlay):
  saliencia por gradiente de entrada -> escalado robusto por percentiles ->
  correccion gamma 0.55 -> colormap viridis con alfa proporcional y umbral de ruido.

Se ejecuta en CPU para no interferir con entrenamientos en GPU.

Salidas (en memoria/tfgs/figs/):
  - app_gradcam_planes.png   : rejilla 3 planos x 5 cortes con overlay
  - app_heatmaps_sorted.png  : sagital, fila superior RM original / inferior overlay
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.models import SwinMultiSliceClassifier

DEVICE = torch.device('cpu')  # GPU ocupada por entrenamiento
CASE = '0105'                 # caso positivo (rotura LCA) del set de validacion
PLANES = ['sagittal', 'coronal', 'axial']
PLANE_ES = {'sagittal': 'Sagital', 'coronal': 'Coronal', 'axial': 'Axial'}
N_SHOW = 5

CKPT = {p: BASE / 'checkpoints' / 'swin_tiny_multiseed' / f'best_{p}_multiseed_final.pth'
        for p in PLANES}

model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def robust_unit_scale(arr, low_q=1, high_q=99):
    low = np.percentile(arr, low_q)
    high = np.percentile(arr, high_q)
    if high - low < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0, 1)


def load_model(plane):
    model = SwinMultiSliceClassifier(pretrained=False).to(DEVICE)
    ck = torch.load(CKPT[plane], map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def saliency_overlay(model, slice_2d):
    """Replica exacta de generate_saliency_overlay de la app para Swin."""
    img = Image.fromarray(slice_2d.astype(np.uint8)).convert('RGB')
    slice_rgb = model_transform(img)
    x = slice_rgb.unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    model.backbone.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    feat_map = model.backbone.forward_features(x)
    feat_vec = model.backbone.forward_head(feat_map, pre_logits=True)
    pooled, _ = model.pooling(feat_vec.unsqueeze(1))
    logits = model.classifier(pooled)
    score = logits.squeeze()
    score.backward()

    saliency = x.grad.detach().abs().mean(dim=1, keepdim=True)
    saliency = F.avg_pool2d(saliency, kernel_size=3, stride=1, padding=1)
    saliency = F.interpolate(saliency, size=slice_rgb.shape[-2:], mode='bilinear', align_corners=False)
    sal_2d = saliency[0, 0].cpu().numpy()

    sal = robust_unit_scale(sal_2d, 1, 99)
    sal = np.power(sal, 0.55)
    rgba = plt.get_cmap('viridis')(sal)
    rgba[..., 3] = np.clip(sal * 1.6, 0.0, 1.0)
    rgba[rgba[..., 3] < 0.15, 3] = 0.0
    prob = float(torch.sigmoid(score).detach().item())
    return rgba, prob


def get_slices(plane):
    vol = np.load(BASE / 'data' / 'val' / plane / f'{CASE}.npy').astype(np.float32)
    with open(BASE / 'data' / 'slice_indices_final' / f'val_{plane}_indices.json') as f:
        idxs = json.load(f)[str(int(CASE))]
    sel = vol[idxs]
    lo, hi = sel.min(), sel.max()
    sel = (sel - lo) / (hi - lo + 1e-8) * 255.0
    return sel.astype(np.uint8), idxs


def attention_weights(model, slices_u8):
    tensors = [model_transform(Image.fromarray(s).convert('RGB')) for s in slices_u8]
    x = torch.stack(tensors).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, attn = model(x)
    prob = float(torch.sigmoid(logits).squeeze())
    return (attn.squeeze(0).numpy() if attn is not None else
            np.full(len(slices_u8), 1.0 / len(slices_u8))), prob


def gray224(slice_u8):
    return np.array(Image.fromarray(slice_u8).convert('L').resize((224, 224), Image.BILINEAR))


def main():
    figs_dir = BASE / 'memoria' / 'tfgs' / 'figs'
    plane_data = {}
    for plane in PLANES:
        print(f'[{plane}] cargando modelo y calculando saliencias...')
        model = load_model(plane)
        slices, idxs = get_slices(plane)
        attn, prob = attention_weights(model, slices)
        order = np.argsort(attn)[::-1][:N_SHOW]   # top-5 por atencion
        order = sorted(order.tolist())
        cells = []
        for j in order:
            rgba, sprob = saliency_overlay(model, slices[j])
            cells.append((gray224(slices[j]), rgba, idxs[j], attn[j], sprob))
        plane_data[plane] = (cells, prob)
        del model

    # ---------- Figura principal: 3 planos x 5 cortes con overlay ----------
    fig, axes = plt.subplots(3, N_SHOW, figsize=(15, 9.3))
    for r, plane in enumerate(PLANES):
        cells, prob = plane_data[plane]
        for c in range(N_SHOW):
            ax = axes[r, c]
            g, rgba, idx, a, sp = cells[c]
            ax.imshow(g, cmap='gray')
            ax.imshow(rgba)
            ax.set_title(f'corte {idx}  |  atención {a:.2f}', fontsize=9)
            ax.axis('off')
        axes[r, 0].text(-0.18, 0.5, f'{PLANE_ES[plane]}\n(p={prob:.2f})',
                        transform=axes[r, 0].transAxes, ha='center', va='center',
                        fontsize=13, fontweight='bold', rotation=90)
    fig.suptitle(f'Mapas de saliencia (Swin-Tiny) — caso {CASE} de validación, rotura de LCA',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    out1 = figs_dir / 'app_gradcam_planes.png'
    fig.savefig(out1, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out1)

    # ---------- Figura anexo: sagital, original vs overlay ----------
    cells, prob = plane_data['sagittal']
    fig, axes = plt.subplots(2, N_SHOW, figsize=(15, 6.6))
    for c in range(N_SHOW):
        g, rgba, idx, a, sp = cells[c]
        axes[0, c].imshow(g, cmap='gray')
        axes[0, c].set_title(f'corte {idx}', fontsize=10)
        axes[1, c].imshow(g, cmap='gray')
        axes[1, c].imshow(rgba)
        axes[1, c].set_title(f'atención {a:.2f}', fontsize=10)
        for r in (0, 1):
            axes[r, c].axis('off')
    axes[0, 0].text(-0.16, 0.5, 'RM original', transform=axes[0, 0].transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axes[1, 0].text(-0.16, 0.5, 'Con saliencia', transform=axes[1, 0].transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.suptitle(f'Plano sagital — cortes originales y mapas de saliencia superpuestos (caso {CASE})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    out2 = figs_dir / 'app_heatmaps_sorted.png'
    fig.savefig(out2, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out2)


if __name__ == '__main__':
    main()
