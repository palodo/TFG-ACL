#!/usr/bin/env python3
"""
Genera la figura de explicabilidad (Grad-CAM / saliencia por gradiente) para los
TRES falsos negativos comunes del conjunto de test (casos 0080, 0087, 0521),
plano sagital, con el modelo Swin-Tiny multi-semilla usado en la memoria.

Mismo algoritmo de saliencia que la app (generate_saliency_overlay).
Salida: TFG/tfgs/figs/outputs/fn_comunes_sagital_gradcam.png

Se ejecuta en CPU para no interferir con la GPU.
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

BASE = Path('/home/palodo2/acl_classifier')
sys.path.insert(0, str(BASE))
from src.models import SwinMultiSliceClassifier

DEVICE = torch.device('cpu')
PLANE = 'sagittal'
CASES = ['0080', '0087', '0521']
N_SHOW = 4
CKPT = BASE / 'checkpoints' / 'swin_multiseed_tiny' / f'best_{PLANE}_multiseed_final.pth'

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


def load_model():
    model = SwinMultiSliceClassifier(pretrained=False).to(DEVICE)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def saliency_overlay(model, slice_2d):
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
    saliency = F.avg_pool2d(saliency, kernel_size=9, stride=1, padding=4)
    saliency = F.interpolate(saliency, size=slice_rgb.shape[-2:], mode='bilinear', align_corners=False)
    sal_2d = saliency[0, 0].cpu().numpy()
    sal = robust_unit_scale(sal_2d, 1, 99)
    sal = np.power(sal, 0.55)
    # Centroide ponderado de la región de mayor saliencia (sin pintar el mapa)
    thr = np.quantile(sal, 0.85)
    mask = sal >= thr
    ys, xs = np.nonzero(mask)
    w = sal[ys, xs] + 1e-8
    cx = float(np.average(xs, weights=w))
    cy = float(np.average(ys, weights=w))
    rx = float(np.sqrt(np.average((xs - cx) ** 2, weights=w)))
    ry = float(np.sqrt(np.average((ys - cy) ** 2, weights=w)))
    radius = float(np.clip(1.8 * max(rx, ry), 26.0, 64.0))
    prob = float(torch.sigmoid(score).detach().item())
    return (cx, cy, radius), prob


def get_slices(case):
    vol = np.load(BASE / 'data' / 'test' / PLANE / f'{case}.npy').astype(np.float32)
    with open(BASE / 'data' / 'slice_indices_final' / f'test_{PLANE}_indices.json') as f:
        idxs = json.load(f)[str(int(case))]
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
    out_dir = BASE / 'TFG' / 'tfgs' / 'figs' / 'outputs'
    model = load_model()
    rows = []
    for case in CASES:
        slices, idxs = get_slices(case)
        attn, prob = attention_weights(model, slices)
        order = sorted(np.argsort(attn)[::-1][:N_SHOW].tolist())
        cells = []
        for j in order:
            circle, sp = saliency_overlay(model, slices[j])
            cells.append((gray224(slices[j]), circle, idxs[j], attn[j]))
        rows.append((case, cells, prob))
        print(f'caso {case}: p_swin={prob:.3f}')

    fig, axes = plt.subplots(len(CASES), N_SHOW, figsize=(3.0 * N_SHOW, 3.0 * len(CASES)))
    for r, (case, cells, prob) in enumerate(rows):
        for c in range(N_SHOW):
            ax = axes[r, c]
            g, (cx, cy, radius), idx, a = cells[c]
            ax.imshow(g, cmap='gray')
            ax.add_patch(plt.Circle((cx, cy), radius, fill=False,
                                    edgecolor='#00e5ff', linewidth=2.2, alpha=0.95))
            ax.set_title(f'corte {idx}  |  atención {a:.2f}', fontsize=9)
            ax.axis('off')
        axes[r, 0].text(-0.20, 0.5, f'Caso {case}\n(p={prob:.2f})',
                        transform=axes[r, 0].transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold', rotation=90, color='#b00020')
    fig.suptitle('Región de mayor saliencia (Swin-Tiny, plano sagital) sobre los tres falsos negativos comunes del test',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    out = out_dir / 'fn_comunes_sagital_gradcam.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out)


if __name__ == '__main__':
    main()
