#!/usr/bin/env python3
"""
Genera una figura COMPARATIVA de explicabilidad para el anexo del TFG, usando
EXACTAMENTE las mismas funciones que la aplicacion web
(scripts/predict_patient.py), de modo que cada arquitectura emplea la tecnica de
interpretabilidad que la app aplica realmente:

  - ResNet50 (CNN) -> Grad-CAM sobre la ultima capa convolucional (layer4),
                      via generate_saliency_overlay (rama isinstance CNN).
  - ViT-Base       -> atencion espacial del token [CLS] de la ultima capa del
                      encoder, via generate_vit_attention_overlay.
  - Swin-Tiny      -> saliencia por gradiente de entrada,
                      via generate_saliency_overlay (rama else).

Se muestran los mismos 5 cortes sagitales del mismo caso de validacion para las
tres arquitecturas. Se ejecuta en CPU para no interferir con entrenamientos.

Salida (en memoria/tfgs/figs/outputs/):
  - explicabilidad_3modelos_sagital.png
"""

import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))

from src.models import (
    CNNMultiSliceClassifier,
    ViTMultiSliceClassifier,
    ViTSmallMultiSliceClassifier,
    SwinMultiSliceClassifier,
)
# Funciones de explicabilidad TAL CUAL las usa la app:
from predict_patient import (
    generate_saliency_overlay,
    generate_vit_attention_overlay,
)

DEVICE = torch.device('cpu')  # GPU ocupada por entrenamiento
CASE = '0105'                 # caso positivo (rotura LCA) del set de validacion
PLANE = 'sagittal'
N_SHOW = 5

# Mismo transform que la app (scripts/predict_patient.py:583)
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# (etiqueta, clase, ruta de checkpoint relativa, funcion de overlay, pretrained)
MODELS = [
    ('ResNet50\n(Grad-CAM)',  CNNMultiSliceClassifier,  f'cnn_multiseed_resnet50/best_{PLANE}_multiseed_final.pth', generate_saliency_overlay, False),
    ('ViT-Small\n(atención)', ViTSmallMultiSliceClassifier, f'vit_small_multiseed/seed_43/best_{PLANE}_seed43_auc.pth', generate_vit_attention_overlay, True),
    ('Swin-Tiny\n(saliencia)', SwinMultiSliceClassifier, f'swin_tiny_multiseed/best_{PLANE}_multiseed_final.pth', generate_saliency_overlay, False),
]


def load_model(cls, ckpt_rel, pretrained=False):
    model = cls(pretrained=pretrained).to(DEVICE)
    ck = torch.load(BASE / 'checkpoints' / ckpt_rel, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_slices():
    vol = np.load(BASE / 'data' / 'val' / PLANE / f'{CASE}.npy').astype(np.float32)
    with open(BASE / 'data' / 'slice_indices_final' / f'val_{PLANE}_indices.json') as f:
        idxs = json.load(f)[str(int(CASE))]
    sel = vol[idxs]
    lo, hi = sel.min(), sel.max()
    sel = (sel - lo) / (hi - lo + 1e-8) * 255.0
    return sel.astype(np.uint8), idxs


def attention_over_slices(model, slices_u8):
    tensors = [model_transform(Image.fromarray(s).convert('RGB')) for s in slices_u8]
    x = torch.stack(tensors).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, attn = model(x)
    return (attn.squeeze(0).numpy() if attn is not None
            else np.full(len(slices_u8), 1.0 / len(slices_u8)))


def overlay_rgba(b64_png, size):
    """Decodifica el PNG base64 que devuelve la app -> array RGBA en [0,1]."""
    img = Image.open(io.BytesIO(base64.b64decode(b64_png))).convert('RGBA')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def gray(slice_u8, size):
    return np.array(Image.fromarray(slice_u8).convert('L').resize((size, size), Image.BILINEAR))


# Opacidad por defecto del visor de la app (App.jsx:333 -> heatmapOpacity = 0.35)
HEATMAP_OPACITY = 0.35


def compose_like_app(gray_2d, rgba):
    """Replica la composicion del frontend de la app:
    overlay con mix-blend-mode: screen y opacity 0.35 sobre la RM en escala de
    grises. El modo 'screen' aclara y nunca oscurece, por lo que las zonas
    oscuras del colormap viridis desaparecen y solo destacan las activaciones."""
    base = (gray_2d.astype(np.float32) / 255.0)[..., None].repeat(3, axis=2)  # (H,W,3)
    over_rgb = rgba[..., :3]
    eff_a = (rgba[..., 3] * HEATMAP_OPACITY)[..., None]                        # alfa efectivo
    screen = 1.0 - (1.0 - base) * (1.0 - over_rgb)                             # blend 'screen'
    out = base * (1.0 - eff_a) + screen * eff_a
    return np.clip(out, 0.0, 1.0)


def main():
    figs_dir = BASE / 'memoria' / 'tfgs' / 'figs' / 'outputs'
    slices, idxs = get_slices()
    SIZE = 256  # la app emite overlays de 256x256

    # 5 cortes de referencia: top-5 por atencion del Swin (coherencia con la
    # figura del cuerpo principal), aplicados a las tres arquitecturas.
    ref = load_model(SwinMultiSliceClassifier, f'swin_tiny_multiseed/best_{PLANE}_multiseed_final.pth')
    attn_ref = attention_over_slices(ref, slices)
    del ref
    order = sorted(np.argsort(attn_ref)[::-1][:N_SHOW].tolist())
    show_idxs = [idxs[j] for j in order]

    rows = []
    for label, cls, ckpt_rel, overlay_fn, pretrained in MODELS:
        print(f'[{label}] cargando modelo y calculando mapas...')
        model = load_model(cls, ckpt_rel, pretrained)
        probs, cells = [], []
        for j in order:
            b64, p = overlay_fn(model, slices[j], model_transform, DEVICE)
            cells.append((gray(slices[j], SIZE), overlay_rgba(b64, SIZE)))
            probs.append(p)
        rows.append((label, float(np.mean(probs)), cells))
        del model

    fig, axes = plt.subplots(len(MODELS), N_SHOW, figsize=(15, 9.3))
    for r, (label, prob, cells) in enumerate(rows):
        for c in range(N_SHOW):
            ax = axes[r, c]
            g, rgba = cells[c]
            ax.imshow(compose_like_app(g, rgba))
            if r == 0:
                ax.set_title(f'corte {show_idxs[c]}', fontsize=10)
            ax.axis('off')
        axes[r, 0].text(-0.20, 0.5, label, transform=axes[r, 0].transAxes,
                        ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.suptitle(f'Explicabilidad por arquitectura — plano sagital, '
                 f'caso {CASE} de validación (rotura de LCA)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    out = figs_dir / 'explicabilidad_3modelos_sagital.png'
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out)


if __name__ == '__main__':
    main()
