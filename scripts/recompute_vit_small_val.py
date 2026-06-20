#!/usr/bin/env python3
"""
Recalcula las métricas del ENSAMBLE multi-vista de ViT-Small (22M, multiseed) en
el conjunto de VALIDACIÓN de MRNet, con el umbral clínico tau* (maximizar recall
con precisión >= 0.75). Mismo pipeline que las demás filas de la Tabla 3.6
(ResNet50/Swin-Tiny): media de las probabilidades de los 3 planos.

Sirve para fijar la fila ViT-Small de la tabla, cuyo run original no se encontró.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path('/home/palodo2/acl_classifier')
sys.path.insert(0, str(BASE))
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import ViTSmallMultiSliceClassifier

DEVICE = torch.device('cpu')
DATA = BASE / 'data'
CKPT_DIR = BASE / 'checkpoints' / 'vit_small_multiseed'
VAL_CSV = DATA / 'val-acl.csv'
PLANES = ['sagittal', 'coronal', 'axial']

# Config por plano (idéntica a la del notebook vit_small_multiseed)
PLANE_CONFIG = {
    'sagittal': {'pooling_mode': 'attention'},
    'coronal':  {'pooling_mode': 'max'},
    'axial':    {'pooling_mode': 'attention'},
}


def calc(labels, probs, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, spec=spec, f1=f1)


def find_threshold(labels, probs, min_precision=0.75):
    best_t, best_r = 0.5, -1.0
    for t in np.linspace(0, 1, 1000):
        m = calc(labels, probs, t)
        if m['prec'] >= min_precision and m['rec'] > best_r:
            best_r, best_t = m['rec'], t
    return best_t


def plane_probs(plane, split='val'):
    csv = VAL_CSV if split == 'val' else (DATA / 'test-acl.csv')
    root = DATA / ('val' if split == 'val' else 'test')
    ds = OptimizedMRNetDataset(
        csv_path=str(csv), data_root=str(root), plane=plane,
        indices_cache_path=str(DATA / 'slice_indices_final' / f'{split}_{plane}_indices.json'),
        transform=get_val_test_transform())
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    # pretrained=True para construir la config correcta de ViT-Small (hidden=384);
    # los pesos se sobreescriben luego con el checkpoint multiseed.
    model = ViTSmallMultiSliceClassifier(pretrained=True,
                                         pooling_mode=PLANE_CONFIG[plane]['pooling_mode']).to(DEVICE)
    ck = torch.load(CKPT_DIR / f'best_{plane}_multiseed_final.pth', map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [{plane}] missing keys: {len(missing)} (p.ej. {missing[:2]})')
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(batch[1].numpy())
    return np.array(probs), np.array(labels)


def ensemble(split):
    from sklearn.metrics import roc_auc_score
    per_plane, labels = {}, None
    for p in PLANES:
        pr, lb = plane_probs(p, split)
        per_plane[p] = pr
        labels = lb if labels is None else labels
    ens = (per_plane['sagittal'] + per_plane['coronal'] + per_plane['axial']) / 3.0
    return ens, labels, roc_auc_score(labels, ens)


def report(split, ens, labels, thr):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, ens)
    m = calc(labels, ens, thr)
    print(f'\n==== ViT-Small ENSAMBLE ({split} MRNet, n={len(labels)}) | tau*={thr:.4f} ====')
    print(f'AUC {auc:.4f}  Acc {m["acc"]:.4f}  Prec {m["prec"]:.4f}  Recall {m["rec"]:.4f}  '
          f'Esp {m["spec"]:.4f}  F1 {m["f1"]:.4f}')
    print(f'VP {m["tp"]}  FN {m["fn"]}  FP {m["fp"]}  VN {m["tn"]}')


def main():
    print('Calculando ensamble de validación...')
    val_ens, val_lb, _ = ensemble('val')
    thr = find_threshold(val_lb, val_ens, 0.75)   # umbral fijado en validación
    report('validación', val_ens, val_lb, thr)
    print('\nCalculando ensamble de test...')
    test_ens, test_lb, _ = ensemble('test')
    report('test', test_ens, test_lb, thr)         # mismo umbral aplicado a test


if __name__ == '__main__':
    main()
