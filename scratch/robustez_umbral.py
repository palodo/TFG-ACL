#!/usr/bin/env python3
"""
Análisis de robustez del umbral de decisión tau* = argmax Recall s.t. Precision>=0.75.

Produce tres evidencias:
  (#1) Estabilidad de tau* entre las 10 semillas (ensamble por semilla, en validación).
  (#2) Sensibilidad al suelo de precisión (0.65..0.85): tau* en val -> métricas en test.
  (#3) Coherencia validación->test: precisión/recall reales en test con el tau* calibrado.

Reproduce la función de calibración del proyecto (scratch/find_cnn_threshold.py).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import SwinMultiSliceClassifier, ViTMultiSliceClassifier, CNNMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = Path('/home/palodo2/acl_classifier'); DATA = BASE/'data'; CK = BASE/'checkpoints'
PLANES = ['sagittal', 'coronal', 'axial']
DIRS = {'cnn': CK/'cnn_multiseed_resnet50', 'vit': CK/'vit_multiseed_base', 'swin': CK/'swin_small_multiseed'}
NAMES = {'cnn': 'ResNet50', 'vit': 'ViT-Base', 'swin': 'Swin-Tiny'}
vit_cfg = {'sagittal': ('attention',0.3,0.2), 'coronal': ('max',0.35,0.25), 'axial': ('attention',0.35,0.25)}
cnn_cfg = {'sagittal': ('attention',0.42,0.32), 'coronal': ('max',0.47,0.37), 'axial': ('attention',0.42,0.32)}


def build(model_key, plane):
    if model_key == 'swin':
        return SwinMultiSliceClassifier(pretrained=False)
    cfg = (vit_cfg if model_key == 'vit' else cnn_cfg)[plane]
    Cls = ViTMultiSliceClassifier if model_key == 'vit' else CNNMultiSliceClassifier
    return Cls(pretrained=False, pooling_mode=cfg[0], dropout_input=cfg[1], dropout_dense=cfg[2])


def load_path(model_key, plane, ckpt_path):
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m = build(model_key, plane); m.load_state_dict(sd, strict=False)
    return m.to(DEVICE)


def predict(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch[0].to(DEVICE)); logits = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.array(probs)


def metrics_at(labels, probs, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred==1)&(labels==1)).sum()); fp = int(((pred==1)&(labels==0)).sum())
    fn = int(((pred==0)&(labels==1)).sum()); tn = int(((pred==0)&(labels==0)).sum())
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec = tp/(tp+fn) if tp+fn>0 else 0.0
    spec = tn/(tn+fp) if tn+fp>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return dict(precision=prec, recall=rec, specificity=spec, f1=f1)


def find_threshold(labels, probs, min_precision=0.75):
    best_thr, best_rec = 0.5, -1
    for thr in np.linspace(0, 1, 1000):
        m = metrics_at(labels, probs, thr)
        if m['precision'] >= min_precision and m['recall'] > best_rec:
            best_rec = m['recall']; best_thr = thr
    return best_thr


def loaders():
    tf = get_val_test_transform(); out = {}
    for split, root, csv in [('val','val','val-acl.csv'), ('test','test','test-acl.csv')]:
        per_plane = {}
        for plane in PLANES:
            ds = OptimizedMRNetDataset(csv_path=str(DATA/csv), data_root=str(DATA/root), plane=plane,
                indices_cache_path=str(DATA/'slice_indices_final'/f'{split}_{plane}_indices.json'), transform=tf)
            per_plane[plane] = DataLoader(ds, batch_size=16, shuffle=False)
        out[split] = per_plane
    return out


def ensemble_probs(model_key, ckpt_fn, ld):
    """ckpt_fn(plane)->path. Devuelve dict split->probs ensamble (promedio 3 planos)."""
    res = {}
    for split in ['val', 'test']:
        acc = np.zeros(len(ld[split]['sagittal'].dataset))
        for plane in PLANES:
            m = load_path(model_key, plane, ckpt_fn(plane)); acc = acc + predict(m, ld[split][plane]); del m
        res[split] = acc/3.0
    return res


def main():
    vlab = pd.read_csv(DATA/'val-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tlab = pd.read_csv(DATA/'test-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    ld = loaders()

    print('#'*70); print('# (#1) ESTABILIDAD DE tau* ENTRE LAS 10 SEMILLAS (calibrado en validación)'); print('#'*70)
    seed_thr = {k: [] for k in DIRS}
    for k in DIRS:
        for s in range(42, 52):
            fn = lambda p, s=s, k=k: DIRS[k]/f'seed_{s}'/f'best_{p}_seed{s}_auc.pth'
            ep = ensemble_probs(k, fn, ld)
            seed_thr[k].append(find_threshold(vlab, ep['val'], 0.75))
        a = np.array(seed_thr[k])
        print(f'  {NAMES[k]:<10} tau*: media={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}  rango={a.max()-a.min():.4f}')

    print('\n'+'#'*70); print('# Ensamble PRINCIPAL (mejor checkpoint por plano) — para #2 y #3'); print('#'*70)
    main_ens = {}
    for k in DIRS:
        fn = lambda p, k=k: DIRS[k]/f'best_{p}_multiseed_final.pth'
        main_ens[k] = ensemble_probs(k, fn, ld)

    print('\n(#3) COHERENCIA validación -> test (tau* calibrado en val, min_prec=0.75)')
    print(f"  {'Modelo':<10}{'tau*':<9}{'Prec val':<10}{'Prec test':<11}{'Recall test':<12}")
    for k in DIRS:
        thr = find_threshold(vlab, main_ens[k]['val'], 0.75)
        mv = metrics_at(vlab, main_ens[k]['val'], thr); mt = metrics_at(tlab, main_ens[k]['test'], thr)
        print(f"  {NAMES[k]:<10}{thr:<9.4f}{mv['precision']:<10.4f}{mt['precision']:<11.4f}{mt['recall']:<12.4f}")

    print('\n(#2) SENSIBILIDAD AL SUELO DE PRECISIÓN (tau* en val -> métricas en test)')
    for k in DIRS:
        print(f'  {NAMES[k]}:')
        print(f"    {'min_prec':<10}{'tau*':<9}{'Recall test':<13}{'Prec test':<11}{'F1 test':<9}")
        for mp in [0.65, 0.70, 0.75, 0.80, 0.85]:
            thr = find_threshold(vlab, main_ens[k]['val'], mp)
            mt = metrics_at(tlab, main_ens[k]['test'], thr)
            print(f"    {mp:<10.2f}{thr:<9.4f}{mt['recall']:<13.4f}{mt['precision']:<11.4f}{mt['f1']:<9.4f}")


if __name__ == '__main__':
    main()
