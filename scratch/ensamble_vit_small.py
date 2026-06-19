#!/usr/bin/env python3
"""
Calcula el ensamble multi-vista del ViT-Small multiseed y lo compara con ViT-Base.
Selecciona el mejor checkpoint por plano (mayor AUC de validación), igual que el
protocolo del TFG, construye el ensamble (promedio de los 3 planos) y reporta AUC
en validación y test, además del umbral tau* y métricas en test.
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
from src.models import ViTSmallMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = Path('/home/palodo2/acl_classifier'); DATA = BASE/'data'
CKDIR = BASE/'checkpoints/vit_small_multiseed'
PLANES = ['sagittal', 'coronal', 'axial']
# pooling y dropout por plano (config del baseline ViT-Small)
cfg = {'sagittal': ('attention',0.2,0.15), 'coronal': ('max',0.25,0.15), 'axial': ('attention',0.25,0.15)}


def best_seed(plane):
    best, bauc = None, -1
    for s in range(42, 52):
        f = CKDIR/f'seed_{s}'/f'best_{plane}_seed{s}_auc.pth'
        va = torch.load(f, map_location='cpu', weights_only=False)['val_auc']
        if va > bauc:
            bauc, best = va, (s, f)
    return best[0], best[1], bauc


def predict(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for b in loader:
            out = model(b[0].to(DEVICE)); lg = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(lg).cpu().numpy().flatten())
    return np.array(probs)


def metrics_at(labels, probs, thr):
    pred = (probs >= thr).astype(int)
    tp=int(((pred==1)&(labels==1)).sum()); fp=int(((pred==1)&(labels==0)).sum())
    fn=int(((pred==0)&(labels==1)).sum()); tn=int(((pred==0)&(labels==0)).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(precision=prec, recall=rec, specificity=spec, f1=f1)


def find_threshold(labels, probs, mp=0.75):
    bt, br = 0.5, -1
    for t in np.linspace(0,1,1000):
        m = metrics_at(labels, probs, t)
        if m['precision'] >= mp and m['recall'] > br:
            br, bt = m['recall'], t
    return bt


def main():
    vlab = pd.read_csv(DATA/'val-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tlab = pd.read_csv(DATA/'test-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tf = get_val_test_transform()
    ens = {'val': np.zeros(len(vlab)), 'test': np.zeros(len(tlab))}

    print('Mejor checkpoint por plano (por AUC de validación):')
    for plane in PLANES:
        s, f, bauc = best_seed(plane)
        print(f'  {plane:9s} seed_{s}  (val_auc={bauc:.4f})')
        pm, di, dd = cfg[plane]
        m = ViTSmallMultiSliceClassifier(pretrained=True, pooling_mode=pm, dropout_input=di, dropout_dense=dd)
        sd = torch.load(f, map_location='cpu', weights_only=False); sd = sd.get('model_state_dict', sd)
        m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False); m.to(DEVICE)
        for split, root, csv in [('val','val','val-acl.csv'), ('test','test','test-acl.csv')]:
            ds = OptimizedMRNetDataset(csv_path=str(DATA/csv), data_root=str(DATA/root), plane=plane,
                indices_cache_path=str(DATA/'slice_indices_final'/f'{split}_{plane}_indices.json'), transform=tf)
            ens[split] = ens[split] + predict(m, DataLoader(ds, batch_size=16, shuffle=False))
        del m
    ens = {k: v/3.0 for k, v in ens.items()}

    auc_val = roc_auc_score(vlab, ens['val']); auc_test = roc_auc_score(tlab, ens['test'])
    thr = find_threshold(vlab, ens['val'], 0.75)
    mt = metrics_at(tlab, ens['test'], thr)
    print(f'\n=== ENSAMBLE ViT-Small (22M) ===')
    print(f'  AUC validación: {auc_val:.4f}')
    print(f'  AUC test:       {auc_test:.4f}')
    print(f'  tau*={thr:.4f}  ->  test: recall={mt["recall"]:.4f} prec={mt["precision"]:.4f} spec={mt["specificity"]:.4f} F1={mt["f1"]:.4f}')
    print(f'\n=== COMPARACIÓN ViT-Small vs ViT-Base (valores TFG) ===')
    print(f"  {'':16s}{'Params':<10}{'AUC val':<10}{'AUC test':<10}")
    print(f"  {'ViT-Small':16s}{'22 M':<10}{auc_val:<10.4f}{auc_test:<10.4f}")
    print(f"  {'ViT-Base (TFG)':16s}{'86 M':<10}{'0.9362':<10}{'0.9364':<10}")


if __name__ == '__main__':
    main()
