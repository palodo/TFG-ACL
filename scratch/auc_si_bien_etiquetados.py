#!/usr/bin/env python3
"""
Calcula cuánto subiría el AUC de test de cada ensamble (ResNet50, ViT-Base,
Swin-Tiny) si los 3 falsos negativos comunes (0080, 0087, 0521) estuvieran
"bien etiquetados", es decir, si su etiqueta real fuese negativa (sin rotura).

Reproduce el ensamble multi-vista del TFG (promedio de los 3 planos, mejor
checkpoint por plano) y compara AUC original vs AUC con esos 3 reetiquetados.
Se ejecuta en CPU para no interferir con entrenamientos en GPU.
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
BASE = Path('/home/palodo2/acl_classifier')
DATA = BASE / 'data'
CK = BASE / 'checkpoints'
TEST_CSV = DATA / 'test-acl.csv'
PLANES = ['sagittal', 'coronal', 'axial']
FLIP = ['0080', '0087', '0521']

DIRS = {'cnn': CK/'cnn_multiseed_resnet50', 'vit': CK/'vit_multiseed_base', 'swin': CK/'swin_small_multiseed'}
NAMES = {'cnn': 'ResNet50', 'vit': 'ViT-Base', 'swin': 'Swin-Tiny'}
vit_cfg = {'sagittal': ('attention',0.3,0.2), 'coronal': ('max',0.35,0.25), 'axial': ('attention',0.35,0.25)}
cnn_cfg = {'sagittal': ('attention',0.42,0.32), 'coronal': ('max',0.47,0.37), 'axial': ('attention',0.42,0.32)}


def predict(model, loader):
    model.eval()
    probs, cids = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch[0].to(DEVICE))
            logits = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            cids.extend(batch[2])
    return np.array(probs), cids


def load(model_key, plane):
    f = DIRS[model_key] / f'best_{plane}_multiseed_final.pth'
    ck = torch.load(str(f), map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    if model_key == 'swin':
        m = SwinMultiSliceClassifier(pretrained=False)
    elif model_key == 'vit':
        pm, di, dd = vit_cfg[plane]; m = ViTMultiSliceClassifier(pretrained=False, pooling_mode=pm, dropout_input=di, dropout_dense=dd)
    else:
        pm, di, dd = cnn_cfg[plane]; m = CNNMultiSliceClassifier(pretrained=False, pooling_mode=pm, dropout_input=di, dropout_dense=dd)
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE)


def main():
    df = pd.read_csv(TEST_CSV, header=None, names=['case', 'label'])
    labels = df['label'].values.astype(int)
    case_ids = df['case'].apply(lambda x: f'{x:04d}').values.tolist()
    tf = get_val_test_transform()

    preds = {k: {} for k in DIRS}
    for plane in PLANES:
        ds = OptimizedMRNetDataset(csv_path=str(TEST_CSV), data_root=str(DATA/'test'),
                                   plane=plane, indices_cache_path=str(DATA/'slice_indices_final'/f'test_{plane}_indices.json'),
                                   transform=tf)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        for k in DIRS:
            print(f'  [{plane}] {NAMES[k]}...', flush=True)
            m = load(k, plane)
            p, cids = predict(m, loader)
            preds[k][plane] = p
            del m
        order = cids  # case order from loader

    # etiquetas reetiquetadas (los 3 casos -> negativo)
    labels_fixed = labels.copy()
    flip_idx = [case_ids.index(c) for c in FLIP]
    for i in flip_idx:
        labels_fixed[i] = 0

    n_pos = int(labels.sum())
    print(f'\nTest: {len(labels)} casos, {n_pos} positivos, {len(labels)-n_pos} negativos')
    print(f'Reetiquetando {FLIP} de positivo -> negativo\n')
    print(f"{'Modelo':<12}{'AUC actual':<14}{'AUC corregido':<16}{'Δ AUC':<10}")
    print('-'*52)
    for k in ['cnn', 'vit', 'swin']:
        ens = (preds[k]['sagittal'] + preds[k]['coronal'] + preds[k]['axial']) / 3.0
        auc0 = roc_auc_score(labels, ens)
        auc1 = roc_auc_score(labels_fixed, ens)
        print(f'{NAMES[k]:<12}{auc0:<14.4f}{auc1:<16.4f}{auc1-auc0:+.4f}')
        # probs de los 3 casos
        for c, i in zip(FLIP, flip_idx):
            pass
    print('\n(Δ = cuánto sube el AUC si esos 3 casos fueran realmente negativos)')


if __name__ == '__main__':
    main()
