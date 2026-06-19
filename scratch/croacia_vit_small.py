#!/usr/bin/env python3
"""
Validación externa zero-shot del modelo SAGITAL ViT-Small en el conjunto de Croacia
(917 estudios), para reemplazar la fila de ViT-Base en la tabla del TFG.
Reproduce el pipeline: OptimizedMRNetDataset sobre los volúmenes croatas + índices
sagitales precalculados; umbral del ViT-Small (0.3804).
"""
import sys, os, tempfile
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
VOL = DATA/'croatia_npy_volumes_full'           # volúmenes completos (32 cortes)
META = DATA/'croatia_npy_volumes_final'/'metadata_final.csv'  # etiquetas
IDX = DATA/'slice_indices_final'/'croatia_sagittal_indices_full.json'
CKPT = BASE/'checkpoints/vit_small_multiseed/seed_43/best_sagittal_seed43_auc.pth'
THR = 0.3804


def main():
    meta = pd.read_csv(META)
    # case = volume_id, label = acl_binary_code
    meta['case'] = meta['volume_id'].astype(int)
    meta['label'] = meta['acl_binary_code'].astype(int)
    lab = meta.sort_values('case')['label'].values.astype(int)
    print(f'Croacia: {len(meta)} estudios, {int(lab.sum())} roto / {int((1-lab).sum()) if lab.dtype!=bool else 0} no roto')

    # CSV temporal sin header (case,label) y symlink data_root/sagittal -> VOL
    tmp = Path(tempfile.mkdtemp())
    csv = tmp/'croacia.csv'
    meta.sort_values('case')[['case','label']].to_csv(csv, index=False, header=False)
    (tmp/'sagittal').symlink_to(VOL)

    ds = OptimizedMRNetDataset(csv_path=str(csv), data_root=str(tmp), plane='sagittal',
                               indices_cache_path=str(IDX), transform=get_val_test_transform())
    ld = DataLoader(ds, batch_size=16, shuffle=False)

    pm, di, dd = ('attention', 0.2, 0.15)
    m = ViTSmallMultiSliceClassifier(pretrained=True, pooling_mode=pm, dropout_input=di, dropout_dense=dd)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False); sd = sd.get('model_state_dict', sd)
    m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False); m.to(DEVICE).eval()

    probs, labels = [], []
    with torch.no_grad():
        for b in ld:
            o = m(b[0].to(DEVICE)); lg = o[0] if isinstance(o, tuple) else o
            probs.extend(torch.sigmoid(lg).cpu().numpy().flatten())
            labels.extend(b[1].numpy().flatten())
    probs = np.array(probs); labels = np.array(labels).astype(int)

    pred = (probs >= THR).astype(int)
    tp=int(((pred==1)&(labels==1)).sum()); fp=int(((pred==1)&(labels==0)).sum())
    fn=int(((pred==0)&(labels==1)).sum()); tn=int(((pred==0)&(labels==0)).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
    acc=(tp+tn)/len(labels)
    print(f'\n=== ViT-Small SAGITAL en Croacia (zero-shot, umbral={THR}) ===')
    print(f'  AUC={roc_auc_score(labels,probs):.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  Esp={spec:.4f}  F1={f1:.4f}')
    print(f'  Matriz: VP={tp}  FN={fn}  FP={fp}  VN={tn}')


if __name__ == '__main__':
    main()
