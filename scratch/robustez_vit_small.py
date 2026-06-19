#!/usr/bin/env python3
"""Estabilidad de tau* del ViT-Small entre las 10 semillas (ensamble val por semilla)."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import ViTSmallMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = Path('/home/palodo2/acl_classifier'); DATA = BASE/'data'
CKDIR = BASE/'checkpoints/vit_small_multiseed'
PLANES = ['sagittal','coronal','axial']
cfg = {'sagittal':('attention',0.2,0.15),'coronal':('max',0.25,0.15),'axial':('attention',0.25,0.15)}


def predict(m, loader):
    m.eval(); pr=[]
    with torch.no_grad():
        for b in loader:
            o=m(b[0].to(DEVICE)); lg=o[0] if isinstance(o,tuple) else o
            pr.extend(torch.sigmoid(lg).cpu().numpy().flatten())
    return np.array(pr)


def find_threshold(lab, probs, mp=0.75):
    bt,br=0.5,-1
    for t in np.linspace(0,1,1000):
        pred=(probs>=t).astype(int)
        tp=((pred==1)&(lab==1)).sum(); fp=((pred==1)&(lab==0)).sum(); fn=((pred==0)&(lab==1)).sum()
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        if prec>=mp and rec>br: br,bt=rec,t
    return bt


def main():
    vlab=pd.read_csv(DATA/'val-acl.csv',header=None,names=['c','l'])['l'].values.astype(int)
    tf=get_val_test_transform()
    loaders={p:DataLoader(OptimizedMRNetDataset(csv_path=str(DATA/'val-acl.csv'),data_root=str(DATA/'val'),
        plane=p,indices_cache_path=str(DATA/'slice_indices_final'/f'val_{p}_indices.json'),transform=tf),
        batch_size=16,shuffle=False) for p in PLANES}
    thrs=[]
    for s in range(42,52):
        ens=np.zeros(len(vlab))
        for p in PLANES:
            pm,di,dd=cfg[p]
            m=ViTSmallMultiSliceClassifier(pretrained=True,pooling_mode=pm,dropout_input=di,dropout_dense=dd)
            sd=torch.load(CKDIR/f'seed_{s}'/f'best_{p}_seed{s}_auc.pth',map_location='cpu',weights_only=False)
            sd=sd.get('model_state_dict',sd); m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()},strict=False); m.to(DEVICE)
            ens=ens+predict(m,loaders[p]); del m
        ens/=3
        thrs.append(find_threshold(vlab,ens,0.75))
    a=np.array(thrs)
    print(f'ViT-Small tau* por semilla: {[f"{t:.3f}" for t in thrs]}')
    print(f'media={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}  rango={a.max()-a.min():.4f}')


if __name__ == '__main__':
    main()
