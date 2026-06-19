#!/usr/bin/env python3
"""
Calcula TODAS las métricas del ViT-Small multiseed en el formato de las tablas del
TFG, para reemplazar a ViT-Base como modelo principal de la familia ViT:
  (A) Tabla multiseed: AUC medio/std/min/max y mejor semilla por plano.
  (B) Tabla individual: AUC val y test del mejor checkpoint por plano.
  (C) Tabla ensamble: AUC, precision, recall, specificity, F1, accuracy en val y test + tau*.
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
cfg = {'sagittal': ('attention',0.2,0.15), 'coronal': ('max',0.25,0.15), 'axial': ('attention',0.25,0.15)}


def predict(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for b in loader:
            out = model(b[0].to(DEVICE)); lg = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(lg).cpu().numpy().flatten())
    return np.array(probs)


def full_metrics(labels, probs, thr):
    pred = (probs >= thr).astype(int)
    tp=int(((pred==1)&(labels==1)).sum()); fp=int(((pred==1)&(labels==0)).sum())
    fn=int(((pred==0)&(labels==1)).sum()); tn=int(((pred==0)&(labels==0)).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
    acc=(tp+tn)/len(labels)
    return dict(auc=roc_auc_score(labels,probs), acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, cm=(tn,fp,fn,tp))


def find_threshold(labels, probs, mp=0.75):
    bt, br = 0.5, -1
    for t in np.linspace(0,1,1000):
        pred=(probs>=t).astype(int)
        tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        if prec>=mp and rec>br: br,bt=rec,t
    return bt


def load(plane, ckpt):
    pm,di,dd = cfg[plane]
    m = ViTSmallMultiSliceClassifier(pretrained=True, pooling_mode=pm, dropout_input=di, dropout_dense=dd)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False); sd = sd.get('model_state_dict', sd)
    m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False)
    return m.to(DEVICE)


def main():
    vlab = pd.read_csv(DATA/'val-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tlab = pd.read_csv(DATA/'test-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tf = get_val_test_transform()

    # (A) multiseed + mejor semilla
    print('='*60); print('(A) TABLA MULTISEED ViT-Small (val_auc por semilla)'); print('='*60)
    best = {}
    for p in PLANES:
        aucs=[]
        for s in range(42,52):
            va=torch.load(CKDIR/f'seed_{s}'/f'best_{p}_seed{s}_auc.pth',map_location='cpu',weights_only=False)['val_auc']
            aucs.append((s,va))
        arr=np.array([a for _,a in aucs]); bs=max(aucs,key=lambda x:x[1])
        best[p]=bs[0]
        print(f'  {p:9s} media={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} mejor_seed={bs[0]}')

    # inferencia mejor checkpoint por plano
    ens={'val':np.zeros(len(vlab)),'test':np.zeros(len(tlab))}
    print('\n'+'='*60); print('(B) TABLA INDIVIDUAL por plano (mejor checkpoint)'); print('='*60)
    for p in PLANES:
        ckpt=CKDIR/f'seed_{best[p]}'/f'best_{p}_seed{best[p]}_auc.pth'
        m=load(p,ckpt)
        pv={}
        for split,root,csv in [('val','val','val-acl.csv'),('test','test','test-acl.csv')]:
            ds=OptimizedMRNetDataset(csv_path=str(DATA/csv),data_root=str(DATA/root),plane=p,
                indices_cache_path=str(DATA/'slice_indices_final'/f'{split}_{p}_indices.json'),transform=tf)
            pr=predict(m,DataLoader(ds,batch_size=16,shuffle=False)); pv[split]=pr; ens[split]=ens[split]+pr
        del m
        av=roc_auc_score(vlab,pv['val']); at=roc_auc_score(tlab,pv['test'])
        print(f'  {p:9s} (seed_{best[p]})  AUC val={av:.4f}  AUC test={at:.4f}')
    ens={k:v/3.0 for k,v in ens.items()}

    # (C) ensamble val + test
    thr=find_threshold(vlab,ens['val'],0.75)
    print('\n'+'='*60); print(f'(C) TABLA ENSAMBLE ViT-Small  (tau*={thr:.4f})'); print('='*60)
    for split,lab in [('val',vlab),('test',tlab)]:
        m=full_metrics(lab,ens[split],thr)
        tn,fp,fn,tp=m['cm']
        print(f"  {split.upper():4s}  AUC={m['auc']:.4f} Acc={m['acc']:.4f} Prec={m['prec']:.4f} "
              f"Rec={m['rec']:.4f} Spec={m['spec']:.4f} F1={m['f1']:.4f}  CM[tn={tn},fp={fp},fn={fn},tp={tp}]")


if __name__ == '__main__':
    main()
