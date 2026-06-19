#!/usr/bin/env python3
"""
Genera las figuras del ViT-Small en el formato del TFG, para reemplazar las de
ViT-Base en figs/outputs/:
  - VIT_{plane}_entrenamiento_multiseed.png : 3 paneles (Loss, AUC, Overfitting), 10 seeds
  - VIT_{plane}_best.png                    : curva del mejor checkpoint del plano
  - VIT_ROC_val.png / VIT_ROC_test.png      : ROC del ensamble en val y test
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import ViTSmallMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = Path('/home/palodo2/acl_classifier'); DATA = BASE/'data'
CKDIR = BASE/'checkpoints/vit_small_multiseed'
OUT = BASE/'TFG/tfgs/figs/outputs'
PLANES = ['sagittal', 'coronal', 'axial']
PLANE_ES = {'sagittal':'SAGITTAL','coronal':'CORONAL','axial':'AXIAL'}
cfg = {'sagittal': ('attention',0.2,0.15), 'coronal': ('max',0.25,0.15), 'axial': ('attention',0.25,0.15)}
SEEDS = list(range(42,52))


def histories(plane):
    hs = {}
    for s in SEEDS:
        ck = torch.load(CKDIR/f'seed_{s}'/f'best_{plane}_seed{s}_auc.pth', map_location='cpu', weights_only=False)
        hs[s] = (ck['history'], ck['val_auc'])
    return hs


def pad(arrs):
    L = max(len(a) for a in arrs)
    M = np.full((len(arrs), L), np.nan)
    for i,a in enumerate(arrs): M[i,:len(a)] = a
    return M


def fig_multiseed(plane):
    hs = histories(plane)
    fig, ax = plt.subplots(1, 3, figsize=(20,6))
    fig.suptitle(f'{PLANE_ES[plane]} - Training vs Validation Comparison (All 10 Seeds)', fontsize=16, fontweight='bold')
    tl = pad([h['train_loss'] for h,_ in hs.values()]); vl = pad([h['val_loss_objective'] for h,_ in hs.values()])
    ta = pad([h['train_auc'] for h,_ in hs.values()]); va = pad([h['val_auc'] for h,_ in hs.values()])
    # Panel 1: Loss
    for h,_ in hs.values():
        ax[0].plot(h['train_loss'], alpha=0.2, lw=1, color='#1f77b4'); ax[0].plot(h['val_loss_objective'], alpha=0.2, lw=1, color='#ff7f0e')
    x=np.arange(tl.shape[1])
    ax[0].plot(x, np.nanmean(tl,0), color='#1f77b4', lw=2.5, label='Train Mean ± Std')
    ax[0].fill_between(x, np.nanmean(tl,0)-np.nanstd(tl,0), np.nanmean(tl,0)+np.nanstd(tl,0), color='#1f77b4', alpha=0.2)
    x=np.arange(vl.shape[1])
    ax[0].plot(x, np.nanmean(vl,0), color='#ff7f0e', lw=2.5, label='Val Mean ± Std')
    ax[0].fill_between(x, np.nanmean(vl,0)-np.nanstd(vl,0), np.nanmean(vl,0)+np.nanstd(vl,0), color='#ff7f0e', alpha=0.2)
    ax[0].set_title('Loss Comparison'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
    # Panel 2: AUC
    for h,_ in hs.values():
        ax[1].plot(h['train_auc'], alpha=0.2, lw=1, color='#2ca02c'); ax[1].plot(h['val_auc'], alpha=0.2, lw=1, color='#d62728')
    x=np.arange(ta.shape[1])
    ax[1].plot(x, np.nanmean(ta,0), color='#2ca02c', lw=2.5, label='Train Mean ± Std')
    ax[1].fill_between(x, np.nanmean(ta,0)-np.nanstd(ta,0), np.nanmean(ta,0)+np.nanstd(ta,0), color='#2ca02c', alpha=0.2)
    x=np.arange(va.shape[1])
    ax[1].plot(x, np.nanmean(va,0), color='#d62728', lw=2.5, label='Val Mean ± Std')
    ax[1].fill_between(x, np.nanmean(va,0)-np.nanstd(va,0), np.nanmean(va,0)+np.nanstd(va,0), color='#d62728', alpha=0.2)
    ax[1].set_title('AUC Comparison'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('AUC'); ax[1].legend(); ax[1].grid(alpha=0.3)
    # Panel 3: Overfitting (gap train-val AUC)
    gap = np.nanmean(ta,0)[:va.shape[1]] - np.nanmean(va,0)
    colors = ['#2ca02c' if g<=0 else '#d62728' for g in gap]
    ax[2].bar(np.arange(len(gap)), gap, color=colors, alpha=0.7)
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set_title(f'Overfitting Analysis\nMin: {np.nanmin(gap):.4f}  Max: {np.nanmax(gap):.4f}  Avg: {np.nanmean(gap):.4f}')
    ax[2].set_xlabel('Epoch'); ax[2].set_ylabel('AUC Gap (Train - Val)'); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT/f'VIT_{plane}_entrenamiento_multiseed.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # best checkpoint del plano
    best_s = max(hs.items(), key=lambda kv: kv[1][1])[0]
    h,_ = hs[best_s]
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle(f'{PLANE_ES[plane]} - Mejor modelo (seed {best_s})', fontsize=14, fontweight='bold')
    ax[0].plot(h['train_loss'], label='Train', lw=2, color='#1f77b4'); ax[0].plot(h['val_loss_objective'], label='Val', lw=2, color='#ff7f0e')
    ax[0].set_title('Loss'); ax[0].set_xlabel('Epoch'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(h['train_auc'], label='Train', lw=2, color='#2ca02c'); ax[1].plot(h['val_auc'], label='Val', lw=2, color='#d62728')
    be = int(np.argmax(h['val_auc']))
    ax[1].axvline(be, ls='--', color='red', label=f'Best ({be+1})'); ax[1].scatter([be],[h['val_auc'][be]], color='red', zorder=5)
    ax[1].set_title('AUC'); ax[1].set_xlabel('Epoch'); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT/f'VIT_{plane}_best.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {plane}: curvas + best (seed {best_s}) OK')


def predict(model, loader):
    model.eval(); probs=[]
    with torch.no_grad():
        for b in loader:
            out=model(b[0].to(DEVICE)); lg=out[0] if isinstance(out,tuple) else out
            probs.extend(torch.sigmoid(lg).cpu().numpy().flatten())
    return np.array(probs)


def fig_roc():
    vlab = pd.read_csv(DATA/'val-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tlab = pd.read_csv(DATA/'test-acl.csv', header=None, names=['c','l'])['l'].values.astype(int)
    tf = get_val_test_transform()
    labs = {'val': vlab, 'test': tlab}
    # predicciones por plano (mejor checkpoint) y ensamble
    per_plane = {'val': {}, 'test': {}}
    ens = {'val': np.zeros(len(vlab)), 'test': np.zeros(len(tlab))}
    for plane in PLANES:
        hs={s:torch.load(CKDIR/f'seed_{s}'/f'best_{plane}_seed{s}_auc.pth',map_location='cpu',weights_only=False)['val_auc'] for s in SEEDS}
        bs=max(hs,key=hs.get)
        pm,di,dd=cfg[plane]
        m=ViTSmallMultiSliceClassifier(pretrained=True,pooling_mode=pm,dropout_input=di,dropout_dense=dd)
        sd=torch.load(CKDIR/f'seed_{bs}'/f'best_{plane}_seed{bs}_auc.pth',map_location='cpu',weights_only=False)
        sd=sd.get('model_state_dict',sd); m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()},strict=False); m.to(DEVICE)
        for split,root,csv in [('val','val','val-acl.csv'),('test','test','test-acl.csv')]:
            ds=OptimizedMRNetDataset(csv_path=str(DATA/csv),data_root=str(DATA/root),plane=plane,
                indices_cache_path=str(DATA/'slice_indices_final'/f'{split}_{plane}_indices.json'),transform=tf)
            pr=predict(m,DataLoader(ds,batch_size=16,shuffle=False))
            per_plane[split][plane]=pr; ens[split]=ens[split]+pr
        del m
    ens={k:v/3 for k,v in ens.items()}
    colors={'sagittal':'#1f77b4','coronal':'#ff7f0e','axial':'#2ca02c'}
    for split in ['val','test']:
        lab=labs[split]
        fig,ax=plt.subplots(figsize=(10,8))
        for plane in PLANES:
            fpr,tpr,_=roc_curve(lab,per_plane[split][plane]); a=roc_auc_score(lab,per_plane[split][plane])
            ax.plot(fpr,tpr,lw=2,color=colors[plane],label=f'{plane.upper()} (AUC={a:.4f})')
        fpr,tpr,_=roc_curve(lab,ens[split]); a=roc_auc_score(lab,ens[split])
        ax.plot(fpr,tpr,lw=3,ls='--',color='darkgreen',label=f'ENSEMBLE (AUC={a:.4f})')
        ax.plot([0,1],[0,1],ls='--',color='gray',lw=1,label='Random (AUC=0.5)')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {"Validation" if split=="val" else "Test"} Set', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT/f'VIT_ROC_{split}.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f'  ROC {split}: ensemble AUC={a:.4f} OK')


def main():
    print('Generando curvas multiseed + best:')
    for p in PLANES: fig_multiseed(p)
    print('Generando ROC:')
    fig_roc()
    print('Listo.')


if __name__ == '__main__':
    main()
