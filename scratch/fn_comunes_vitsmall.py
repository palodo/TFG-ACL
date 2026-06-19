#!/usr/bin/env python3
"""
Recalcula los falsos negativos comunes a los tres modelos usando ViT-Small (en vez
de ViT-Base) como representante de la familia ViT. Ensamble multi-vista por modelo,
con el umbral calibrado de cada uno; FN comunes = GT=1 y los 3 predicen 0.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/palodo2/acl_classifier')
from src.data_loader import OptimizedMRNetDataset, get_val_test_transform
from src.models import CNNMultiSliceClassifier, ViTSmallMultiSliceClassifier, SwinMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = Path('/home/palodo2/acl_classifier'); DATA = BASE/'data'; CK = BASE/'checkpoints'
PLANES = ['sagittal','coronal','axial']
THR = {'cnn':0.5445, 'vit':0.3804, 'swin':0.3734}
NAMES = {'cnn':'ResNet50','vit':'ViT-Small','swin':'Swin-Tiny'}
vit_best = {'sagittal':43,'coronal':51,'axial':48}
vit_cfg = {'sagittal':('attention',0.2,0.15),'coronal':('max',0.25,0.15),'axial':('attention',0.25,0.15)}
cnn_cfg = {'sagittal':('attention',0.42,0.32),'coronal':('max',0.47,0.37),'axial':('attention',0.42,0.32)}


def predict(m, loader):
    m.eval(); pr=[]
    with torch.no_grad():
        for b in loader:
            o=m(b[0].to(DEVICE)); lg=o[0] if isinstance(o,tuple) else o
            pr.extend(torch.sigmoid(lg).cpu().numpy().flatten())
    return np.array(pr)


def build_load(key, plane):
    if key=='cnn':
        pm,di,dd=cnn_cfg[plane]; m=CNNMultiSliceClassifier(pretrained=False,pooling_mode=pm,dropout_input=di,dropout_dense=dd)
        f=CK/'cnn_multiseed_resnet50'/f'best_{plane}_multiseed_final.pth'
    elif key=='swin':
        m=SwinMultiSliceClassifier(pretrained=False)
        f=CK/'swin_small_multiseed'/f'best_{plane}_multiseed_final.pth'
    else:
        pm,di,dd=vit_cfg[plane]; m=ViTSmallMultiSliceClassifier(pretrained=True,pooling_mode=pm,dropout_input=di,dropout_dense=dd)
        f=CK/'vit_small_multiseed'/f'seed_{vit_best[plane]}'/f'best_{plane}_seed{vit_best[plane]}_auc.pth'
    sd=torch.load(f,map_location='cpu',weights_only=False); sd=sd.get('model_state_dict',sd)
    m.load_state_dict({k.replace('module.',''):v for k,v in sd.items()},strict=False)
    return m.to(DEVICE)


def main():
    df=pd.read_csv(DATA/'test-acl.csv',header=None,names=['case','label'])
    lab=df['label'].values.astype(int); cids=df['case'].apply(lambda x:f'{x:04d}').values
    tf=get_val_test_transform()
    ens={k:np.zeros(len(lab)) for k in THR}
    for plane in PLANES:
        ds=OptimizedMRNetDataset(csv_path=str(DATA/'test-acl.csv'),data_root=str(DATA/'test'),plane=plane,
            indices_cache_path=str(DATA/'slice_indices_final'/f'test_{plane}_indices.json'),transform=tf)
        ld=DataLoader(ds,batch_size=16,shuffle=False)
        for k in THR:
            m=build_load(k,plane); ens[k]=ens[k]+predict(m,ld); del m
    ens={k:v/3 for k,v in ens.items()}
    pred={k:(ens[k]>=THR[k]).astype(int) for k in THR}
    # FN comunes: GT=1 y los 3 predicen 0
    fn_common=[i for i in range(len(lab)) if lab[i]==1 and all(pred[k][i]==0 for k in THR)]
    print(f'Test: {len(lab)} casos, {int(lab.sum())} positivos')
    print(f'Umbrales: ResNet50={THR["cnn"]}, ViT-Small={THR["vit"]}, Swin-Tiny={THR["swin"]}')
    print(f'\nFalsos negativos comunes a los 3 modelos: {len(fn_common)}')
    print(f"  {'Caso':<8}{'ResNet50':<10}{'ViT-Small':<11}{'Swin-Tiny':<10}")
    for i in fn_common:
        print(f"  {cids[i]:<8}{ens['cnn'][i]:<10.4f}{ens['vit'][i]:<11.4f}{ens['swin'][i]:<10.4f}")
    # FN por modelo (para contexto)
    for k in THR:
        fn=[cids[i] for i in range(len(lab)) if lab[i]==1 and pred[k][i]==0]
        print(f'  FN {NAMES[k]}: {len(fn)} -> {fn}')
    # AUC reetiquetando los 3 FN comunes como negativos
    from sklearn.metrics import roc_auc_score
    lab_fix = lab.copy()
    for i in fn_common: lab_fix[i]=0
    print('\nAUC test actual vs reetiquetando los 3 casos como negativos:')
    for k in THR:
        a0=roc_auc_score(lab,ens[k]); a1=roc_auc_score(lab_fix,ens[k])
        print(f'  {NAMES[k]:<10} {a0:.4f} -> {a1:.4f}  (+{a1-a0:.4f})')


if __name__ == '__main__':
    main()
