#!/usr/bin/env python3
"""
Evaluación ligera de un estudio con los TRES modelos a la vez (ResNet50, ViT-Small,
Swin-Tiny) para la vista comparativa de la app. No genera heatmaps ni imágenes por
corte: solo las probabilidades por plano, el ensamble por modelo y el veredicto.

Reutiliza el pipeline de scripts/predict_patient.py (extracción, planos, selectores).
Salida: NDJSON (eventos de progreso + un evento final 'comparison').
"""
import os, sys, json, shutil, tempfile, warnings
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings('ignore')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
CKPT = Path(os.environ.get('CHECKPOINTS_DIR', BASE / 'checkpoints'))

import importlib.util
_spec = importlib.util.spec_from_file_location('pp', str(BASE / 'scripts' / 'predict_patient.py'))
pp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pp)
from src.models import CNNMultiSliceClassifier, ViTSmallMultiSliceClassifier, SwinMultiSliceClassifier

DEVICE = pp.select_device()
TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

MODELS = ['cnn', 'vit', 'swin']
LABELS = {'cnn': 'ResNet50', 'vit': 'ViT-Small', 'swin': 'Swin-Tiny'}
THRESH = {'swin': 0.3734, 'vit': 0.3804, 'cnn': 0.5455}
DIRS = {'cnn': 'cnn_multiseed_resnet50', 'vit': 'vit_small_multiseed', 'swin': 'swin_tiny_multiseed'}
POOL = {'sagittal': 'attention', 'coronal': 'max', 'axial': 'attention'}


def emit(obj):
    print(json.dumps(obj)); sys.stdout.flush()


def progress(msg, step=0):
    emit({'type': 'progress', 'message': msg, 'step_index': step})


def build_model(arch, plane):
    pm = POOL[plane]
    if arch == 'cnn':
        m = CNNMultiSliceClassifier(num_classes=1, pretrained=False, pooling_mode=pm,
                                    dropout_input=0.42, dropout_dense=0.32)
    elif arch == 'vit':
        m = ViTSmallMultiSliceClassifier(num_classes=1, pretrained=True, pooling_mode=pm,
                                         dropout_input=0.25, dropout_dense=0.15)
    else:
        m = SwinMultiSliceClassifier()
    ck = torch.load(str(CKPT / DIRS[arch] / f'best_{plane}_multiseed_final.pth'),
                    map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()


def plane_probability(model, vol, indices):
    ts = [TF(Image.fromarray(vol[i].astype(np.uint8)).convert('RGB')) for i in indices]
    x = torch.stack(ts).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
    return float(torch.sigmoid(logits).squeeze().item())


def calibrate(ensemble, thr):
    if ensemble >= thr:
        return 0.5 + 0.5 * (ensemble - thr) / (1.0 - thr + 1e-8)
    return 0.5 * ensemble / (thr + 1e-8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp()
    try:
        progress('Descomprimiendo serie DICOM...', 0)
        pp.secure_zip_extract(args.zip, tmp)
        progress('Reconstruyendo volúmenes por plano anatómico...', 0)
        vols, patient_info = pp.load_dicom_from_dir(tmp)
        if not vols:
            emit({'type': 'error', 'error': 'No se han encontrado cortes DICOM válidos de rodilla.'})
            return
        emit({'type': 'patient_info', 'patient_info': patient_info})

        # Selección de cortes (compartida por los 3 modelos)
        progress('Localizando cortes relevantes (selector MobileNetV2)...', 1)
        sel_s = pp.SimpleCNNSelector(str(CKPT / 'acl_slice_classifier_v3' / 'best_model_f1.pth'), device=DEVICE).eval()
        sel_c = pp.SimpleCNNSelector(str(CKPT / 'acl_slice_classifier_coronal' / 'best_model_f1.pth'), device=DEVICE).eval()
        norm = lambda v: (v.astype(np.float32) - v.min()) / (v.max() - v.min() + 1e-8)
        idx = {}
        if 'sagittal' in vols: idx['sagittal'] = sel_s.select_top_k_slices(norm(vols['sagittal']), k=5)[0]
        if 'coronal' in vols: idx['coronal'] = sel_c.select_top_k_slices(norm(vols['coronal']), k=10)[0]
        if 'axial' in vols: idx['axial'] = pp.center_consecutive_indices(vols['axial'].shape[0], k=10)

        results = {}
        for step, arch in enumerate(MODELS, start=2):
            progress(f'Evaluando {LABELS[arch]} en los tres planos...', step)
            planes = {}
            for plane in ['sagittal', 'coronal', 'axial']:
                if plane not in vols or plane not in idx:
                    continue
                model = build_model(arch, plane)
                planes[plane] = plane_probability(model, vols[plane], idx[plane])
                del model
            ensemble = float(np.mean(list(planes.values()))) if planes else 0.0
            thr = THRESH[arch]
            results[arch] = {
                'label': LABELS[arch],
                'planes': planes,
                'ensemble_probability': ensemble,
                'prediction': 'ACL INJURY' if ensemble >= thr else 'HEALTHY',
                'calibrated_confidence': calibrate(ensemble, thr),
                'threshold': thr,
            }
            emit({'type': 'model_result', 'model': arch, 'result': results[arch]})

        # Consenso informal (no es una métrica validada; solo para la vista)
        votes = sum(1 for r in results.values() if r['prediction'] == 'ACL INJURY')
        consensus = 'ACL INJURY' if votes >= 2 else 'HEALTHY'
        emit({'type': 'comparison', 'patient_info': patient_info, 'models': results,
              'consensus': {'prediction': consensus, 'votes_injury': votes, 'n_models': len(results)}})
        progress('Comparativa completada.', 6)
    except Exception as e:
        import traceback
        emit({'type': 'error', 'error': str(e), 'traceback': traceback.format_exc()})
    finally:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
