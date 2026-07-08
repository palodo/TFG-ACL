"""Fine-tuning del Swin-Tiny sagital canónico sobre Croacia.

Objetivo: superar el AUC de test de Croacia previo (~0.8993).

- Parte del checkpoint canónico multiseed (best_sagittal_multiseed_final.pth),
  que en zero-shot sobre el test de Croacia da AUC ~0.897.
- Usa los splits e índices de cortes ya cacheados (train/val/test) para que el
  resultado sea comparable con la corrida anterior.
- Receta: transfer learning en 2 fases (congela backbone unas épocas y luego
  descongela con LR discriminativo), pos_weight para el desbalance, selección
  del mejor epoch por AUC de validación, early stopping.
- Robustez: varias semillas + test-time augmentation (hflip) + ensamble de las
  semillas. Reporta AUC por semilla y del ensamble.
"""
import os, sys, json, copy, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

BASE = Path('/home/palodo2/acl_classifier')
sys.path.insert(0, str(BASE))
os.chdir(BASE)
from src.models import SwinMultiSliceClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROATIA_DIR = BASE / 'data' / 'croatia_npy_volumes_full'
SPLITS = CROATIA_DIR / 'splits'
IDX = SPLITS / 'slice_indices'
CANON = BASE / 'checkpoints' / 'swin_tiny_multiseed' / 'best_sagittal_multiseed_final.pth'
OUT = SPLITS / 'finetune_v2'
OUT.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm = transforms.Normalize(mean=MEAN, std=STD)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    norm,
])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), norm])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_vol(p):
    v = np.load(p).astype(np.float32)
    lo, hi = v.min(), v.max()
    if hi > lo:
        v = (v - lo) / (hi - lo)
    return v


class CroatiaSplit(Dataset):
    def __init__(self, csv, idx_json, tf):
        self.df = pd.read_csv(csv)
        self.idx = json.load(open(idx_json))
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = load_vol(CROATIA_DIR / r['filename'])
        sl = vol[self.idx[str(r['volume_id'])]]
        sl = torch.from_numpy(sl).float().unsqueeze(1).repeat(1, 3, 1, 1)
        sl = torch.stack([self.tf(s) for s in sl], 0)
        return sl, torch.tensor(float(r['label'])), str(r['volume_id'])


def make_loaders():
    tr = CroatiaSplit(SPLITS / 'train.csv', IDX / 'train_sagittal_indices.json', train_tf)
    va = CroatiaSplit(SPLITS / 'val.csv', IDX / 'val_sagittal_indices.json', eval_tf)
    te = CroatiaSplit(SPLITS / 'test.csv', IDX / 'test_sagittal_indices.json', eval_tf)
    g = torch.Generator(); g.manual_seed(0)
    return (DataLoader(tr, 16, shuffle=True, num_workers=4, pin_memory=True, generator=g),
            DataLoader(va, 16, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(te, 16, shuffle=False, num_workers=4, pin_memory=True))


def build_model():
    m = SwinMultiSliceClassifier()
    ck = torch.load(CANON, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE)


def fwd(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict(model, loader, tta=False):
    model.eval()
    ps, ys = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        logit = fwd(model, x)
        prob = torch.sigmoid(logit)
        if tta:
            prob = 0.5 * (prob + torch.sigmoid(fwd(model, torch.flip(x, dims=[-1]))))
        ps.append(prob.cpu().numpy().ravel()); ys.append(y.numpy().ravel())
    return np.concatenate(ps), np.concatenate(ys)


def pos_weight(loader):
    ys = np.concatenate([y.numpy().ravel() for _, y, _ in loader])
    n_pos = ys.sum(); n_neg = len(ys) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)


def train_one(seed, epochs, freeze_epochs, lr_head, lr_bb, wd, patience):
    set_seed(seed)
    tr, va, te = make_loaders()
    model = build_model()
    pw = pos_weight(tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    groups = [
        {'params': [p for n, p in model.named_parameters() if n.startswith('backbone.')], 'lr': lr_bb},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone.')], 'lr': lr_head},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def set_backbone(freeze):
        for p in model.backbone.parameters():
            p.requires_grad = not freeze

    best_auc, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(1, epochs + 1):
        set_backbone(ep <= freeze_epochs)
        model.train()
        for x, y, _ in tr:
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, 1)
            opt.zero_grad()
            loss = crit(fwd(model, x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vp, vy = predict(model, va)
        vauc = roc_auc_score(vy, vp)
        flag = ''
        if vauc > best_auc:
            best_auc, best_state, best_ep, bad = vauc, copy.deepcopy(model.state_dict()), ep, 0
            flag = ' *'
        else:
            bad += 1
        print(f'  seed {seed} ep{ep:02d} valAUC={vauc:.4f}{flag}', flush=True)
        if bad >= patience:
            print(f'  seed {seed} early stop @ep{ep} (best ep{best_ep} valAUC={best_auc:.4f})', flush=True)
            break

    model.load_state_dict(best_state)
    tp, ty = predict(model, te, tta=True)
    vp, vy = predict(model, va, tta=True)
    tauc = roc_auc_score(ty, tp)
    print(f'  seed {seed} -> best valAUC={best_auc:.4f} (ep{best_ep}) | test AUC(TTA)={tauc:.4f}', flush=True)
    torch.save({'model_state_dict': best_state, 'seed': seed, 'val_auc': best_auc,
                'test_auc_tta': tauc}, OUT / f'ft_swin_croatia_seed{seed}.pth')
    return {'seed': seed, 'best_ep': best_ep, 'val_auc': float(best_auc),
            'test_auc_tta': float(tauc), 'test_probs': tp, 'test_labels': ty,
            'val_probs': vp, 'val_labels': vy}


def metrics_at_thr(y, p, thr):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y.astype(int), pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    spec = tn / (tn + fp) if tn + fp else 0
    return {'auc': float(roc_auc_score(y, p)), 'precision': float(prec), 'recall': float(rec),
            'specificity': float(spec), 'threshold': float(thr), 'cm': cm.tolist()}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--freeze', type=int, default=3)
    ap.add_argument('--lr_head', type=float, default=3e-4)
    ap.add_argument('--lr_bb', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=7)
    args = ap.parse_args()

    print(f'Device={DEVICE} | seeds={args.seeds} | epochs={args.epochs} '
          f'freeze={args.freeze} lr_head={args.lr_head} lr_bb={args.lr_bb}', flush=True)

    runs = []
    for s in args.seeds:
        print(f'\n=== SEED {s} ===', flush=True)
        runs.append(train_one(s, args.epochs, args.freeze, args.lr_head, args.lr_bb, args.wd, args.patience))

    ty = runs[0]['test_labels']; vy = runs[0]['val_labels']

    # RESULTADO REPORTADO: seleccionar el modelo por AUC de validación, medir test una vez.
    selected = max(runs, key=lambda r: r['val_auc'])
    sel_thr = best_f1_threshold(vy, selected['val_probs'])
    sel_metrics = metrics_at_thr(ty, selected['test_probs'], sel_thr)

    # Ensamble (solo referencia)
    ens_test = np.mean([r['test_probs'] for r in runs], axis=0)
    ens_metrics = metrics_at_thr(ty, ens_test, best_f1_threshold(vy, np.mean([r['val_probs'] for r in runs], axis=0)))

    per_seed = [{'seed': r['seed'], 'best_ep': r['best_ep'], 'val_auc': r['val_auc'],
                 'test_auc_tta': r['test_auc_tta']} for r in runs]
    aucs = [r['test_auc_tta'] for r in runs]

    summary = {
        'target_to_beat': 0.8993,
        'zero_shot_test_auc': 0.8968,
        'per_seed': per_seed,
        'selected_by_val': {'seed': selected['seed'], 'val_auc': selected['val_auc'], **sel_metrics},
        'seed_test_auc_mean': float(np.mean(aucs)),
        'seed_test_auc_std': float(np.std(aucs)),
        'ensemble_test_metrics_reference': ens_metrics,
    }
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
    pd.DataFrame({'label': ty, 'prob': selected['test_probs']}).to_csv(OUT / 'selected_test_predictions.csv', index=False)

    print('\n' + '=' * 70)
    print('RESUMEN')
    print('=' * 70)
    for r in per_seed:
        mark = '  <= mejor val' if r['seed'] == selected['seed'] else ''
        print(f"  seed {r['seed']}: val {r['val_auc']:.4f} | test(TTA) {r['test_auc_tta']:.4f}{mark}")
    print(f"  >> SELECCIONADO POR VALIDACION: seed {selected['seed']} (val {selected['val_auc']:.4f}) "
          f"-> TEST {sel_metrics['auc']:.4f} (rec={sel_metrics['recall']:.3f} "
          f"prec={sel_metrics['precision']:.3f} thr={sel_thr:.3f})")
    print(f"  estabilidad (media 5 semillas) = {summary['seed_test_auc_mean']:.4f} ± {summary['seed_test_auc_std']:.4f}")
    print(f"  ensamble (solo referencia) = {ens_metrics['auc']:.4f}")
    print(f"  objetivo a batir = 0.8993 | zero-shot = 0.8968")
    print('=' * 70)


if __name__ == '__main__':
    main()
