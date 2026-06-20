#!/usr/bin/env python3
"""
Genera los diagramas de arquitectura del TFG, fieles a la implementacion real:

  1) pipeline_general.png : volumen RM -> seleccion de cortes por plano
     (MobileNetV2 K=5/K=10, axial central) -> clasificador multi-corte
     (backbone + Attention Pooling) -> promedio simple 1/3 -> umbral -> diagnostico.

  2) attention_pooling.png : mecanismo de Attention Pooling con sus ecuaciones
     y la cabeza de clasificacion real (768 -> 256 -> 1).

Incrusta cortes de RM reales del caso 0105 de validacion para hacerlo visual.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[1]
FIGS = BASE / 'memoria' / 'tfgs' / 'figs'
CASE = '0105'

# Paleta sobria
C_INPUT = '#dbeafe'   # azul claro
C_SEL = '#fef3c7'     # ambar claro
C_NET = '#dcfce7'     # verde claro
C_FUSE = '#ede9fe'    # violeta claro
C_OUT_OK = '#bbf7d0'  # verde
C_OUT_BAD = '#fecaca' # rojo
C_EDGE = '#334155'


def box(ax, x, y, w, h, text, fc, fontsize=9.5, bold=False, ec=C_EDGE):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.012',
                       linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, **kw):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                        linewidth=1.3, color=C_EDGE, zorder=1, **kw)
    ax.add_patch(a)


def mri_slice(plane):
    vol = np.load(BASE / 'data' / 'val' / plane / f'{CASE}.npy').astype(np.float32)
    s = vol[vol.shape[0] // 2]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    ax.text(50, 97.0, 'Sistema de detección de rotura del LCA en RM de rodilla',
            ha='center', fontsize=14, fontweight='bold')

    # --- Entrada ---
    box(ax, 1.5, 40, 11, 22, 'Estudio RM\n(DICOM)\n\nVolúmenes 3D\n$V\\in\\mathbb{R}^{S\\times H\\times W}$',
        C_INPUT, fontsize=9, bold=False)

    lanes = {'sagittal': 76, 'coronal': 47, 'axial': 18}
    sel_text = {
        'sagittal': 'Selector CNN\nMobileNetV2\n$K=5$ cortes',
        'coronal':  'Selector CNN\nMobileNetV2\n$K=10$ cortes',
        'axial':    'Selección geométrica\n$K=10$ cortes\ncentrales',
    }
    plane_es = {'sagittal': 'Plano sagital', 'coronal': 'Plano coronal', 'axial': 'Plano axial'}

    for plane, yc in lanes.items():
        # Miniatura RM real
        im_ax = fig.add_axes(ax_to_fig(ax, fig, 16.5, yc - 9, 10, 18))
        im_ax.imshow(mri_slice(plane), cmap='gray')
        im_ax.set_xticks([]); im_ax.set_yticks([])
        for s in im_ax.spines.values():
            s.set_color(C_EDGE)
        im_ax.set_title(plane_es[plane], fontsize=9, fontweight='bold', pad=2)

        # Seleccion
        box(ax, 30, yc - 8, 15.5, 16, sel_text[plane], C_SEL, fontsize=8.8)
        # Clasificador multi-corte
        box(ax, 49.5, yc - 8, 21, 16,
            'Clasificador multi-corte\nBackbone 2D compartido\n+ Attention Pooling\n'
            r'$\rightarrow\ p_{\mathrm{%s}}$' % plane_es[plane].split()[-1],
            C_NET, fontsize=8.8)

        # Flechas de la fila
        arrow(ax, 12.7, 51, 15.8, yc)            # entrada -> miniatura
        arrow(ax, 27.0, yc, 29.7, yc)            # miniatura -> seleccion
        arrow(ax, 45.7, yc, 49.2, yc)            # seleccion -> clasificador
        arrow(ax, 70.7, yc, 74.3, 51)            # clasificador -> fusion

    # --- Fusion y diagnostico ---
    box(ax, 74.5, 41, 12.5, 20,
        'Fusión multi-vista\npromedio simple\n\n'
        r'$p=\frac{p_{sag}+p_{cor}+p_{ax}}{3}$',
        C_FUSE, fontsize=8.8)
    arrow(ax, 87.2, 51, 89.6, 51)
    box(ax, 89.8, 44, 9, 14, 'Umbral $\\tau$\ncalibrado en\nvalidación', C_FUSE, fontsize=8.6)

    arrow(ax, 94.3, 58.2, 94.3, 66.5)
    box(ax, 89.8, 67, 9, 9, 'Rotura LCA\n$(p\\geq\\tau)$', C_OUT_BAD, fontsize=8.6, bold=True)
    arrow(ax, 94.3, 43.8, 94.3, 35.5)
    box(ax, 89.8, 26.5, 9, 9, 'Sano\n$(p<\\tau)$', C_OUT_OK, fontsize=8.6, bold=True)

    # Nota de backbones evaluados
    box(ax, 49.5, 1.5, 21, 9,
        'Backbones evaluados:\nResNet50 (23.5M) · ViT-Base (86M) · Swin-Tiny (27.8M)',
        '#f1f5f9', fontsize=8.2)

    out = FIGS / 'pipeline_general.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out)


def ax_to_fig(ax, fig, x, y, w, h):
    """Convierte coords de datos del eje (0-100) a coords de figura [l,b,w,h]."""
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x + w, y + h))
    inv = fig.transFigure.inverted()
    (l, b), (r, t) = inv.transform(p0), inv.transform(p1)
    return [l, b, r - l, t - b]


def attention_diagram():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    ax.text(50, 97, 'Attention Pooling sobre los $K$ cortes seleccionados',
            ha='center', fontsize=14, fontweight='bold')

    # Cortes de entrada (miniaturas reales del plano sagital)
    vol = np.load(BASE / 'data' / 'val' / 'sagittal' / f'{CASE}.npy').astype(np.float32)
    mids = np.linspace(vol.shape[0] * 0.35, vol.shape[0] * 0.65, 3).astype(int)
    ys = [70, 48, 26]
    for i, (sidx, yc) in enumerate(zip(mids, ys), start=1):
        s = vol[sidx]; s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        im_ax = fig.add_axes(ax_to_fig(ax, fig, 2.5, yc - 7, 8, 14))
        im_ax.imshow(s, cmap='gray'); im_ax.set_xticks([]); im_ax.set_yticks([])
        lbl = f'corte {i}' if i < 3 else 'corte $K$'
        im_ax.set_title(lbl, fontsize=9, pad=2)
        if i == 2:
            ax.text(6.5, 38.5, r'$\vdots$', fontsize=16, ha='center')

        # backbone compartido
        arrow(ax, 11.0, yc, 14.6, yc)
        # vector de caracteristicas
        box(ax, 26, yc - 4, 12, 8, r'$f_{%s}\in\mathbb{R}^{768}$' % (str(i) if i < 3 else 'K'),
            '#e2e8f0', fontsize=9)
        arrow(ax, 38.2, yc, 41.8, yc)
        # puntuacion de atencion
        box(ax, 42, yc - 4, 16, 8,
            r'$a_{%s}=w_2^{\top}\tanh(W_1 f_{%s})$' % ((str(i),)*2 if i < 3 else ('K',)*2),
            C_SEL, fontsize=8.6)
        arrow(ax, 58.2, yc, 62.0, 48 if i != 2 else yc)

    # Backbone compartido (caja vertical)
    box(ax, 14.8, 19, 11, 60, 'Backbone 2D\ncompartido\n(ResNet50 /\nViT-Base /\nSwin-Tiny)',
        C_NET, fontsize=8.8, bold=False)
    for yc in ys:
        arrow(ax, 25.9, yc, 25.95, yc)  # backbone -> f_i (visual)

    # Softmax + suma ponderada
    box(ax, 62.2, 40, 13, 16, 'softmax\n$\\alpha_i = \\dfrac{e^{a_i}}{\\sum_j e^{a_j}}$',
        C_FUSE, fontsize=8.8)
    arrow(ax, 75.4, 48, 78.6, 48)
    box(ax, 78.8, 40, 13.5, 16, 'Suma ponderada\n$z=\\sum_{i=1}^{K}\\alpha_i\\, f_i$',
        C_FUSE, fontsize=8.8)

    # Cabeza de clasificacion real
    arrow(ax, 85.5, 39.8, 85.5, 31.0)
    box(ax, 64, 12, 33, 18,
        'Clasificador\nDropout $\\rightarrow$ Linear $768\\!\\to\\!256$ $\\rightarrow$ ReLU '
        '$\\rightarrow$ Dropout $\\rightarrow$ Linear $256\\!\\to\\!1$\n'
        r'$p=\sigma(\mathrm{logit})\in[0,1]$',
        C_NET, fontsize=8.8)

    # Nota
    ax.text(50, 3.5, 'Los pesos $\\alpha_i$ indican la contribución de cada corte a la decisión '
                     'y son los mostrados por la aplicación como "peso de atención".',
            ha='center', fontsize=8.5, style='italic', color='#475569')

    out = FIGS / 'attention_pooling.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', out)


if __name__ == '__main__':
    pipeline_diagram()
    attention_diagram()
