#!/usr/bin/env python3
"""
Genera las matrices de confusión gráficas de la validación externa zero-shot en
Croacia (plano sagital, sin reajuste de pesos), con el MISMO formato visual que
las matrices por modelo de MRNet (*_cm_grafica.png).

Los valores (VN, FP, FN, VP) provienen de la Tabla de matrices de confusión de
Croacia zero-shot de la memoria (memoria/tfgs/tex/resultados.tex). No requiere modelo
ni datos: solo dibuja.

Salida: memoria/tfgs/figs/outputs/CROACIA_cm_grafica.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path('/home/palodo2/acl_classifier')
OUT = BASE / 'memoria' / 'tfgs' / 'figs' / 'outputs' / 'CROACIA_cm_grafica.png'

# Paleta tomada de las matrices de MRNet (*_cm_grafica.png)
GREEN = '#52B788'   # VN (verdadero negativo)
TEAL = '#A8DADC'    # VP (verdadero positivo)
PINK = '#F6C7CC'    # FP / FN (errores)
TEXT = '#1d3557'    # texto oscuro

# (modelo, VN, FP, FN, VP)  -- zero-shot test de Croacia (917 estudios)
MODELS = [
    ('ResNet50', 671, 19, 100, 127),
    ('ViT-Small', 620, 70, 72, 155),
    ('Swin-Tiny', 665, 25, 98, 129),
]


def draw_cm(ax, title, vn, fp, fn, vp):
    # celdas: fila 0 = Real Sano (VN | FP), fila 1 = Real Roto (FN | VP)
    cells = [
        (0, 1, vn, 'VN', GREEN),   # col 0, fila superior
        (1, 1, fp, 'FP', PINK),    # col 1, fila superior
        (0, 0, fn, 'FN', PINK),    # col 0, fila inferior
        (1, 0, vp, 'VP', TEAL),    # col 1, fila inferior
    ]
    for cx, cy, val, lab, color in cells:
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=color,
                               edgecolor='white', linewidth=3))
        ax.text(cx + 0.5, cy + 0.60, f'{val}', ha='center', va='center',
                fontsize=26, fontweight='bold', color=TEXT)
        ax.text(cx + 0.5, cy + 0.28, lab, ha='center', va='center',
                fontsize=13, color=TEXT)
    # ejes / etiquetas
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # cabecera columnas
    ax.text(1.0, 2.42, 'Predicho', ha='center', va='center',
            fontsize=15, fontweight='bold', color=TEXT)
    ax.text(1.0, 2.20, title, ha='center', va='center',
            fontsize=16, fontweight='bold', color=TEXT)
    ax.text(0.5, 2.05, 'Sano', ha='center', va='center', fontsize=13, color=TEXT)
    ax.text(1.5, 2.05, 'Roto', ha='center', va='center', fontsize=13, color=TEXT)
    # etiquetas filas
    ax.text(-0.42, 1.0, 'Real', ha='center', va='center', rotation=90,
            fontsize=15, fontweight='bold', color=TEXT)
    ax.text(-0.15, 1.5, 'Sano', ha='right', va='center', fontsize=13, color=TEXT)
    ax.text(-0.15, 0.5, 'Roto', ha='right', va='center', fontsize=13, color=TEXT)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (name, vn, fp, fn, vp) in zip(axes, MODELS):
        draw_cm(ax, name, vn, fp, fn, vp)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.04, wspace=0.45)
    fig.savefig(OUT, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('OK ->', OUT)


if __name__ == '__main__':
    main()
