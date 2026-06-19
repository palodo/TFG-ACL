#!/usr/bin/env python3
"""Igual que el pipeline compilado, pero con la parte de backbones rehecha."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import generate_tfg_diagrams as G   # reutiliza helpers y colores
from matplotlib.patches import FancyBboxPatch

box, arrow, ax_to_fig, mri_slice = G.box, G.arrow, G.ax_to_fig, G.mri_slice
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'TFG'/'tfgs'/'figs'/'pipeline_general_v3.png'

fig, ax = plt.subplots(figsize=(13.5, 8.2)); ax.set_xlim(0,100); ax.set_ylim(-10,100); ax.axis('off')
ax.text(50,97,'Sistema de detección de rotura del LCA en RM de rodilla',ha='center',fontsize=14,fontweight='bold')

# Entrada
box(ax,1.5,40,11,22,'Estudio RM\n(DICOM)\n\nVolúmenes 3D\n$V\\in\\mathbb{R}^{S\\times H\\times W}$',G.C_INPUT,fontsize=9)
lanes={'sagittal':76,'coronal':47,'axial':18}
sel_text={'sagittal':'Selector CNN\nMobileNetV2\n$K=5$ cortes',
          'coronal':'Selector CNN\nMobileNetV2\n$K=10$ cortes',
          'axial':'Selección geométrica\n$K=10$ cortes\ncentrales'}
plane_es={'sagittal':'Plano sagital','coronal':'Plano coronal','axial':'Plano axial'}
for plane,yc in lanes.items():
    im=fig.add_axes(ax_to_fig(ax,fig,16.5,yc-9,10,18)); im.imshow(mri_slice(plane),cmap='gray')
    im.set_xticks([]); im.set_yticks([])
    for s in im.spines.values(): s.set_color(G.C_EDGE)
    im.set_title(plane_es[plane],fontsize=9,fontweight='bold',pad=2)
    box(ax,30,yc-8,15.5,16,sel_text[plane],G.C_SEL,fontsize=8.8)
    box(ax,49.5,yc-8,21,16,
        'Clasificador multi-corte\nBackbone 2D ($\\bigstar$)\n'
        r'$\rightarrow\ p_{\mathrm{%s}}$'%plane_es[plane].split()[-1],G.C_NET,fontsize=8.8)
    arrow(ax,12.7,51,15.8,yc); arrow(ax,27.0,yc,29.7,yc); arrow(ax,45.7,yc,49.2,yc); arrow(ax,70.7,yc,74.3,51)

# Fusion y diagnostico
box(ax,74.5,41,12.5,20,'Fusión multi-vista\npromedio simple\n\n'r'$p=\frac{p_{sag}+p_{cor}+p_{ax}}{3}$',G.C_FUSE,fontsize=8.8)
arrow(ax,87.2,51,89.6,51)
box(ax,89.8,44,9,14,'Umbral $\\tau$\ncalibrado en\nvalidación',G.C_FUSE,fontsize=8.6)
arrow(ax,94.3,58.2,94.3,66.5); box(ax,89.8,67,9,9,'Rotura LCA\n$(p\\geq\\tau)$',G.C_OUT_BAD,fontsize=8.6,bold=True)
arrow(ax,94.3,43.8,94.3,35.5); box(ax,89.8,26.5,9,9,'Sano\n$(p<\\tau)$',G.C_OUT_OK,fontsize=8.6,bold=True)

# ===== PARTE DE BACKBONES REHECHA (debajo de todo, sin pisar el plano axial) =====
cont=FancyBboxPatch((24,-9),64,10.5,boxstyle='round,pad=0.012,rounding_size=0.015',
    lw=1.3,edgecolor='#7c3aed',facecolor='#faf5ff',zorder=2,linestyle='--')
ax.add_patch(cont)
ax.text(26,-0.7,r'$\bigstar$  Backbone 2D intercambiable',fontsize=9.5,fontweight='bold',color='#6d28d9',va='center')
ax.text(26,-3.2,'Se compara una arquitectura cada vez, bajo el mismo pipeline:',fontsize=8.2,color='#475569',va='center')
chips=[('ResNet50','23.5 M','#fee2e2','#b91c1c'),
       ('ViT-Small','22 M','#dbeafe','#1d4ed8'),
       ('Swin-Tiny','27.8 M','#dcfce7','#15803d')]
cx=[34,52,70]
for (name,par,fc,ec),x in zip(chips,cx):
    b=FancyBboxPatch((x,-7.8),16,3.4,boxstyle='round,pad=0.01,rounding_size=0.02',
        lw=1.1,edgecolor=ec,facecolor=fc,zorder=3); ax.add_patch(b)
    ax.text(x+8,-6.1,f'{name}  ·  {par}',ha='center',va='center',fontsize=8.4,fontweight='bold',zorder=4)

fig.savefig(OUT,dpi=200,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('OK ->',OUT)
