import json
from pathlib import Path

def main():
    notebook_path = Path('/home/palodo2/acl_classifier/Análisis_Falsos_Negativos.ipynb')
    
    # Define the cells for the notebook
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Análisis de Falsos Negativos Comunes en el Test Set\n",
                "\n",
                "Este Notebook presenta la evaluación en el conjunto de prueba (Test Set) y el análisis visual interactivo de los **Falsos Negativos (FN)** que son compartidos por los tres modelos principales del proyecto:\n",
                "* **Swin Small**\n",
                "* **ViT Base**\n",
                "* **CNN ResNet50**\n",
                "\n",
                "### ¿Qué es un Falso Negativo (FN)?\n",
                "Un Falso Negativo ocurre cuando un paciente que **realmente tiene una rotura de LCA** (Ground Truth = 1) es diagnosticado incorrectamente por los modelos como **Sano** (Predicción = 0).\n",
                "Es el tipo de error más crítico en el ámbito médico, ya que puede retrasar un tratamiento quirúrgico o la terapia física adecuada."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup y Carga de Módulos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "\n",
                "# Insertar la ruta raíz para importar src\n",
                "sys.path.insert(0, '/home/palodo2/acl_classifier')\n",
                "\n",
                "# Configurar visualización de gráficos\n",
                "%matplotlib inline\n",
                "plt.rcParams['figure.figsize'] = (20, 8.5)\n",
                "print(\"Entorno inicializado correctamente.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Análisis del Solapamiento de Errores\n",
                "\n",
                "Cargamos el archivo CSV detallado de fallos generado en el Test Set para verificar las probabilidades y predicciones obtenidas por cada modelo."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "detailed_csv_path = Path('/home/palodo2/acl_classifier/scratch/detailed_failures.csv')\n",
                "if detailed_csv_path.exists():\n",
                "    df_failures = pd.read_csv(detailed_csv_path)\n",
                "    # Filtrar únicamente los Falsos Negativos compartidos por los 3 modelos\n",
                "    # (GroundTruth == 1 y fallan los 3, lo que significa que predijeron 0)\n",
                "    df_fn = df_failures[\n",
                "        (df_failures['GroundTruth'] == 1) & \n",
                "        (df_failures['SwinPred'] == 0) & \n",
                "        (df_failures['ViTPred'] == 0) & \n",
                "        (df_failures['CNNPred'] == 0)\n",
                "    ]\n",
                "    print(f\"✓ Se encontraron {len(df_fn)} Falsos Negativos comunes:\")\n",
                "    display(df_fn)\n",
                "else:\n",
                "    print(\"Error: No se encontró el archivo detailed_failures.csv en scratch.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Función de Visualización Detallada (Sin Solapar Textos)\n",
                "\n",
                "Definimos la función para graficar los 25 cortes analizados para un caso determinado (5 sagitales, 10 coronales y 10 axiales). \n",
                "Los nombres de los planos anatómicos se colocan a la izquierda de cada fila y los números de los cortes arriba, dejando las imágenes de resonancia magnética completamente libres de texto superpuesto."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plot_all_analyzed_slices(case_id, data_dir=Path('/home/palodo2/acl_classifier/data')):\n",
                "    planes = ['sagittal', 'coronal', 'axial']\n",
                "    \n",
                "    # Cargar caches de índices precalculados\n",
                "    caches = {}\n",
                "    for plane in planes:\n",
                "        cache_path = data_dir / 'slice_indices_final' / f'test_{plane}_indices.json'\n",
                "        with open(cache_path, 'r') as f:\n",
                "            caches[plane] = json.load(f)\n",
                "            \n",
                "    fig, axes = plt.subplots(3, 10, figsize=(20, 8.5))\n",
                "    \n",
                "    plane_display_names = {\n",
                "        'sagittal': 'SAGITAL\\n(5 cortes)',\n",
                "        'coronal': 'CORONAL\\n(10 cortes)',\n",
                "        'axial': 'AXIAL\\n(10 cortes)'\n",
                "    }\n",
                "    \n",
                "    for row_idx, plane in enumerate(planes):\n",
                "        vol_path = data_dir / 'test' / plane / f\"{int(case_id):04d}.npy\"\n",
                "        if not vol_path.exists():\n",
                "            print(f\"No se encontró el volumen: {vol_path}\")\n",
                "            continue\n",
                "            \n",
                "        vol = np.load(vol_path)\n",
                "        selected_indices = caches[plane][str(int(case_id))]\n",
                "        num_slices = len(selected_indices)\n",
                "        \n",
                "        for col_idx in range(10):\n",
                "            ax = axes[row_idx, col_idx]\n",
                "            \n",
                "            # Ocultar bordes de la subfigura y marcas de los ejes para limpieza visual\n",
                "            ax.spines['top'].set_visible(False)\n",
                "            ax.spines['right'].set_visible(False)\n",
                "            ax.spines['bottom'].set_visible(False)\n",
                "            ax.spines['left'].set_visible(False)\n",
                "            ax.set_xticks([])\n",
                "            ax.set_yticks([])\n",
                "            \n",
                "            if col_idx < num_slices:\n",
                "                slice_idx = selected_indices[col_idx]\n",
                "                slice_img = vol[slice_idx]\n",
                "                ax.imshow(slice_img, cmap='bone')\n",
                "                ax.set_title(f\"Corte {slice_idx}\", fontsize=9, pad=4)\n",
                "            else:\n",
                "                # Desactivar celdas vacías (plano sagital solo tiene 5 cortes)\n",
                "                ax.axis('off')\n",
                "                \n",
                "            # Colocar la etiqueta del plano a la izquierda de la primera columna\n",
                "            if col_idx == 0:\n",
                "                ax.set_ylabel(plane_display_names[plane], fontsize=12, fontweight='bold', \n",
                "                              labelpad=25, rotation=0, va='center', ha='right')\n",
                "                \n",
                "    fig.suptitle(f\"Caso {case_id} (Falso Negativo Común - Real: Roto, IA: Sano)\", \n",
                "                 fontsize=16, fontweight='bold', y=0.98)\n",
                "    \n",
                "    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.12, right=0.98, hspace=0.35, wspace=0.15)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visualización de los Casos de Falso Negativo"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Caso 0080"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plot_all_analyzed_slices(\"0080\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Caso 0087"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plot_all_analyzed_slices(\"0087\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Caso 0521"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plot_all_analyzed_slices(\"0521\")"
            ]
        }
    ]
    
    # Construct the full notebook dictionary
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Write to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2)
        
    print(f"Successfully generated notebook: {notebook_path}")

if __name__ == '__main__':
    main()
