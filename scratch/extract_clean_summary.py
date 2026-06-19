import json
import re
from pathlib import Path

def get_clean_metrics(nb_path):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        return f"File {nb_path} not found.\n"
        
    summary = []
    summary.append(f"\n==============================================")
    summary.append(f"NOTEBOOK: {nb_path}")
    summary.append(f"==============================================")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type != 'code':
            continue
            
        outputs = cell.get('outputs', [])
        for out in outputs:
            text = ""
            if 'text' in out:
                text = "".join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                text = "".join(out['data']['text/plain'])
                
            # Filter out progress bars or long lines
            clean_lines = []
            for line in text.split('\n'):
                # Ignore lines like "Progreso: ..." or loading status unless it's a summary
                if "progreso" in line.lower() or "loading" in line.lower() or "loaded" in line.lower() or "dataset cargado" in line.lower():
                    continue
                if line.strip():
                    clean_lines.append(line)
            
            clean_text = "\n".join(clean_lines)
            if not clean_text.strip():
                continue
                
            # If the text has summary keywords
            if any(term in clean_text.lower() for term in ['roc_auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'threshold', 'croatia', 'croacia', 'confusion_matrix', 'matriz de confusión', 'resumen comparativo', 'comparativa de ensambles', 'caída auc']):
                summary.append(f"\n[Cell {i}]:")
                summary.append(clean_text)
                summary.append("-" * 30)
                
    return "\n".join(summary)

if __name__ == '__main__':
    notebooks = {
        'CNN': 'models/cnn/Threshold_Tuning_and_External_Validation.ipynb',
        'ViT': 'models/vit/Threshold_Tuning_and_External_Validation.ipynb',
        'Swin Small (Final Model)': 'final_model/Threshold_Tuning_and_External_Validation.ipynb'
    }
    
    with open('scratch/clean_summaries.txt', 'w', encoding='utf-8') as f:
        for name, path in notebooks.items():
            f.write(get_clean_metrics(path))
            
    print("Wrote clean summaries to scratch/clean_summaries.txt")
