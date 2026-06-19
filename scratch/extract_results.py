import json
import re
from pathlib import Path

def inspect_notebook(nb_path, out_file):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        out_file.write(f"\nFile not found: {nb_path}\n")
        return
    
    out_file.write(f"\n==========================================\n")
    out_file.write(f"INSPECTING: {nb_path}\n")
    out_file.write(f"==========================================\n")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    out_file.write(f"Total cells: {len(cells)}\n")
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        source_text = "".join(source)
        
        # Check if it's a markdown header
        if cell_type == 'markdown':
            lines = [l.strip() for l in source if l.strip().startswith('#')]
            if lines:
                out_file.write(f"Cell {i} (MD Header): {lines[0]}\n")
                
        # Check if the cell output contains metrics we want
        outputs = cell.get('outputs', [])
        for output in outputs:
            text_list = []
            if 'text' in output:
                text_list = output.get('text', [])
            elif 'data' in output and 'text/plain' in output['data']:
                text_list = output['data']['text/plain']
                
            out_text = "".join(text_list)
            # Match performance metrics pattern or tables
            if any(term in out_text.lower() for term in ['auc', 'precision', 'recall', 'f1-score', 'accuracy', 'specificity', 'confusion matrix', 'seed', 'multiseed', 'sensibilidad', 'sens:', 'especificidad', 'esp:', 'threshold']):
                code_snippet = source_text[:200].replace('\n', ' ')
                out_file.write(f"  [Cell {i} {cell_type.upper()}] Code: {code_snippet}...\n")
                out_file.write(f"  [Output]:\n")
                lines = out_text.split('\n')
                for line in lines:
                    if line.strip():
                        out_file.write(f"    {line}\n")
                out_file.write("-" * 40 + "\n")

if __name__ == '__main__':
    notebooks = [
        'models/vit/Threshold_Tuning_and_External_Validation.ipynb',
        'final_model/Threshold_Tuning_and_External_Validation.ipynb',
        'models/cnn/Threshold_Tuning_and_External_Validation.ipynb',
        'models/vit/vit_multiseed.ipynb',
        'models/cnn/cnn_multiseed.ipynb'
    ]
    with open('scratch/extracted_results_report.txt', 'w', encoding='utf-8') as out_file:
        for nb in notebooks:
            inspect_notebook(nb, out_file)
    print("Completed inspection and wrote to scratch/extracted_results_report.txt")
