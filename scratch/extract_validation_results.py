import json
from pathlib import Path

def analyze_validation_notebook(nb_path):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"File not found: {nb_path}")
        return
        
    print(f"\n========================================================")
    print(f"EXAMINING: {nb_path}")
    print(f"========================================================")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source_text = "".join(cell.get('source', []))
        
        # Check if cell type is code and it has outputs
        if cell_type == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                text = ""
                if 'text' in out:
                    text = "".join(out['text'])
                elif 'data' in out and 'text/plain' in out['data']:
                    text = "".join(out['data']['text/plain'])
                
                # Check for signs of final results
                if any(x in text.lower() for x in ['roc_auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'threshold', 'croatia', 'croacia', 'confusion_matrix', 'matriz de confusión', 'test set', 'validation set']):
                    # Print context of what cell was doing
                    print(f"Cell {i}:")
                    print(f"Code snippet: {source_text[:100].strip()}...")
                    print("Output:")
                    print(text.strip())
                    print("-" * 50)

if __name__ == '__main__':
    analyze_validation_notebook('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')
    analyze_validation_notebook('models/vit/Threshold_Tuning_and_External_Validation.ipynb')
    analyze_validation_notebook('final_model/Threshold_Tuning_and_External_Validation.ipynb')
