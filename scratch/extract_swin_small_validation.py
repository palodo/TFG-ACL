import json
from pathlib import Path

def analyze_swin_sel(nb_path):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"File not found: {nb_path}")
        return
        
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    print(f"\n==========================================")
    print(f"INSPECTING Swin Selection Notebook: {nb_path}")
    print(f"==========================================")
    
    for idx, cell in enumerate(nb.get('cells', [])):
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
                
            if any(term in text.lower() for term in ['swin_small_multiseed', 'mean', 'std', 'auc', 'f1', 'resumen', 'seed']):
                print(f"--- Cell {idx} output ---")
                print(text.strip())
                print("*" * 50)

if __name__ == '__main__':
    analyze_swin_sel('final_model/Swin_Small_Model_Selection_Validation.ipynb')
