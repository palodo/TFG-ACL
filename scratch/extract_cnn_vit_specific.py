import json
from pathlib import Path

def analyze_nb(nb_path):
    print(f"\n==========================================")
    print(f"FILE: {nb_path}")
    print(f"==========================================")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
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
            
            # Print if contains validation/test/croatia results summary
            if "resumen comparativo" in text.lower() or "comparativa de ensambles" in text.lower() or "informe de validación externa" in text.lower() or "métricas:" in text.lower():
                print(f"Cell {idx} output:")
                print(text.strip())
                print("*" * 40)

print("--- CNN ---")
analyze_nb('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')

print("\n--- ViT ---")
analyze_nb('models/vit/Threshold_Tuning_and_External_Validation.ipynb')
