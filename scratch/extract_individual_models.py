import json
from pathlib import Path

def get_final_results_summary(nb_path, model_name):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"File not found: {nb_path}")
        return
        
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    print(f"\n========================================================")
    print(f"RESULTS SUMMARY FOR {model_name} ({nb_path.name})")
    print(f"========================================================")
    
    # We want to search for cell outputs that display comparative tables or final metrics
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
                
            # If the text has summary keywords
            if "resumen comparativo" in text.lower() or "comparativa de ensambles" in text.lower() or "informe de validación externa" in text.lower() or "métricas en croacia" in text.lower():
                print(f"[Cell {idx} Output]:")
                print(text.strip())
                print("-" * 50)
            elif "optimal threshold" in text.lower() and "precision" in text.lower():
                print(f"[Cell {idx} Output (Threshold)]: ")
                print(text.strip())
                print("-" * 50)
            elif "matriz de confusión" in text.lower() and "vp:" in text.lower():
                print(f"[Cell {idx} Output (Confusion Matrix)]: ")
                print(text.strip())
                print("-" * 50)

if __name__ == '__main__':
    get_final_results_summary('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', 'CNN')
    get_final_results_summary('models/vit/Threshold_Tuning_and_External_Validation.ipynb', 'ViT')
    get_final_results_summary('final_model/Threshold_Tuning_and_External_Validation.ipynb', 'Swin Small (Final Model)')
