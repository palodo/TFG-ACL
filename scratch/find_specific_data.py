import json
from pathlib import Path

def print_nb_outputs(nb_path, keywords_any, keywords_all=None):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"File not found: {nb_path}")
        return
        
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    print(f"\n==========================================")
    print(f"FILE: {nb_path}")
    print(f"==========================================")
    
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type != 'code':
            continue
            
        outputs = cell.get('outputs', [])
        found_in_outputs = False
        text_out = ""
        
        for out in outputs:
            text_list = []
            if 'text' in out:
                text_list = out.get('text', [])
            elif 'data' in out and 'text/plain' in out['data']:
                text_list = out['data']['text/plain']
            text_out += "".join(text_list)
            
        if not text_out.strip():
            continue
            
        # Match keywords
        match = False
        text_out_lower = text_out.lower()
        if keywords_all:
            match = all(k.lower() in text_out_lower for k in keywords_all)
        else:
            match = any(k.lower() in text_out_lower for k in keywords_any)
            
        if match:
            print(f"\n--- Cell {idx} ---")
            source_lines = "".join(cell.get('source', []))
            print(f"Code: {source_lines[:150].strip()}...")
            print("Output:")
            print(text_out.strip())
            print("-" * 50)

if __name__ == '__main__':
    # Let's inspect each notebook for specific keywords
    
    # 1. CNN Threshold Tuning & External Validation
    print("=== CNN TUNING & VALIDATION ===")
    print_nb_outputs('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', ['threshold', 'croatia', 'confusion', 'auc', 'test'], ['optimal'])
    print_nb_outputs('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', ['croatia'])
    
    # 2. ViT Threshold Tuning & External Validation
    print("\n=== ViT TUNING & VALIDATION ===")
    print_nb_outputs('models/vit/Threshold_Tuning_and_External_Validation.ipynb', ['threshold', 'croatia', 'confusion', 'auc', 'test'], ['optimal'])
    print_nb_outputs('models/vit/Threshold_Tuning_and_External_Validation.ipynb', ['croatia'])
    
    # 3. Final Model Threshold Tuning & External Validation
    print("\n=== FINAL MODEL (SWIN SMALL) ===")
    print_nb_outputs('final_model/Threshold_Tuning_and_External_Validation.ipynb', ['threshold', 'croatia', 'confusion', 'auc', 'test'], ['optimal'])
    print_nb_outputs('final_model/Threshold_Tuning_and_External_Validation.ipynb', ['croatia'])
    
    # 4. ViT Multiseed
    print("\n=== ViT MULTISEED ===")
    print_nb_outputs('models/vit/vit_multiseed.ipynb', ['resumen', 'mean', 'std', 'estadístico', 'average'], ['mean', 'std'])
    
    # 5. CNN Multiseed
    print("\n=== CNN MULTISEED ===")
    print_nb_outputs('models/cnn/cnn_multiseed.ipynb', ['resumen', 'mean', 'std', 'estadístico', 'average'], ['mean', 'std'])
