import json
from pathlib import Path

def main():
    in_path = Path('final_model/Threshold_Tuning_and_External_Validation.ipynb')
    out_path = Path('models/vit/Threshold_Tuning_and_External_Validation.ipynb')
    
    with open(in_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        # Clear cell outputs and execution counts to make it a fresh notebook
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
            
        source = cell.get('source', [])
        new_source = []
        for line in source:
            # Replace working directories and output directories
            line = line.replace("os.chdir('/home/palodo2/acl_classifier/final_model')", "os.chdir('/home/palodo2/acl_classifier/models/vit')")
            line = line.replace("RESULTS_DIR = BASE_DIR / 'final_model' / 'results_threshold'", "RESULTS_DIR = BASE_DIR / 'models' / 'vit' / 'results_threshold'")
            
            # Replace checkoint paths and variable names
            line = line.replace("SWIN_SMALL_MULTISEED_DIR", "VIT_MULTISEED_DIR")
            line = line.replace("swin_small_multiseed", "vit_multiseed_base")
            line = line.replace("SWIN_SMALL_MODELS", "VIT_MODELS")
            
            # Replace model classes and variable names
            line = line.replace("SwinMultiSliceClassifier", "ViTMultiSliceClassifier")
            line = line.replace("swin_small_models", "vit_models")
            line = line.replace("swin_small_model", "vit_model")
            line = line.replace("swin_small", "vit")
            
            # Text / comment updates
            line = line.replace("Swin Small", "ViT Base")
            line = line.replace("Swin", "ViT")
            line = line.replace("swin", "vit")
            
            # Fix data folder path mapping if needed
            line = line.replace("Path('data/croatia_npy_volumes_full/splits/finetune_results')", "DATA_DIR / 'croatia_npy_volumes_full' / 'splits' / 'finetune_results'")
            
            new_source.append(line)
            
        # Specific model initialization replacement in the loading block
        cell_code = "".join(new_source)
        if "model = ViTMultiSliceClassifier()" in cell_code:
            old_loop = """for plane in PLANES:
    model_path = VIT_MODELS[plane]
    
    model = ViTMultiSliceClassifier()"""
            
            new_loop = """# Configuraciones específicas de ViT por plano (pooling y dropout)
VIT_PLANE_CONFIGS = {
    'sagittal': {
        'pooling_mode': 'attention',
        'dropout_input': 0.3,
        'dropout_dense': 0.2
    },
    'coronal': {
        'pooling_mode': 'max',
        'dropout_input': 0.35,
        'dropout_dense': 0.25
    },
    'axial': {
        'pooling_mode': 'attention',
        'dropout_input': 0.35,
        'dropout_dense': 0.25
    }
}

for plane in PLANES:
    model_path = VIT_MODELS[plane]
    
    cfg = VIT_PLANE_CONFIGS[plane]
    model = ViTMultiSliceClassifier(
        model_name="google/vit-base-patch16-224",
        num_classes=1,
        pretrained=True,
        pooling_mode=cfg['pooling_mode'],
        dropout_input=cfg['dropout_input'],
        dropout_dense=cfg['dropout_dense']
    )"""
            cell_code = cell_code.replace(old_loop, new_loop)
            new_source = [l + '\n' for l in cell_code.split('\n')][:-1]
            
        cell['source'] = new_source
        
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully converted notebook and saved clean version to {out_path}")

if __name__ == '__main__':
    main()
