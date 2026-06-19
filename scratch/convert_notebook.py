import json
from pathlib import Path

def main():
    in_path = Path('final_model/Threshold_Tuning_and_External_Validation.ipynb')
    out_dir = Path('models/vit')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'Threshold_Tuning_and_External_Validation.ipynb'
    
    with open(in_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        if cell_type == 'markdown':
            # Update titles and texts to mention ViT instead of Swin Small
            source = cell.get('source', [])
            new_source = []
            for line in source:
                line = line.replace('Swin Small', 'ViT Base')
                line = line.replace('swin_small_models', 'vit_models')
                line = line.replace('swin_small', 'vit')
                line = line.replace('Swin Small', 'ViT Base')
                line = line.replace('Swin', 'ViT')
                new_source.append(line)
            cell['source'] = new_source
            
        elif cell_type == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # 1. working directory
                line = line.replace("os.chdir('/home/palodo2/acl_classifier/final_model')", "os.chdir('/home/palodo2/acl_classifier/models/vit')")
                # 2. results directory
                line = line.replace("RESULTS_DIR = BASE_DIR / 'final_model' / 'results_threshold'", "RESULTS_DIR = BASE_DIR / 'models' / 'vit' / 'results_threshold'")
                # 3. Model weights configuration
                line = line.replace("SWIN_SMALL_MULTISEED_DIR = CHECKPOINTS_DIR / 'swin_small_multiseed'", "VIT_MULTISEED_DIR = CHECKPOINTS_DIR / 'vit_multiseed_base'")
                line = line.replace("SWIN_SMALL_MODELS = {", "VIT_MODELS = {")
                line = line.replace("SWIN_SMALL_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth'", "VIT_MULTISEED_DIR / 'best_sagittal_multiseed_final.pth'")
                line = line.replace("SWIN_SMALL_MULTISEED_DIR / 'best_coronal_multiseed_final.pth'", "VIT_MULTISEED_DIR / 'best_coronal_multiseed_final.pth'")
                line = line.replace("SWIN_SMALL_MULTISEED_DIR / 'best_axial_multiseed_final.pth'", "VIT_MULTISEED_DIR / 'best_axial_multiseed_final.pth'")
                
                # 4. Model import and initialization
                line = line.replace("from src.models import SimpleCNNSelector, SwinMultiSliceClassifier", "from src.models import SimpleCNNSelector, ViTMultiSliceClassifier")
                
                # 5. Load model block
                line = line.replace("print(\"\\nCargando modelos Swin Small Multiseed...\")", "print(\"\\nCargando modelos ViT Multiseed...\")")
                line = line.replace("swin_small_models = {}", "vit_models = {}")
                
                line = line.replace("model_path = SWIN_SMALL_MODELS[plane]", "model_path = VIT_MODELS[plane]")
                
                # 6. Model class name replacement
                line = line.replace("model = SwinMultiSliceClassifier()", "model = ViTMultiSliceClassifier()") # We'll replace this whole loop if needed or just handle it. Let's do a more precise replacement below.
                
                # 7. Model variables
                line = line.replace("swin_small_models[plane] = model", "vit_models[plane] = model")
                line = line.replace("swin_small_models['sagittal']", "vit_models['sagittal']")
                
                # 8. Path modifications
                line = line.replace("Path('data/croatia_npy_volumes_full/splits/finetune_results')", "DATA_DIR / 'croatia_npy_volumes_full' / 'splits' / 'finetune_results'")
                
                new_source.append(line)
            
            # Now let's handle the specific cell block where model is loaded to pass plane-specific config.
            cell_code = "".join(new_source)
            if "model = ViTMultiSliceClassifier()" in cell_code:
                # We want to replace the standard loop with a plane-config aware loop
                old_loop = """for plane in PLANES:
    model_path = VIT_MODELS[plane]
    
    model = ViTMultiSliceClassifier()"""
                
                new_loop = """# Configuraciones específicas de ViT por plano ( pooling y dropout)
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
                new_source = [line + '\n' for line in cell_code.split('\n')][:-1]
                
            cell['source'] = new_source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully converted notebook and saved to {out_path}")

if __name__ == '__main__':
    main()
