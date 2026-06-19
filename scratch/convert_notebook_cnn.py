import json
from pathlib import Path

def main():
    in_path = Path('models/vit/Threshold_Tuning_and_External_Validation.ipynb')
    out_dir = Path('models/cnn')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'Threshold_Tuning_and_External_Validation.ipynb'
    
    with open(in_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
            
        source = cell.get('source', [])
        new_source = []
        for line in source:
            # Replaces for directory paths
            line = line.replace("os.chdir('/home/palodo2/acl_classifier/models/vit')", "os.chdir('/home/palodo2/acl_classifier/models/cnn')")
            line = line.replace("RESULTS_DIR = BASE_DIR / 'models' / 'vit' / 'results_threshold'", "RESULTS_DIR = BASE_DIR / 'models' / 'cnn' / 'results_threshold'")
            
            # Replaces for checkpoint naming
            line = line.replace("VIT_MULTISEED_DIR", "CNN_MULTISEED_DIR")
            line = line.replace("vit_multiseed_base", "cnn_multiseed_resnet50")
            line = line.replace("VIT_MODELS", "CNN_MODELS")
            
            # Replaces for model names and classes
            line = line.replace("ViTMultiSliceClassifier", "CNNMultiSliceClassifier")
            line = line.replace("vit_models", "cnn_models")
            line = line.replace("vit_model", "cnn_model")
            line = line.replace("vit", "cnn")
            
            # Text changes
            line = line.replace("ViT Base", "CNN ResNet50")
            line = line.replace("ViT", "CNN")
            line = line.replace("VIT", "CNN")
            
            new_source.append(line)
            
        cell_code = "".join(new_source)
        
        # Replace the specific model initialization code to match CNN class signature and dropouts
        if "model = CNNMultiSliceClassifier(" in cell_code:
            old_block = """# Configuraciones específicas de CNN por plano (pooling y dropout)
CNN_PLANE_CONFIGS = {
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
    model_path = CNN_MODELS[plane]
    
    cfg = CNN_PLANE_CONFIGS[plane]
    model = CNNMultiSliceClassifier(
        model_name="google/vit-base-patch16-224",
        num_classes=1,
        pretrained=True,
        pooling_mode=cfg['pooling_mode'],
        dropout_input=cfg['dropout_input'],
        dropout_dense=cfg['dropout_dense']
    )"""
            
            new_block = """# Configuraciones específicas de CNN por plano (pooling y dropout)
CNN_PLANE_CONFIGS = {
    'sagittal': {
        'pooling_mode': 'attention',
        'dropout_input': 0.42,
        'dropout_dense': 0.32
    },
    'coronal': {
        'pooling_mode': 'max',
        'dropout_input': 0.47,
        'dropout_dense': 0.37
    },
    'axial': {
        'pooling_mode': 'attention',
        'dropout_input': 0.42,
        'dropout_dense': 0.32
    }
}

for plane in PLANES:
    model_path = CNN_MODELS[plane]
    
    cfg = CNN_PLANE_CONFIGS[plane]
    model = CNNMultiSliceClassifier(
        num_classes=1,
        pretrained=True,
        pooling_mode=cfg['pooling_mode'],
        dropout_input=cfg['dropout_input'],
        dropout_dense=cfg['dropout_dense']
    )"""
            cell_code = cell_code.replace(old_block, new_block)
            new_source = [l + '\n' for l in cell_code.split('\n')][:-1]
            
        cell['source'] = new_source
        
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Successfully converted notebook and saved clean version to {out_path}")

if __name__ == '__main__':
    main()
