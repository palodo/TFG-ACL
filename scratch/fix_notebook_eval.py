import json
from pathlib import Path

def main():
    nb_path = Path('models/vit/Threshold_Tuning_and_External_Validation.ipynb')
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            cell_code = "".join(source)
            
            # 1. Update Cell 34 to run baseline and post-test on Croatia test split
            if "baseline_test_probs, baseline_test_labels = get_predictions_on_dataset(" in cell_code:
                # Replace baseline and post-test data loader
                cell_code = cell_code.replace(
                    "baseline_test_probs, baseline_test_labels = get_predictions_on_dataset(\n    vit_models['sagittal'],\n    test_dataloaders['sagittal'],\n    DEVICE,\n)",
                    "baseline_test_probs, baseline_test_labels = get_predictions_on_dataset(\n    vit_models['sagittal'],\n    test_loader_ft,\n    DEVICE,\n)"
                )
                cell_code = cell_code.replace(
                    "post_test_probs, post_test_labels = get_predictions_on_dataset(\n    ft_model,\n    test_dataloaders['sagittal'],\n    DEVICE,\n)",
                    "post_test_probs, post_test_labels = get_predictions_on_dataset(\n    ft_model,\n    test_loader_ft,\n    DEVICE,\n)"
                )
                
                # Append validation prediction generation at the end of the cell
                if "val_probs_ft, val_labels_ft =" not in cell_code:
                    cell_code += "\n\n# Obtener predicciones en la validación de Croacia para tunear el threshold\nval_probs_ft, val_labels_ft = get_predictions_on_dataset(\n    ft_model,\n    val_loader_ft,\n    DEVICE,\n)\nprint(f'Generadas predicciones de validación para Croatia (val_probs_ft): {len(val_probs_ft)}')"
                
                source = [line + '\n' for line in cell_code.split('\n')][:-1]
                cell['source'] = source
                
            # 2. Update Cell 38 to prioritize Croatia validation split for threshold tuning
            if "probs = _get(['val_ensemble_probs'" in cell_code:
                cell_code = cell_code.replace(
                    "probs = _get(['val_ensemble_probs','val_probs','val_probs_ft','val_ensemble_probs','val_probs'])",
                    "probs = _get(['val_probs_ft', 'val_ensemble_probs', 'val_probs'])"
                )
                cell_code = cell_code.replace(
                    "labels = _get(['val_all_labels','val_labels','val_true','val_ensemble_labels','val_labels_ft'])",
                    "labels = _get(['val_labels_ft', 'val_all_labels', 'val_labels'])"
                )
                source = [line + '\n' for line in cell_code.split('\n')][:-1]
                cell['source'] = source
                
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print("Successfully adjusted evaluation and tuning targets to Croatia dataset.")

if __name__ == '__main__':
    main()
