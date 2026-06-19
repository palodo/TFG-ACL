import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import sys

def main():
    nb_path = Path('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')
    print(f"Reading notebook {nb_path}...", flush=True)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # Replace cuda with cpu in code cells
    print("Modifying notebook to force CPU execution...", flush=True)
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            # Replace device assignment
            if 'torch.device("cuda"' in source or "torch.device('cuda'" in source:
                source = source.replace('torch.device("cuda" if torch.cuda.is_available() else "cpu")', 'torch.device("cpu")')
                source = source.replace("torch.device('cuda' if torch.cuda.is_available() else 'cpu')", "torch.device('cpu')")
                source = source.replace('device = "cuda"', 'device = "cpu"')
                source = source.replace("device = 'cuda'", "device = 'cpu'")
                cell['source'] = source
                
    print("Executing notebook on CPU...", flush=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent.absolute())}})
        print("Notebook executed successfully on CPU!", flush=True)
        
        print(f"Writing notebook back to {nb_path}...", flush=True)
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Saved notebook with outputs!", flush=True)
    except Exception as e:
        print("ERROR during CPU notebook execution:", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
