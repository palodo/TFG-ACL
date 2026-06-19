import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import sys

def main():
    nb_path = Path('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')
    print(f"Reading notebook {nb_path}...", flush=True)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing notebook...", flush=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent.absolute())}})
        print("Notebook executed successfully!", flush=True)
        
        print(f"Writing notebook back to {nb_path}...", flush=True)
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Saved notebook with outputs!", flush=True)
    except Exception as e:
        print("ERROR during notebook execution:", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
