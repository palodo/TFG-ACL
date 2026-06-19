import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def main():
    nb_path = Path('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')
    print(f"Reading notebook {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing notebook...")
    # Execute with path set to the directory of the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # We execute it in its parent directory
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent.absolute())}})
    
    print(f"Writing notebook back to {nb_path}...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Notebook executed and saved successfully!")

if __name__ == '__main__':
    main()
