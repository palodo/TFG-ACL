import json
import sys
from pathlib import Path

def main():
    nb_path = Path('models/cnn/Threshold_Tuning_and_External_Validation.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            code = "".join(cell.get('source', []))
            code_cells.append(code)
            
    # Combine code cells, but remove plotting or interactive blocks to prevent hanging
    combined_code = []
    for code in code_cells:
        # Skip plotting code or modify it to not call plt.show()
        if "plt.show" in code:
            code = code.replace("plt.show()", "# plt.show()")
        if "fig, ax =" in code or "plt.subplots" in code:
            # We can still run the code, just make sure matplotlib is in non-interactive mode
            pass
        combined_code.append(code)
        
    full_script = "import matplotlib\nmatplotlib.use('Agg')\n" + "\n# CELL_BOUNDARY\n".join(combined_code)
    
    script_path = Path('scratch/cnn_temp_exec.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(full_script)
        
    print(f"Written combined script to {script_path}. Running it and logging outputs...")
    
if __name__ == '__main__':
    main()
