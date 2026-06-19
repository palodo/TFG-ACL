import json
import sys

def main():
    nb_path = 'final_model/Threshold_Tuning_and_External_Validation.ipynb'
    out_path = 'scratch/Threshold_Tuning_and_External_Validation_code.txt'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type')
            out.write(f"\n# ==========================================\n")
            out.write(f"# CELL {i} ({cell_type.upper()})\n")
            out.write(f"# ==========================================\n")
            source = "".join(cell.get('source', []))
            out.write(source)
            out.write("\n")
            
    print(f"Extracted notebook cells to {out_path}")

if __name__ == '__main__':
    main()
