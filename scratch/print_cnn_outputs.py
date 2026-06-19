import json
with open('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        if outputs:
            print(f"Cell {i} has {len(outputs)} outputs:")
            for out in outputs:
                text = "".join(out.get('text', [])) or str(out.get('data', {}))
                print(text[:200])
