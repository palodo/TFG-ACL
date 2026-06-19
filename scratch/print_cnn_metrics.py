import json

with open('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        print(f"\n--- Cell {idx} ---")
        print("Source:", "".join(cell.get('source', []))[:100].strip())
        print("Outputs count:", len(outputs))
        for o_idx, out in enumerate(outputs):
            text = ""
            if 'text' in out:
                text = "".join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                text = "".join(out['data']['text/plain'])
            print(f"  Output {o_idx}: {text[:150].strip()}")
