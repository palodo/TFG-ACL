import json

with open('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        if outputs:
            print(f"\n--- Cell {idx} ---")
            for out in outputs:
                text_list = []
                if 'text' in out:
                    text_list = out.get('text', [])
                elif 'data' in out and 'text/plain' in out['data']:
                    text_list = out['data']['text/plain']
                
                text_out = "".join(text_list)
                print(f"Output type: {out.get('output_type')}")
                print(text_out[:500])
