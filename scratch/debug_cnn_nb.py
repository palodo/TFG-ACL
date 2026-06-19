import json

with open('models/cnn/Threshold_Tuning_and_External_Validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in CNN validation nb: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])
    source_str = "".join(source).strip()
    outputs = cell.get('outputs', [])
    
    # We want to see cells that are code and have outputs
    if cell_type == 'code':
        print(f"\nCell {idx}: type={cell_type}")
        print(f"Source: {source_str[:150]}...")
        print(f"Number of outputs: {len(outputs)}")
        for o_idx, out in enumerate(outputs):
            text = ""
            if 'text' in out:
                text = "".join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                text = "".join(out['data']['text/plain'])
            print(f"  Output {o_idx} type={out.get('output_type')}: {text[:200].strip()}...")
