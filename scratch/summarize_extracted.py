import re

with open('scratch/extracted_results_report.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's search for patterns in the text and print them
print("Searching for sections and metrics in extracted report:")
print("=" * 60)

lines = text.split('\n')
i = 0
while i < len(lines):
    line = lines[i]
    # If the line contains a notebook name, print it
    if "INSPECTING:" in line:
        print("\n" + line)
        print("=" * len(line))
        i += 1
        continue
    
    # If line is a header
    if "Cell" in line and "(MD Header):" in line:
        print("  " + line.strip())
        i += 1
        continue
        
    # If the line is an output block with metrics, look for specific keys
    if "  [Output]:" in line:
        # Collect lines of the output block
        output_lines = []
        i += 1
        while i < len(lines) and not lines[i].startswith("----------------------------------------") and not lines[i].startswith("===") and not lines[i].startswith("INSPECTING:"):
            output_lines.append(lines[i])
            i += 1
            
        output_text = "\n".join(output_lines)
        # Check if the output has relevant keywords
        keywords = ['auc', 'precision', 'recall', 'f1', 'specificity', 'accuracy', 'threshold', 'croatia', 'croacia', 'confusion', 'matrix', 'seed', 'std', 'mean', '±', 'sensibilidad', 'especificidad']
        if any(kw in output_text.lower() for kw in keywords):
            # Print the output block (only lines with content)
            clean_lines = [l.strip() for l in output_lines if l.strip()]
            if clean_lines:
                print("    [Output Block]:")
                # print first few lines of interest or the whole thing if short
                for cl in clean_lines[:15]:
                    print("      " + cl)
                if len(clean_lines) > 15:
                    print(f"      ... (truncated {len(clean_lines) - 15} lines)")
        continue
    i += 1
