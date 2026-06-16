import csv

def read_summary(filepath):
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'computed':
                results.append(row)
    return results

rows1 = read_summary('results/original-20260617_022147/original_summary.csv')
rows2 = read_summary('results/original-20260617_022920/original_summary.csv')

# Exclude failed reld and neuopt from rows1
filtered_rows1 = [r for r in rows1 if r['method'] not in ['reld', 'neuopt']]
all_rows = filtered_rows1 + rows2

# Format markdown table
print("| Method | Problem | Size Group | Count | Avg Gap (%) | Avg Time (s) |")
print("|---|---|---|---|---|---|")
for r in sorted(all_rows, key=lambda x: (x['problem'], x['method'], x['size_group'])):
    print(f"| {r['method']} | {r['problem']} | {r['size_group']} | {r['count']} | {float(r['avg_gap_percent']):.2f} | {float(r['avg_time_seconds']):.2f} |")
