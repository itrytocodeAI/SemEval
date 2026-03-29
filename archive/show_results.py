import json, csv
data = json.load(open('logs/baseline_results.json'))
print(f"{'Model':<22} {'Domain':<12} {'RMSE_VA':>8} {'PCC_V':>8} {'PCC_A':>8}")
print("-"*62)
for d in data:
    pv = f"{d['PCC_V']:.4f}" if d.get('PCC_V') is not None else "NaN"
    pa = f"{d['PCC_A']:.4f}" if d.get('PCC_A') is not None else "NaN"
    print(f"{d['model']:<22} {d['domain']:<12} {d['RMSE_VA']:>8.4f} {pv:>8} {pa:>8}")
