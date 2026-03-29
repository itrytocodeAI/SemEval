import json, numpy as np

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def parse_va(s):
    v, a = s.split("#")
    return float(v), float(a)

pred_data = load_jsonl("predictions/v3_p2_task1_restaurant.jsonl")
pv, pa = [], []
for e in pred_data:
    for asp in e["Aspect_VA"]:
        v, a = parse_va(asp["VA"])
        pv.append(v)
        pa.append(a)

print(f"Valence: mean={np.mean(pv):.2f}, std={np.std(pv):.2f}, min={np.min(pv):.2f}, max={np.max(pv):.2f}")
print(f"Arousal: mean={np.mean(pa):.2f}, std={np.std(pa):.2f}, min={np.min(pa):.2f}, max={np.max(pa):.2f}")

# Check first 10
print("\nFirst 10 V-A pairs:")
for i in range(10):
    print(f"{pv[i]:.2f}#{pa[i]:.2f}")
