import json, os

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

gold_path = "task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl"
pred_path = "predictions/v3_p2_task1_restaurant.jsonl"

gold_data = load_jsonl(gold_path)
pred_data = load_jsonl(pred_path)

gold_keys = set()
for e in gold_data:
    for asp in e["Aspect_VA"]:
        gold_keys.add((e["Text"].strip(), asp["Aspect"]))

pred_keys = set()
for e in pred_data:
    for asp in e["Aspect_VA"]:
        pred_keys.add((e["Text"].strip(), asp["Aspect"]))

print(f"Gold keys: {len(gold_keys)}")
print(f"Pred keys: {len(pred_keys)}")
print(f"Intersection: {len(gold_keys.intersection(pred_keys))}")

# Check first 5 differences
diff = gold_keys - pred_keys
if diff:
    print("\nSample gold keys missing in pred:")
    for d in list(diff)[:5]:
        print(d)

diff2 = pred_keys - gold_keys
if diff2:
    print("\nSample pred keys missing in gold:")
    for d in list(diff2)[:5]:
        print(d)
