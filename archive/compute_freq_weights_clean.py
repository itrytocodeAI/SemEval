import json, os, numpy as np

def compute(domain):
    data_dir = "task-dataset/track_a/subtask_1/eng"
    train_path = os.path.join(data_dir, f"eng_{domain}_train_alltasks.jsonl")
    data = [json.loads(l) for l in open(train_path, "r", encoding="utf-8")]
    v_c = np.zeros(10); a_c = np.zeros(10)
    for e in data:
        for asp in (e.get("Aspect_VA", []) or e.get("Quadruplet", [])):
            v, a = map(float, asp["VA"].split("#"))
            v_c[int(round(v))] += 1; a_c[int(round(a))] += 1
    def get_w(c):
        c = c[1:10]; f = c / (c.sum() + 1e-6); w = 1.0 / (np.sqrt(f) + 1e-6)
        w = w / (w.mean() + 1e-6); w = np.clip(w, 0.5, 3.0)
        return [round(float(x), 2) for x in w]
    return {"v": get_w(v_c), "a": get_w(a_c)}

print("REST:", compute("restaurant"))
print("LAP:", compute("laptop"))
