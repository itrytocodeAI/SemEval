import json, os, numpy as np

def compute(domain):
    data_dir = "task-dataset/track_a/subtask_1/eng"
    train_path = os.path.join(data_dir, f"eng_{domain}_train_alltasks.jsonl")
    data = [json.loads(l) for l in open(train_path, "r", encoding="utf-8")]
    
    # Bucket [1, 9] into 9 bins
    v_counts = np.zeros(10) # 0-9
    a_counts = np.zeros(10)
    
    total = 0
    for e in data:
        asps = e.get("Aspect_VA", []) or e.get("Quadruplet", [])
        for asp in asps:
            v, a = map(float, asp["VA"].split("#"))
            v_counts[int(round(v))] += 1
            a_counts[int(round(a))] += 1
            total += 1
            
    # Smoothing and inverse frequency
    # We use sqrt(1/freq) and normalize so that the max weight is capped
    # To be simple, we can just return the raw counts or normalized weights
    def get_weights(counts):
        counts = counts[1:10] # ignore bin 0
        freq = counts / (counts.sum() + 1e-6)
        weights = 1.0 / (np.sqrt(freq) + 1e-6)
        # Normalize weights to mean 1.0 to keep loss scale similar
        weights = weights / (weights.mean() + 1e-6)
        # Cap at 3.0
        weights = np.clip(weights, 0.5, 3.0)
        return weights.tolist()

    return {
        "v_weights": get_weights(v_counts),
        "a_weights": get_weights(a_counts)
    }

print("Restaurant:", compute("restaurant"))
print("Laptop:", compute("laptop"))
