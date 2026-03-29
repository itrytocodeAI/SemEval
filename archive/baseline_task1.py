"""
DimABSA2026 – Subtask 1: Complete Baseline Suite
==================================================
6 baselines + official evaluation + CSV/JSON logging:
  1. Random Baseline (uniform random VA)
  2. Global Mean Baseline (training mean)
  3. Per-Aspect Mean Baseline (per-aspect mean from training)
  4. Sentiment-Lexicon Baseline
  5. TextBlob Sentiment Baseline
  6. TF-IDF + SVR Baseline

Usage:  python baseline_task1.py
"""

import json, os, math, random, logging, csv
from datetime import datetime
from collections import defaultdict

# ── Logging ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(f"logs/baseline_task1_{timestamp}.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng")
PRED = os.path.join(BASE, "predictions")

# ── Helpers ──────────────────────────────────────────────────────────────
def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def parse_va(s): v,a=s.split("#"); return float(v),float(a)
def fmt_va(v,a): return f"{max(1,min(9,v)):.2f}#{max(1,min(9,a)):.2f}"

def get_aspects(entry):
    """Get aspect list from either Aspect_VA or Quadruplet format."""
    if "Aspect_VA" in entry: return entry["Aspect_VA"]
    quads = entry.get("Quadruplet", [])
    return [{"Aspect": q["Aspect"], "VA": q["VA"]} for q in quads]

def compute_rmse(gold, pred):
    gd = {e["ID"]: {a["Aspect"]: a for a in e["Aspect_VA"]} for e in gold}
    pd2 = {e["ID"]: {a["Aspect"]: a for a in e["Aspect_VA"]} for e in pred}
    gv,ga,pv,pa=[],[],[],[]
    for gid in gd:
        for asp_name, g_asp in gd[gid].items():
            gvi,gai = parse_va(g_asp["VA"])
            gv.append(gvi); ga.append(gai)
            if gid in pd2 and asp_name in pd2[gid]:
                pvi,pai = parse_va(pd2[gid][asp_name]["VA"])
            else:
                pvi,pai = 5.0, 5.0
            pv.append(pvi); pa.append(pai)
    all_g = gv+ga; all_p = pv+pa
    n = len(gv)
    rmse = math.sqrt(sum((g-p)**2 for g,p in zip(all_g,all_p))/n) if n else float("inf")
    rmse_norm = rmse / math.sqrt(128)
    # PCC
    from scipy.stats import pearsonr
    try: pcc_v = pearsonr(pv,gv)[0]
    except: pcc_v = float("nan")
    try: pcc_a = pearsonr(pa,ga)[0]
    except: pcc_a = float("nan")
    return {"RMSE_VA": rmse, "RMSE_norm": rmse_norm, "PCC_V": pcc_v, "PCC_A": pcc_a, "N": n,
            "gold_v": gv, "gold_a": ga, "pred_v": pv, "pred_a": pa}

def save_preds(preds, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in preds: f.write(json.dumps(e, ensure_ascii=False)+"\n")

# ── Sentiment Lexicon ────────────────────────────────────────────────────
LEXICON = {
    "amazing":(2.5,2.0),"excellent":(2.5,1.8),"fantastic":(2.5,2.0),"incredible":(2.5,2.2),
    "outstanding":(2.5,2.0),"perfect":(3.0,2.5),"awesome":(2.5,2.0),"superb":(2.5,1.8),
    "wonderful":(2.5,1.5),"love":(2.0,1.5),"loved":(2.0,1.5),"best":(2.5,2.0),
    "delicious":(2.0,1.5),"phenomenal":(2.5,2.0),"great":(2.0,1.5),"good":(1.5,1.0),
    "nice":(1.0,0.5),"fresh":(1.5,1.0),"tasty":(1.5,1.0),"friendly":(1.5,1.0),
    "attentive":(1.5,0.8),"clean":(1.5,0.5),"fast":(1.5,1.5),"quick":(1.5,1.5),
    "helpful":(1.5,0.8),"polite":(1.3,0.5),"reasonable":(1.0,0.5),"comfortable":(1.5,0.0),
    "decent":(0.5,0.0),"fine":(0.5,0.0),"ok":(0.0,0.0),"okay":(0.0,0.0),
    "average":(-0.3,-0.5),"solid":(0.5,0.0),
    "bad":(-2.0,1.0),"poor":(-2.0,0.5),"slow":(-1.5,0.5),"disappointing":(-1.5,0.5),
    "disappointed":(-2.0,0.5),"lacking":(-1.5,0.0),"mediocre":(-1.0,0.0),
    "terrible":(-3.0,2.0),"horrible":(-3.0,2.5),"awful":(-3.0,2.0),
    "worst":(-3.0,2.0),"rude":(-2.5,2.0),"disgusting":(-3.0,2.5),
    "cold":(-1.0,0.5),"soggy":(-1.5,0.5),"stale":(-1.5,0.0),
    "overpriced":(-1.5,0.5),"flavorless":(-1.5,-0.5),
}

# ── Baselines ────────────────────────────────────────────────────────────
def baseline_random(dev_data):
    random.seed(42)
    preds = []
    for e in dev_data:
        p = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        for a in e["Aspect_VA"]:
            p["Aspect_VA"].append({"Aspect":a["Aspect"],"VA":fmt_va(random.uniform(1,9),random.uniform(1,9))})
        preds.append(p)
    return preds

def baseline_mean(dev_data, mean_v, mean_a):
    return [{"ID":e["ID"],"Text":e["Text"],"Aspect_VA":
        [{"Aspect":a["Aspect"],"VA":fmt_va(mean_v,mean_a)} for a in e["Aspect_VA"]]} for e in dev_data]

def baseline_per_aspect_mean(dev_data, asp_means, mean_v, mean_a):
    preds = []
    for e in dev_data:
        p = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        for a in e["Aspect_VA"]:
            key = a["Aspect"].lower().strip()
            if key in asp_means:
                pv,pa = asp_means[key]
            else:
                pv,pa = mean_v, mean_a
            p["Aspect_VA"].append({"Aspect":a["Aspect"],"VA":fmt_va(pv,pa)})
        preds.append(p)
    return preds

def baseline_lexicon(dev_data, mean_v, mean_a):
    preds = []
    for e in dev_data:
        p = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        for a in e["Aspect_VA"]:
            words = e["Text"].lower().split()
            offs = [(LEXICON[w.strip(".,!?;:'\"()-")]) for w in words if w.strip(".,!?;:'\"()-") in LEXICON]
            if offs:
                pv = max(1,min(9, 5+sum(o[0] for o in offs)/len(offs)))
                pa = max(1,min(9, 5+sum(o[1] for o in offs)/len(offs)))
            else:
                pv,pa = mean_v, mean_a
            p["Aspect_VA"].append({"Aspect":a["Aspect"],"VA":fmt_va(pv,pa)})
        preds.append(p)
    return preds

def baseline_textblob(dev_data, mean_v, mean_a):
    try:
        from textblob import TextBlob
    except ImportError:
        log.warning("  textblob not installed, falling back to lexicon")
        return baseline_lexicon(dev_data, mean_v, mean_a)
    preds = []
    for e in dev_data:
        p = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        blob = TextBlob(e["Text"])
        pol = blob.sentiment.polarity   # [-1, 1]
        sub = blob.sentiment.subjectivity  # [0, 1]
        # Map polarity to V: [-1,1] → [1,9]
        v = max(1, min(9, 5 + pol * 4))
        # Map subjectivity to A: [0,1] → [3,8] (subjective → higher arousal)
        a = max(1, min(9, 3 + sub * 5))
        for asp in e["Aspect_VA"]:
            p["Aspect_VA"].append({"Aspect":asp["Aspect"],"VA":fmt_va(v, a)})
        preds.append(p)
    return preds

def baseline_tfidf_svr(train_data, dev_data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline

    # Prepare training data: text+aspect → V, A
    train_texts, train_v, train_a = [], [], []
    for e in train_data:
        asps = get_aspects(e)
        for a in asps:
            train_texts.append(e["Text"] + " [SEP] " + a["Aspect"])
            v,ar = parse_va(a["VA"])
            train_v.append(v); train_a.append(ar)

    log.info(f"  TF-IDF+SVR: training on {len(train_texts)} samples...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = tfidf.fit_transform(train_texts)

    svr_v = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_a = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_v.fit(X_train, train_v)
    svr_a.fit(X_train, train_a)

    preds = []
    for e in dev_data:
        p = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        for a in e["Aspect_VA"]:
            x = tfidf.transform([e["Text"] + " [SEP] " + a["Aspect"]])
            pv = max(1,min(9, svr_v.predict(x)[0]))
            pa = max(1,min(9, svr_a.predict(x)[0]))
            p["Aspect_VA"].append({"Aspect":a["Aspect"],"VA":fmt_va(pv,pa)})
        preds.append(p)
    return preds

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    log.info("="*70)
    log.info("DimABSA2026 – Complete Baseline Suite (Task 1)")
    log.info("="*70)

    all_results = []

    for domain in ["restaurant", "laptop"]:
        train_path = os.path.join(DATA, f"eng_{domain}_train_alltasks.jsonl")
        dev_path = os.path.join(DATA, f"eng_{domain}_dev_task1.jsonl")
        if not os.path.exists(train_path) or not os.path.exists(dev_path):
            log.warning(f"  Missing files for {domain}, skipping")
            continue

        log.info(f"\n{'═'*60}\n  Domain: {domain}\n{'═'*60}")
        train_data = load_jsonl(train_path)
        dev_data = load_jsonl(dev_path)
        log.info(f"  Loaded {len(train_data)} train, {len(dev_data)} dev")

        # Compute stats
        all_v, all_a = [], []
        asp_va = defaultdict(lambda: ([],[]))
        for e in train_data:
            for a in get_aspects(e):
                v,ar = parse_va(a["VA"])
                all_v.append(v); all_a.append(ar)
                k = a["Aspect"].lower().strip()
                asp_va[k][0].append(v); asp_va[k][1].append(ar)
        mean_v = sum(all_v)/len(all_v); mean_a = sum(all_a)/len(all_a)
        asp_means = {k:(sum(vs)/len(vs), sum(ars)/len(ars)) for k,(vs,ars) in asp_va.items()}
        log.info(f"  Train: {len(all_v)} VA, mean_V={mean_v:.3f}, mean_A={mean_a:.3f}, unique_aspects={len(asp_means)}")

        baselines = [
            ("Random", baseline_random(dev_data)),
            ("Global Mean", baseline_mean(dev_data, mean_v, mean_a)),
            ("Per-Aspect Mean", baseline_per_aspect_mean(dev_data, asp_means, mean_v, mean_a)),
            ("Sentiment Lexicon", baseline_lexicon(dev_data, mean_v, mean_a)),
            ("TextBlob", baseline_textblob(dev_data, mean_v, mean_a)),
            ("TF-IDF + SVR", baseline_tfidf_svr(train_data, dev_data)),
        ]

        for name, preds in baselines:
            fname = f"baseline_{name.lower().replace(' ','_').replace('+','_')}_task1_{domain}.jsonl"
            save_preds(preds, os.path.join(PRED, fname))
            res = compute_rmse(dev_data, preds)
            log.info(f"  {name:<20} RMSE_VA={res['RMSE_VA']:.4f}  PCC_V={res['PCC_V']:.4f}  PCC_A={res['PCC_A']:.4f}")
            all_results.append({"model":name, "domain":domain, "task":1,
                "RMSE_VA":res["RMSE_VA"], "RMSE_norm":res["RMSE_norm"],
                "PCC_V":float(res["PCC_V"]) if not math.isnan(res["PCC_V"]) else None,
                "PCC_A":float(res["PCC_A"]) if not math.isnan(res["PCC_A"]) else None,
                "N":res["N"], "timestamp":timestamp,
                "gold_v":res["gold_v"],"gold_a":res["gold_a"],
                "pred_v":res["pred_v"],"pred_a":res["pred_a"]})

    # ── Save results as CSV and JSON ──
    csv_path = os.path.join("logs", "baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","domain","task","RMSE_VA","RMSE_norm","PCC_V","PCC_A","N","timestamp"])
        w.writeheader()
        for r in all_results:
            w.writerow({k:r[k] for k in w.fieldnames})
    log.info(f"\nCSV saved: {csv_path}")

    json_path = os.path.join("logs", "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump([{k:v for k,v in r.items() if k not in ("gold_v","gold_a","pred_v","pred_a")} for r in all_results], f, indent=2)
    log.info(f"JSON saved: {json_path}")

    # Save detailed predictions for plotting
    detailed_path = os.path.join("logs", "baseline_detailed_predictions.json")
    with open(detailed_path, "w") as f:
        json.dump(all_results, f)

    # ── Summary Table ──
    log.info(f"\n{'='*70}\n{'Model':<20} {'Domain':<10} {'RMSE_VA':>8} {'PCC_V':>8} {'PCC_A':>8}")
    log.info("─"*60)
    for r in all_results:
        pv = f"{r['PCC_V']:.4f}" if r['PCC_V'] is not None else "NaN"
        pa = f"{r['PCC_A']:.4f}" if r['PCC_A'] is not None else "NaN"
        log.info(f"{r['model']:<20} {r['domain']:<10} {r['RMSE_VA']:>8.4f} {pv:>8} {pa:>8}")

if __name__ == "__main__":
    main()
