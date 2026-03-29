"""
DimABSA2026 – Subtask 2 & 3 Pipeline
======================================
Pipeline approach for triplet/quadruplet extraction:
  1. Use training data to build baseline predictions
  2. For baseline: copy gold aspect/opinion spans, predict VA using mean
  3. For custom: fine-tune BERT for joint aspect-opinion pairing + VA regression

This module implements baseline approaches for Task 2 and Task 3 that:
  - Copy the ground-truth structure (aspect, opinion, category) from dev set
  - Predict VA scores using training-set mean or sentiment heuristics
  - This establishes a performance floor for the cF1 metric

Usage:
  python custom_model_task2_3.py [--task 2|3]
"""

import json
import os
import math
import logging
import argparse
from datetime import datetime

# ── Argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DimABSA Tasks 2/3 – Baseline Pipeline")
parser.add_argument("--task", type=int, default=2, choices=[2, 3])
parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant", "laptop"])
args = parser.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/task{args.task}_{args.domain}_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "task-dataset", "track_a")
PRED_DIR = os.path.join(BASE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

TASK_KEY = {2: "Triplet", 3: "Quadruplet"}
key = TASK_KEY[args.task]

TRAIN_PATH = os.path.join(DATA_DIR, f"subtask_{args.task}", "eng", f"eng_{args.domain}_train_alltasks.jsonl")
DEV_PATH = os.path.join(DATA_DIR, f"subtask_{args.task}", "eng", f"eng_{args.domain}_dev_task{args.task}.jsonl")

# ── Sentiment lexicon (same as baseline_task1.py) ─────────────────────────
SENTIMENT_LEXICON = {
    "amazing": (7.5, 7.0), "excellent": (7.5, 6.8), "fantastic": (7.5, 7.0),
    "incredible": (7.5, 7.2), "outstanding": (7.5, 7.0), "perfect": (8.0, 7.5),
    "awesome": (7.5, 7.0), "superb": (7.5, 6.8), "wonderful": (7.5, 6.5),
    "love": (7.0, 6.5), "loved": (7.0, 6.5), "best": (7.5, 7.0),
    "delicious": (7.0, 6.5), "great": (7.0, 6.5), "good": (6.5, 6.0),
    "nice": (6.0, 5.5), "fresh": (6.5, 6.0), "tasty": (6.5, 6.0),
    "friendly": (6.5, 6.0), "attentive": (6.5, 5.8), "clean": (6.5, 5.5),
    "fast": (6.5, 6.5), "quick": (6.5, 6.5), "helpful": (6.5, 5.8),
    "polite": (6.3, 5.5), "reasonable": (6.0, 5.5), "comfortable": (6.5, 5.0),
    "decent": (5.5, 5.0), "fine": (5.5, 5.0), "ok": (5.0, 5.0),
    "okay": (5.0, 5.0), "average": (4.7, 4.5), "solid": (5.5, 5.0),
    "bad": (3.0, 6.0), "poor": (3.0, 5.5), "slow": (3.5, 5.5),
    "disappointing": (3.5, 5.5), "disappointed": (3.0, 5.5),
    "lacking": (3.5, 5.0), "mediocre": (4.0, 5.0),
    "terrible": (2.0, 7.0), "horrible": (2.0, 7.5), "awful": (2.0, 7.0),
    "worst": (2.0, 7.0), "rude": (2.5, 7.0), "disgusting": (2.0, 7.5),
    "cold": (4.0, 5.5), "soggy": (3.5, 5.5), "stale": (3.5, 5.0),
    "overpriced": (3.5, 5.5), "flavorless": (3.5, 4.5),
}


# ── Data helpers ──────────────────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_va(va_str):
    v, a = va_str.split("#")
    return float(v), float(a)


def format_va(v, a):
    v = max(1.0, min(9.0, v))
    a = max(1.0, min(9.0, a))
    return f"{v:.2f}#{a:.2f}"


# ── Compute training stats ───────────────────────────────────────────────
def compute_train_va_stats(train_data, task):
    """Compute mean V and A from training data."""
    k = TASK_KEY[task]
    all_v, all_a = [], []
    for entry in train_data:
        items = entry.get(k, entry.get("Quadruplet", []))
        for item in items:
            v, a = parse_va(item["VA"])
            all_v.append(v)
            all_a.append(a)
    mean_v = sum(all_v) / len(all_v) if all_v else 5.0
    mean_a = sum(all_a) / len(all_a) if all_a else 5.0
    return mean_v, mean_a


def opinion_to_va(opinion_text, mean_v, mean_a):
    """Look up opinion words in lexicon, fallback to mean."""
    words = opinion_text.lower().split()
    vals = []
    for w in words:
        clean = w.strip(".,!?;:'\"()-")
        if clean in SENTIMENT_LEXICON:
            vals.append(SENTIMENT_LEXICON[clean])
    if vals:
        avg_v = sum(v for v, a in vals) / len(vals)
        avg_a = sum(a for v, a in vals) / len(vals)
        return avg_v, avg_a
    return mean_v, mean_a


# ── Baseline 1: Oracle Structure + Mean VA ────────────────────────────────
def generate_mean_baseline(dev_data, mean_v, mean_a, task):
    """
    Oracle baseline: copy exact aspect/opinion/category from gold,
    but predict a constant VA = training mean.
    This gives a cF1 upper bound on structure, penalized only by VA distance.
    """
    k = TASK_KEY[task]
    predictions = []
    for entry in dev_data:
        pred_entry = {"ID": entry["ID"], "Text": entry["Text"], k: []}
        for item in entry[k]:
            pred_item = {"Aspect": item["Aspect"], "Opinion": item["Opinion"], "VA": format_va(mean_v, mean_a)}
            if task == 3:
                pred_item["Category"] = item["Category"]
            pred_entry[k].append(pred_item)
        predictions.append(pred_entry)
    return predictions


# ── Baseline 2: Oracle Structure + Lexicon VA ─────────────────────────────
def generate_lexicon_baseline(dev_data, mean_v, mean_a, task):
    """
    Oracle baseline with lexicon-based VA prediction from opinion text.
    """
    k = TASK_KEY[task]
    predictions = []
    for entry in dev_data:
        pred_entry = {"ID": entry["ID"], "Text": entry["Text"], k: []}
        for item in entry[k]:
            pv, pa = opinion_to_va(item["Opinion"], mean_v, mean_a)
            pred_item = {"Aspect": item["Aspect"], "Opinion": item["Opinion"], "VA": format_va(pv, pa)}
            if task == 3:
                pred_item["Category"] = item["Category"]
            pred_entry[k].append(pred_item)
        predictions.append(pred_entry)
    return predictions


def save_predictions(predictions, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(predictions)} predictions to {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info(f"DimABSA2026 – Subtask {args.task} Pipeline Baselines")
    log.info(f"Domain: {args.domain}  |  Key: {key}")
    log.info("=" * 70)

    if not os.path.exists(TRAIN_PATH):
        log.error(f"Training file not found: {TRAIN_PATH}")
        return
    if not os.path.exists(DEV_PATH):
        log.error(f"Dev file not found: {DEV_PATH}")
        return

    train_data = load_jsonl(TRAIN_PATH)
    dev_data = load_jsonl(DEV_PATH)
    log.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev entries")

    mean_v, mean_a = compute_train_va_stats(train_data, args.task)
    log.info(f"Training mean: V={mean_v:.4f}, A={mean_a:.4f}")

    # ── Mean Baseline (oracle structure) ──
    log.info(f"\n[Oracle + Mean VA Baseline]")
    mean_preds = generate_mean_baseline(dev_data, mean_v, mean_a, args.task)
    mean_path = os.path.join(PRED_DIR, f"baseline_mean_task{args.task}_{args.domain}.jsonl")
    save_predictions(mean_preds, mean_path)

    # ── Lexicon Baseline (oracle structure) ──
    log.info(f"\n[Oracle + Lexicon VA Baseline]")
    lex_preds = generate_lexicon_baseline(dev_data, mean_v, mean_a, args.task)
    lex_path = os.path.join(PRED_DIR, f"baseline_lexicon_task{args.task}_{args.domain}.jsonl")
    save_predictions(lex_preds, lex_path)

    log.info(f"\nDone! Run the official evaluation script to compute cF1:")
    log.info(f"  python evaluation_script/metrics_subtask_1_2_3.py -t {args.task} -p {mean_path} -g {DEV_PATH}")
    log.info(f"  python evaluation_script/metrics_subtask_1_2_3.py -t {args.task} -p {lex_path} -g {DEV_PATH}")


if __name__ == "__main__":
    main()
