"""
DimABSA2026 – Master Experiment Runner
=======================================
Orchestrates all baselines and custom models, runs official evaluation,
and produces comparison tables.

Usage:
  python run_experiments.py [--skip_bert] [--epochs 10]
"""

import json
import os
import sys
import subprocess
import logging
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/experiments_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "task-dataset", "track_a")
PRED_DIR = os.path.join(BASE, "predictions")
EVAL_SCRIPT = os.path.join(BASE, "evaluation_script", "metrics_subtask_1_2_3.py")

os.makedirs(PRED_DIR, exist_ok=True)


def run_cmd(cmd, desc=""):
    """Run a command and capture output."""
    log.info(f"  → Running: {desc or cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log.info(f"    {line}")
    if result.returncode != 0:
        log.error(f"  ✗ Command failed (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                log.error(f"    {line}")
    return result


def run_official_eval(task, pred_path, gold_path, desc=""):
    """Run the official evaluation script and capture results."""
    cmd = [sys.executable, EVAL_SCRIPT, "-t", str(task), "-p", pred_path, "-g", gold_path]
    log.info(f"\n  📊 Official Eval: {desc}")
    result = run_cmd(cmd, desc)
    return result.stdout


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_bert", action="store_true", help="Skip BERT training (use existing predictions)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("DimABSA2026 – Master Experiment Runner")
    log.info("=" * 80)

    all_results = []

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Task 1 Baselines
    # ════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'═' * 80}")
    log.info("PHASE 1: Task 1 – VA Regression Baselines")
    log.info(f"{'═' * 80}")

    result = run_cmd([sys.executable, os.path.join(BASE, "baseline_task1.py")], "Task 1 Baselines")

    # Run official evaluation on baseline predictions
    for domain in ["restaurant", "laptop"]:
        gold_path = os.path.join(DATA_DIR, "subtask_1", "eng", f"eng_{domain}_dev_task1.jsonl")
        if not os.path.exists(gold_path):
            log.warning(f"  Dev file not found for {domain}, skipping")
            continue

        for baseline_type in ["mean", "lexicon"]:
            pred_path = os.path.join(PRED_DIR, f"baseline_{baseline_type}_task1_{domain}.jsonl")
            if os.path.exists(pred_path):
                output = run_official_eval(1, pred_path, gold_path,
                                           f"Task1 {baseline_type} baseline ({domain})")
                all_results.append(f"Task1/{baseline_type}/{domain}:\n{output}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Task 1 Custom BERT Model
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_bert:
        log.info(f"\n{'═' * 80}")
        log.info("PHASE 2: Task 1 – Custom BERT VA Regression")
        log.info(f"{'═' * 80}")

        for domain in ["restaurant", "laptop"]:
            train_path = os.path.join(DATA_DIR, "subtask_1", "eng", f"eng_{domain}_train_alltasks.jsonl")
            if not os.path.exists(train_path):
                log.warning(f"  Training file not found for {domain}, skipping")
                continue

            log.info(f"\n  Training BERT for {domain}...")
            result = run_cmd(
                [sys.executable, os.path.join(BASE, "custom_model_task1.py"),
                 "--domain", domain, "--epochs", str(args.epochs)],
                f"BERT training ({domain})"
            )

            gold_path = os.path.join(DATA_DIR, "subtask_1", "eng", f"eng_{domain}_dev_task1.jsonl")
            pred_path = os.path.join(PRED_DIR, f"custom_bert_task1_{domain}.jsonl")
            if os.path.exists(pred_path) and os.path.exists(gold_path):
                output = run_official_eval(1, pred_path, gold_path,
                                           f"Task1 BERT custom ({domain})")
                all_results.append(f"Task1/BERT/{domain}:\n{output}")
    else:
        log.info("\n  Skipping BERT training (--skip_bert)")
        for domain in ["restaurant", "laptop"]:
            pred_path = os.path.join(PRED_DIR, f"custom_bert_task1_{domain}.jsonl")
            gold_path = os.path.join(DATA_DIR, "subtask_1", "eng", f"eng_{domain}_dev_task1.jsonl")
            if os.path.exists(pred_path) and os.path.exists(gold_path):
                output = run_official_eval(1, pred_path, gold_path,
                                           f"Task1 BERT custom ({domain})")
                all_results.append(f"Task1/BERT/{domain}:\n{output}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: Task 2 & 3 Baselines
    # ════════════════════════════════════════════════════════════════════════
    for task_id in [2, 3]:
        log.info(f"\n{'═' * 80}")
        log.info(f"PHASE 3: Task {task_id} – {'Triplet' if task_id == 2 else 'Quadruplet'} Baselines")
        log.info(f"{'═' * 80}")

        for domain in ["restaurant", "laptop"]:
            train_path = os.path.join(DATA_DIR, f"subtask_{task_id}", "eng", f"eng_{domain}_train_alltasks.jsonl")
            if not os.path.exists(train_path):
                log.warning(f"  Training file not found for task{task_id}/{domain}, skipping")
                continue

            result = run_cmd(
                [sys.executable, os.path.join(BASE, "custom_model_task2_3.py"),
                 "--task", str(task_id), "--domain", domain],
                f"Task {task_id} baselines ({domain})"
            )

            gold_path = os.path.join(DATA_DIR, f"subtask_{task_id}", "eng", f"eng_{domain}_dev_task{task_id}.jsonl")
            if not os.path.exists(gold_path):
                continue

            for baseline_type in ["mean", "lexicon"]:
                pred_path = os.path.join(PRED_DIR, f"baseline_{baseline_type}_task{task_id}_{domain}.jsonl")
                if os.path.exists(pred_path):
                    output = run_official_eval(task_id, pred_path, gold_path,
                                               f"Task{task_id} {baseline_type} baseline ({domain})")
                    all_results.append(f"Task{task_id}/{baseline_type}/{domain}:\n{output}")

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'═' * 80}")
    log.info("ALL RESULTS SUMMARY")
    log.info(f"{'═' * 80}")
    for r in all_results:
        log.info(r)

    # Save results
    results_path = os.path.join("logs", f"all_results_{timestamp}.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(all_results))
    log.info(f"\nResults saved to {results_path}")
    log.info("Done!")


if __name__ == "__main__":
    main()
