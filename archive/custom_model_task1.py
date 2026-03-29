"""
DimABSA2026 – Subtask 1 Custom Model
=====================================
Fine-tune BERT for aspect-level VA regression.

Architecture:
  Input:  [CLS] <text> [SEP] <aspect> [SEP]
  Model:  BERT → [CLS] hidden → 2-layer MLP → (V, A)
  Loss:   MSE on V and A
  Training: AdamW, linear warmup, 10 epochs

Usage:
  python custom_model_task1.py [--epochs 10] [--batch_size 16] [--lr 2e-5]
"""

import json
import os
import sys
import math
import logging
import argparse
import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ── Argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DimABSA Task 1 – BERT VA Regression")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant", "laptop"])
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--device", type=str, default="auto")
args = parser.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/custom_task1_{args.domain}_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
log.info(f"Device: {device}")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng")
TRAIN_PATH = os.path.join(DATA_DIR, f"eng_{args.domain}_train_alltasks.jsonl")
DEV_PATH = os.path.join(DATA_DIR, f"eng_{args.domain}_dev_task1.jsonl")
PRED_DIR = os.path.join(BASE, "predictions")
CKPT_DIR = os.path.join(BASE, "checkpoints")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Import transformers ──────────────────────────────────────────────────
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ── Data ──────────────────────────────────────────────────────────────────
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


class AspectVADataset(Dataset):
    """
    Each sample is (text, aspect) → (V, A).
    One review may produce multiple samples (one per aspect).
    """
    def __init__(self, data, max_len=128, is_train=True):
        self.samples = []
        self.max_len = max_len
        for entry in data:
            text = entry["Text"]
            # Training data uses Quadruplet key, dev uses Aspect_VA
            aspects = entry.get("Aspect_VA", [])
            if not aspects:
                # Try Quadruplet (training format)
                quads = entry.get("Quadruplet", [])
                for q in quads:
                    v, a = parse_va(q["VA"])
                    self.samples.append({
                        "text": text,
                        "aspect": q["Aspect"],
                        "valence": v,
                        "arousal": a,
                        "id": entry["ID"],
                    })
            else:
                for asp in aspects:
                    v, a = parse_va(asp["VA"])
                    self.samples.append({
                        "text": text,
                        "aspect": asp["Aspect"],
                        "valence": v,
                        "arousal": a,
                        "id": entry["ID"],
                    })
        log.info(f"  Dataset: {len(self.samples)} aspect-level samples from {len(data)} entries")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoding = tokenizer(
            s["text"],
            s["aspect"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "valence": torch.tensor(s["valence"], dtype=torch.float),
            "arousal": torch.tensor(s["arousal"], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────
class BertVARegressor(nn.Module):
    """BERT + 2-layer MLP regression head for (V, A) prediction."""

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # predict (V, A)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        va = self.regressor(cls_output)  # (batch, 2)
        return va


# ── Training ──────────────────────────────────────────────────────────────
def get_linear_schedule(optimizer, num_warmup_steps, num_training_steps):
    """Simple linear warmup then linear decay scheduler."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        targets = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds_v, all_preds_a = [], []
    all_gold_v, all_gold_a = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            targets = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device)

            preds = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            n_batches += 1

            all_preds_v.extend(preds[:, 0].cpu().numpy().tolist())
            all_preds_a.extend(preds[:, 1].cpu().numpy().tolist())
            all_gold_v.extend(batch["valence"].numpy().tolist())
            all_gold_a.extend(batch["arousal"].numpy().tolist())

    avg_loss = total_loss / n_batches if n_batches else 0

    # RMSE_VA (matching official script)
    gold_va = all_gold_v + all_gold_a
    pred_va = all_preds_v + all_preds_a
    sum_sq = sum((g - p) ** 2 for g, p in zip(gold_va, pred_va))
    n = len(all_gold_v)
    rmse = math.sqrt(sum_sq / n) if n > 0 else float("inf")
    rmse_norm = rmse / math.sqrt(128)

    return avg_loss, rmse, rmse_norm


def generate_predictions(model, dev_data, device, max_len=128):
    """Generate predictions for the dev set in the official JSONL format."""
    model.eval()
    predictions = []

    for entry in dev_data:
        pred_entry = {
            "ID": entry["ID"],
            "Text": entry["Text"],
            "Aspect_VA": [],
        }
        aspects = entry.get("Aspect_VA", [])
        for asp in aspects:
            encoding = tokenizer(
                entry["Text"],
                asp["Aspect"],
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                token_type_ids = encoding["token_type_ids"].to(device)
                pred = model(input_ids, attention_mask, token_type_ids)
                pv = pred[0, 0].item()
                pa = pred[0, 1].item()

            pred_entry["Aspect_VA"].append({
                "Aspect": asp["Aspect"],
                "VA": format_va(pv, pa),
            })
        predictions.append(pred_entry)

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info(f"DimABSA2026 – Subtask 1 Custom BERT Model")
    log.info(f"Domain: {args.domain} | Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")
    log.info("=" * 70)

    # Load data
    train_data = load_jsonl(TRAIN_PATH)
    dev_data = load_jsonl(DEV_PATH)
    log.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev entries")

    # Create datasets
    train_dataset = AspectVADataset(train_data, max_len=args.max_len)
    dev_dataset = AspectVADataset(dev_data, max_len=args.max_len, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = BertVARegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule(optimizer, num_warmup_steps, num_training_steps)

    log.info(f"Training steps: {num_training_steps}, Warmup: {num_warmup_steps}")
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_rmse = float("inf")
    best_epoch = -1

    # Training log
    train_log = []

    for epoch in range(1, args.epochs + 1):
        log.info(f"\n{'─' * 50}")
        log.info(f"Epoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        dev_loss, dev_rmse, dev_rmse_norm = evaluate(model, dev_loader, criterion, device)

        log.info(f"  Train Loss: {train_loss:.6f}")
        log.info(f"  Dev   Loss: {dev_loss:.6f}  |  RMSE_VA: {dev_rmse:.4f}  |  RMSE_norm: {dev_rmse_norm:.4f}")

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_rmse": dev_rmse,
            "dev_rmse_norm": dev_rmse_norm,
        })

        # Save best
        if dev_rmse < best_rmse:
            best_rmse = dev_rmse
            best_epoch = epoch
            ckpt_path = os.path.join(CKPT_DIR, f"best_task1_{args.domain}.pt")
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"  ★ New best model saved (RMSE_VA={best_rmse:.4f})")

    log.info(f"\n{'=' * 50}")
    log.info(f"Training complete. Best epoch: {best_epoch}, Best RMSE_VA: {best_rmse:.4f}")

    # Load best and generate predictions
    best_path = os.path.join(CKPT_DIR, f"best_task1_{args.domain}.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    predictions = generate_predictions(model, dev_data, device, args.max_len)

    pred_path = os.path.join(PRED_DIR, f"custom_bert_task1_{args.domain}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"Predictions saved to {pred_path}")

    # Save training log
    log_path = os.path.join("logs", f"training_curve_task1_{args.domain}_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    # Update summary CSV
    csv_path = os.path.join("logs", "baseline_results.csv")
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","domain","task","RMSE_VA","RMSE_norm","PCC_V","PCC_A","N","timestamp"])
        if mode == "w": w.writeheader()
        w.writerow({
            "model": "BERT-base (custom)",
            "domain": args.domain,
            "task": 1,
            "RMSE_VA": best_rmse,
            "RMSE_norm": best_rmse / math.sqrt(128),
            "PCC_V": None, # Will add if needed
            "PCC_A": None,
            "N": len(dev_dataset),
            "timestamp": timestamp
        })
    log.info(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
