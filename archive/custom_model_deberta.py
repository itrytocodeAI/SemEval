"""
DimABSA2026 – DeBERTa-v3 Custom Model (SOTA-Inspired)
=======================================================
Inspired by AILS-NTUA (transformer encoder) + LogSigma (uncertainty-weighted multitask).

Architecture:
  Input:  [CLS] <text> [SEP] <aspect> [SEP]
  Model:  DeBERTa-v3-base → [CLS] → 2-layer MLP → (V, A)
  Loss:   Uncertainty-weighted MSE: L = (1/2σ²_v)·MSE_v + (1/2σ²_a)·MSE_a + log(σ_v·σ_a)
  Training: AdamW, cosine annealing, fp16 mixed precision, GPU

Usage:
  python custom_model_deberta.py --domain restaurant --epochs 15
"""

import json, os, sys, math, logging, argparse, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant","laptop"])
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--accumulation_steps", type=int, default=2)
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(f"logs/deberta_{args.domain}_{ts}.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device} ({'GPU: '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng")

from transformers import AutoTokenizer, AutoModel
MODEL_NAME = "microsoft/deberta-v3-base"
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Data ──────────────────────────────────────────────────────────────────
def load_jsonl(p):
    with open(p,"r",encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def parse_va(s): v,a=s.split("#"); return float(v),float(a)
def fmt_va(v,a): return f"{max(1,min(9,v)):.2f}#{max(1,min(9,a)):.2f}"

class AspectVADataset(Dataset):
    def __init__(self, data, max_len=128):
        self.samples = []
        for e in data:
            asps = e.get("Aspect_VA", [])
            if not asps:
                for q in e.get("Quadruplet",[]):
                    v,a = parse_va(q["VA"])
                    self.samples.append({"text":e["Text"],"aspect":q["Aspect"],"v":v,"a":a,"id":e["ID"]})
            else:
                for asp in asps:
                    v,a = parse_va(asp["VA"])
                    self.samples.append({"text":e["Text"],"aspect":asp["Aspect"],"v":v,"a":a,"id":e["ID"]})
        self.max_len = max_len
        log.info(f"  Dataset: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        enc = tokenizer(s["text"], s["aspect"], max_length=self.max_len,
                        padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "v": torch.tensor(s["v"],dtype=torch.float),
                "a": torch.tensor(s["a"],dtype=torch.float)}

# ── Model ─────────────────────────────────────────────────────────────────
class DeBERTaVARegressor(nn.Module):
    """DeBERTa + uncertainty-weighted VA regression (LogSigma-inspired)."""
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 2)  # (V, A)
        )
        # Learnable log-variance for uncertainty weighting
        self.log_var_v = nn.Parameter(torch.zeros(1))
        self.log_var_a = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        va = self.head(cls)
        return va

    def uncertainty_loss(self, pred, target):
        """Uncertainty-weighted MSE: L = (1/2σ²)·MSE + log(σ)"""
        mse_v = nn.functional.mse_loss(pred[:,0], target[:,0])
        mse_a = nn.functional.mse_loss(pred[:,1], target[:,1])
        loss_v = 0.5 * torch.exp(-self.log_var_v) * mse_v + 0.5 * self.log_var_v
        loss_a = 0.5 * torch.exp(-self.log_var_a) * mse_a + 0.5 * self.log_var_a
        return loss_v + loss_a, mse_v.item(), mse_a.item()

# ── Training ──────────────────────────────────────────────────────────────
def main():
    log.info("="*70)
    log.info(f"DeBERTa-v3 VA Regression | {args.domain} | {args.epochs} epochs | BS={args.batch_size}")
    log.info("="*70)

    train = load_jsonl(os.path.join(DATA, f"eng_{args.domain}_train_alltasks.jsonl"))
    dev = load_jsonl(os.path.join(DATA, f"eng_{args.domain}_dev_task1.jsonl"))
    train_ds = AspectVADataset(train, args.max_len)
    dev_ds = AspectVADataset(dev, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DeBERTaVARegressor().to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs // args.accumulation_steps
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-7)
    scaler = GradScaler()

    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Steps: {total_steps}, Warmup: {warmup}")

    best_rmse = float("inf")
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss, ep_mse_v, ep_mse_a, nb = 0, 0, 0, 0
        optim.zero_grad()

        for step, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tgt = torch.stack([batch["v"], batch["a"]], dim=1).to(device)

            with autocast(dtype=torch.float16):
                pred = model(ids, mask)
                loss, mv, ma = model.uncertainty_loss(pred, tgt)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()

            ep_loss += loss.item() * args.accumulation_steps
            ep_mse_v += mv; ep_mse_a += ma; nb += 1

        train_loss = ep_loss/nb
        # ── Eval ──
        model.eval()
        pv_all, pa_all, gv_all, ga_all = [], [], [], []
        dev_loss_total = 0
        with torch.no_grad():
            for batch in dev_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tgt = torch.stack([batch["v"], batch["a"]], dim=1).to(device)
                pred = model(ids, mask)
                loss, _, _ = model.uncertainty_loss(pred, tgt)
                dev_loss_total += loss.item()
                pv_all.extend(pred[:,0].cpu().numpy().tolist())
                pa_all.extend(pred[:,1].cpu().numpy().tolist())
                gv_all.extend(batch["v"].numpy().tolist())
                ga_all.extend(batch["a"].numpy().tolist())

        dev_loss = dev_loss_total / len(dev_dl)
        # Clamp predictions
        pv_all = [max(1,min(9,x)) for x in pv_all]
        pa_all = [max(1,min(9,x)) for x in pa_all]
        g_all = gv_all + ga_all; p_all = pv_all + pa_all
        rmse = math.sqrt(sum((g-p)**2 for g,p in zip(g_all,p_all))/len(gv_all))
        rmse_norm = rmse / math.sqrt(128)

        from scipy.stats import pearsonr
        try: pcc_v = pearsonr(pv_all, gv_all)[0]
        except: pcc_v = float("nan")
        try: pcc_a = pearsonr(pa_all, ga_all)[0]
        except: pcc_a = float("nan")

        σv = math.exp(0.5 * model.log_var_v.item())
        σa = math.exp(0.5 * model.log_var_a.item())

        log.info(f"Ep {epoch:>2}/{args.epochs} | TrainL={train_loss:.4f} DevL={dev_loss:.4f} "
                 f"RMSE={rmse:.4f} norm={rmse_norm:.4f} PCC_V={pcc_v:.4f} PCC_A={pcc_a:.4f} "
                 f"σV={σv:.3f} σA={σa:.3f}")

        history.append({"epoch":epoch, "train_loss":train_loss, "dev_loss":dev_loss,
            "rmse":rmse, "rmse_norm":rmse_norm, "pcc_v":float(pcc_v), "pcc_a":float(pcc_a),
            "sigma_v":σv, "sigma_a":σa, "lr":scheduler.get_last_lr()[0],
            "pred_v":pv_all, "pred_a":pa_all, "gold_v":gv_all, "gold_a":ga_all})

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join("checkpoints", f"best_deberta_{args.domain}.pt"))
            log.info(f"  ★ New best (RMSE={best_rmse:.4f})")

    # ── Generate predictions ──
    model.load_state_dict(torch.load(os.path.join("checkpoints", f"best_deberta_{args.domain}.pt"),
                                      map_location=device, weights_only=True))
    model.eval()
    predictions = []
    for e in dev:
        pe = {"ID":e["ID"],"Text":e["Text"],"Aspect_VA":[]}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=args.max_len,
                           padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                pred = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            pe["Aspect_VA"].append({"Aspect":asp["Aspect"],
                "VA":fmt_va(pred[0,0].item(), pred[0,1].item())})
        predictions.append(pe)

    pred_path = os.path.join("predictions", f"custom_deberta_task1_{args.domain}.jsonl")
    with open(pred_path,"w",encoding="utf-8") as f:
        for p in predictions: f.write(json.dumps(p,ensure_ascii=False)+"\n")
    log.info(f"Predictions: {pred_path}")

    # ── Save training history ──
    hist_path = os.path.join("logs", f"deberta_history_{args.domain}_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Update CSV
    csv_path = os.path.join("logs", "baseline_results.csv")
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","domain","task","RMSE_VA","RMSE_norm","PCC_V","PCC_A","N","timestamp"])
        if mode == "w": w.writeheader()
        w.writerow({"model":"DeBERTa-v3 (ours)","domain":args.domain,"task":1,
            "RMSE_VA":best_rmse,"RMSE_norm":best_rmse/math.sqrt(128),
            "PCC_V":history[-1]["pcc_v"],"PCC_A":history[-1]["pcc_a"],
            "N":len(gv_all),"timestamp":ts})
    log.info(f"Results appended to {csv_path}")
    log.info(f"\n{'='*50}\nBest RMSE_VA: {best_rmse:.4f}\n{'='*50}")

if __name__ == "__main__":
    main()
