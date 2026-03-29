"""
DimABSA2026 – V3 Custom Model (Phase 1)
=========================================
Research-grade upgrades over V2 DeBERTa:

1. Multi-task learning: regression + polarity + arousal intensity (soft labels)
2. Arousal feature injection: 8-dim linguistic features + LayerNorm
3. Label smoothing: Gaussian noise N(0, 0.1) on regression targets
4. Hard example mining: 2× weight on top-20% error samples after epoch 3
5. Uncertainty-weighted MSE (carried over from V2)

Architecture:
  Input:  [CLS] <text> [SEP] <aspect> [SEP]
  Backbone: DeBERTa-v3-base
  Features: [CLS_hidden; arousal_features_8d]
  Heads:  (V,A) regression + polarity_3 + arousal_level_3
  Loss:   L_reg + 0.1·L_pol + 0.1·L_aro (anneal after ep 5)

Usage:
  python custom_model_v3.py --domain restaurant --epochs 15
"""

import json, os, sys, math, logging, argparse, csv
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# Local imports
from arousal_features import extract_features_batch, ArousalFeatureNorm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant","laptop"])
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--accumulation_steps", type=int, default=2)
parser.add_argument("--label_noise_std", type=float, default=0.1)
parser.add_argument("--hard_mining_start", type=int, default=3)
parser.add_argument("--hard_mining_topk", type=float, default=0.2)
parser.add_argument("--lambda_pol", type=float, default=0.1)
parser.add_argument("--lambda_aro", type=float, default=0.1)
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(f"logs/v3_{args.domain}_{ts}.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device} ({'GPU: '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng")

from transformers import AutoTokenizer, AutoModel
MODEL_NAME = "microsoft/deberta-v3-base"
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Soft Label Generation ────────────────────────────────────────────────
def compute_soft_polarity(v):
    """Map valence → soft polarity distribution (neg, neu, pos)."""
    p_neg = torch.sigmoid(-(v - 4.0))   # high when V < 4
    p_pos = torch.sigmoid(v - 6.0)      # high when V > 6
    p_neu = 1.0 - p_neg - p_pos
    p_neu = torch.clamp(p_neu, min=0.01)  # prevent negative
    total = p_neg + p_neu + p_pos
    return torch.stack([p_neg/total, p_neu/total, p_pos/total], dim=-1)


def compute_soft_arousal_level(a):
    """Map arousal → soft intensity distribution (low, med, high)."""
    p_low = torch.sigmoid(-(a - 4.0))
    p_high = torch.sigmoid(a - 6.0)
    p_med = 1.0 - p_low - p_high
    p_med = torch.clamp(p_med, min=0.01)
    total = p_low + p_med + p_high
    return torch.stack([p_low/total, p_med/total, p_high/total], dim=-1)


# ── Data ──────────────────────────────────────────────────────────────────
def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def parse_va(s):
    v, a = s.split("#")
    return float(v), float(a)

def fmt_va(v, a):
    return f"{max(1,min(9,v)):.2f}#{max(1,min(9,a)):.2f}"


class AspectVADatasetV3(Dataset):
    """Enhanced dataset with text storage for feature extraction."""
    def __init__(self, data, max_len=128):
        self.samples = []
        for e in data:
            asps = e.get("Aspect_VA", [])
            if not asps:
                for q in e.get("Quadruplet", []):
                    v, a = parse_va(q["VA"])
                    self.samples.append({"text": e["Text"], "aspect": q["Aspect"],
                                         "v": v, "a": a, "id": e["ID"]})
            else:
                for asp in asps:
                    v, a = parse_va(asp["VA"])
                    self.samples.append({"text": e["Text"], "aspect": asp["Aspect"],
                                         "v": v, "a": a, "id": e["ID"]})
        self.max_len = max_len
        log.info(f"  Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        enc = tokenizer(s["text"], s["aspect"], max_length=self.max_len,
                        padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "v": torch.tensor(s["v"], dtype=torch.float),
            "a": torch.tensor(s["a"], dtype=torch.float),
            "text": s["text"],  # kept for arousal feature extraction
        }


# ── Model ─────────────────────────────────────────────────────────────────
class DeBERTaV3MultiTask(nn.Module):
    """V3: DeBERTa + multi-task + arousal features + uncertainty weighting."""

    def __init__(self, hidden_size=768, n_arousal_feats=8, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(dropout)

        # Arousal feature normalization (LayerNorm on raw counts)
        self.arousal_norm = ArousalFeatureNorm(n_arousal_feats)

        # Fused dim = DeBERTa hidden + arousal features
        fused_dim = hidden_size + n_arousal_feats

        # Head 1: VA regression
        self.va_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 2),  # (V, A)
        )

        # Head 2: Polarity classifier (neg, neu, pos)
        self.polarity_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Head 3: Arousal intensity classifier (low, med, high)
        self.arousal_cls_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Learnable uncertainties (from V2)
        self.log_var_v = nn.Parameter(torch.zeros(1))
        self.log_var_a = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, arousal_feats):
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            arousal_feats: (B, 8) raw linguistic features
        Returns:
            va_pred: (B, 2)
            pol_logits: (B, 3)
            aro_logits: (B, 3)
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = self.drop(out.last_hidden_state[:, 0, :])  # (B, 768)

        # Normalize and concatenate arousal features
        aro_normed = self.arousal_norm(arousal_feats)  # (B, 8)
        fused = torch.cat([cls_hidden, aro_normed], dim=-1)  # (B, 776)

        va_pred = self.va_head(fused)
        pol_logits = self.polarity_head(fused)
        aro_logits = self.arousal_cls_head(fused)

        return va_pred, pol_logits, aro_logits

    def compute_loss(self, va_pred, pol_logits, aro_logits,
                     va_target, pol_soft, aro_soft,
                     sample_weights=None, lambda_pol=0.1, lambda_aro=0.1):
        """
        Combined loss with uncertainty weighting + soft label classification.

        Args:
            va_pred: (B, 2) predicted V, A
            pol_logits: (B, 3) polarity logits
            aro_logits: (B, 3) arousal intensity logits
            va_target: (B, 2) gold V, A (possibly noise-smoothed)
            pol_soft: (B, 3) soft polarity labels
            aro_soft: (B, 3) soft arousal level labels
            sample_weights: (B,) per-sample weights for hard mining
            lambda_pol: classification loss weight
            lambda_aro: arousal classification loss weight
        """
        # ── Uncertainty-weighted regression loss ──
        mse_v = F.mse_loss(va_pred[:, 0], va_target[:, 0], reduction='none')
        mse_a = F.mse_loss(va_pred[:, 1], va_target[:, 1], reduction='none')

        if sample_weights is not None:
            mse_v = mse_v * sample_weights
            mse_a = mse_a * sample_weights

        mse_v_mean = mse_v.mean()
        mse_a_mean = mse_a.mean()

        loss_v = 0.5 * torch.exp(-self.log_var_v) * mse_v_mean + 0.5 * self.log_var_v
        loss_a = 0.5 * torch.exp(-self.log_var_a) * mse_a_mean + 0.5 * self.log_var_a
        loss_reg = loss_v + loss_a

        # ── Soft-label cross-entropy for polarity ──
        pol_log_probs = F.log_softmax(pol_logits, dim=-1)
        loss_pol = -(pol_soft * pol_log_probs).sum(dim=-1).mean()

        # ── Soft-label cross-entropy for arousal intensity ──
        aro_log_probs = F.log_softmax(aro_logits, dim=-1)
        loss_aro = -(aro_soft * aro_log_probs).sum(dim=-1).mean()

        total = loss_reg + lambda_pol * loss_pol + lambda_aro * loss_aro

        return total, mse_v_mean.item(), mse_a_mean.item(), loss_pol.item(), loss_aro.item()


# ── Training ──────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info(f"DimABSA V3 Phase 1 | {args.domain} | {args.epochs} epochs | BS={args.batch_size}")
    log.info(f"  Upgrades: multi-task, arousal_feats, label_smoothing, hard_mining")
    log.info("=" * 70)

    train_data = load_jsonl(os.path.join(DATA, f"eng_{args.domain}_train_alltasks.jsonl"))
    dev_data = load_jsonl(os.path.join(DATA, f"eng_{args.domain}_dev_task1.jsonl"))
    train_ds = AspectVADatasetV3(train_data, args.max_len)
    dev_ds = AspectVADatasetV3(dev_data, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    model = DeBERTaV3MultiTask().to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-7)
    scaler = GradScaler()

    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Steps: {total_steps}")

    best_rmse = float("inf")
    history = []

    # Track per-sample losses for hard example mining
    sample_losses = torch.zeros(len(train_ds))

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_mse_v, ep_mse_a, ep_pol, ep_aro, nb = 0, 0, 0, 0, 0, 0
        optim.zero_grad()

        # Anneal classification λ after epoch 5
        cur_lambda_pol = args.lambda_pol * max(0.3, 1.0 - (epoch - 5) * 0.1) if epoch > 5 else args.lambda_pol
        cur_lambda_aro = args.lambda_aro * max(0.3, 1.0 - (epoch - 5) * 0.1) if epoch > 5 else args.lambda_aro

        for step, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            # VA targets with optional label smoothing
            v_target = batch["v"].clone()
            a_target = batch["a"].clone()
            if args.label_noise_std > 0:
                v_target += torch.randn_like(v_target) * args.label_noise_std
                a_target += torch.randn_like(a_target) * args.label_noise_std
                v_target = v_target.clamp(1.0, 9.0)
                a_target = a_target.clamp(1.0, 9.0)
            va_target = torch.stack([v_target, a_target], dim=1).to(device)

            # Soft labels from original (unsmoothed) values
            pol_soft = compute_soft_polarity(batch["v"]).to(device)
            aro_soft = compute_soft_arousal_level(batch["a"]).to(device)

            # Arousal linguistic features
            aro_feats = extract_features_batch(batch["text"]).to(device)

            # Hard example mining weights
            sample_w = None
            if epoch >= args.hard_mining_start and sample_losses.sum() > 0:
                # Get indices for this batch
                batch_start = step * args.batch_size
                batch_end = min(batch_start + len(ids), len(sample_losses))
                if batch_end > batch_start:
                    batch_losses = sample_losses[batch_start:batch_end]
                    threshold = torch.quantile(sample_losses[sample_losses > 0],
                                               1.0 - args.hard_mining_topk)
                    sample_w = torch.where(batch_losses >= threshold,
                                           torch.tensor(2.0), torch.tensor(1.0)).to(device)

            with autocast(dtype=torch.float16):
                va_pred, pol_logits, aro_logits = model(ids, mask, aro_feats)
                loss, mv, ma, lp, la = model.compute_loss(
                    va_pred, pol_logits, aro_logits,
                    va_target, pol_soft, aro_soft,
                    sample_weights=sample_w,
                    lambda_pol=cur_lambda_pol,
                    lambda_aro=cur_lambda_aro,
                )
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()

            # Track per-sample losses for hard mining
            with torch.no_grad():
                per_sample = ((va_pred[:, 0] - va_target[:, 0])**2
                              + (va_pred[:, 1] - va_target[:, 1])**2)
                batch_start = step * args.batch_size
                batch_end = min(batch_start + len(per_sample), len(sample_losses))
                if batch_end > batch_start:
                    sample_losses[batch_start:batch_end] = per_sample[:batch_end-batch_start].cpu()

            ep_loss += loss.item() * args.accumulation_steps
            ep_mse_v += mv; ep_mse_a += ma
            ep_pol += lp; ep_aro += la; nb += 1

        train_loss = ep_loss / nb

        # ── Evaluation ──
        model.eval()
        pv_all, pa_all, gv_all, ga_all = [], [], [], []
        dev_loss_total = 0

        with torch.no_grad():
            for batch in dev_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                va_target = torch.stack([batch["v"], batch["a"]], dim=1).to(device)
                pol_soft = compute_soft_polarity(batch["v"]).to(device)
                aro_soft = compute_soft_arousal_level(batch["a"]).to(device)
                aro_feats = extract_features_batch(batch["text"]).to(device)

                va_pred, pol_logits, aro_logits = model(ids, mask, aro_feats)
                loss, _, _, _, _ = model.compute_loss(
                    va_pred, pol_logits, aro_logits,
                    va_target, pol_soft, aro_soft,
                )
                dev_loss_total += loss.item()
                pv_all.extend(va_pred[:, 0].cpu().numpy().tolist())
                pa_all.extend(va_pred[:, 1].cpu().numpy().tolist())
                gv_all.extend(batch["v"].numpy().tolist())
                ga_all.extend(batch["a"].numpy().tolist())

        dev_loss = dev_loss_total / len(dev_dl)

        # Clamp predictions
        pv_all = [max(1, min(9, x)) for x in pv_all]
        pa_all = [max(1, min(9, x)) for x in pa_all]

        # RMSE_VA
        g_all = gv_all + ga_all
        p_all = pv_all + pa_all
        rmse = math.sqrt(sum((g-p)**2 for g, p in zip(g_all, p_all)) / len(gv_all))
        rmse_norm = rmse / math.sqrt(128)

        from scipy.stats import pearsonr
        try: pcc_v = pearsonr(pv_all, gv_all)[0]
        except: pcc_v = float("nan")
        try: pcc_a = pearsonr(pa_all, ga_all)[0]
        except: pcc_a = float("nan")

        sigma_v = math.exp(0.5 * model.log_var_v.item())
        sigma_a = math.exp(0.5 * model.log_var_a.item())

        log.info(f"Ep {epoch:>2}/{args.epochs} | TrainL={train_loss:.4f} DevL={dev_loss:.4f} "
                 f"RMSE={rmse:.4f} PCC_V={pcc_v:.4f} PCC_A={pcc_a:.4f} "
                 f"σV={sigma_v:.3f} σA={sigma_a:.3f} λ_pol={cur_lambda_pol:.3f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
            "rmse": rmse, "rmse_norm": rmse_norm,
            "pcc_v": float(pcc_v), "pcc_a": float(pcc_a),
            "sigma_v": sigma_v, "sigma_a": sigma_a,
            "lr": scheduler.get_last_lr()[0],
            "lambda_pol": cur_lambda_pol, "lambda_aro": cur_lambda_aro,
            "pred_v": pv_all, "pred_a": pa_all, "gold_v": gv_all, "gold_a": ga_all,
        })

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join("checkpoints", f"best_v3_{args.domain}.pt"))
            log.info(f"  ★ New best (RMSE={best_rmse:.4f})")

    log.info(f"\n{'='*50}\nBest RMSE_VA: {best_rmse:.4f}\n{'='*50}")

    # ── Generate predictions with best model ──
    model.load_state_dict(torch.load(os.path.join("checkpoints", f"best_v3_{args.domain}.pt"),
                                      map_location=device, weights_only=True))
    model.eval()
    predictions = []
    for e in dev_data:
        pe = {"ID": e["ID"], "Text": e["Text"], "Aspect_VA": []}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=args.max_len,
                           padding="max_length", truncation=True, return_tensors="pt")
            aro_feats = extract_features_batch([e["Text"]]).to(device)
            with torch.no_grad():
                va_pred, _, _ = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    aro_feats,
                )
            pe["Aspect_VA"].append({
                "Aspect": asp["Aspect"],
                "VA": fmt_va(va_pred[0, 0].item(), va_pred[0, 1].item()),
            })
        predictions.append(pe)

    pred_path = os.path.join("predictions", f"v3_task1_{args.domain}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info(f"Predictions: {pred_path}")

    # ── Save history ──
    hist_path = os.path.join("logs", f"v3_history_{args.domain}_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # ── Update CSV ──
    csv_path = os.path.join("logs", "baseline_results.csv")
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "domain", "task", "RMSE_VA", "RMSE_norm",
            "PCC_V", "PCC_A", "N", "timestamp",
        ])
        if mode == "w":
            w.writeheader()
        w.writerow({
            "model": "V3-Phase1 (ours)",
            "domain": args.domain,
            "task": 1,
            "RMSE_VA": best_rmse,
            "RMSE_norm": best_rmse / math.sqrt(128),
            "PCC_V": history[-1]["pcc_v"],
            "PCC_A": history[-1]["pcc_a"],
            "N": len(gv_all),
            "timestamp": ts,
        })
    log.info(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
