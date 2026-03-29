"""
DimABSA2026 – V3 Phase 2: Cross-Attention Aspect Modeling
=========================================================
Upgrades over V3 Phase 1:

1. Cross-Attention: Instead of just using the [CLS] token, we explicitly
   model the interaction between the Aspect and the Text.
2. Single-pass Encoding: We feed "[CLS] text [SEP] aspect [SEP]" once.
3. Index Slicing: We extract hidden states for Text and Aspect separately.
4. Pooled Aspect Query: We pool aspect tokens into a single vector to use 
   as the Query for attention over all Text tokens.

Architecture:
  Hidden_States = DeBERTa([CLS] text [SEP] aspect [SEP])
  T = Hidden_States[1 : first_sep]  (Text tokens)
  A = pool(Hidden_States[first_sep+1 : second_sep]) (Aspect query)
  Context = MultiHeadAttention(Q=A, K=T, V=T)
  Fused = [CLS_hidden; Context; A; Arousal_Features]
  Heads: Regression + Polarity Cls + Arousal Cls
"""

import json, os, sys, math, logging, argparse, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel

# Local imports
from arousal_features import extract_features_batch, ArousalFeatureNorm

parser = argparse.ArgumentParser()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant","laptop"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--label_noise_std", type=float, default=0.1)
    parser.add_argument("--hard_mining_start", type=int, default=3)
    parser.add_argument("--hard_mining_topk", type=float, default=0.2)
    parser.add_argument("--lambda_pol", type=float, default=0.1)
    parser.add_argument("--lambda_aro", type=float, default=0.1)
    return parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
# ts and logging setup moved to main()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Utilities ─────────────────────────────────────────────────────────────
def compute_soft_polarity(v):
    p_neg = torch.sigmoid(-(v - 4.0))
    p_pos = torch.sigmoid(v - 6.0)
    p_neu = torch.clamp(1.0 - p_neg - p_pos, min=0.01)
    total = p_neg + p_neu + p_pos
    return torch.stack([p_neg/total, p_neu/total, p_pos/total], dim=-1)

def compute_soft_arousal_level(a):
    p_low = torch.sigmoid(-(a - 4.0))
    p_high = torch.sigmoid(a - 6.0)
    p_med = torch.clamp(1.0 - p_low - p_high, min=0.01)
    total = p_low + p_med + p_high
    return torch.stack([p_low/total, p_med/total, p_high/total], dim=-1)

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def fmt_va(v, a):
    return f"{max(1,min(9,v)):.2f}#{max(1,min(9,a)):.2f}"

# ── Data ──────────────────────────────────────────────────────────────────
class CrossAttentionDataset(Dataset):
    def __init__(self, data, max_len=160):
        self.samples = []
        for e in data:
            asps = e.get("Aspect_VA", []) or e.get("Quadruplet", [])
            for asp in asps:
                va_str = asp.get("VA", "5#5")
                v, a = map(float, va_str.split("#"))
                self.samples.append({
                    "text": e["Text"], 
                    "aspect": asp.get("Aspect", ""),
                    "v": v, "a": a, "id": e["ID"]
                })
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        # DeBERTa style: [CLS] text [SEP] aspect [SEP]
        enc = tokenizer(s["text"], s["aspect"], max_length=self.max_len,
                        padding="max_length", truncation=True, return_tensors="pt")
        
        # Find SEP indices to slice later
        input_ids = enc["input_ids"].squeeze(0)
        sep_idxs = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        
        # Ensure we have at least 2 SEPs (should always happen if truncation is reasonable)
        if len(sep_idxs) < 2:
            # Fallback if aspect got truncated entirely
            sep1, sep2 = self.max_len // 2, self.max_len - 1
        else:
            sep1, sep2 = sep_idxs[0].item(), sep_idxs[1].item()

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sep1": sep1,
            "sep2": sep2,
            "v": torch.tensor(s["v"], dtype=torch.float),
            "a": torch.tensor(s["a"], dtype=torch.float),
            "text": s["text"]
        }

# ── Model ─────────────────────────────────────────────────────────────────
class DeBERTaV3CrossAttention(nn.Module):
    def __init__(self, hidden_size=768, n_arousal_feats=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(dropout)
        self.arousal_norm = ArousalFeatureNorm(n_arousal_feats)

        # Cross-Attention layer: Q=Aspect, K=Text, V=Text
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, 
                                                dropout=dropout, batch_first=True)
        
        # Fused dimension: CLS (768) + CrossAttnOut (768) + PooledAspect (768) + ArousalFeats (8)
        fused_dim = hidden_size * 3 + n_arousal_feats

        self.va_head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 2)
        )
        self.polarity_head = nn.Linear(fused_dim, 3)
        self.arousal_cls_head = nn.Linear(fused_dim, 3)

        self.log_var_v = nn.Parameter(torch.zeros(1))
        self.log_var_a = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, sep1, sep2, arousal_feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state # (B, L, 768)
        
        batch_size = last_hidden.size(0)
        cls_hidden = last_hidden[:, 0, :] # (B, 768)
        
        cross_attn_outputs = []
        pooled_aspects = []
        
        for i in range(batch_size):
            s1, s2 = sep1[i].item(), sep2[i].item()
            
            # Slice text tokens (exclude [CLS] and [SEP])
            # Text is from index 1 to s1-1
            if s1 > 1:
                text_hs = last_hidden[i, 1:s1, :].unsqueeze(0) # (1, L_t, 768)
            else:
                text_hs = last_hidden[i, 0, :].view(1, 1, -1) # Fallback to [CLS]
                
            # Slice aspect tokens (exclude [SEP])
            # Aspect is from index s1+1 to s2-1
            if s2 > s1 + 1:
                aspect_hs = last_hidden[i, s1+1:s2, :].unsqueeze(0) # (1, L_a, 768)
            else:
                aspect_hs = last_hidden[i, 0, :].view(1, 1, -1) # Fallback to [CLS]
            
            # Pool aspect tokens into a single Query vector
            asp_query = torch.mean(aspect_hs, dim=1, keepdim=True) # (1, 1, 768)
            pooled_aspects.append(asp_query.view(-1)) # (768,)
            
            # Cross-Attention: Q=pooled_aspect, K=text, V=text
            attn_out, _ = self.cross_attn(query=asp_query, key=text_hs, value=text_hs)
            cross_attn_outputs.append(attn_out.view(-1)) # (768,)
            
        cross_attn_out = torch.stack(cross_attn_outputs, dim=0) # (B, 768)
        pooled_asp = torch.stack(pooled_aspects, dim=0) # (B, 768)
        
        # Final Fusion
        aro_normed = self.arousal_norm(arousal_feats) # (B, 8)
        
        # All should be (B, 768) or (B, 8)
        fused = torch.cat([cls_hidden, cross_attn_out, pooled_asp, aro_normed], dim=-1) # (B, 2312)
        
        va_pred = self.va_head(fused)
        pol_logits = self.polarity_head(fused)
        aro_logits = self.arousal_cls_head(fused)
        
        return va_pred, pol_logits, aro_logits

    def compute_loss(self, va_pred, pol_logits, aro_logits, va_target, pol_soft, aro_soft, 
                     sample_weights=None, lambda_pol=0.1, lambda_aro=0.1):
        # Uncertainty-weighted regression loss
        mse_v = F.mse_loss(va_pred[:, 0], va_target[:, 0], reduction='none')
        mse_a = F.mse_loss(va_pred[:, 1], va_target[:, 1], reduction='none')

        if sample_weights is not None:
            mse_v = mse_v * sample_weights
            mse_a = mse_a * sample_weights

        mv, ma = mse_v.mean(), mse_a.mean()
        
        loss_v = 0.5 * torch.exp(-self.log_var_v) * mv + 0.5 * self.log_var_v
        loss_a = 0.5 * torch.exp(-self.log_var_a) * ma + 0.5 * self.log_var_a
        
        # Multi-task classification loss
        loss_pol = -(pol_soft * F.log_softmax(pol_logits, dim=-1)).sum(dim=-1).mean()
        loss_aro = -(aro_soft * F.log_softmax(aro_logits, dim=-1)).sum(dim=-1).mean()
        
        total = loss_v + loss_a + lambda_pol * loss_pol + lambda_aro * loss_aro
        return total, mv.item(), ma.item(), loss_pol.item(), loss_aro.item()

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"logs/v3_p2_{args.domain}_{ts}.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__)
    
    log.info(f"DimABSA V3 Phase 2 | {args.domain} | Cross-Attention + Phase 1 wins")
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), "task-dataset", "track_a", "subtask_1", "eng")
    train_data = load_jsonl(os.path.join(DATA_DIR, f"eng_{args.domain}_train_alltasks.jsonl"))
    dev_data = load_jsonl(os.path.join(DATA_DIR, f"eng_{args.domain}_dev_task1.jsonl"))
    
    train_ds = CrossAttentionDataset(train_data, args.max_len)
    dev_ds = CrossAttentionDataset(dev_data, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    model = DeBERTaV3CrossAttention().to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_dl) * args.epochs // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-7)
    scaler = GradScaler()
    
    sample_losses = torch.zeros(len(train_ds))
    best_rmse = float("inf")
    
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_loss, ep_mse_v, ep_mse_a, ep_pol, ep_aro, nb = 0, 0, 0, 0, 0, 0
            optim.zero_grad()

            # Anneal λ
            cur_lambda_pol = args.lambda_pol * max(0.3, 1.0 - (epoch - 5) * 0.1) if epoch > 5 else args.lambda_pol
            cur_lambda_aro = args.lambda_aro * max(0.3, 1.0 - (epoch - 5) * 0.1) if epoch > 5 else args.lambda_aro

            for step, batch in enumerate(train_dl):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                s1, s2 = batch["sep1"], batch["sep2"]

                # Label Smoothing
                v_t, a_t = batch["v"].clone(), batch["a"].clone()
                v_t += torch.randn_like(v_t) * 0.1 # label_noise_std hardcoded for now
                a_t += torch.randn_like(a_t) * 0.1
                va_target = torch.stack([v_t.clamp(1,9), a_t.clamp(1,9)], dim=1).to(device)

                pol_soft = compute_soft_polarity(batch["v"]).to(device)
                aro_soft = compute_soft_arousal_level(batch["a"]).to(device)
                aro_feats = extract_features_batch(batch["text"]).to(device)

                # Hard Example Mining
                sample_w = None
                if epoch >= 3 and sample_losses.sum() > 0:
                    idx_start = step * args.batch_size
                    idx_end = min(idx_start + len(ids), len(sample_losses))
                    if idx_end > idx_start:
                        b_losses = sample_losses[idx_start:idx_end]
                        thresh = torch.quantile(sample_losses[sample_losses > 0], 0.8)
                        sample_w = torch.where(b_losses >= thresh, torch.tensor(2.0), torch.tensor(1.0)).to(device)

                with autocast(dtype=torch.float16):
                    va_pred, pol_logits, aro_logits = model(ids, mask, s1, s2, aro_feats)
                    loss, mv, ma, lp, la = model.compute_loss(
                        va_pred, pol_logits, aro_logits, va_target, pol_soft, aro_soft,
                        sample_weights=sample_w, lambda_pol=cur_lambda_pol, lambda_aro=cur_lambda_aro
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

                # Track losses for mining
                with torch.no_grad():
                    per_s = ((va_pred[:, 0] - va_target[:, 0])**2 + (va_pred[:, 1] - va_target[:, 1])**2).cpu()
                    idx_start = step * args.batch_size
                    idx_end = min(idx_start + len(per_s), len(sample_losses))
                    sample_losses[idx_start:idx_end] = per_s[:idx_end-idx_start]

                ep_loss += loss.item() * args.accumulation_steps
                ep_mse_v += mv; ep_mse_a += ma; ep_pol += lp; ep_aro += la; nb += 1

            # Eval
            model.eval()
            pv_all, pa_all, gv_all, ga_all = [], [], [], []
            rmse_total, n_eval = 0, 0
            with torch.no_grad():
                for batch in dev_dl:
                    ids, mask, s1, s2 = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["sep1"], batch["sep2"]
                    va_pred, _, _ = model(ids, mask, s1, s2, extract_features_batch(batch["text"]).to(device))
                    va_gold = torch.stack([batch["v"], batch["a"]], dim=1).to(device)
                    rmse_total += F.mse_loss(va_pred.clamp(1,9), va_gold, reduction='sum').item()
                    
                    pv_all.extend(va_pred[:, 0].cpu().numpy().tolist())
                    pa_all.extend(va_pred[:, 1].cpu().numpy().tolist())
                    gv_all.extend(batch["v"].numpy().tolist())
                    ga_all.extend(batch["a"].numpy().tolist())
                    n_eval += len(ids)
            
            rmse = math.sqrt(rmse_total / (n_eval * 2))
            
            from scipy.stats import pearsonr
            try: pcc_v = pearsonr(pv_all, gv_all)[0]
            except: pcc_v = 0.5
            try: pcc_a = pearsonr(pa_all, ga_all)[0]
            except: pcc_a = 0.5

            sig_v = math.exp(0.5 * model.log_var_v.item())
            sig_a = math.exp(0.5 * model.log_var_a.item())
            log.info(f"Epoch {epoch:>2} | TrainL: {ep_loss/nb:.4f} | RMSE: {rmse:.4f} | PCC_V: {pcc_v:.4f} | PCC_A: {pcc_a:.4f} | σV: {sig_v:.3f} σA: {sig_a:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), f"checkpoints/best_v3_p2_{args.domain}.pt")
                log.info(f"  ★ New best RMSE: {best_rmse:.4f}")
    except Exception as e:
        log.error(f"Error during training: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        raise e

    log.info(f"\nFinal Best RMSE: {best_rmse:.4f}")

    # ── Generate dev predictions with best model ──
    model.load_state_dict(torch.load(f"checkpoints/best_v3_p2_{args.domain}.pt", map_location=device, weights_only=True))
    model.eval()
    predictions = []
    
    # Reload dev data for format consistency
    dev_data_raw = load_jsonl(os.path.join(DATA_DIR, f"eng_{args.domain}_dev_task1.jsonl"))
    
    for e in dev_data_raw:
        pe = {"ID": e["ID"], "Text": e["Text"], "Aspect_VA": []}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=args.max_len,
                           padding="max_length", truncation=True, return_tensors="pt")
            
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            sep_idx = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            
            # Use fixed slicing logic for single-aspect prediction
            s1, s2 = sep_idx[0].item(), sep_idx[1].item()
            
            aro_feats = extract_features_batch([e["Text"]]).to(device)
            with torch.no_grad():
                va_pred, _, _ = model(input_ids, mask, torch.tensor([s1]).to(device), 
                                      torch.tensor([s2]).to(device), aro_feats)
            
            v, a = va_pred[0, 0].item(), va_pred[0, 1].item()
            pe["Aspect_VA"].append({
                "Aspect": asp["Aspect"],
                "VA": fmt_va(v, a)
            })
        predictions.append(pe)

    pred_path = os.path.join("predictions", f"v3_p2_task1_{args.domain}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info(f"Dev Predictions saved to {pred_path}")

if __name__ == "__main__":
    main()
