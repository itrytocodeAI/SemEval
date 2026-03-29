"""
DimABSA2026 – V3 Phase 3 (Robust Distribution-Aware Regression)
================================================================
Consolidates all improvements:
1. Cross-Attention (from Phase 2)
2. Distribution-Aware Huber Loss (sqrt-frequency weights)
3. Biased Initialization (Bias=5.0 to prevent uncertainty trap)
4. Delayed Uncertainty Weighting (Epochs 4+)
5. Fixed SEP token indexing for inference
"""

import os, json, math, logging, argparse, random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from arousal_features import extract_features_batch, ArousalFeatureNorm
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--domain", type=str, default="restaurant", choices=["restaurant","laptop"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--label_noise_std", type=float, default=0.1)
    parser.add_argument("--hard_mining_start", type=int, default=3)
    return parser.parse_args()

# ── Model ───────────────────────────────────────────────────────────────────
class DeBERTaV3Phase3(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Cross-Attention: Aspect as Query, Text as Key/Value
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        # Arousal Feature Norm
        self.arousal_norm = ArousalFeatureNorm(8)
        
        # Fusion LayerNorm
        fused_dim = hidden_size * 3 + 8 # CLS + CrossAttn + PooledAsp + ArousalFeats
        self.fusion_norm = nn.LayerNorm(fused_dim)
        
        # Regression Head (V, A)
        self.va_head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 2)
        )
        
        # Biased Initialization: targets are [1, 9], so bias to 5.0
        nn.init.constant_(self.va_head[-1].bias, 5.0)
        
        # Multi-task classification headers (Polarity 3, Arousal-Intensity 5)
        self.pol_head = nn.Linear(fused_dim, 3)
        self.aro_head = nn.Linear(fused_dim, 5)
        
        # Uncertainty weights (learnable log-variances)
        self.log_var_v = nn.Parameter(torch.zeros(1))
        self.log_var_a = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, sep1, sep2, arousal_feats):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state # (B, L, 768)
        batch_size = input_ids.size(0)
        
        cls_hidden = last_hidden[:, 0, :] # (B, 768)
        
        cross_attn_outputs = []
        pooled_aspects = []
        
        for i in range(batch_size):
            s1_idx, s2_idx = sep1[i].item(), sep2[i].item()
            
            # Slice text (between [CLS] and first [SEP]) and aspect (between SEPs)
            text_hs = last_hidden[i, 1:s1_idx, :].unsqueeze(0) # (1, L_t, 768)
            aspect_hs = last_hidden[i, s1_idx+1:s2_idx, :].unsqueeze(0) # (1, L_a, 768)
            
            # Pool aspect
            asp_query = aspect_hs.mean(dim=1, keepdim=True) # (1, 1, 768)
            pooled_aspects.append(asp_query.squeeze(1))
            
            # Cross-attention: Query=AspectPool, Key/Value=Text
            attn_out, _ = self.cross_attn(query=asp_query, key=text_hs, value=text_hs)
            cross_attn_outputs.append(attn_out.squeeze(1)) # (1, 768)
            
        cross_attn_out = torch.cat(cross_attn_outputs, dim=0) # (B, 768)
        pooled_asp = torch.cat(pooled_aspects, dim=0) # (B, 768)
        
        # 8-dim arousal features
        aro_normed = self.arousal_norm(arousal_feats) # (B, 8)
        
        # Combine
        fused = torch.cat([cls_hidden, cross_attn_out, pooled_asp, aro_normed], dim=-1)
        fused = self.fusion_norm(fused)
        
        va_pred = self.va_head(fused)
        pol_logits = self.pol_head(fused)
        aro_logits = self.aro_head(fused)
        
        return va_pred, pol_logits, aro_logits

def compute_loss(va_pred, va_gold, pol_logits, pol_soft, aro_logits, aro_soft, 
                 log_var_v, log_var_a, sample_weights, use_uncertainty=True):
    
    # Bucketized Sample Weights for Huber
    # va_gold: (B, 2)
    # sample_weights: dict with 'v' and 'a' lists of length 9
    
    # 1. Regression (Huber Loss)
    huber = nn.HuberLoss(delta=1.0, reduction='none')
    
    loss_v_raw = huber(va_pred[:, 0], va_gold[:, 0])
    loss_a_raw = huber(va_pred[:, 1], va_gold[:, 1])
    
    # Apply freq weights based on gold bucket
    bw_v, bw_a = [], []
    for i in range(va_gold.size(0)):
        bv = int(torch.round(va_gold[i, 0]).item()) - 1
        ba = int(torch.round(va_gold[i, 1]).item()) - 1
        bw_v.append(sample_weights["v"][max(0, min(8, bv))])
        bw_a.append(sample_weights["a"][max(0, min(8, ba))])
    
    w_v = torch.tensor(bw_v, device=va_pred.device)
    w_a = torch.tensor(bw_a, device=va_pred.device)
    
    loss_v = (loss_v_raw * w_v).mean()
    loss_a = (loss_a_raw * w_a).mean()
    
    # Multi-task
    loss_pol = -(pol_soft * F.log_softmax(pol_logits, dim=-1)).sum(dim=-1).mean()
    loss_aro = -(aro_soft * F.log_softmax(aro_logits, dim=-1)).sum(dim=-1).mean()

    # Uncertainty weighting
    if use_uncertainty:
        weighted_v = loss_v * torch.exp(-log_var_v) + log_var_v
        weighted_a = loss_a * torch.exp(-log_var_a) + log_var_a
        l_total = weighted_v + weighted_a + 0.1 * (loss_pol + loss_aro)
    else:
        l_total = loss_v + loss_a + 0.1 * (loss_pol + loss_aro)
        weighted_v, weighted_a = loss_v, loss_a
        
    return l_total, weighted_v, weighted_a, loss_pol, loss_aro

# ── Data ───────────────────────────────────────────────────────────────────
class Phase3Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=160):
        self.data, self.tokenizer, self.max_len = data, tokenizer, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        enc = self.tokenizer(s["text"], s["aspect"], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        
        # Identify SEPs for Cross-Attention
        sep_idxs = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        s1, s2 = sep_idxs[0].item(), sep_idxs[1].item()
        
        # Soft Labels for Multi-task
        v, a = s["v"], s["a"]
        p_neg = torch.sigmoid(torch.tensor(-(v - 4.0)))
        p_pos = torch.sigmoid(torch.tensor(v - 6.0))
        p_neu = 1.0 - p_neg - p_pos
        pol_soft = torch.tensor([p_neg, p_neu, p_pos])
        
        aro_soft = torch.zeros(5)
        # Intensity bins: [1-2, 3-4, 5-6, 7-8, 9] mapped to 0-4
        bin_idx = min(4, int((a - 1) // 2))
        aro_soft[bin_idx] = 1.0 # Simple hard label for intensity for now or smooth it
        
        # Linguistic features (Bbatch extraction)
        aro_feats = extract_features_batch([s["text"]]).squeeze(0)
        
        return {
            "input_ids": input_ids, "attention_mask": mask, "s1": s1, "s2": s2,
            "v": torch.tensor(v, dtype=torch.float), "a": torch.tensor(a, dtype=torch.float),
            "pol_soft": pol_soft, "aro_soft": aro_soft, "aro_feats": aro_feats,
            "text": s["text"], "aspect": s["aspect"], "id": s["id"]
        }

def load_data(domain):
    path = f"task-dataset/track_a/subtask_1/eng/eng_{domain}_train_alltasks.jsonl"
    raw = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    data = []
    for e in raw:
        asps = e.get("Aspect_VA", []) or e.get("Quadruplet", [])
        for asp in asps:
            v, a = map(float, asp["VA"].split("#"))
            data.append({"id": e["ID"], "text": e["Text"], "aspect": asp["Aspect"], "v": v, "a": a})
    return data

def get_freq_weights(data):
    v_c = np.zeros(10); a_c = np.zeros(10)
    for s in data:
        v_c[int(round(s["v"]))] += 1; a_c[int(round(s["a"]))] += 1
    def get_w(c):
        c = c[1:10]; f = c / (c.sum() + 1e-6); w = 1.0 / (np.sqrt(f) + 1e-6)
        w = w / (w.mean() + 1e-6); w = np.clip(w, 0.5, 3.0)
        return [float(x) for x in w]
    return {"v": get_w(v_c), "a": get_w(a_c)}

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"logs/v3_p3_{args.domain}_{ts}.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    train_data = load_data(args.domain)
    # Dev data loading
    dev_path = f"task-dataset/track_a/subtask_1/eng/eng_{args.domain}_dev_task1.jsonl"
    dev_raw = [json.loads(l) for l in open(dev_path, "r", encoding="utf-8")]
    dev_list = []
    for e in dev_raw:
        for asp in e["Aspect_VA"]:
            v, a = map(float, asp["VA"].split("#"))
            dev_list.append({"id": e["ID"], "text": e["Text"], "aspect": asp["Aspect"], "v": v, "a": a})
            
    sample_weights = get_freq_weights(train_data)
    log.info(f"Freq Weights V: {sample_weights['v']}")
    log.info(f"Freq Weights A: {sample_weights['a']}")

    train_ds = Phase3Dataset(train_data, tokenizer)
    dev_ds = Phase3Dataset(dev_list, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size)

    model = DeBERTaV3Phase3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps=int(len(train_dl)*args.epochs*args.warmup_ratio), 
                num_training_steps=len(train_dl)*args.epochs)
    
    scaler = torch.amp.GradScaler('cuda')
    best_rmse = 9.9
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0, 0
        use_uncertainty = (epoch >= 3) # Delayed uncertainty weighting
        
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            s1, s2 = batch["s1"].to(device), batch["s2"].to(device)
            v_gold, a_gold = batch["v"].to(device), batch["a"].to(device)
            va_gold = torch.stack([v_gold, a_gold], dim=1)
            pol_soft, aro_soft = batch["pol_soft"].to(device), batch["aro_soft"].to(device)
            aro_feats = batch["aro_feats"].to(device)
            
            with torch.amp.autocast('cuda'):
                vp, pl, al = model(ids, mask, s1, s2, aro_feats)
                loss, mv, ma, lp, la = compute_loss(vp, va_gold, pl, pol_soft, al, aro_soft,
                                                    model.log_var_v, model.log_var_a, 
                                                    sample_weights, use_uncertainty)
                loss = loss / args.accumulation_steps
            
            scaler.scale(loss).backward()
            if (i+1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            total_loss += loss.item() * args.accumulation_steps
            n_batches += 1
            if i % 50 == 0:
                log.info(f"EP {epoch} | B {i} | Loss: {loss.item():.4f} | σV: {torch.exp(0.5*model.log_var_v).item():.3f} | σA: {torch.exp(0.5*model.log_var_a).item():.3f}")

        # Eval
        model.eval()
        rmse_sum, n_total = 0, 0
        with torch.no_grad():
            for batch in dev_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                s1, s2 = batch["s1"].to(device), batch["s2"].to(device)
                aro_feats = batch["aro_feats"].to(device)
                v_g, a_g = batch["v"].to(device), batch["a"].to(device)
                va_gold = torch.stack([v_g, a_g], dim=1)
                
                vp, _, _ = model(ids, mask, s1, s2, aro_feats)
                # Official competition RMSE is sqrt(avg squared error over all dim)
                # But here we sum squared errors first
                sq_err = (vp.clamp(1,9) - va_gold)**2
                rmse_sum += sq_err.sum().item()
                n_total += len(v_g)
        
        # Matches official metric: sqrt(sum_squared_errors / (N_samples * 2_dimensions)) * sqrt(2)
        # Wait, official is sqrt(avg_MSE_v + avg_MSE_a) = sqrt(2 * avg_MSE_all)
        rmse = math.sqrt(rmse_sum / n_total) # This is sqrt( (sum_sq_v + sum_sq_a)/N )
        log.info(f"EP {epoch} DEV RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"checkpoints/best_v3_p3_{args.domain}.pt")
            log.info(f"  ★ New Best RMSE!")

    # Final Prediction for Dev Calibration
    model.load_state_dict(torch.load(f"checkpoints/best_v3_p3_{args.domain}.pt", weights_only=True))
    model.eval()
    predictions = []
    dev_data_raw = [json.loads(l) for l in open(dev_path, "r", encoding="utf-8")]
    for e in dev_data_raw:
        pe = {"ID": e["ID"], "Text": e["Text"], "Aspect_VA": []}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=160, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            s_idxs = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[1]
            s1, s2 = s_idxs[0].item(), s_idxs[1].item()
            af = extract_features_batch([e["Text"]]).to(device)
            with torch.no_grad():
                vp, _, _ = model(input_ids, mask, torch.tensor([s1]).to(device), torch.tensor([s2]).to(device), af)
            v, a = vp[0, 0].clamp(1,9).item(), vp[0, 1].clamp(1,9).item()
            pe["Aspect_VA"].append({"Aspect": asp["Aspect"], "VA": f"{v:.2f}#{a:.2f}"})
        predictions.append(pe)
    
    pred_path = os.path.join("predictions", f"v3_p3_task1_{args.domain}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions: f.write(json.dumps(p) + "\n")
    log.info(f"Phase 3 Dev Predictions saved: {pred_path}")

if __name__ == "__main__":
    main()
