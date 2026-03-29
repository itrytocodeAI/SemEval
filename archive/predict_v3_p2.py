"""
Standalone Prediction Script for V3 Phase 2
============================================
Loads a trained Cross-Attention model checkpoint and generates dev predictions.
"""

import json, os, sys, math, torch
from transformers import AutoTokenizer
from custom_model_v3_phase2 import DeBERTaV3CrossAttention, extract_features_batch

def get_device(use_cpu=False):
    if use_cpu: return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_NAME = "microsoft/deberta-v3-base"

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def fmt_va(v, a):
    return f"{max(1,min(9,v)):.2f}#{max(1,min(9,a)):.2f}"

def predict(domain, checkpoint_path, device):
    print(f"Generating predictions for {domain} using {checkpoint_path} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = DeBERTaV3CrossAttention().to(device)
    msg = model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Load info: {msg}")
    model.eval()
    
    data_dir = os.path.join(os.path.dirname(__file__), "task-dataset", "track_a", "subtask_1", "eng")
    dev_path = os.path.join(data_dir, f"eng_{domain}_dev_task1.jsonl")
    dev_data = load_jsonl(dev_path)
    
    predictions = []
    for e in dev_data:
        pe = {"ID": e["ID"], "Text": e["Text"], "Aspect_VA": []}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=160,
                           padding="max_length", truncation=True, return_tensors="pt")
            
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            
            # Since input_ids is (1, L), nonzero returns (row_idxs, col_idxs)
            # We want col_idxs where condition is true
            all_nonzero = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)
            sep_idx = all_nonzero[1] 
            
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

    out_path = os.path.join("predictions", f"v3_p2_task1_{domain}.jsonl")
    os.makedirs("predictions", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    device = get_device(args.cpu)
    predict(args.domain, args.ckpt, device)
