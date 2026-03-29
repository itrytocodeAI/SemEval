import torch, json, os, logging
from transformers import AutoTokenizer
from custom_model_v3_phase3 import DeBERTaV3Phase3
from arousal_features import extract_features_batch

import torch, json, os, argparse
from transformers import AutoTokenizer
from custom_model_v3_phase3 import DeBERTaV3Phase3
from arousal_features import extract_features_batch

def predict(domain):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = DeBERTaV3Phase3().to(device)
    
    ckpt_path = f"checkpoints/best_v3_p3_{domain}.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    dev_path = f"task-dataset/track_a/subtask_1/eng/eng_{domain}_dev_task1.jsonl"
    if not os.path.exists(dev_path):
        print(f"ERROR: Dev file {dev_path} not found.")
        return
        
    data = [json.loads(l) for l in open(dev_path, "r", encoding="utf-8")]
    
    predictions = []
    print(f"Running inference on {domain} dev set...")
    for i, e in enumerate(data):
        if i % 50 == 0: print(f"Processing {i}/{len(data)}...")
        pe = {"ID": e["ID"], "Text": e["Text"], "Aspect_VA": []}
        for asp in e["Aspect_VA"]:
            enc = tokenizer(e["Text"], asp["Aspect"], max_length=160, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"]
            mask = enc["attention_mask"]
            
            # Find SEP indices for index slicing
            s_idxs = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[1]
            s1, s2 = s_idxs[0].item(), s_idxs[1].item()
            
            af = extract_features_batch([e["Text"]])
            with torch.no_grad():
                vp, _, _ = model(input_ids, mask, torch.tensor([s1]), torch.tensor([s2]), af)
            
            v, a = vp[0, 0].clamp(1,9).item(), vp[0, 1].clamp(1,9).item()
            pe["Aspect_VA"].append({"Aspect": asp["Aspect"], "VA": f"{v:.2f}#{a:.2f}"})
        predictions.append(pe)
        
    out_dir = "predictions"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/v3_p3_task1_{domain}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions: f.write(json.dumps(p) + "\n")
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="laptop", choices=["restaurant", "laptop"])
    args = parser.parse_args()
    predict(args.domain)

