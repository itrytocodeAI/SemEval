import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tqdm

class DeBERTaV3BIO(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

def evaluate_extractor(domain):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    label_map = {'O': 0, 'B-ASP': 1, 'I-ASP': 2, 'B-OPN': 3, 'I-OPN': 4}
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Load Model
    model = DeBERTaV3BIO('microsoft/deberta-v3-base', 5).to(device)
    model.load_state_dict(torch.load(f'checkpoints/extractor_{domain}.pt', map_location=device, weights_only=True))
    model.eval()
    
    # Load Data (Dev set for Task 2/3 extraction)
    # Using Subtask 2 dev set to evaluate boundaries if available
    test_file = f'task-dataset/track_a/subtask_2/eng/eng_{domain}_dev_task2.jsonl'
    if not os.path.exists(test_file):
        test_file = f'task-dataset/track_a/subtask_1/eng/eng_{domain}_dev_task1.jsonl'
        if not os.path.exists(test_file):
            print(f"Skipping {domain}, dev file not found.")
            return None
            
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    all_preds = []
    all_golds = []
    
    for item in tqdm.tqdm(data, desc=f"Evaluating {domain}"):
        text = item['Text']
        
        # We need to construct gold tags if testing on Subtask 2 annotations
        # If testing blindly we might just use Subtask 1 dev.
        # But this script is mostly for XAI / confusion matrix for whatever we can align
        # Since exact gold offsets are complex, we'll just run inference and report stats
        # To get a confusion matrix, we need gold tags. 
        # Let's align Subtask 2 'Quadruplet'/'Aspect' fields to BIO if present.
        
        target_list = item.get('Quadruplet', item.get('Aspect_VA', []))
        
        tokens = tokenizer.tokenize(text)
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', padding=False, truncation=True)
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        offsets = enc['offset_mapping'][0].tolist()
        
        gold_labels = [0] * len(ids[0])
        for t in target_list:
            asp = t.get('Aspect', 'NULL')
            opn = t.get('Opinion', 'NULL')
            
            # Simple soft matching for gold construction
            for target_str, label_prefix in [(asp, 'ASP'), (opn, 'OPN')]:
                if target_str == 'NULL': continue
                
                start_idx = text.find(target_str)
                if start_idx != -1:
                    end_idx = start_idx + len(target_str)
                    
                    found_start = False
                    for i, (tok_start, tok_end) in enumerate(offsets):
                        if tok_start == tok_end: continue # special token
                        
                        if tok_start >= start_idx and tok_end <= end_idx:
                            if not found_start:
                                gold_labels[i] = label_map[f'B-{label_prefix}']
                                found_start = True
                            else:
                                gold_labels[i] = label_map[f'I-{label_prefix}']

        with torch.no_grad():
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
        # exclude special tokens
        for i, (ts, te) in enumerate(offsets):
            if ts != te:
                all_preds.append(preds[i])
                all_golds.append(gold_labels[i])
                
    cm = confusion_matrix(all_golds, all_preds, labels=[0, 1, 2, 3, 4])
    report = classification_report(all_golds, all_preds, labels=[0, 1, 2, 3, 4], target_names=['O', 'B-ASP', 'I-ASP', 'B-OPN', 'I-OPN'], output_dict=True)
    
    os.makedirs('plots/xai', exist_ok=True)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['O', 'B-ASP', 'I-ASP', 'B-OPN', 'I-OPN'], yticklabels=['O', 'B-ASP', 'I-ASP', 'B-OPN', 'I-OPN'])
    plt.title(f'BIO Extraction Confusion Matrix ({domain.capitalize()})')
    plt.xlabel('Predicted')
    plt.ylabel('Gold')
    plt.tight_layout()
    plt.savefig(f'plots/xai/bio_confusion_matrix_{domain}.png', dpi=300)
    plt.close()
    
    # Save CSV metrics
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'logs/extractor_metrics_{domain}.csv')
    
    print(f"[{domain}] Extractor Metrics Saved:")
    print(df)
    
    return report

if __name__ == "__main__":
    evaluate_extractor('restaurant')
    evaluate_extractor('laptop')
