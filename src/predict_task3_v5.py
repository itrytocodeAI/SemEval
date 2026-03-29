import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import tqdm
from arousal_features import extract_features_batch
from custom_model_v3_phase3 import DeBERTaV3Phase3, ArousalFeatureNorm # Need to import these to load the model

# Reuse the BIO model architecture
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

def get_spans(text, tokens, labels, inv_label_map, offsets):
    spans = []
    current_span = None
    
    for i, label_id in enumerate(labels):
        label = inv_label_map[label_id]
        if label.startswith('B-'):
            if current_span:
                spans.append(current_span)
            start, end = offsets[i]
            current_span = {'type': label[2:], 'start': start, 'end': end, 'text': text[start:end]}
        elif label.startswith('I-') and current_span and label[2:] == current_span['type']:
            _, end = offsets[i]
            current_span['end'] = end
            current_span['text'] = text[current_span['start']:end]
        else:
            if current_span:
                spans.append(current_span)
                current_span = None
    if current_span:
        spans.append(current_span)
    return spans

def pair_spans(asps, opns):
    pairs = []
    used_opns = set()
    
    for asp in asps:
        best_opn = None
        min_dist = float('inf')
        
        # Midpoint of aspect
        asp_mid = (asp['start'] + asp['end']) / 2
        
        for i, opn in enumerate(opns):
            opn_mid = (opn['start'] + opn['end']) / 2
            dist = abs(asp_mid - opn_mid)
            if dist < min_dist:
                min_dist = dist
                best_opn = opn
        
        if best_opn:
            pairs.append((asp['text'], best_opn['text']))
        else:
            pairs.append((asp['text'], 'NULL'))
            
    return pairs

def predict_task3(domain='restaurant'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    # Load Extractor
    extractor = DeBERTaV3BIO('microsoft/deberta-v3-base', 5).to(device)
    extractor.load_state_dict(torch.load(f'checkpoints/extractor_{domain}.pt', map_location=device))
    extractor.eval()
    
    # Load Regressor (V3 Phase 3)
    regressor = DeBERTaV3Phase3().to(device)
    regressor.load_state_dict(torch.load(f'checkpoints/best_v3_p3_{domain}.pt', map_location=device))
    regressor.eval()
    
    test_file = f'task-dataset/track_a/subtask_2/eng/eng_{domain}_test_task3.jsonl'
    if not os.path.exists(test_file):
        # Fallback to dev if test not available for testing the script
        test_file = f'task-dataset/track_a/subtask_2/eng/eng_{domain}_dev_task2.jsonl'
        
    output_file = f'predictions/v3_p5_task3_{domain}.jsonl'
    inv_label_map = {0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'}
    
    results = []
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Running Task 3 Inference for {domain}...")
    for line in tqdm.tqdm(lines):
        item = json.loads(line)
        text = item['Text']
        
        # Step 1: Extraction
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', padding=True, truncation=True)
        ids = encoding['input_ids'].to(device)
        mask = encoding['attention_mask'].to(device)
        offsets = encoding['offset_mapping'][0].tolist()
        
        with torch.no_grad():
            logits = extractor(ids, mask)
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
        spans = get_spans(text, None, preds, inv_label_map, offsets)
        asps = [s for s in spans if s['type'] == 'ASP']
        opns = [s for s in spans if s['type'] == 'OPN']
        
        pairs = pair_spans(asps, opns)
        
        # Step 2: Regression
        quads = []
        for aspect_text, opinion_text in pairs:
            # Format for Regressor: [CLS] text [SEP] aspect [SEP]
            enc_reg = tokenizer(text, aspect_text, max_length=160, padding='max_length', truncation=True, return_tensors='pt')
            ids_reg = enc_reg['input_ids'].to(device)
            mask_reg = enc_reg['attention_mask'].to(device)
            
            # Robust SEP search: DeBERTaV3 uses [CLS] text [SEP] aspect [SEP]
            sep_ids = (ids_reg[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_ids) >= 2:
                s1, s2 = sep_ids[0].item(), sep_ids[1].item()
            else:
                # Fallback if SEP missing (e.g. truncation extreme)
                s1, s2 = 80, 159 # Heuristic middle and end
            
            aro_feats = extract_features_batch([text]).to(device)
            
            with torch.no_grad():
                # Correctly pass batch-dim tensors
                s1_t, s2_t = torch.tensor([s1]).to(device), torch.tensor([s2]).to(device)
                va_pred, _, _ = regressor(ids_reg, mask_reg, s1_t, s2_t, aro_feats)
            
            v, a = va_pred[0, 0].clamp(1,9).item(), va_pred[0, 1].clamp(1,9).item()
            quads.append({
                "Aspect": aspect_text,
                "Opinion": opinion_text,
                "Category": "RESTAURANT#GENERAL", # Category prediction is Task 2 specific, defaulting for now
                "VA": f"{v:.2f}#{a:.2f}"
            })
            
        results.append({
            "ID": item["ID"],
            "Text": text,
            "Quadruplet": quads
        })
        
    os.makedirs('predictions', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    # Ensure dependencies are available (circular imports etc)
    # custom_model_v3_phase3 must be in the same dir
    predict_task3('restaurant')
    predict_task3('laptop')
