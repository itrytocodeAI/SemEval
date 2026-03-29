import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    print("Successfully imported transformers components")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Configuration
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',
    'max_len': 128,
    'batch_size': 8,
    'epochs': 5,
    'lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'labels': {'O': 0, 'B-ASP': 1, 'I-ASP': 2, 'B-OPN': 3, 'I-OPN': 4},
    'inv_labels': {0: 'O', 1: 'B-ASP', 2: 'I-ASP', 3: 'B-OPN', 4: 'I-OPN'}
}

class BIODataset(Dataset):
    def __init__(self, file_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config['max_len']
        self.label_map = config['labels']
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['Text']
        
        # Tokenize with offsets
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        offsets = encoding['offset_mapping'][0].tolist()
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        # Initialize labels with O (0)
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        # Mark special tokens as -100 for CrossEntropy ignore_index
        for i, (start, end) in enumerate(offsets):
            if start == end == 0:
                labels[i] = -100
        
        # Find spans for Aspects and Opinions
        quads = item.get('Quadruplet', item.get('Aspect_VA', []))
        for quad in quads:
            asp = quad.get('Aspect', 'NULL')
            opn = quad.get('Opinion', 'NULL')
            
            if asp != 'NULL':
                self._mark_span(text, asp, labels, offsets, 'ASP')
            if opn != 'NULL':
                self._mark_span(text, opn, labels, offsets, 'OPN')
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _mark_span(self, text, span, labels, offsets, tag_type):
        start_char = text.find(span)
        if start_char == -1: return
        end_char = start_char + len(span)
        
        first = True
        for i, (s, e) in enumerate(offsets):
            if s == e == 0: continue # Skip special tokens
            
            # overlap logic
            if s >= start_char and e <= end_char:
                if first:
                    labels[i] = self.label_map[f'B-{tag_type}']
                    first = False
                else:
                    labels[i] = self.label_map[f'I-{tag_type}']

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

def train(domain='restaurant'):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    train_file = f'task-dataset/track_a/subtask_2/eng/eng_{domain}_train_alltasks.jsonl'
    dev_file = f'task-dataset/track_a/subtask_2/eng/eng_{domain}_dev_task2.jsonl'
    
    train_ds = BIODataset(train_file, tokenizer, CONFIG)
    dev_ds = BIODataset(dev_file, tokenizer, CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=CONFIG['batch_size'])
    
    model = DeBERTaV3BIO(CONFIG['model_name'], len(CONFIG['labels'])).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_f1 = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(CONFIG['device'])
            mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            logits = model(ids, mask)
            loss = criterion(logits.view(-1, 5), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        # Validation
        val_f1 = evaluate(model, dev_loader, CONFIG)
        print(f"Validation F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'checkpoints/extractor_{domain}.pt')
            print(f"Saved best model with F1: {val_f1:.4f}")

def evaluate(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(config['device'])
            mask = batch['attention_mask'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1)
            
            # Mask out special tokens
            mask_flat = (labels.view(-1) != -100)
            all_preds.extend(preds.view(-1)[mask_flat].cpu().numpy())
            all_labels.extend(labels.view(-1)[mask_flat].cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1

if __name__ == "__main__":
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    train('restaurant')
    train('laptop')
