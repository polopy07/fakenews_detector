import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    FunnelTokenizer,
    FunnelModel,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
from tqdm.auto import tqdm
import time

# FunnelModel 기반 분류 모델 정의
class FunnelForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name="kykim/funnel-kor-base", num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        
        # Funnel Transformer 로드
        self.funnel = FunnelModel.from_pretrained(model_name)
        
        # 드롭아웃 추가
        self.dropout = torch.nn.Dropout(0.3)
        
        # 분류기 레이어
        self.classifier = torch.nn.Linear(self.funnel.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Funnel-Transformer 수행
        outputs = self.funnel(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] 토큰 표현 사용
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # 드롭아웃 및 분류
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 손실 계산
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }

# 데이터셋 클래스 정의
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 학습 및 평가 기능
def train_and_evaluate(train_dataset, val_dataset, tokenizer, model_name="kykim/funnel-kor-base", batch_size=16, epochs=3):
    """
    Funnel-Transformer 모델 학습 및 평가 함수
    
    최적 하이퍼파라미터:
    - 학습률: 5e-5
    - 배치크기: 16
    - 에폭: 3
    - 드롭아웃: 0.3
    - 옵티마이저: AdamW (weight_decay=0.01)
    - 스케줄러: Linear warmup (10%)
    """
    # 모델 생성
    model = FunnelForSequenceClassification(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # DataLoader 설정
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True
    )
    
    # 옵티마이저 설정
    optimizer = AdamW(
        model.parameters(), 
        lr=5e-5,
        weight_decay=0.01
    )
    
    # 스케줄러 설정
    num_training_steps = epochs * len(train_dataloader)
    warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 학습 루프
    best_f1 = 0.0
    
    for epoch in range(epochs):
        # 학습
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_dataloader)
        
        # 평가
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
                
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].cpu().numpy())
        
        val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Epoch {epoch+1}/{epochs} 결과:")
        print(f"  학습 손실: {train_loss:.4f}")
        print(f"  검증 손실: {val_loss:.4f}")
        print(f"  정확도: {accuracy:.4f}")
        print(f"  F1 점수: {f1:.4f}")
        
        # 최고 성능 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            print("  New best F1 score! Saving model...")
    
    return model, best_f1, accuracy



print("\n✅ Funnel-Transformer 모델 구현 완료")
