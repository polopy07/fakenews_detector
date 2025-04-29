import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
     AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 1. 데이터 불러오기
with open('json파일 경로', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# 2. 데이터 분리 (train/val/test)
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. 토크나이저 및 데이터셋 정의
model_name = 'skt/kobert-base-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)

# 4. 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. 평가 지표 함수 정의
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 6. 트레이닝 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_total_limit=1
)

# 7. Trainer 정의 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# 8. 테스트셋 성능 확인
predictions = trainer.predict(test_dataset)
print("📊 테스트셋 정확도:", predictions.metrics["test_accuracy"])

# 9. 모델 저장
model.save_pretrained('./best_model')
tokenizer.save_pretrained('./best_model')

print("✅ KoBERT 학습 완료 및 저장 완료!")
