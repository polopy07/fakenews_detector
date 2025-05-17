import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# ✅ 시드 고정
set_seed(42)

# ✅ 데이터 로드
with open("C:/Users/배승환/OneDrive/바탕 화면/news_dataset/fake_news_dataset_all_rewritten.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
texts = [item['text'] for item in data if item['text']]
labels = [item['label'] for item in data if item['text']]

# ✅ 데이터 분할
X_train, X_valtest, y_train, y_valtest = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

# ✅ 모델과 토크나이저
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ Dataset 정의
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)

# ✅ 클래스 가중치 계산
from collections import Counter
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = [total / class_counts[i] for i in range(2)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ✅ 커스텀 Trainer 정의
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ✅ 메트릭 정의
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50
)

# ✅ Trainer 실행
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ✅ 테스트셋 평가
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)

print("📊 테스트셋 정확도:", accuracy_score(y_test, y_pred))
print("📊 테스트셋 F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["진짜 뉴스", "가짜 뉴스"]))

# ✅ 모델 저장
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")
