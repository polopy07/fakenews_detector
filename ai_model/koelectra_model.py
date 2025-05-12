import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# ✅ 설정
set_seed(42)
model_name = "monologg/koelectra-base-discriminator"

# ✅ 데이터 로드
with open("C:/Users/배승환/OneDrive/바탕 화면/git/news_fake_detector/dataset/processed/balanced_preprocessed_news.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# ✅ 토크나이저 및 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ Dataset 정의
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# ✅ 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)

# ✅ 메트릭 함수 정의
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./koelectra_model",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    save_total_limit=1,
    report_to="none",
    seed=42
)

# ✅ Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ✅ 학습 실행
trainer.train()

# ✅ 모델 저장
model.save_pretrained("./koelectra_model")
tokenizer.save_pretrained("./koelectra_model")
