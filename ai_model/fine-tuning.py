import json
import random
import torch
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

# ✅ GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 디바이스:", device)

# ✅ 데이터 로드
with open("추가할 학습용 데이터터", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# ✅ train/val 분할
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# ✅ 모델 및 토크나이저 로드 (기존 모델 이어서 학습)
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)

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

# ✅ 메트릭
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="./results_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    logging_dir="./logs_finetune",
    logging_steps=10
)

# ✅ Trainer 정의 및 학습
trainer = Trainer(
    model=model,  # 이미 GPU에 올라간 모델
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ✅ 모델 저장
model.save_pretrained("./best_model_finetuned")
tokenizer.save_pretrained("./best_model_finetuned")

print("✅ 추가 학습 완료 (fine-tuned) 및 저장됨")
