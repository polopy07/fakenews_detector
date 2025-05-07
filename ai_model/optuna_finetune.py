import json
import torch
import random
import optuna
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# ✅ 시드 고정
set_seed(42)

# ✅ 데이터 로드
with open("C:/Users/배승환/OneDrive/바탕 화면/news_dataset/fake_news_dataset_all_rewritten.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
texts = [item["text"] for item in data if item["text"]]
labels = [item["label"] for item in data if item["text"]]

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

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

# ✅ 클래스 가중치 계산
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = [total / class_counts[i] for i in range(2)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ✅ 커스텀 Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ✅ 메트릭
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ✅ Optuna 목적 함수
def objective(trial):
    # 하이퍼파라미터 샘플링
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    epochs = trial.suggest_int("epochs", 3, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # 모델 및 토크나이저 로딩
    model_name = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 데이터셋 생성
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)

    # 학습 설정
    training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    logging_dir="./logs",
    disable_tqdm=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)


    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1"]

# ✅ 튜닝 시작
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ✅ 최적 조합 출력
print("🎯 Best hyperparameters:", study.best_params)
print("🏆 Best F1-score:", study.best_value)
