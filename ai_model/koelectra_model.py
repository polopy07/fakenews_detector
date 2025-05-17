import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from collections import Counter
import random

# ✅ 설정
set_seed(42)
model_name = "monologg/koelectra-base-discriminator"

# ✅ 데이터 로드
with open("C:/Users/배승환/OneDrive/바탕 화면/git/news_fake_detector/dataset/processed/news_data_balanced_1to1.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
random.shuffle(data)
texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

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

# ✅ 클래스 가중치
def get_class_weights(y):
    class_counts = Counter(y)
    total = sum(class_counts.values())
    return torch.tensor([total / class_counts[i] for i in range(2)], dtype=torch.float)

# ✅ WeightedTrainer 정의
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ✅ 메트릭 함수
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ✅ 학습 데이터 준비
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)
class_weights = get_class_weights(y_train)

# ✅ Trial 0번 최적 파라미터로 학습 설정
training_args = TrainingArguments(
    output_dir="./kloectra_model",
    learning_rate=1.455456922178936e-05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.08826005668801223,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    save_total_limit=1,
    report_to="none",
    seed=42,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# ✅ 학습 시작
trainer.train()

# ✅ 평가
results = trainer.evaluate()
print("📊 Final Eval Results:", results)

# ✅ 혼동 행렬 시각화
pred_output = trainer.predict(val_dataset)
y_pred = pred_output.predictions.argmax(-1)
y_true = pred_output.label_ids
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (0)", "Fake (1)"]).plot(
    cmap="Blues", values_format="d"
)
plt.title("Confusion Matrix (Trial 0 Val Set)")
plt.grid(False)
plt.show()

# ✅ 저장
model.save_pretrained("./kloectra_model")
tokenizer.save_pretrained("./kloectra_model")

