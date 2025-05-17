import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 1. 데이터 로딩
with open("C:/Users/배승환/OneDrive/바탕 화면/git/news_fake_detector/ai_model/all_news_with_fake_cleaned.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data if item['text']]
labels = [item['label'] for item in data if item['text']]

# 2. 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. 토크나이저 및 모델 로드
model_name = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Dataset 클래스 정의
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

# 5. 정확도 계산 함수 정의 (Trainer에 전달)
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_total_limit=1
)

# 7. Trainer 정의 (여기서 compute_metrics 전달이 핵심!)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 8. 학습 시작
trainer.train()

# 9. 테스트셋 평가
predictions = trainer.predict(test_dataset)
print("📊 테스트셋 메트릭:", predictions.metrics)
print("📊 테스트셋 정확도:", predictions.metrics.get("test_accuracy", "정확도 없음"))

# 10. 모델 저장
trainer.save_model('./best_model')             # ✅ Trainer가 관리하는 모델 저장
tokenizer.save_pretrained('./best_model')

print("✅ KoBERT 학습 및 저장 완료")

