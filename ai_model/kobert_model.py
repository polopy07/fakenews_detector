import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. 데이터 불러오기
with open('all_news_with_fake_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# 2. 데이터 분리
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. KoBERT 모델 및 토크나이저 사용
model_name = 'monologg/kobert'  # KoBERT 사전학습 모델

tokenizer = BertTokenizer.from_pretrained(model_name)

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

# 4. 모델 불러오기
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. 트레이닝 설정
training_args = TrainingArguments(
    output_dir='./results',          # 결과 저장 폴더
    evaluation_strategy='epoch',     # 에폭마다 검증
    save_strategy='epoch',           # 에폭마다 저장
    num_train_epochs=3,              # 학습할 에폭 수
    per_device_train_batch_size=8,   # 학습 배치 사이즈
    per_device_eval_batch_size=16,   # 검증 배치 사이즈
    logging_dir='./logs',             # 로그 저장 폴더
    logging_steps=10,
    load_best_model_at_end=True,      # 가장 좋은 모델 저장
    metric_for_best_model='accuracy', # 평가 기준
    save_total_limit=1                # 저장 모델 수 제한
)

# 6. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 7. 학습 시작
trainer.train()

# 8. 모델 저장
model.save_pretrained('./best_model')
tokenizer.save_pretrained('./best_model')

print("\u2705 KoBERT 학습 완료! 모델 저장됨.")