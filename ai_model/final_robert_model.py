import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# ✅ 1. 데이터 로드 (예시로 JSONL 파일을 사용)
import json
with open("/content/fake_news_dataset_all_rewritten.jsonl", "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f]

# 텍스트와 라벨 추출
texts = [item["text"] for item in data_list]
labels = [int(item["label"]) for item in data_list]  # 라벨은 반드시 int로 변환

# ✅ 2. Dataset 생성
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ✅ 3. 모델 및 토크나이저 로드
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ 4. 토큰화 함수 정의
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# ✅ 5. 평가 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1)
    }

# ✅ 6. 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./roberta_fake_news_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    seed=42,
    report_to="none"  # WandB 비활성화
)

# ✅ 7. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# ✅ 8. 모델 학습
print("🚀 모델 학습을 시작합니다...")
trainer.train()
print("✅ 모델 학습 완료!")

# ✅ 9. 모델 저장
model_dir = "./final_roberta_model"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

# ✅ 10. 모델 zip 압축
import shutil
shutil.make_archive("final_roberta_model", 'zip', model_dir)
print("✅ 모델이 'final_roberta_model.zip'으로 저장되었습니다. VSCode 등에서 다운로드해 사용하세요.")
