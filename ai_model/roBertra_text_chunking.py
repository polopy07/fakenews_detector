import os
# 1. 클럭 제한 (선택 사항)
os.system("nvidia-smi -lgc 1000,1600")
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import copy

# 2. JSONL 파일 불러오기
file_path = "news_balanced_50k.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 3. train/eval split
train_data, eval_data = train_test_split(data, test_size=0.1, stratify=[item["label"] for item in data], random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "eval": Dataset.from_list(eval_data)
})

# 4. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", trust_remote_code=True)

# 한국어 문장 토큰화가 잘 안 될 경우, 아래처럼 간단한 규칙으로 대체 가능
def korean_sent_tokenize(text):
    return [s.strip() for s in text.strip().replace("?", ".").replace("!", ".").split(".") if s.strip()]

# ✅ 문장 기반 청크 생성 함수
def sentence_window_tokenize(text, max_tokens=512, min_tokens=50):
    sentences = korean_sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        tentative = current_chunk + sent + " "
        tokenized = tokenizer(tentative, truncation=False)
        length = len(tokenized["input_ids"])

        if length <= max_tokens:
            current_chunk = tentative
        else:
            if current_chunk:
                tokenized_current = tokenizer(current_chunk, truncation=False)
                if len(tokenized_current["input_ids"]) >= min_tokens:
                    chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    if current_chunk:
        tokenized_final = tokenizer(current_chunk, truncation=False)
        if len(tokenized_final["input_ids"]) >= min_tokens:
            chunks.append(current_chunk.strip())
    
    return chunks


# ✅ 수정된 토큰화 및 평탄화 함수
def tokenize_and_flatten(dataset_split):
    new_examples = {"input_ids": [], "attention_mask": [], "labels": [], "article_id": []}
    article_id_counter = 0

    for example in tqdm(dataset_split, desc="Tokenizing"):
        sentence_chunks = sentence_window_tokenize(example['text'], max_tokens=512)

        for chunk_text in sentence_chunks:
            tokens = tokenizer(chunk_text, padding='max_length', truncation=True, max_length=512)
            new_examples["input_ids"].append(tokens["input_ids"])
            new_examples["attention_mask"].append(tokens["attention_mask"])
            new_examples["labels"].append(example["label"])
            new_examples["article_id"].append(article_id_counter)

        article_id_counter += 1

    return Dataset.from_dict(new_examples)

# 7. 토큰화 적용
tokenized_dataset = DatasetDict({
    "train": tokenize_and_flatten(dataset["train"]),
    "eval": tokenize_and_flatten(dataset["eval"])
})

# ✅ 모델 정의
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=2, trust_remote_code=True)

# ✅ 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results_weights_v3",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    load_best_model_at_end=False,  # chunk_accuracy 기준 저장 비활성화
    metric_for_best_model=None,    # chunk 기준 저장도 끔
    logging_steps=50,
    fp16=False,
    gradient_accumulation_steps=2,
    no_cuda=False
)

# ✅ chunk 단위 평가 함수 (기본 용도)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"chunk_accuracy": accuracy}

# ✅ 콜백 클래스: 기사 단위 정확도 기준 저장
class SaveBestArticleModelCallback(TrainerCallback):
    def __init__(self, eval_dataset, save_path="best_article_model.pth"):
        self.eval_dataset = eval_dataset
        self.best_accuracy = 0.0
        self.save_path = save_path

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs['trainer']
        model = trainer.model

        raw_preds = trainer.predict(self.eval_dataset)
        logits = torch.tensor(raw_preds.predictions)
        labels = raw_preds.label_ids
        article_ids = self.eval_dataset["article_id"]

        article_logits = defaultdict(list)
        article_labels = {}
        for i in range(len(article_ids)):
            aid = article_ids[i]
            article_logits[aid].append(logits[i])
            article_labels[aid] = labels[i]

        correct = 0
        for aid, chunk_logits in article_logits.items():
            probs = torch.nn.functional.softmax(torch.stack(chunk_logits), dim=-1)
            weights = torch.linspace(1.0, 2.0, len(probs)).unsqueeze(1)
            weighted_probs = probs * weights
            avg_probs = weighted_probs.mean(dim=0)
            pred = torch.argmax(avg_probs).item()
            if pred == article_labels[aid]:
                correct += 1

        article_accuracy = correct / len(article_labels)
        print(f"✅ [콜백] 기사 단위 정확도: {article_accuracy:.4f}")

        if article_accuracy > self.best_accuracy:
            self.best_accuracy = article_accuracy
            torch.save(copy.deepcopy(model.state_dict()), self.save_path)
            print(f"📦 새 최고 기사 정확도 모델 저장됨: {self.save_path}")

# ✅ Trainer 정의 (콜백 포함)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestArticleModelCallback(tokenized_dataset["eval"], save_path="best_article_model.pth")]
)

# ✅ 이어서 학습 시작
print("🔁 학습 시작")
trainer.train(resume_from_checkpoint=True)
print("✅ 학습 완료")



# 1. 현재 모델로 청크 단위 예측 수행
raw_preds = trainer.predict(tokenized_dataset["eval"])
logits = torch.tensor(raw_preds.predictions)
labels = raw_preds.label_ids
article_ids = tokenized_dataset["eval"]["article_id"]

# 2. 기사 단위 정확도 계산
article_logits = defaultdict(list)
article_labels = {}

for i in range(len(article_ids)):
    aid = article_ids[i]
    article_logits[aid].append(logits[i])
    article_labels[aid] = labels[i]

correct = 0
for aid, chunk_logits in article_logits.items():
    probs = torch.nn.functional.softmax(torch.stack(chunk_logits), dim=-1)
    weights = torch.linspace(1.0, 2.0, len(probs)).unsqueeze(1)
    weighted_probs = probs * weights
    avg_probs = weighted_probs.mean(dim=0)
    pred = torch.argmax(avg_probs).item()
    if pred == article_labels[aid]:
        correct += 1

article_accuracy = correct / len(article_labels)
print(f"✅ 기사 단위 정확도: {article_accuracy:.4f}")

# 3. 가장 좋은 모델 저장 디렉토리 설정
BEST_MODEL_DIR = "./best_article_model"

# 4. 기사 단위 정확도가 기존보다 높으면 모델 저장
if not os.path.exists(BEST_MODEL_DIR):
    os.makedirs(BEST_MODEL_DIR)
    trainer.save_model(BEST_MODEL_DIR)
    print(f"📦 기사 단위 기준 베스트 모델 저장 (처음 저장): {BEST_MODEL_DIR}")

else:
    # 이미 저장된 모델이 있다면 비교해서 덮어쓸지 결정
    # 이 경우, 정확도 기록을 저장한 파일이 필요함 (예: best_article_accuracy.txt)
    acc_file = os.path.join(BEST_MODEL_DIR, "best_article_accuracy.txt")
    prev_best = 0.0
    if os.path.exists(acc_file):
        with open(acc_file, "r") as f:
            prev_best = float(f.read().strip())

    if article_accuracy > prev_best:
        trainer.save_model(BEST_MODEL_DIR)
        with open(acc_file, "w") as f:
            f.write(str(article_accuracy))
        print(f"✅ 기사 단위 기준으로 기존보다 정확도가 높아 새로 저장됨: {article_accuracy:.4f}")
    else:
        print(f"ℹ 기존 기사 단위 정확도({prev_best:.4f})보다 낮아 저장하지 않음.")