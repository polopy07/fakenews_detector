#최종수정
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Model,
    GPT2Config,
    GPT2PreTrainedModel,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import torch.nn as nn

# ✅ 1. 데이터 로딩
with open("fake_news_dataset_all_rewritten.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

texts = [item["text"] for item in data if item["text"]]
labels = [item["label"] for item in data if item["text"]]

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# ✅ 2. 정확한 KoGPT 토크나이저 사용
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    model_name,
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>'
)

# ✅ 3. Dataset 정의
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)

# ✅ 4. GPT2 기반 분류 모델 정의
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.hidden_size, 2)
        self.config = config

        # ✅ Trainer 오류 방지용
        self.model_parallel = False
        self.is_parallelizable = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, -1, :]
        logits = self.score(cls_token)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# ✅ 5. 모델 로딩
config = GPT2Config.from_pretrained(model_name)
config.pad_token_id = tokenizer.pad_token_id  # pad_token 적용
model = GPT2ForSequenceClassification.from_pretrained(model_name, config=config)

# ✅ 6. Trainer 설정
training_args = TrainingArguments(
    output_dir="./kogpt_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./kogpt_logs",
    load_best_model_at_end=True,
    save_strategy="epoch",
    metric_for_best_model="f1",
)

def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# ✅ 7. 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
