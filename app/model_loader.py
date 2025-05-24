import html
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os

# 환경변수에서 모델 이름 가져오기
model_name = os.getenv("MODEL_NAME", "olopy/fakenews")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from_pretrained에 torch_dtype 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
model.to(device)
model.eval()

# 키워드 기반 간단 규칙
fake_keywords = [
    "음모론", "외계인", "좀비", "백신 사망", "기지 건설", "정부가 숨겼다",
    "조작된 자료", "무조건 죽는다", "극비 문서", "세계정부", "DNA 변형",
    "5G 감염", "불로장생", "암 치료", "미국이 지진 유발"
]
max_rule_score = len(fake_keywords)

def korean_sent_tokenize(text):
    return [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]

def sentence_window_tokenize(text, max_tokens=512, min_tokens=50):
    sentences = korean_sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sent in sentences:
        tentative = current_chunk + sent + " "
        if len(tokenizer(tentative, truncation=False)["input_ids"]) <= max_tokens:
            current_chunk = tentative
        else:
            if len(tokenizer(current_chunk)["input_ids"]) >= min_tokens:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    if len(tokenizer(current_chunk)["input_ids"]) >= min_tokens:
        chunks.append(current_chunk.strip())
    return chunks

def rule_based_score(text):
    return sum(1 for kw in fake_keywords if kw in text)


def predict_fake_news(text):
    clean_text = html.unescape(text).replace("\r", " ").replace("\n", " ").strip()

    rule_score = rule_based_score(clean_text)
    rule_score_norm = rule_score / max_rule_score

    chunks = sentence_window_tokenize(clean_text)
    chunk_logits = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            chunk_logits.append(logits.squeeze(0))

    if not chunk_logits:
        return {"error": "청크를 생성할 수 없습니다."}

    logits = torch.stack(chunk_logits)
    weights = torch.linspace(1.0, 2.0, len(logits)).unsqueeze(1).to(device)
    weighted_logits = logits * weights
    avg_logits = weighted_logits.mean(dim=0)
    avg_probs = softmax(avg_logits, dim=0)

    real_prob = float(avg_probs[0])
    fake_prob = float(avg_probs[1])
    final_score = 0.7 * fake_prob + 0.3 * rule_score_norm
    label = 1 if final_score >= 0.5 else 0

    result_msg = (
        "🔴 가짜 뉴스로 판단됨 (딥러닝 + 키워드 일치)" if label == 1 and rule_score >= 1 else
        "🔴 가짜 뉴스로 판단됨 (확률 기반)" if label == 1 else
        "⚠️ 판단 유보" if final_score >= 0.4 else
        "🟢 진짜 뉴스로 판단됨"
    )

    print("[DEBUG] 최종 label:", label, "| result:", result_msg)

    return {
        "label": label,
        "confidence": round(final_score, 4),
        "result": result_msg,
        "probabilities": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4)
        },
        "rule_score": rule_score
    }


