import html
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# 모델 경로
model_path = "C:/Users/sasha/OneDrive/Desktop/best_article_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 위험 키워드
fake_keywords = [
    "음모론", "외계인", "좀비", "백신 사망", "기지 건설", "정부가 숨겼다",
    "조작된 자료", "무조건 죽는다", "극비 문서", "세계정부", "DNA 변형",
    "5G 감염", "불로장생", "물 한 방울로 암 치료", "미국이 지진 유발"
]
max_rule_score = len(fake_keywords)

# 문장 단위로 나누기
def korean_sent_tokenize(text):
    return [s.strip() for s in text.strip().replace("?", ".").replace("!", ".").split(".") if s.strip()]

# 청크 분할 함수
def sentence_window_tokenize(text, tokenizer, max_tokens=512, min_tokens=50):
    sentences = korean_sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        tentative = current_chunk + sent + " "
        tokenized = tokenizer(tentative, truncation=False)
        if len(tokenized["input_ids"]) <= max_tokens:
            current_chunk = tentative
        else:
            if len(tokenizer(current_chunk)["input_ids"]) >= min_tokens:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    if len(tokenizer(current_chunk)["input_ids"]) >= min_tokens:
        chunks.append(current_chunk.strip())

    return chunks

# 키워드 기반 점수
def rule_based_score(text):
    return sum(1 for kw in fake_keywords if kw in text)

# 예측 함수
def predict_fake_news(text):
    clean_text = html.unescape(text).replace("\r", " ").replace("\n", " ").strip()
    rule_score = rule_based_score(clean_text)
    rule_score_norm = rule_score / max_rule_score

    chunks = sentence_window_tokenize(clean_text, tokenizer)
    chunk_logits = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            chunk_logits.append(output.logits.squeeze(0))

    if not chunk_logits:
        return {"error": "청크를 생성할 수 없습니다. 입력이 너무 짧거나 토큰화 실패."}

    # 기사 단위 softmax 평균 (가중치 포함)
    logits = torch.stack(chunk_logits)
    weights = torch.linspace(1.0, 2.0, len(logits)).unsqueeze(1).to(device)
    weighted_logits = logits * weights
    avg_logits = weighted_logits.mean(dim=0)
    avg_probs = torch.nn.functional.softmax(avg_logits, dim=0)
    
    real_prob = float(avg_probs[0])
    fake_prob = float(avg_probs[1])

    # 최종 하이브리드 점수
    final_score = 0.7 * fake_prob + 0.3 * rule_score_norm
    label = 1 if final_score >= 0.5 else 0

    # 메시지
    if label == 1 and rule_score >= 1:
        result = "🔴 가짜 뉴스로 판단됨 (딥러닝 + 키워드 일치)"
    elif label == 1:
        result = "🔴 가짜 뉴스로 판단됨 (확률 기반)"
    elif final_score >= 0.4:
        result = "⚠️ 판단 유보"
    else:
        result = "🟢 진짜 뉴스로 판단됨"

    print(f"[DEBUG] Softmax: {avg_probs.tolist()} | Rule score: {rule_score} | Hybrid score: {final_score:.4f}")
    print(f"[DEBUG] fake_prob: {fake_prob}, rule_score: {rule_score}, final_score: {final_score}, label: {label}")

    return {
        "label": label,
        "confidence": round(final_score, 4),
        "result": result,
        "probabilities": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4)
        },
        "rule_score": rule_score
    }
