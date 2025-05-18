import html
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 경로 설정
model_path = "C:/Users/sasha/OneDrive/Desktop/best_article_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 가짜뉴스 관련 위험 키워드
fake_keywords = [
    "음모론", "외계인", "좀비", "백신 사망", "기지 건설", "정부가 숨겼다",
    "조작된 자료", "무조건 죽는다", "극비 문서", "세계정부", "DNA 변형",
    "5G 감염", "불로장생", "물 한 방울로 암 치료", "미국이 지진 유발"
]

max_rule_score = len(fake_keywords)  # 정규화에 사용

def rule_based_score(text):
    return sum(1 for kw in fake_keywords if kw in text)

def predict_fake_news(text):
    # 텍스트 정리
    try:
        clean_text = html.unescape(text).replace("\r", " ").replace("\n", " ").strip()
    except Exception as e:
        print(f"[ERROR] 텍스트 정리 실패: {e}")
        clean_text = text

    # Rule-based 점수 계산
    rule_score = rule_based_score(clean_text)
    rule_score_norm = rule_score / max_rule_score  # 0~1로 정규화

    # 모델 예측
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        real_prob = float(probs[0][0])
        fake_prob = float(probs[0][1])

    # 앙상블 점수 계산
    final_score = 0.7 * fake_prob + 0.3 * rule_score_norm
    label = 1 if final_score >= 0.5 else 0

    # 결과 메시지
    if label == 1 and rule_score >= 1:
        result = "🔴 가짜 뉴스로 판단됨 (딥러닝 + 키워드 일치)"
    elif label == 1:
        result = "🔴 가짜 뉴스로 판단됨 (확률 기반)"
    elif final_score >= 0.4:
        result = "⚠️ 판단 유보"
    else:
        result = "🟢 진짜 뉴스로 판단됨"

    print(f"[DEBUG] Softmax: {probs.tolist()} | Rule score: {rule_score} | Hybrid score: {final_score:.4f}")

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
