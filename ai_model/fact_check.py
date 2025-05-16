
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 1. 모델 및 토크나이저 로딩 (학습된 체크포인트 경로 사용)
model_path = ""
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2. 입력할 뉴스 기사 리스트
news_articles = [
      # 가짜 뉴스 예시
]

# 3. 토크나이징 및 모델 예측
inputs = tokenizer(news_articles, padding=True, truncation=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

# 4. 결과 출력
for i, article in enumerate(news_articles):
    label = "가짜 뉴스" if preds[i].item() == 1 else "진짜 뉴스"
    confidence = probs[i][preds[i]].item()
    print(f"\n기사 {i+1}:\n{article}\n→ 분류: {label} (신뢰도: {confidence:.2f})")


