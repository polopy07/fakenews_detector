from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 저장된 모델 경로
model_path = "./best_model"

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# GPU 있으면 GPU로 옮기기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def predict_fake_news(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return {
        "label": pred,
        "confidence": float(probs[0][pred]),
        "result": "📰 진짜 뉴스" if pred == 0 else "⚠️ 가짜 뉴스"
    }
text = "정부는 오늘 긴급재난지원금을 추가로 지급하기로 발표했다."
result = predict_fake_news(text)
print("예측 결과:", result)
