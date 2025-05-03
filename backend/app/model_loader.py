import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "../best_model"  # ← backend/app 기준 상대경로
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

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
